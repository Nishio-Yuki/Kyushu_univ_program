# ============================================================
# 1) パラメータ（ここだけ変更すればOK）
# ============================================================
INPUT_DIR        = "/home/pj25000162/ku50001814/scRNA-seq_data/Adrenal_scRNA-seq_MS"      # .h5 / .h5ad が入っているフォルダ
OUTPUT_DIR       = "/home/pj25000162/ku50001814/scRNA-seq_data/Adrenal_scRNA-seq_MS/output_arrow_test"  # save_to_disk 先
VAR_GENEKEY      = "gene_ids"          # var の最終キー（例: 'gene_symbol','gene','features','gene_ids','ensembl_id'）
USE_RAW          = True                # geneの抽出・順位付けに raw を使うか

# 既存: symbol<->ensembl マッピング（任意）
MAPPING_PICKLE   = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"

# 追加: Ensembl ID → 整数ID の対応表（学習のボキャブラリ）。必須。
# 例: { "ENSMUSG00000017167": 1234, "ENSMUSG00000000001": 42, ... }
ENSEMBL2ID_PATH  = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1.pkl"

# 出力列を「2カラム（input_ids, length）」のまま保つ
# = True にすると rank_values も書き出す（3カラム目）
SAVE_RVE_COLUMN  = False

# Rank Value Encoding のモード（順序は常に「発現高→低」）
# "none": 値は保存しない（順序のみ使用） / "linear": 1.0..(降順で)→0.0 へ線形
# "log": 線形より上位を強調（1/log(rank+1) を正規化）
RVE_MODE         = "linear"  # "none" | "linear" | "log"
MAX_LEN          = 2048      # 例: 2048 など。None で全非ゼロ遺伝子
MIN_LEN          = 1         # 0 のセルはスキップ

# QC モード: "basic" か "sigma"
QC_MODE = "basic"
QC_MIN_GENES         = 200
QC_MAX_GENES         = None
QC_MIN_COUNTS        = None
QC_MAX_COUNTS        = None
QC_MAX_MITO_FRAC     = 0.20
QC_MIN_CELLS_PER_GENE= 3
QC_TOTAL_CAP         = 20000
QC_SIGMA_K           = 3.0

# 表示
LIST_KEYS       = True
LIST_KEYS_TOPN  = 30

# ======== 遺伝子条件（アクション付き）========
# 形式: {"gene": str, "op": "<=|<|>=|>|==|!=", "value": float, "action": "keep|delete", [optional] "key": "gene_symbol|ensembl_id|gene_ids|symbol"}
# 例: A>=10 をキープ, B<=6 を削除
GENE_EXPR_CONSTRAINTS = [
     {"gene": "Cyp11b1", "op": ">=", "value": 8, "action": "keep"},
    # {"gene": "Lyz2",  "op": "<=", "value": 6,  "action": "delete"},
]
GENE_MATCH_CASE_SENSITIVE = False  # False なら大文字小文字を無視

# ============================================================
# 2) 実装（トークナイズ版）
# ============================================================
import os, glob, sys, math, pickle, time, json, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

_t0 = time.perf_counter()
def log(msg: str):
    now = time.perf_counter() - _t0
    print(f"[{now:8.2f}s] {msg}")
    sys.stdout.flush()

CANDIDATE_GENE_KEYS = [
    "gene_symbol", "gene", "features", "feature_name",
    "symbol", "Gene", "gene_id", "gene_ids", "ensembl_id", "Ensembl",
]

def _preview_list(name: str, items, topn: int):
    items = list(items) if items is not None else []
    n = len(items); head = items[:topn]
    log(f"  - {name}: count={n}")
    if n: log(f"    head({min(topn,n)}): {head}")

def summarize_adata_keys(adata: ad.AnnData, topn: int = 30):
    X = adata.X
    x_type = type(X).__name__
    if sp.issparse(X):
        x_type += f" (format={'CSC' if sp.isspmatrix_csc(X) else 'CSR' if sp.isspmatrix_csr(X) else 'sparse'})"
    log("=== AnnData KEY SUMMARY ===")
    log(f"shape: cells={adata.n_obs}, genes={adata.n_vars}, X={x_type}, dtype={getattr(X,'dtype',getattr(getattr(X,'data',None),'dtype',None))}")
    _preview_list("obs.columns", adata.obs.columns, topn)
    _preview_list("var.columns", adata.var.columns, topn)
    if adata.raw is not None:
        try: _preview_list("raw.var.columns", adata.raw.var.columns, topn)
        except Exception as e: log(f"  - raw.var.columns: <unavailable> ({e})")
    else:
        log("  - raw: None")
    for name, getter in [("layers", lambda: adata.layers.keys()),
                         ("obsm",   lambda: adata.obsm.keys()),
                         ("varm",   lambda: adata.varm.keys()),
                         ("uns",    lambda: adata.uns.keys())]:
        try: _preview_list(name, getter(), topn)
        except Exception as e: log(f"  - {name}: <unavailable> ({e})")
    gkeys_present = [k for k in CANDIDATE_GENE_KEYS if k in adata.var.columns]
    log(f"  - candidate gene keys present in var: {gkeys_present}")
    log("=== END SUMMARY ===")

# ---------- IO ----------
def _list_inputs(folder: str):
    h5    = sorted(glob.glob(os.path.join(folder, "*.h5")))
    h5ad  = sorted(glob.glob(os.path.join(folder, "*.h5ad")))
    if not h5 and not h5ad:
        raise FileNotFoundError(f"No .h5 or .h5ad files under {folder}")
    return h5, h5ad

def _normalize_indices(adata: ad.AnnData, prefer_var_key: Optional[str] = None, drop_empty: bool = True) -> ad.AnnData:
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    if prefer_var_key and (prefer_var_key in adata.var.columns):
        new_names = adata.var[prefer_var_key].astype(str).fillna("")
        if drop_empty:
            mask = new_names.values != ""
            if mask.sum() < len(new_names):
                adata = adata[:, mask].copy()
                new_names = new_names[mask]
        adata.var_names = new_names.values
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    return adata

def _load_one(path: str) -> ad.AnnData:
    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    else:
        try:
            adata = sc.read_10x_h5(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path} as 10x .h5; if it's an AnnData file, use .h5ad") from e
    pref_key = VAR_GENEKEY if 'VAR_GENEKEY' in globals() else None
    candidates = [k for k in [pref_key, "ensembl_id", "gene_ids", "gene_symbol", "gene", "features"] if k]
    for k in candidates:
        if k in adata.var.columns:
            adata = _normalize_indices(adata, prefer_var_key=k, drop_empty=True)
            break
    else:
        adata = _normalize_indices(adata, prefer_var_key=None, drop_empty=False)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata

def _concat_anndatas(paths: List[str]) -> ad.AnnData:
    ads = []
    for p in paths:
        adata = _load_one(p)
        adata.obs["__source__"] = os.path.basename(p)
        if not adata.var_names.is_unique:
            log(f"[WARN] {os.path.basename(p)}: var_names still not unique after normalization.")
        if not adata.obs_names.is_unique:
            log(f"[WARN] {os.path.basename(p)}: obs_names still not unique after normalization.")
        ads.append(adata)
    if len(ads) == 1:
        return ads[0]
    return ad.concat(ads, join="outer", merge="same")

# ---------- symbol<->ensembl ----------
def _is_ensembl_like(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("ENSMUSG", "ENSG"))

def load_symbol_ensembl_mapping(pkl_path: str):
    if not pkl_path or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Mapping pickle not found: {pkl_path}")
    m = pickle.load(open(pkl_path, "rb"))
    if isinstance(m, dict) and "symbol2ens" in m and "ens2symbol" in m:
        return dict(m["symbol2ens"]), dict(m["ens2symbol"])
    if isinstance(m, dict):
        sample = list(m.items())[:50]
        key_has_ens = sum(_is_ensembl_like(k) for k,_ in sample)
        val_has_ens = sum(_is_ensembl_like(v) for _,v in sample if isinstance(v,str))
        if val_has_ens > key_has_ens:
            s2e = {str(k): str(v) for k,v in m.items() if isinstance(v,str)}
            e2s = {v:k for k,v in s2e.items()}
            return s2e, e2s
        else:
            e2s = {str(k): str(v) for k,v in m.items() if isinstance(v,str)}
            s2e = {v:k for k,v in e2s.items()}
            return s2e, e2s
    raise ValueError("Unsupported pickle format for mapping.")

def ensure_var_has_columns(var: pd.DataFrame, symbol2ens: Dict[str, str], ens2symbol: Dict[str, str]):
    var = var.copy()
    if "gene_symbol" not in var.columns and "symbol" in var.columns:
        var["gene_symbol"] = var["symbol"].astype(str)
    if "ensembl_id" not in var.columns and "gene_ids" in var.columns:
        var["ensembl_id"] = var["gene_ids"].astype(str)
    if "gene_symbol" not in var.columns and "ensembl_id" in var.columns:
        var["gene_symbol"] = var["ensembl_id"].map(ens2symbol).fillna("")
    if "ensembl_id" not in var.columns and "gene_symbol" in var.columns:
        var["ensembl_id"] = var["gene_symbol"].map(symbol2ens).fillna("")
    for k in ["gene_symbol", "ensembl_id"]:
        if k in var.columns:
            var[k] = var[k].astype(str).fillna("")
    return var

# ---------- Ensembl → IntID ローダ ----------
def load_ensembl_to_id(path: str) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Ensembl→ID mapping not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pkl", ".pickle"]:
        m = pickle.load(open(path, "rb"))
        if not isinstance(m, dict): raise ValueError("Pickle must be a dict")
        return {str(k): int(v) for k, v in m.items()}
    if ext == ".json":
        m = json.load(open(path, "r"))
        return {str(k): int(v) for k, v in m.items()}
    # TSV/CSV: 1列目=ensembl_id, 2列目=int_id（ヘッダ可）
    if ext in [".tsv", ".csv"]:
        sep = "\t" if ext == ".tsv" else ","
        out = {}
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader)
            # ヘッダ判定
            try:
                int(header[1])
                # ヘッダではなくデータ行だった
                row = header
                if len(row) >= 2:
                    out[str(row[0])] = int(row[1])
            except Exception:
                pass  # ヘッダあり
            for row in reader:
                if len(row) < 2: continue
                out[str(row[0])] = int(row[1])
        return out
    raise ValueError(f"Unsupported mapping extension: {ext}")

# ---------- QC ----------
def compute_qc_columns(adata: ad.AnnData) -> ad.AnnData:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    is_sparse = sp.issparse(X)
    X = X.tocsr() if is_sparse else np.asarray(X)

    if "gene_symbol" in adata.var.columns:
        names_raw = adata.var["gene_symbol"].to_numpy()
    elif "symbol" in adata.var.columns:
        names_raw = adata.var["symbol"].to_numpy()
    else:
        names_raw = np.asarray(adata.var_names)

    names_series = pd.Series(names_raw, dtype="object").astype(str).fillna("")
    var_names = np.asarray(names_series.to_numpy(), dtype="U")
    mt_flag = np.char.startswith(np.char.lower(var_names), "mt-")
    mt_any  = bool(mt_flag.any())

    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = (np.asarray(X.sum(axis=1)).ravel() if is_sparse
                                 else X.sum(axis=1).astype(np.float64))
    if "n_genes_by_counts" not in adata.obs.columns:
        if is_sparse:
            adata.obs["n_genes_by_counts"] = np.diff(X.indptr).astype(np.int32)
        else:
            adata.obs["n_genes_by_counts"] = (X > 0).sum(axis=1).astype(np.int32)
    if "mt_counts" not in adata.obs.columns:
        if mt_any:
            adata.obs["mt_counts"] = (np.asarray(X[:, mt_flag].sum(axis=1)).ravel() if is_sparse
                                      else X[:, mt_flag].sum(axis=1).astype(np.float64))
        else:
            adata.obs["mt_counts"] = 0.0
    return adata

def qc_basic_filter(adata: ad.AnnData, min_genes=200, max_genes=None, min_counts=None, max_counts=None,
                    max_mito_frac=0.2, min_cells_per_gene=3) -> ad.AnnData:
    adata = compute_qc_columns(adata)
    c = adata.obs["n_counts"].values.astype(float)
    g = adata.obs["n_genes_by_counts"].values.astype(int)
    m = adata.obs["mt_counts"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mito_frac = np.where(c > 0, m / c, 0.0)
    adata.obs["pct_mt"] = mito_frac

    mask = np.ones(adata.n_obs, dtype=bool)
    if min_genes is not None:  mask &= g >= int(min_genes)
    if max_genes is not None:  mask &= g <= int(max_genes)
    if min_counts is not None: mask &= c >= float(min_counts)
    if max_counts is not None: mask &= c <= float(max_counts)
    if max_mito_frac is not None: mask &= mito_frac <= float(max_mito_frac)

    ad_f = adata[mask].copy()

    if min_cells_per_gene is not None and min_cells_per_gene > 1 and ad_f.n_obs > 0:
        X = ad_f.layers["counts"] if "counts" in ad_f.layers else ad_f.X
        detected = ((X > 0).sum(axis=0).A1 if sp.issparse(X) else (X > 0).sum(axis=0))
        keep_genes = detected >= int(min_cells_per_gene)
        ad_f = ad_f[:, keep_genes].copy()

    log(f"[QC basic] cells: {adata.n_obs:,} → {ad_f.n_obs:,} ; genes: {adata.n_vars:,} → {ad_f.n_vars:,}")
    return ad_f

def qc_sigma_filter(adata: ad.AnnData, k=3.0, total_cap=20000, min_genes=7) -> ad.AnnData:
    adata = compute_qc_columns(adata)
    c = adata.obs["n_counts"].astype(float).values
    m = adata.obs["mt_counts"].astype(float).values
    g = adata.obs["n_genes_by_counts"].astype(int).values

    mu_c, sd_c = float(c.mean()), float(c.std())
    mu_m, sd_m = float(m.mean()), float(m.std())

    lo_c = max(0.0, mu_c - k * sd_c); hi_c = mu_c + k * sd_c
    lo_m = max(0.0, mu_m - k * sd_m); hi_m = mu_m + k * sd_m

    mask = ((c >= lo_c) & (c <= hi_c) & (m >= lo_m) & (m <= hi_m) &
            (c > 0) & (c <= float(total_cap)) & (g >= int(min_genes)))
    ad_f = adata[mask].copy()

    removed = int(adata.n_obs - ad_f.n_obs)
    rate = 100.0 * removed / max(int(adata.n_obs), 1)
    log("=== [QC sigma thresholds] ===")
    log(f"n_counts in [{lo_c:,.2f}, {hi_c:,.2f}]  (μ={mu_c:,.2f}, σ={sd_c:,.2f})")
    log(f"mt_counts in [{lo_m:,.2f}, {hi_m:,.2f}] (μ={mu_m:,.2f}, σ={sd_m:.2f})")
    log(f"0 < total ≤ {total_cap:,}, n_genes_by_counts ≥ {min_genes}")
    log(f"cells: {adata.n_obs:,} → {ad_f.n_obs:,} (removed {removed:,}, {rate:.2f}%)")
    return ad_f

# ---------- 遺伝子発現条件（keep/delete） ----------
PREFERRED_GENE_KEYS = ("ensembl_id", "gene_symbol", "gene_ids", "symbol")

def _ensembl_core(s: str) -> str:
    return s.split(".", 1)[0] if isinstance(s, str) else s

def _resolve_gene_column(var_df: pd.DataFrame, gene: str, case_sensitive: bool = True) -> Tuple[Optional[int], Optional[str]]:
    if gene is None or not isinstance(gene, str):
        return None, None
    gene_cmp = gene if case_sensitive else gene.lower()
    gene_core = _ensembl_core(gene_cmp)
    for key in PREFERRED_GENE_KEYS:
        if key not in var_df.columns: 
            continue
        series = var_df[key].astype(str).fillna("")
        vals = series if case_sensitive else series.str.lower()
        # 完全一致
        hit = np.where(vals.values == gene_cmp)[0]
        if hit.size:
            return int(hit[0]), key
        # Ensembl コアID一致（版数無視）
        if gene_cmp.startswith(("ENSMUSG", "ENSG", "ensmusg", "ensg")):
            vals_core = vals.map(_ensembl_core)
            hit2 = np.where(vals_core.values == gene_core)[0]
            if hit2.size:
                return int(hit2[0]), key
    return None, None

def _apply_gene_expr_constraints(adata: ad.AnnData, constraints: Optional[List[Dict[str, Any]]]) -> ad.AnnData:
    """
    constraints: [{"gene": str, "op": "<=|<|>=|>|==|!=", "value": float, "action": "keep|delete", [optional] "key": "gene_symbol|ensembl_id|gene_ids|symbol"}, ...]
    - アクションは以下のように解釈（順序は指定順で逐次適用）:
        keep   : 現在の候補集合に対して条件を満たすセルのみ残す（AND）
        delete : 条件を満たすセルを候補から除外する（NOT）
    - いずれも満たすセルを効率的に列スキャンで求めます
    """
    if not constraints:
        return adata

    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    is_sparse = sp.issparse(X)
    var_df = adata.var

    # 現在の「候補セル」マスク（最初は全True）
    mask = np.ones(adata.n_obs, dtype=bool)

    def _compare(vec: np.ndarray, op: str, val: float) -> np.ndarray:
        if op == "<=": return vec <= val
        if op == "<":  return vec <  val
        if op == ">=": return vec >= val
        if op == ">":  return vec >  val
        if op == "==": return vec == val
        if op == "!=": return vec != val
        raise ValueError(f"Unsupported op: {op}")

    for idx, c in enumerate(constraints, 1):
        gene = c.get("gene")
        op   = c.get("op")
        val  = float(c.get("value"))
        act  = (c.get("action") or "keep").lower()
        explicit_key = c.get("key")

        if act not in {"keep", "delete"}:
            raise ValueError(f"[Constraint #{idx}] action must be 'keep' or 'delete', got: {act}")

        # 列を決定
        if explicit_key:
            if explicit_key not in var_df.columns:
                raise KeyError(f"[Constraint #{idx}] var['{explicit_key}'] が存在しません。")
            series = var_df[explicit_key].astype(str).fillna("")
            vals = series if GENE_MATCH_CASE_SENSITIVE else series.str.lower()
            gene_cmp = gene if GENE_MATCH_CASE_SENSITIVE else str(gene).lower()
            hit = np.where(vals.values == gene_cmp)[0]
            if hit.size == 0 and gene_cmp.startswith(("ensmusg", "ensg")):
                vals_core = vals.map(_ensembl_core)
                hit = np.where(vals_core.values == _ensembl_core(gene_cmp))[0]
            j = int(hit[0]) if hit.size else -1
            resolved_key = explicit_key
        else:
            j, resolved_key = _resolve_gene_column(var_df, gene, case_sensitive=GENE_MATCH_CASE_SENSITIVE)

        if j is None or j < 0:
            log(f"[WARN] Constraint #{idx}: gene not found -> skip | gene={gene}")
            continue

        # 列ベクトル抽出（全細胞）
        if is_sparse:
            expr_full = np.asarray(X[:, j].toarray()).ravel()
        else:
            expr_full = np.asarray(X[:, j]).ravel()

        # 現在の候補に限定
        expr = expr_full[mask]
        cond_local = _compare(expr, op, val)  # 候補集合上の条件

        before = int(mask.sum())
        if act == "keep":
            # 候補の中から条件を満たすものだけ残す
            mask_indices = np.where(mask)[0]
            mask_sub = np.zeros_like(mask, dtype=bool)
            mask_sub[mask_indices[cond_local]] = True
            mask = mask & mask_sub
        else:  # delete
            # 候補の中から条件を満たすものを除去
            mask_indices = np.where(mask)[0]
            to_delete = mask_indices[cond_local]
            mask_del = mask.copy()
            mask_del[to_delete] = False
            mask = mask_del

        after = int(mask.sum())
        affected = before - after if act == "delete" else after  # deleteは除外数、keepは残存数
        if act == "keep":
            rate = 100.0 * after / max(before, 1)
            log(f"[Constraint #{idx} KEEP] gene={gene} (col={resolved_key}, idx={j})  expr {op} {val}  -> kept {after:,}/{before:,} ({rate:.2f}%)")
        else:
            rate = 100.0 * (before - after) / max(before, 1)
            log(f"[Constraint #{idx} DELETE] gene={gene} (col={resolved_key}, idx={j})  expr {op} {val}  -> removed {before-after:,}/{before:,} ({rate:.2f}%)")

        if after == 0:
            log(f"[Constraint #{idx}] All cells filtered out; stopping further constraints.")
            break

    ad_f = adata[mask].copy()
    removed = int(adata.n_obs - ad_f.n_obs)
    rate_r = 100.0 * removed / max(adata.n_obs, 1)
    log(f"[GeneExpr Summary] cells: {adata.n_obs:,} → {ad_f.n_obs:,} (removed {removed:,}, {rate_r:.2f}%)")
    return ad_f

# ---------- トークナイズ（行=セル） ----------
def _get_matrix_and_var(adata: ad.AnnData, use_raw: bool):
    if use_raw and adata.raw is not None:
        return adata.raw.X, adata.raw.var, "raw"
    return adata.X, adata.var, "X"

def _linear_rve(rank_idx: np.ndarray, L: int) -> np.ndarray:
    if L <= 1: return np.ones_like(rank_idx, dtype=np.float32)
    return (1.0 - (rank_idx.astype(np.float32) / (L - 1))).astype(np.float32)

def _log_rve(rank_idx: np.ndarray, L: int) -> np.ndarray:
    r = rank_idx.astype(np.float64) + 1.0
    val = 1.0 / np.log1p(r)
    if L > 0:
        val = (val - val.min()) / (val.max() - val.min() + 1e-12)
    return val.astype(np.float32)

def _tokenize_rows(
    X,
    ensembl_series: pd.Series,
    ensembl2id: Dict[str, int],
    rve_mode: str = "none",
    max_len: Optional[int] = None,
    min_len: int = 1
):
    """
    X: cells x genes
    ensembl_series: var["ensembl_id"]（str）
    戻り値: list of {"input_ids": List[int], "length": int, (optional) "rank_values": List[float]}
    """
    if sp.issparse(X) and not sp.isspmatrix_csr(X):
        X = X.tocsr()
    elif not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))

    n_cells, n_genes = X.shape
    ens = ensembl_series.astype(str).fillna("").values
    gid = np.full(n_genes, -1, dtype=np.int64)
    for j, e in enumerate(ens):
        if e and e in ensembl2id:
            gid[j] = int(ensembl2id[e])
    mapped_mask = gid >= 0

    data = X.data
    indices = X.indices
    indptr = X.indptr

    out = []
    dropped_noexpr = 0
    dropped_nomap = 0

    for i in range(n_cells):
        start, end = indptr[i], indptr[i+1]
        jj = indices[start:end]
        vv = data[start:end]

        if jj.size == 0:
            dropped_noexpr += 1
            continue

        m = mapped_mask[jj]
        if not m.any():
            dropped_nomap += 1
            continue
        jj = jj[m]; vv = vv[m]
        ids = gid[jj]

        order = np.lexsort((ids, -vv))
        ids_sorted = ids[order]

        if max_len is not None:
            ids_sorted = ids_sorted[:max_len]

        L = ids_sorted.size
        if L < min_len:
            continue

        item = {"input_ids": ids_sorted.tolist(), "length": int(L)}

        if rve_mode and rve_mode.lower() != "none":
            ranks = np.arange(L, dtype=np.int64)  # 0..L-1
            if rve_mode == "linear":
                vals = _linear_rve(ranks, L)
            elif rve_mode == "log":
                vals = _log_rve(ranks, L)
            else:
                raise ValueError(f"Unknown RVE mode: {rve_mode}")
            if SAVE_RVE_COLUMN:
                item["rank_values"] = vals.tolist()
        out.append(item)

    if dropped_noexpr:
        log(f"[INFO] skipped {dropped_noexpr} cells with no nonzero expression")
    if dropped_nomap:
        log(f"[INFO] skipped {dropped_nomap} cells with no mappable genes (check ENSEMBL2ID_PATH)")
    return out

# ---------- パイプライン ----------
def run_pipeline(
    input_dir: str,
    output_dir: str,
    var_key_for_symbols: str = "gene_symbol",
    use_raw_for_gene: bool = True,
    mapping_pickle: Optional[str] = None,
    list_keys: bool = True,
    list_keys_topn: int = 30,
    qc_mode: str = "basic",
    # basic
    qc_min_genes: int = 200,
    qc_max_genes: Optional[int] = None,
    qc_min_counts: Optional[int] = None,
    qc_max_counts: Optional[int] = None,
    qc_max_mito_frac: Optional[float] = 0.2,
    qc_min_cells_per_gene: Optional[int] = 3,
    # sigma
    qc_sigma_k: float = 3.0,
    qc_total_cap: int = 20000,
    # tokenization
    ensembl2id_path: str = "",
    rve_mode: str = "none",
    max_len: Optional[int] = None,
    min_len: int = 1,
    # 遺伝子条件（keep/delete）
    gene_expr_constraints: Optional[List[Dict[str, Any]]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 入力
    h5, h5ad = _list_inputs(input_dir)
    paths = h5 + h5ad
    log(f"[INFO] files: {len(paths)}")
    for p in paths[:5]: log(" - " + p)
    if len(paths) > 5: log(f"   ... (+{len(paths)-5} more)")

    adata = _concat_anndatas(paths)
    log(f"[INFO] concatenated AnnData: cells={adata.n_obs:,}, genes={adata.n_vars:,}")

    # 行アクセスのため CSR 推奨
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(X := adata.X):
        adata.X = X.tocsr()
        log("[INFO] converted X to CSR for fast row ops")

    if use_raw_for_gene and adata.raw is None:
        log("[WARN] USE_RAW=True ですが adata.raw がありません。adata.X/adata.var を使用します。")

    if list_keys:
        summarize_adata_keys(adata, topn=list_keys_topn)

    # symbol<->ensembl（任意）
    if mapping_pickle:
        symbol2ens, ens2symbol = load_symbol_ensembl_mapping(mapping_pickle)
    else:
        symbol2ens, ens2symbol = {}, {}

    # var を補完
    adata.var = ensure_var_has_columns(adata.var, symbol2ens, ens2symbol)

    # ---------- QC ----------
    cells_before_qc = int(adata.n_obs)
    if qc_mode.lower() == "basic":
        adata_qc = qc_basic_filter(
            adata, qc_min_genes, qc_max_genes, qc_min_counts, qc_max_counts,
            qc_max_mito_frac, qc_min_cells_per_gene
        )
    elif qc_mode.lower() == "sigma":
        adata_qc = qc_sigma_filter(
            adata, k=qc_sigma_k, total_cap=qc_total_cap,
            min_genes=max(7, qc_min_genes if qc_min_genes is not None else 7)
        )
        if qc_min_cells_per_gene and qc_min_cells_per_gene > 1 and adata_qc.n_obs > 0:
            X = adata_qc.layers["counts"] if "counts" in adata_qc.layers else adata_qc.X
            detected = ((X > 0).sum(axis=0).A1 if sp.issparse(X) else (X > 0).sum(axis=0))
            keep_genes = detected >= int(qc_min_cells_per_gene)
            adata_qc = adata_qc[:, keep_genes].copy()
            log(f"[QC sigma+gene] genes filtered by min_cells_per_gene={qc_min_cells_per_gene}")
    else:
        raise ValueError(f"Unknown qc_mode: {qc_mode}")

    cells_after_qc = int(adata_qc.n_obs)
    qc_removed = cells_before_qc - cells_after_qc
    qc_removed_rate = 100.0 * qc_removed / max(cells_before_qc, 1)
    log(f"[INFO] after QC: cells={cells_after_qc:,}, genes={adata_qc.n_vars:,}")
    log(f"[SUMMARY] QC removed: {qc_removed:,}/{cells_before_qc:,} cells ({qc_removed_rate:.2f}%)")

    # ---------- ユーザー条件（keep/delete） ----------
    cells_before_user = int(adata_qc.n_obs)
    adata_qc = _apply_gene_expr_constraints(adata_qc, gene_expr_constraints)
    cells_after_user = int(adata_qc.n_obs)
    user_removed = cells_before_user - cells_after_user
    user_removed_rate = 100.0 * user_removed / max(cells_before_user, 1)
    if gene_expr_constraints:
        log(f"[SUMMARY] User constraints removed: {user_removed:,}/{cells_before_user:,} cells ({user_removed_rate:.2f}%)")
    else:
        log("[SUMMARY] User constraints: none (no additional removal)")

    if adata_qc.n_obs == 0:
        raise RuntimeError("No cells left after QC and/or gene expression constraints.")

    # ---------- トークナイズ & 保存 ----------
    ensembl2id = load_ensembl_to_id(ensembl2id_path)

    X, var, layer = _get_matrix_and_var(adata_qc, use_raw_for_gene)
    if "ensembl_id" not in var.columns:
        raise KeyError("var['ensembl_id'] が存在しません。VAR_GENEKEY と symbol/ensembl 補完を確認してください。")

    rve_mode = (rve_mode or "none").lower()
    if rve_mode not in {"none", "linear", "log"}:
        raise ValueError(f"Unsupported RVE_MODE: {rve_mode}")

    token_rows = _tokenize_rows(
        X=X,
        ensembl_series=var["ensembl_id"],
        ensembl2id=ensembl2id,
        rve_mode=rve_mode,
        max_len=max_len,
        min_len=min_len,
    )

    if not token_rows:
        raise RuntimeError("No rows to save. すべてフィルタ落ちか、マッピング不一致の可能性があります。")

    from datasets import Dataset
    if SAVE_RVE_COLUMN:
        ds = Dataset.from_list(token_rows)  # input_ids, length, rank_values
    else:
        ds = Dataset.from_dict({
            "input_ids": [r["input_ids"] for r in token_rows],
            "length":    [r["length"] for r in token_rows],
        })
    ds.save_to_disk(output_dir)

    log("[INFO] example rows:")
    for row in ds.select(range(min(3, len(ds)))):
        log(str(row))
    log(f"[DONE] saved tokenized dataset to: {output_dir}")

    # --------- 最終サマリー ---------
    log("===== FINAL SUMMARY =====")
    log(f"Input cells           : {cells_before_qc:,}")
    log(f"After QC              : {cells_after_qc:,}  (removed {qc_removed:,}, {qc_removed_rate:.2f}%)")
    log(f"After User constraints: {cells_after_user:,}  (removed {user_removed:,}, {user_removed_rate:.2f}%)")
    log(f"Saved rows (Dataset)  : {len(ds):,}")

# ============================================================
# 3) 実行
# ============================================================
if __name__ == "__main__":
    run_pipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        var_key_for_symbols=VAR_GENEKEY,
        use_raw_for_gene=USE_RAW,
        mapping_pickle=MAPPING_PICKLE,
        list_keys=LIST_KEYS,
        list_keys_topn=LIST_KEYS_TOPN,
        qc_mode=QC_MODE,
        qc_min_genes=QC_MIN_GENES,
        qc_max_genes=QC_MAX_GENES,
        qc_min_counts=QC_MIN_COUNTS,
        qc_max_counts=QC_MAX_COUNTS,
        qc_max_mito_frac=QC_MAX_MITO_FRAC,
        qc_min_cells_per_gene=QC_MIN_CELLS_PER_GENE,
        qc_sigma_k=QC_SIGMA_K,
        qc_total_cap=QC_TOTAL_CAP,
        ensembl2id_path=ENSEMBL2ID_PATH,
        rve_mode=RVE_MODE,
        max_len=MAX_LEN,
        min_len=MIN_LEN,
        gene_expr_constraints=GENE_EXPR_CONSTRAINTS,
    )
