#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_virtual_scRNA_data.py (with progress bars, robust logging, and metadata columns)
=========================================================================================

- 入力: 1列目=遺伝子名、2列目以降=サンプル列（ヘッダ=サンプル名）の発現「合計」行列（CSV/TSV/TXT/XLSX/XLS）。
- 仕様: 各サンプル列につき N_CELLS_PER_SAMPLE 仮想セルを生成 → 連結（合計 = サンプル数 × N_CELLS_PER_SAMPLE）。
- 中央値: 生成した「全仮想セル」から gene median を自動計算（既定: non-zero のみ）。
- 出力:
  * フル行列 CSV（行=セル, 列=遺伝子 + n_counts + filter_pass + sample）をチャンク書き出し（進捗表示）。
  * HuggingFace Dataset(save_to_disk) with columns: input_ids, length, disease, cell_types, organ_major。
- 依存: numpy, pandas（任意: datasets, pyarrow, tqdm, openpyxl）。
- 進捗: フェーズ境界ログ + CSV書き出しチャンクの進捗 + 中央値計算の列バッチ進捗。
"""

import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================================
# ユーザー設定セクション（ここだけ編集すればOK）
# ============================================================================

# 入力（1列目=遺伝子名、2列目以降=サンプル列）
BULK_MATRIX_PATH: str = "/home/pj25000162/ku50001814/scRNA-seq_data/virtual_data_raw/zF_cells_gene_expression.xlsx"  # .csv/.tsv/.txt/.xlsx/.xls すべて対応
GENE_TO_ENSEMBL_PICKLE: str = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"  # {symbol -> ensembl_id}
ENSEMBL_TO_TOKEN_PICKLE: str = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1.pkl"  # {ensembl_id -> int token}

# 出力
OUTPUT_DIR: str = "/home/pj25000162/ku50001814/test"
COUNTS_CSV_NAME: str = "virtual_cells_counts.csv"    # フル行列 + n_counts + filter_pass + sample
DATASET_DIR_NAME: str = "virtual_cells.dataset"      # HuggingFace Dataset(save_to_disk)
DATASET_BASENAME: str = "virtual_cells"              # ログ表示用の名前

# 生成・前処理パラメータ
N_CELLS_PER_SAMPLE: int = 10_000   # ★ サンプル列ごとに生成するセル数
MEAN_COUNTS: int = 5_000           # Poisson(lambda) for total UMI per cell
FILTER_THRESHOLD: int = 1_000      # 総UMIのQCしきい値
MODEL_INPUT_SIZE: int = 2_048      # Geneformer既定
TARGET_SUM: int = 10_000           # rank-value 正規化のスケーリング
MEDIAN_NONZERO_ONLY: bool = True   # 0を除外して中央値算出（未検出は中央値に含めない）
BASE_RANDOM_STATE: int = 42        # RNGシード基点（サンプルごとにオフセット付与）

# 進捗・I/Oチューニング
CSV_CHUNK_SIZE: int = 10_000       # CSV書き出しの行チャンクサイズ（進捗を細かく）
MEDIAN_COL_BATCH: int = 2_000      # 中央値計算の列バッチ幅（メモリ節約＆進捗表示）
USE_TQDM: bool = True              # tqdmが無ければ自動的にFalseにフォールバック

# ログ設定
LOG_LEVEL: str = "INFO"            # "DEBUG" または "INFO"
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
PROGRESS_EVERY_CELLS: int = 1_000  # セル単位の進捗ログ間隔（各サンプル内）
PROGRESS_EVERY_GENES: int = 10_000 # 遺伝子単位（大規模時）での進捗ログ間隔
# ============================================================================


# ---------------------------
# ログ初期化
# ---------------------------
def _init_logging() -> None:
    level = logging.DEBUG if LOG_LEVEL.upper() == "DEBUG" else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


# ---------------------------
# ユーティリティ
# ---------------------------
def _require_file(path: str, tag: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{tag} not found: {path}")


def _get_tqdm():
    """tqdm を取得（無ければ None）。"""
    if not USE_TQDM:
        return None
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


# ---------------------------
# I/O: バルク「行列」を読み込む
# ---------------------------
def load_bulk_matrix(file_path: str) -> pd.DataFrame:
    """1列目=gene、2列目以降=サンプル列の行列を読み込む（CSV/TSV/TXT/XLSX/XLS）。
    返り値: index=gene_symbol, columns=[sample1, sample2, ...] の DataFrame（int）
    """
    t0 = time.perf_counter()
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        try:
            import openpyxl  # noqa: F401
        except Exception:
            logging.error("Reading Excel requires 'openpyxl'. Install via: pip install openpyxl")
            raise
        logging.info(f"[load_bulk_matrix] Detected Excel file: {file_path}")
        df = pd.read_excel(file_path)
    else:
        # CSV/TSV/TXT: 区切り自動判定 + 可変空白フォールバック
        read_ok = False
        for sep in [",", "\t"]:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if df.shape[1] >= 2:
                    read_ok = True
                    break
            except Exception:
                continue
        if not read_ok:
            try:
                df = pd.read_csv(file_path, delim_whitespace=True, engine="python")
                if df.shape[1] >= 2:
                    read_ok = True
            except Exception:
                pass
        if not read_ok:
            raise ValueError(f"Unable to parse bulk counts matrix: {file_path}")

    # 1列目を gene、以降をサンプル列とみなす
    if df.shape[1] < 2:
        raise ValueError("Bulk matrix must have >=2 columns: [gene, sample1, sample2, ...]")
    gene_col = df.columns[0]
    df = df.set_index(gene_col)

    # 数値化
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    t1 = time.perf_counter()
    logging.info(f"[load_bulk_matrix] genes={df.shape[0]}, samples={df.shape[1]}; elapsed={t1-t0:.2f}s")
    return df


# ---------------------------
# 仮想セル生成（サンプルごと）と結合
# ---------------------------
def generate_virtual_cells_per_sample(
    bulk_matrix: pd.DataFrame,
    n_cells_per_sample: int,
    mean_counts: int,
    base_random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """各サンプル列につき n_cells_per_sample 個の仮想セルを生成し、縦に結合。
    返り値:
      - cell_counts_df_all: (total_cells x n_genes) カウント行列
      - total_counts_all  : (total_cells,) 各セル総UMI
      - sample_labels     : (total_cells,) 各セルのサンプル名
    """
    t0 = time.perf_counter()
    genes = bulk_matrix.index
    sample_names = list(bulk_matrix.columns)
    n_genes = len(genes)
    total_cells = n_cells_per_sample * len(sample_names)

    logging.info(f"[generate_virtual_cells_per_sample] start: samples={len(sample_names)}, "
                 f"n_cells_per_sample={n_cells_per_sample}, total_cells={total_cells}, n_genes={n_genes}")

    counts_blocks: List[pd.DataFrame] = []
    totals_blocks: List[np.ndarray] = []
    labels_blocks: List[List[str]] = []

    for s_idx, s_name in enumerate(sample_names):
        rng = np.random.default_rng(base_random_state + s_idx)  # サンプルごとにシードをずらす
        counts = bulk_matrix[s_name].values.astype(float)
        if counts.sum() <= 0:
            logging.warning(f"[generate_virtual_cells_per_sample] sample '{s_name}' has zero total count; generating zeros.")
        gene_prob = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts) / len(counts)

        total_counts = rng.poisson(lam=mean_counts, size=n_cells_per_sample)
        zero_total = int((total_counts == 0).sum())
        total_counts[total_counts == 0] = 1
        if zero_total:
            logging.info(f"[generate_virtual_cells_per_sample] [{s_name}] total_counts had {zero_total} zeros -> set to 1")

        cell_counts = np.zeros((n_cells_per_sample, n_genes), dtype=int)
        for i, tot in enumerate(total_counts):
            cell_counts[i] = rng.multinomial(tot, gene_prob)
            if (i + 1) % PROGRESS_EVERY_CELLS == 0 or (i + 1) == n_cells_per_sample:
                logging.info(f"[{s_name}] simulated {i+1}/{n_cells_per_sample} cells")

        counts_df = pd.DataFrame(cell_counts, columns=genes)
        counts_blocks.append(counts_df)
        totals_blocks.append(total_counts)
        labels_blocks.append([s_name] * n_cells_per_sample)

    cell_counts_df_all = pd.concat(counts_blocks, axis=0, ignore_index=True)
    total_counts_all = np.concatenate(totals_blocks, axis=0)
    sample_labels = sum(labels_blocks, [])  # フラット化

    t1 = time.perf_counter()
    logging.info(f"[generate_virtual_cells_per_sample] done; total_cells={len(sample_labels)}; elapsed={t1-t0:.2f}s")
    return cell_counts_df_all, total_counts_all, sample_labels


def apply_filter(total_counts: np.ndarray, threshold: int) -> np.ndarray:
    passed = int((total_counts >= threshold).sum())
    logging.info(f"[apply_filter] threshold={threshold}; pass={passed}/{len(total_counts)}")
    return (total_counts >= threshold).astype(int)


# ---------------------------
# CSV: チャンク書き出し（進捗表示）
# ---------------------------
def save_virtual_counts_with_sample_chunked(
    cell_counts_df: pd.DataFrame,
    total_counts: np.ndarray,
    filter_pass: np.ndarray,
    sample_labels: List[str],
    output_path: str,
    chunksize: int = 10_000,
    use_tqdm: bool = True,
) -> None:
    """巨大CSVを行チャンクで書き出し（進捗表示）。"""
    import math
    tqdm = _get_tqdm() if use_tqdm else None

    t0 = time.perf_counter()
    n = cell_counts_df.shape[0]
    cols_genes = list(cell_counts_df.columns)
    out_cols = cols_genes + ["n_counts", "filter_pass", "sample"]

    n_chunks = math.ceil(n / chunksize)
    iterator = range(0, n, chunksize)
    if tqdm is not None:
        iterator = tqdm(iterator, total=n_chunks, desc="Writing CSV (chunks)", unit="chunk", mininterval=0.5, leave=False)

    wrote_header = False
    for start in iterator:
        end = min(start + chunksize, n)
        block = cell_counts_df.iloc[start:end].copy()
        block["n_counts"] = total_counts[start:end]
        block["filter_pass"] = filter_pass[start:end]
        block["sample"] = sample_labels[start:end]
        mode = "w" if not wrote_header else "a"
        header = not wrote_header
        block.to_csv(
            output_path,
            mode=mode,
            header=header,
            index=False,
            columns=out_cols,
        )
        wrote_header = True
        if tqdm is None:
            logging.info(f"[save_csv_chunked] wrote rows {start}:{end} / {n}")

    t1 = time.perf_counter()
    logging.info(f"[save_csv_chunked] wrote CSV to {output_path}; rows={n}; elapsed={t1-t0:.2f}s")


# ---------------------------
# gene median（列バッチで進捗表示）
# ---------------------------
def compute_gene_medians_from_virtual_batched(
    cell_counts_df: pd.DataFrame,
    total_counts: np.ndarray,
    target_sum: int,
    use_nonzero_only: bool,
    cols_batch_size: int = 2_000,
    use_tqdm: bool = True,
) -> pd.Series:
    """列をバッチに分割して中央値を計算（進捗表示 & メモリ節約）。"""
    tqdm = _get_tqdm() if use_tqdm else None

    t0 = time.perf_counter()
    logging.info(f"[compute_medians_batched] start: target_sum={target_sum}, nonzero_only={use_nonzero_only}, "
                 f"batch={cols_batch_size}")
    denom = total_counts.astype(float)
    cols = list(cell_counts_df.columns)
    n_cols = len(cols)
    med_parts = []

    iterator = range(0, n_cols, cols_batch_size)
    if tqdm is not None:
        import math
        iterator = tqdm(iterator, total=math.ceil(n_cols/cols_batch_size),
                        desc="Computing medians (col-batches)", unit="batch", mininterval=0.5, leave=False)

    for start in iterator:
        end = min(start + cols_batch_size, n_cols)
        sub = cell_counts_df.iloc[:, start:end]
        norm = sub.div(denom, axis=0) * float(target_sum)
        if use_nonzero_only:
            med_sub = norm.mask(norm <= 0).median(axis=0, skipna=True).fillna(0.0)
        else:
            med_sub = norm.median(axis=0, skipna=True).fillna(0.0)
        med_parts.append(med_sub)

    med = pd.concat(med_parts, axis=0)
    med = med[cols]  # 元の列順に並べ直し

    n_zero = int((med <= 0).sum())
    t1 = time.perf_counter()
    logging.info(f"[compute_medians_batched] done; zero-or-missing medians={n_zero}/{len(med)}; elapsed={t1-t0:.2f}s")
    return med


# ---------------------------
# マッピング（辞書読み込み／構築）
# ---------------------------
def load_mapping_dicts(
    gene_to_ensembl_path: str,
    ensembl_to_token_path: str,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    t0 = time.perf_counter()
    with open(gene_to_ensembl_path, "rb") as f:
        gene_to_ensembl: Dict[str, str] = pickle.load(f)
    with open(ensembl_to_token_path, "rb") as f:
        ensembl_to_token: Dict[str, int] = pickle.load(f)
    t1 = time.perf_counter()
    logging.info(
        f"[load_mapping_dicts] loaded: gene->ensembl={len(gene_to_ensembl)}, "
        f"ensembl->token={len(ensembl_to_token)}; elapsed={t1-t0:.2f}s"
    )
    return gene_to_ensembl, ensembl_to_token


def build_gene_mapping(
    genes: List[str],
    gene_to_ensembl: Dict[str, str],
    ensembl_to_token: Dict[str, int],
    gene_median_by_symbol: pd.Series,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    n_total = len(genes)
    gene_indices: List[int] = []
    token_ids: List[int] = []
    norm_factors: List[float] = []

    for idx, gene_name in enumerate(genes):
        ensembl_id = gene_to_ensembl.get(gene_name)
        if ensembl_id is None:
            continue
        token = ensembl_to_token.get(ensembl_id)
        median_val = float(gene_median_by_symbol.get(gene_name, 0.0))
        if token is None or median_val <= 0.0:
            continue
        gene_indices.append(idx)
        token_ids.append(token)
        norm_factors.append(median_val)

        if (idx + 1) % PROGRESS_EVERY_GENES == 0 or (idx + 1) == n_total:
            logging.info(f"[build_gene_mapping] processed {idx+1}/{n_total} genes; kept={len(gene_indices)}")

    t1 = time.perf_counter()
    logging.info(f"[build_gene_mapping] kept {len(gene_indices)}/{n_total} genes; elapsed={t1-t0:.2f}s")
    return gene_indices, np.array(token_ids, dtype=int), np.array(norm_factors, dtype=float)


# ---------------------------
# トークナイズ（rank-value encoding）
# ---------------------------
def rank_genes(
    normalized_counts: np.ndarray, token_ids: np.ndarray, model_input_size: int
) -> List[int]:
    nonzero_mask = normalized_counts > 0
    if not np.any(nonzero_mask):
        return []
    nz_counts = normalized_counts[nonzero_mask]
    nz_tokens = token_ids[nonzero_mask]
    sort_order = np.argsort(-nz_counts)
    top_tokens = nz_tokens[sort_order][:model_input_size]
    return top_tokens.tolist()


def tokenize_cells(
    cell_counts: np.ndarray,
    total_counts: np.ndarray,
    gene_indices: List[int],
    token_ids: np.ndarray,
    norm_factors: np.ndarray,
    target_sum: int,
    model_input_size: int,
) -> List[List[int]]:
    t0 = time.perf_counter()
    n_cells = cell_counts.shape[0]
    tokenized_cells: List[List[int]] = []
    denom = total_counts.astype(float)
    sub_counts = cell_counts[:, gene_indices]

    for i in range(sub_counts.shape[0]):
        norm_vec = (sub_counts[i].astype(float) / denom[i]) * target_sum
        norm_vec = norm_vec / norm_factors
        tokens = rank_genes(norm_vec, token_ids, model_input_size=model_input_size)
        tokenized_cells.append(tokens)

        if (i + 1) % PROGRESS_EVERY_CELLS == 0 or (i + 1) == n_cells:
            lengths = [len(x) for x in tokenized_cells]
            mean_len = sum(lengths) / len(lengths)
            zero_len = sum(1 for x in tokenized_cells if len(x) == 0)
            logging.info(
                f"[tokenize_cells] tokenized {i+1}/{n_cells} cells; "
                f"mean_len={mean_len:.1f}, zero_len={zero_len}"
            )

    t1 = time.perf_counter()
    logging.info(f"[tokenize_cells] done; elapsed={t1-t0:.2f}s")
    return tokenized_cells


# ---------------------------
# ★ 修正済み: サンプル名から disease を抽出
# ---------------------------
def infer_disease_from_sample(sample_name: str) -> str:
    """
    例:
      'Control_female_rep1' -> 'Control_female'
      'Control_male_rep3'   -> 'Control_male'
      'Cas+Oil_rep2'        -> 'Cas+Oil'
      'Ovx+DHT_rep10'       -> 'Ovx+DHT'
      'Cas+E2_r1'           -> 'Cas+E2'
    末尾の replicate 指定だけを除去（_repN / -repN / _rN / -rN）。中間の '_' は残す。
    """
    base = re.sub(r'([_-]?rep\d+|[_-]r\d+)$', '', sample_name, flags=re.IGNORECASE)
    base = re.sub(r'[_-]+$', '', base)  # 末尾に余った区切りを掃除
    return base


# ---------------------------
# 保存ユーティリティ（HuggingFace Dataset）
# ---------------------------
def save_token_dataset(
    tokenized_cells: List[List[int]],
    output_dir: str,
    dataset_dir_name: str,
    dataset_name_for_log: str,
    include_length: bool = True,
    extra_columns: Optional[Dict[str, List[Any]]] = None,
) -> None:
    """HuggingFace Dataset（save_to_disk形式）で保存。datasets/pyarrow未導入なら警告のみ。"""
    try:
        import datasets  # type: ignore
    except ImportError:
        logging.warning("datasets or pyarrow not available; skipping Dataset export.")
        return

    t0 = time.perf_counter()
    data: Dict[str, Any] = {"input_ids": tokenized_cells}
    if include_length:
        data["length"] = [len(x) for x in tokenized_cells]

    # 追加カラム（メタデータ）も保存
    if extra_columns:
        for k, v in extra_columns.items():
            if len(v) != len(tokenized_cells):
                raise ValueError(f"extra column '{k}' length mismatch: {len(v)} vs {len(tokenized_cells)}")
            data[k] = v

    dset = datasets.Dataset.from_dict(data)
    arrow_dir = os.path.join(output_dir, dataset_dir_name)
    dset.save_to_disk(arrow_dir)
    t1 = time.perf_counter()
    logging.info(
        f"[save_token_dataset] saved Dataset({dataset_name_for_log}) to {arrow_dir}; "
        f"rows={len(tokenized_cells)}; cols={list(data.keys())}; elapsed={t1-t0:.2f}s"
    )


# ---------------------------
# メイン
# ---------------------------
def main() -> None:
    _init_logging()

    # 入力ファイル存在チェック
    _require_file(BULK_MATRIX_PATH, "bulk_matrix")
    _require_file(GENE_TO_ENSEMBL_PICKLE, "gene_to_ensembl_pickle")
    _require_file(ENSEMBL_TO_TOKEN_PICKLE, "ensembl_to_token_pickle")

    # 出力ディレクトリ
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"[main] output_dir={OUTPUT_DIR}")

    # 1) バルク「行列」読込（index=gene, columns=sample）
    bulk_mat = load_bulk_matrix(BULK_MATRIX_PATH)

    # 2) サンプルごとに仮想セル生成 → 連結（合計 = samples × N_CELLS_PER_SAMPLE）
    logging.info("[main] START: generating virtual cells per sample ...")
    cell_counts_df, total_counts, sample_labels = generate_virtual_cells_per_sample(
        bulk_matrix=bulk_mat,
        n_cells_per_sample=N_CELLS_PER_SAMPLE,
        mean_counts=MEAN_COUNTS,
        base_random_state=BASE_RANDOM_STATE,
    )
    logging.info("[main] DONE : generating virtual cells per sample.")

    # 3) QC
    filter_pass = apply_filter(total_counts, threshold=FILTER_THRESHOLD)

    # 4) フル行列CSV 保存（sample列つき）— チャンク書き出し＋進捗
    logging.info("[main] START: saving full counts CSV (chunked) ...")
    counts_csv_path = os.path.join(OUTPUT_DIR, COUNTS_CSV_NAME)
    save_virtual_counts_with_sample_chunked(
        cell_counts_df, total_counts, filter_pass, sample_labels, counts_csv_path,
        chunksize=CSV_CHUNK_SIZE, use_tqdm=USE_TQDM
    )
    logging.info("[main] DONE : saving full counts CSV.")

    # 5) gene median（全仮想セルから内製）— 列バッチ＋進捗
    logging.info("[main] START: computing gene medians (batched) ...")
    gene_median_by_symbol = compute_gene_medians_from_virtual_batched(
        cell_counts_df=cell_counts_df,
        total_counts=total_counts,
        target_sum=TARGET_SUM,
        use_nonzero_only=MEDIAN_NONZERO_ONLY,
        cols_batch_size=MEDIAN_COL_BATCH,
        use_tqdm=USE_TQDM,
    )
    logging.info("[main] DONE : computing gene medians (batched).")

    # 6) マッピング辞書ロード
    gene_to_ensembl, ensembl_to_token = load_mapping_dicts(
        GENE_TO_ENSEMBL_PICKLE, ENSEMBL_TO_TOKEN_PICKLE
    )

    # 7) マッピング構築
    logging.info("[main] START: building gene mapping ...")
    gene_indices, token_ids, norm_factors = build_gene_mapping(
        list(bulk_mat.index), gene_to_ensembl, ensembl_to_token, gene_median_by_symbol
    )
    if len(gene_indices) == 0:
        logging.warning(
            "[main] No genes were mapped to tokens with median>0. "
            "Check dictionaries, gene naming, and median settings."
        )
    logging.info("[main] DONE : building gene mapping.")

    # 8) トークナイズ
    logging.info("[main] START: tokenizing cells ...")
    cell_counts_array = cell_counts_df.values.astype(int)
    tokenized_cells = tokenize_cells(
        cell_counts=cell_counts_array,
        total_counts=total_counts,
        gene_indices=gene_indices,
        token_ids=np.asarray(token_ids),
        norm_factors=np.asarray(norm_factors, dtype=float),
        target_sum=TARGET_SUM,
        model_input_size=MODEL_INPUT_SIZE,
    )
    logging.info("[main] DONE : tokenizing cells.")

    # 追加メタデータ列を作成（全セル分）
    n_cells_total = len(tokenized_cells)
    disease_labels = [infer_disease_from_sample(s) for s in sample_labels]
    if len(disease_labels) != n_cells_total:
        raise RuntimeError(f"disease labels length mismatch: {len(disease_labels)} vs {n_cells_total}")
    cell_types_col = ["nan"] * n_cells_total
    organ_major_col = ["adrenal"] * n_cells_total

    # 9) Dataset保存（input_ids + length + disease + cell_types + organ_major）
    logging.info("[main] START: saving tokenized dataset with metadata ...")
    save_token_dataset(
        tokenized_cells=tokenized_cells,
        output_dir=OUTPUT_DIR,
        dataset_dir_name=DATASET_DIR_NAME,
        dataset_name_for_log=DATASET_BASENAME,
        include_length=True,
        extra_columns={
            "disease": disease_labels,
            "cell_types": cell_types_col,
            "organ_major": organ_major_col,
        },
    )
    logging.info("[main] DONE : saving tokenized dataset with metadata.")

    logging.info("[main] Finished.")


if __name__ == "__main__":
    main()
