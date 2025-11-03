# tokenizer_fixed.py
"""
Geneformer tokenizer (minimal, per-loom median support).

Row attribute:
  Prefer "ensembl_id", but also accept "gene_ids", "gene_id", "ensembl", "var_names".
Optional col attribute:
  "filter_pass" (1 = use)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import glob
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import loompy as lp
from datasets import Dataset, concatenate_datasets

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
logger = logging.getLogger(__name__)

# === User-modifiable defaults ===
GENE_MEDIAN_FILE = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"  # 修正
TOKEN_DICTIONARY_FILE = "/home/pj25000162/ku50001814/pkl_data/MLM-re_token_dictionary_v1.pkl" # 修正

# --------------------------- utils ---------------------------

def _safe_pickle_load(path: str):
    """空なら {} を返す safe load"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def _to_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    return str(x)

def _strip_ver(s: str) -> str:
    # ENSMUSG00000000001.1 -> ENSMUSG00000000001
    return s.rsplit(".", 1)[0] if "." in s and s.split(".")[-1].isdigit() else s

def _normalize_id_array(arr) -> np.ndarray:
    return np.array([_strip_ver(_to_str(v)).strip() for v in arr], dtype=object)

def rank_genes(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> np.ndarray:
    """rank by value (desc), top-2048"""
    sorted_indices = np.argsort(-gene_vector)[:2048]
    return gene_tokens[sorted_indices]

def tokenize_cell(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> List[int]:
    """Return token-id list for one cell."""
    nz = np.nonzero(gene_vector)[0]
    if nz.size == 0:
        return []
    ranked = rank_genes(gene_vector[nz], gene_tokens[nz])
    return [int(x) for x in ranked]

# --------------------------- main class ---------------------------

class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict: Optional[Dict[str, str]] = None,
        nproc: int = 1,
        gene_median_file: str = GENE_MEDIAN_FILE,
        token_dictionary_file: str = TOKEN_DICTIONARY_FILE,
        gene_median_file_path_dict: Optional[Dict[str, str]] = None,  # {loom_abs_path: median.pkl}
    ):
        self.custom_attr_name_dict = custom_attr_name_dict or {}
        self.nproc = int(nproc)

        # Token辞書（Ensembl -> token_id）を正規化してロード
        if not os.path.exists(token_dictionary_file):
            raise FileNotFoundError(f"token_dictionary_file not found: {token_dictionary_file}")
        raw_tok = _safe_pickle_load(token_dictionary_file)
        if not isinstance(raw_tok, dict):
            raise ValueError("token_dictionary_file must be a dict of {ensembl_id: token_id}")
        self.gene_token_dict: Dict[str, int] = { _strip_ver(_to_str(k)): int(v) for k, v in raw_tok.items() }

        # median 辞書（空でもOK）。正規化しておく
        raw_med = _safe_pickle_load(gene_median_file)
        if not isinstance(raw_med, dict):
            raise ValueError("gene_median_file must be a dict of {ensembl_id: median_value}")
        self.gene_median_dict: Dict[str, float] = { _strip_ver(_to_str(k)): float(v) for k, v in raw_med.items() }

        # per-loom median のマッピング
        self.gene_median_file_path_dict = None
        if gene_median_file_path_dict:
            self.gene_median_file_path_dict = {
                os.path.realpath(str(k)): str(v)
                for k, v in gene_median_file_path_dict.items()
            }

        # これらは tokenize_loom() 内で都度更新
        self.gene_keys: List[str] = list(self.gene_median_dict.keys())
        self.genelist_dict: Dict[str, bool] = {k: True for k in self.gene_keys}

    # ---- 各loomに対応する median をロード（なければフォールバック維持） ----
    def _load_gene_median_for(self, loom_file_path: str) -> None:
        if not self.gene_median_file_path_dict:
            return
        key = os.path.realpath(str(loom_file_path))
        median_pkl = self.gene_median_file_path_dict.get(key)
        if median_pkl is None:
            raise KeyError(f"[median_map] not found for loom: {loom_file_path}")
        md = _safe_pickle_load(median_pkl)
        if not isinstance(md, dict):
            raise ValueError(f"median file must be dict: {median_pkl}")
        # 正規化
        self.gene_median_dict = { _strip_ver(_to_str(k)): float(v) for k, v in md.items() }
        self.gene_keys = list(self.gene_median_dict.keys())
        self.genelist_dict = {k: True for k in self.gene_keys}

    # ---- .loom の row IDs を取得（複数候補を自動検出） ----
    @staticmethod
    def _get_row_ids(data: lp.LoomConnection) -> np.ndarray:
        candidates = ["ensembl_id", "gene_ids", "gene_id", "ensembl", "var_names"]
        for key in candidates:
            if key in data.ra:
                arr = _normalize_id_array(data.ra[key])
                logger.info(f"Using row attribute '{key}'")
                return arr
        # 一部のツールは var_names を row attr ではなく特別扱いしていることがあるので念のため
        if hasattr(data, "row_attrs") and "var_names" in data.row_attrs:
            arr = _normalize_id_array(data.row_attrs["var_names"])
            logger.info("Using row attribute 'var_names' (row_attrs)")
            return arr
        raise KeyError(f"{data.filename} requires a row attribute of gene IDs (e.g., 'ensembl_id' or 'gene_ids').")

    # ---- 1つの .loom をトークナイズ ----
    def tokenize_loom(self, loom_file_path: str, target_sum: int = 10_000):
        file_cell_metadata: Optional[Dict[str, List]] = None
        if self.custom_attr_name_dict:
            file_cell_metadata = {dst: [] for dst in self.custom_attr_name_dict.keys()}

        # 各loomに対応する median を読み込み
        self._load_gene_median_for(loom_file_path)

        # 語彙セット
        token_keys = set(self.gene_token_dict.keys())
        median_keys = set(self.gene_median_dict.keys())

        with lp.connect(str(loom_file_path)) as data:
            # row IDs を取り出す（自動検出）
            row_ids = self._get_row_ids(data)  # np.ndarray[str], version なし

            # median が空なら「全部 1.0」として扱う
            has_median = len(self.gene_median_dict) > 0

            # 採用可否（token は必須。median は有れば使う）
            if has_median:
                valid_mask = np.array([(gid in token_keys) and (gid in median_keys) for gid in row_ids])
            else:
                valid_mask = np.array([(gid in token_keys) for gid in row_ids])

            coding_loc = np.where(valid_mask)[0]
            if coding_loc.size == 0:
                cells_counts = int(data.shape[1])
                logger.warning(f"No overlapping genes with token dict (and median dict) in {loom_file_path}")
                return [], (file_cell_metadata if file_cell_metadata else None), cells_counts

            coding_ids = row_ids[coding_loc]
            coding_tokens = np.array([self.gene_token_dict[g] for g in coding_ids], dtype=np.int64)

            if has_median:
                norm_factor_vector = np.array([self.gene_median_dict[g] for g in coding_ids], dtype=np.float64)
            else:
                norm_factor_vector = np.ones(coding_ids.shape[0], dtype=np.float64)

            # フィルタ列（任意）
            if "filter_pass" in data.ca:
                filter_pass_loc = np.where([int(i) == 1 for i in data.ca["filter_pass"]])[0]
            else:
                filter_pass_loc = np.arange(data.shape[1], dtype=np.int64)

            n_overlap_tok = coding_loc.size
            n_overlap_med = (np.isin(coding_ids, list(median_keys))).sum() if has_median else 0
            logger.info(
                f"[{Path(loom_file_path).name}] genes: total={data.shape[0]}, overlap(token)={n_overlap_tok}"
                + (f", overlap(median)={n_overlap_med}" if has_median else ", median: fallback=1.0")
                + f"; cells(selected)={filter_pass_loc.size}/{data.shape[1]}"
            )

            tokenized_cells: List[List[int]] = []

            # セル方向（axis=1）に走査
            for (_ix, _sel, view) in data.scan(items=filter_pass_loc, axis=1):
                subview = view.view[coding_loc, :]  # 必要行のみ
                arr = subview[:, :].astype(np.float64, copy=False)  # dense

                # cell-wise total で正規化し target_sum を掛ける
                totals = np.sum(arr, axis=0)
                totals[totals == 0] = 1.0
                subview_norm = (arr / totals) * float(target_sum)

                # gene median で割る
                subview_norm = subview_norm / norm_factor_vector[:, None]

                # 各細胞をトークナイズ
                for j in range(subview_norm.shape[1]):
                    tokenized_cells.append(tokenize_cell(subview_norm[:, j], coding_tokens))

                # メタデータ（必要なら）
                if file_cell_metadata is not None:
                    for dst in file_cell_metadata.keys():
                        src = self.custom_attr_name_dict[dst]
                        if src not in subview.ca:
                            raise KeyError(f"Column attribute '{src}' not found in {loom_file_path}")
                        file_cell_metadata[dst] += subview.ca[src].tolist()

        cells_counts = int(filter_pass_loc.size)
        return tokenized_cells, (file_cell_metadata if file_cell_metadata else None), cells_counts

    # ---- Dataset 作成（トランケート＆長さ付与）----
    def create_dataset(self, tokenized_cells: List[List[int]], cell_metadata: Optional[Dict[str, List]], use_generator: bool = False) -> Dataset:
        dataset_dict = {"input_ids": tokenized_cells}
        if cell_metadata:
            dataset_dict.update(cell_metadata)

        if use_generator:
            def dict_gen():
                n = len(tokenized_cells)
                keys = list(dataset_dict.keys())
                for i in range(n):
                    yield {k: dataset_dict[k][i] for k in keys}
            ds = Dataset.from_generator(dict_gen, num_proc=self.nproc)
        else:
            ds = Dataset.from_dict(dataset_dict)

        # 2048 トークンに切り詰め
        def _truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example
        ds = ds.map(_truncate, num_proc=self.nproc)

        # 長さを付与
        def _length(example):
            example["length"] = len(example["input_ids"])
            return example
        ds = ds.map(_length, num_proc=self.nproc)
        return ds

    # ---- ディレクトリ内の .loom をまとめて処理し、Arrow で保存 ----
    def tokenize_data(self, data_directory: str, output_directory: str, output_prefix: str = "output") -> None:
        data_directory = os.path.realpath(str(data_directory))
        output_directory = os.path.realpath(str(output_directory))
        os.makedirs(output_directory, exist_ok=True)

        loom_paths = sorted(glob.glob(os.path.join(data_directory, "*.loom")))
        if not loom_paths:
            raise FileNotFoundError(f"No .loom files found in: {data_directory}")

        all_datasets: List[Dataset] = []
        total_cells = 0

        for lp_path in loom_paths:
            logger.info(f"Tokenizing: {lp_path}")
            tokenized_cells, file_cell_metadata, n_cells = self.tokenize_loom(lp_path)
            total_cells += int(n_cells)
            if len(tokenized_cells) == 0:
                logger.warning(f"No tokenized cells from: {lp_path}")
                continue
            ds = self.create_dataset(tokenized_cells, file_cell_metadata, use_generator=False)
            all_datasets.append(ds)

        if not all_datasets:
            raise RuntimeError("No tokenized data produced from any .loom file.")

        merged = all_datasets[0]
        for ds in all_datasets[1:]:
            merged = concatenate_datasets([merged, ds])

        save_path = os.path.join(output_directory, output_prefix)
        merged.save_to_disk(save_path)
        logger.info(f"Saved dataset to: {save_path} (cells={len(merged)})")
