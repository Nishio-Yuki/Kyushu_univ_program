# ============================================================
# merge_multi_arrow.py  (複数データセットを列和集合でマージ)
# 変更点:
#  - 各データセットの細胞数（行数）を集計
#  - 合計細胞数、およびマージ後・シャッフル後の行数をログ出力
# ============================================================

import os
import glob

from typing import List, Dict, Optional

from datasets import load_from_disk, concatenate_datasets, Dataset, Features, Sequence, Value, ClassLabel

# =========================
# 1) 設定（ここだけ編集）
# =========================

BASE_DIR = ""

INPUT_DIRS = sorted(glob.glob(f"{BASE_DIR}/merged_batch_*"))

OUTPUT_DIR = f"{BASE_DIR}/merged_all"
# マージ後にシャッフルしたい場合は seed（例: 42）、不要なら None
SHUFFLE_SEED: Optional[int] = None

# 「必須列のみ（input_ids, length だけ）」に制限する場合は True
KEEP_ONLY_REQUIRED = False

# 必須列とオプション列（存在すれば揃える）
REQUIRED_COLS = {
    "input_ids": Sequence(Value("int64")),
    "length": Value("int32"),
}
OPTIONAL_SEQ_COLS = {
    "rank_values": Sequence(Value("float64")),  # あれば揃える（なければ作らない）
}

# =========================
# 2) 実装
# =========================
def is_dataset_dir(p: str) -> bool:
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json"))

# ---- 型併合（ダウンキャストしない） ----
_NUM_ORDER = {"bool": 0, "int32": 1, "int64": 2, "float32": 3, "float64": 4}
def _value_supertype(a: Value, b: Value) -> Value:
    ta, tb = a.dtype, b.dtype
    # どちらかが string → string
    if "string" in (ta, tb):
        return Value("string")
    # どちらかが float → float の上位
    if "float" in ta or "float" in tb:
        rk = max(_NUM_ORDER.get(ta, 0), _NUM_ORDER.get(tb, 0))
        return Value("float64" if rk >= _NUM_ORDER["float64"] else "float32")
    # 数値（bool は最下位として int32 に寄せる）
    if ta == "bool": ta = "int32"
    if tb == "bool": tb = "int32"
    # int32/int64 → 上位へ
    if ta in _NUM_ORDER and tb in _NUM_ORDER:
        rk = max(_NUM_ORDER[ta], _NUM_ORDER[tb])
        return Value("int64" if rk >= _NUM_ORDER["int64"] else "int32")
    # それ以外は文字列へフォールバック
    return Value("string")

def _feature_supertype(fa, fb):
    # 同型ならそのまま
    if type(fa) is type(fb):
        if isinstance(fa, Value):
            return _value_supertype(fa, fb)
        if isinstance(fa, Sequence):
            # element_type は Value or ClassLabel 前提
            ea = fa.feature
            eb = fb.feature
            if isinstance(ea, Value) and isinstance(eb, Value):
                return Sequence(_value_supertype(ea, eb))
            # 型が合わない配列は文字列配列にフォールバック
            return Sequence(Value("string"))
        if isinstance(fa, ClassLabel):
            # 異なるラベル集合は合わせにくいので文字列化
            return Value("string")
        # 未対応の複合型 → 文字列
        return Value("string")
    # 型が違う場合の総フォールバック
    if isinstance(fa, Sequence) or isinstance(fb, Sequence):
        return Sequence(Value("string"))
    if isinstance(fa, ClassLabel) or isinstance(fb, ClassLabel):
        return Value("string")
    if isinstance(fa, Value) and isinstance(fb, Value):
        return _value_supertype(fa, fb)
    return Value("string")

def _union_features_keep_all(datasets: List[Dataset]) -> Features:
    # すべての列名の和集合 + 型は上位互換に
    colnames = set()
    for ds in datasets:
        colnames.update(ds.column_names)
    # 必須列は最低限含める
    colnames.update(REQUIRED_COLS.keys())
    # オプション列（rank_values 等）は、どれかに存在するときだけ含める
    for opt_col in OPTIONAL_SEQ_COLS:
        if any(opt_col in ds.column_names for ds in datasets):
            colnames.add(opt_col)

    merged: Dict[str, object] = {}
    for col in sorted(colnames):
        feats = []
        for ds in datasets:
            f = ds.features.get(col)
            if f is not None:
                feats.append(f)
        if not feats:
            # どの ds にも無い（基本来ない）は既定の型へ
            if col in REQUIRED_COLS:
                merged[col] = REQUIRED_COLS[col]
            elif col in OPTIONAL_SEQ_COLS:
                merged[col] = OPTIONAL_SEQ_COLS[col]
            else:
                merged[col] = Value("string")
            continue
        # 複数 ds からの型を順に併合
        f0 = feats[0]
        for fk in feats[1:]:
            f0 = _feature_supertype(f0, fk)
        merged[col] = f0
    return Features(merged)

def _union_features_required_only(datasets: List[Dataset]) -> Features:
    # 必須 + （存在すれば）OPTIONAL_SEQ_COLS だけに制限
    need = dict(REQUIRED_COLS)
    for opt_col, feat in OPTIONAL_SEQ_COLS.items():
        if any(opt_col in ds.column_names for ds in datasets):
            # 型差があれば上位互換（ここは全て Sequence(Value) 前提）
            # feats 抽出
            feats = []
            for ds in datasets:
                f = ds.features.get(opt_col)
                if f is not None:
                    feats.append(f)
            if feats:
                f0 = feats[0]
                for fk in feats[1:]:
                    f0 = _feature_supertype(f0, fk)
                need[opt_col] = f0
            else:
                need[opt_col] = feat
    return Features(need)

# ---- 欠損列の補完値 ----
def _default_for_feature(ft):
    if isinstance(ft, Sequence):
        return []
    if isinstance(ft, Value):
        dt = ft.dtype
        if dt.startswith("int"):
            return 0
        if dt.startswith("float"):
            return 0.0
        if dt == "bool":
            return False
        if dt == "string":
            return ""
        return ""
    if isinstance(ft, ClassLabel):
        # とりあえず 0（未知ラベルは 0 とする）
        return 0
    # 未知は文字列空
    return ""

def _ensure_columns_and_cast(ds: Dataset, target: Features) -> Dataset:
    # 1) ない列を補完
    for col, ft in target.items():
        if col not in ds.column_names:
            ds = ds.add_column(col, [_default_for_feature(ft) for _ in range(len(ds))])
    # 2) 余計な列は残して OK（Features にないと cast が失敗するので先に型を拡張）
    # 3) 型を target に合わせて cast（ダウンキャストはしていない想定）
    return ds.cast(target)

def _maybe_drop_extra_columns(ds: Dataset, target: Features) -> Dataset:
    drop_cols = [c for c in ds.column_names if c not in target]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds

def _fmt(n: int) -> str:
    return f"{n:,}"

def merge_arrow_datasets(input_dirs: List[str], output_dir: str, shuffle_seed: Optional[int] = None):
    if not input_dirs:
        raise ValueError("入力フォルダが指定されていません。")

    datasets = []
    load_stats = []  # [(path, rows, columns)]
    for p in input_dirs:
        if not is_dataset_dir(p):
            raise ValueError(f"HF Datasets の save_to_disk ではありません: {p}")
        ds = load_from_disk(p)
        if isinstance(ds, dict):
            if "train" not in ds:
                raise ValueError(f"{p} は DatasetDict ですが 'train' がありません。")
            ds = ds["train"]
        rows = len(ds)
        cols = list(ds.column_names)
        print(f"[INFO] Loaded: {p} ({_fmt(rows)} rows) cols={cols}")
        datasets.append(ds)
        load_stats.append((p, rows, cols))

    # ロード済みデータの細胞数集計を表示
    print("--------------- per-dataset cell counts ---------------")
    total_cells_before = 0
    for p, rows, _ in load_stats:
        print(f"[COUNT] {p}: {_fmt(rows)}")
        total_cells_before += rows
    print(f"[COUNT] Total cells across inputs: {_fmt(total_cells_before)}")
    print("-------------------------------------------------------")

    # 特徴量スキーマを決定
    if KEEP_ONLY_REQUIRED:
        target = _union_features_required_only(datasets)
        print("[INFO] Mode: KEEP_ONLY_REQUIRED → columns:", list(target.keys()))
        # 先に余分な列を削る（行数は変わらない）
        datasets = [_maybe_drop_extra_columns(ds, target) for ds in datasets]
    else:
        target = _union_features_keep_all(datasets)
        print("[INFO] Mode: KEEP_ALL_COLUMNS → columns:", list(target.keys()))

    # 列補完 + 型そろえ（行数は変わらない）
    datasets = [_ensure_columns_and_cast(ds, target) for ds in datasets]

    # マージ
    merged = concatenate_datasets(datasets, axis=0)
    merged_len = len(merged)
    print(f"[INFO] After concatenate_datasets: {_fmt(merged_len)} rows")

    # シャッフル（任意）
    if shuffle_seed is not None:
        merged = merged.shuffle(seed=shuffle_seed)
        print(f"[INFO] After shuffle(seed={shuffle_seed}): {_fmt(len(merged))} rows")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    merged.save_to_disk(output_dir)

    # サマリ
    print("===========================================")
    print(f"[DONE] merged {len(datasets)} datasets → {output_dir}")
    print("columns:", merged.column_names)
    print("size (rows):", _fmt(len(merged)))
    # 整合性チェック（結合後の行数が入力合計と一致するか）
    if merged_len != total_cells_before:
        print(f"[WARN] Row count mismatch: concatenated={_fmt(merged_len)} vs inputs_total={_fmt(total_cells_before)}")
    else:
        print(f"[OK] Row counts match: {_fmt(merged_len)}")
    print("===========================================")

# =========================
# 3) 実行
# =========================
if __name__ == "__main__":
    merge_arrow_datasets(INPUT_DIRS, OUTPUT_DIR, shuffle_seed=SHUFFLE_SEED)
