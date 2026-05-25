# -*- coding: utf-8 -*-
"""
Fixed-boundary point cloud splitting (IDE-run version)

How to use:
1) Edit paths and parameters in the configuration section below.
2) Run this script directly in the IDE.

Outputs:
- meta json: fixed split boundaries defined by the reference point cloud
- optional parts: one txt file per bin
- optional train/test: train/test outputs based on the same meta boundaries
"""

import json
from pathlib import Path
import numpy as np


# =========================
# Configuration
# =========================

# 1) Reference point cloud path for generating fixed-boundary metadata
REF_TXT_PATH = r"./recon/eye_04/04_gt.txt"

# 2) Point clouds to split; all reuse the same metadata boundaries
INPUT_TXT_PATHS = [
    r"./recon/04/04_gt.txt",
    r"./recon/04/04_snow.txt",
    r"./recon/04/04_rie.txt",
    r"./recon/04/04_diff.txt",
    r"./recon/04/04_shape.txt",
    r"./recon/04/04_shape2vecset.txt"
]

# 3) Output root directory
OUT_ROOT_DIR = r"./recon/eye_04"

# 4) Metadata output path
META_JSON_PATH = r"./recon/04/04_meta.json"

# 5) Split parameters; boundaries are determined by REF_TXT_PATH
SPLIT_DIM_ID = 1   # 0:x, 1:y, 2:z
SPLIT_N = 10

# 6) Save per-bin point clouds
SAVE_PARTS = True
SAVE_EMPTY_BINS = False

# 7) Also generate train/test splits
MAKE_TRAIN_TEST = True
VALI_TRAIN_RATIO = 7  # Segment-count threshold, not a fractional ratio
DATASET = ""          # Example: "david"; leave empty if not using dataset rules
OBJECTS = ["head", "mid", "tail"]  # Generate head/mid/tail in sequence
MID_START = 3         # Used when object="mid"; test interval is [MID_START, VALI_TRAIN_RATIO)


# =========================
# Implementation
# =========================

def load_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError(f"Need >=3 columns (x y z). Got shape={arr.shape} for {path}")
    return arr


def save_txt(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ["%.6f"] * arr.shape[1]
    np.savetxt(str(path), arr, fmt=fmt)


def make_meta_from_ref(ref: np.ndarray, split_dim_id: int, split_n: int) -> dict:
    coord = ref[:, split_dim_id]
    minv = float(coord.min())
    maxv = float(coord.max())
    span = maxv - minv

    if split_n <= 0:
        raise ValueError("SPLIT_N must be > 0")

    if np.isclose(span, 0.0):
        step = 0.0
        edges = [minv] * (split_n + 1)
    else:
        step = span / float(split_n)
        edges = [minv + i * step for i in range(split_n + 1)]

    return {
        "split_dim_id": int(split_dim_id),
        "split_n": int(split_n),
        "ref_min": minv,
        "ref_max": maxv,
        "ref_span": float(span),
        "ref_step": float(step),
        "edges": edges,  # For logging/visualization only
    }


def assign_bins_with_meta(data: np.ndarray, meta: dict) -> np.ndarray:
    split_dim_id = int(meta["split_dim_id"])
    split_n = int(meta["split_n"])
    minv = float(meta["ref_min"])
    step = float(meta["ref_step"])

    x = data[:, split_dim_id]

    if np.isclose(step, 0.0):
        return np.zeros((data.shape[0],), dtype=np.int64)

    raw = np.floor((x - minv) / step).astype(np.int64)
    bins = np.clip(raw, 0, split_n - 1)
    return bins


def split_train_test_masks(bin_indices: np.ndarray,
                          split_n: int,
                          dataset: str,
                          obj: str,
                          vali_train_ratio: int,
                          mid_start: int):
    if not (0 <= vali_train_ratio <= split_n):
        raise ValueError(f"VALI_TRAIN_RATIO must be in [0, {split_n}], got {vali_train_ratio}")

    dataset = (dataset or "").lower()
    obj = (obj or "").lower()

    if dataset == "david":
        # Last x bins are train; first split_n-x bins are test, following the original logic
        train_mask = bin_indices >= vali_train_ratio
        test_mask = bin_indices < vali_train_ratio

    elif obj == "head":
        train_mask = bin_indices < vali_train_ratio
        test_mask = bin_indices >= vali_train_ratio

    elif obj == "tail":
        train_mask = bin_indices >= (split_n - vali_train_ratio)
        test_mask = bin_indices < (split_n - vali_train_ratio)

    elif obj == "mid":
        train_mask = (bin_indices < mid_start) | (bin_indices >= vali_train_ratio)
        test_mask = (bin_indices >= mid_start) & (bin_indices < vali_train_ratio)

    else:
        raise ValueError('Please set DATASET="david" OR OBJECT in {"head","tail","mid"}.')

    return train_mask, test_mask


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run():
    base_out_root = Path(OUT_ROOT_DIR)
    base_out_root.mkdir(parents=True, exist_ok=True)

    # Step A: generate fixed-boundary metadata from the reference point cloud
    ref_path = Path(REF_TXT_PATH)
    ref = load_txt(ref_path)
    meta = make_meta_from_ref(ref, SPLIT_DIM_ID, SPLIT_N)
    meta.update({
        "ref_path": str(ref_path),
        "ref_num_points": int(ref.shape[0]),
        "ref_num_cols": int(ref.shape[1]),
    })

    meta_path = Path(META_JSON_PATH)
    write_json(meta_path, meta)
    print(f"[OK] Meta saved: {meta_path}")

    # Step B: split each input point cloud with the metadata
    for obj, p in ((obj, p) for obj in OBJECTS for p in INPUT_TXT_PATHS):
        in_path = Path(p)
        data = load_txt(in_path)
        bins = assign_bins_with_meta(data, meta)
        split_n = int(meta["split_n"])

        # Create one subdirectory per input file to keep outputs separate
        out_dir = base_out_root / obj / in_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) parts
        if SAVE_PARTS:
            parts_dir = out_dir / f"parts_dim{meta['split_dim_id']}_n{split_n}"
            parts_dir.mkdir(parents=True, exist_ok=True)
            for i in range(split_n):
                part = data[bins == i]
                if part.shape[0] == 0 and not SAVE_EMPTY_BINS:
                    continue
                save_txt(parts_dir / f"{in_path.stem}_part_{i:02d}.txt", part)

        # 2) train/test
        if MAKE_TRAIN_TEST:
            train_mask, test_mask = split_train_test_masks(
                bin_indices=bins,
                split_n=split_n,
                dataset=DATASET,
                obj=obj,
                vali_train_ratio=VALI_TRAIN_RATIO,
                mid_start=MID_START
            )
            save_txt(out_dir / f"{in_path.stem}_train_refmeta.txt", data[train_mask])
            save_txt(out_dir / f"{in_path.stem}_test_refmeta.txt", data[test_mask])

        # 3) stat
        counts = [int((bins == i).sum()) for i in range(split_n)]
        stat = {
            "input": str(in_path),
            "meta": str(meta_path),
            "num_points": int(data.shape[0]),
            "num_cols": int(data.shape[1]),
            "bin_counts": counts,
            "split_dim_id": int(meta["split_dim_id"]),
            "split_n": split_n,
            "ref_min": float(meta["ref_min"]),
            "ref_max": float(meta["ref_max"]),
            "ref_step": float(meta["ref_step"]),
            "SAVE_PARTS": SAVE_PARTS,
            "MAKE_TRAIN_TEST": MAKE_TRAIN_TEST,
            "DATASET": DATASET,
            "OBJECT": obj,
            "VALI_TRAIN_RATIO": VALI_TRAIN_RATIO,
            "MID_START": MID_START,
        }
        write_json(out_dir / f"{in_path.stem}_refmeta_stat.json", stat)
        print(f"[OK] Split done ({obj}): {in_path} -> {out_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    run()
