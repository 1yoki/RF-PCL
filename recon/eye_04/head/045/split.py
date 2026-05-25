import argparse
import json
from pathlib import Path
import numpy as np


def load_point_cloud_txt(path: Path) -> np.ndarray:
    """
    Load point cloud from txt.
    Supports whitespace-delimited numeric columns.
    Returns: (N, C) float64
    """
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        # single line
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"Expect at least 3 columns (x y z), got shape={data.shape}")
    return data


def compute_bins(data: np.ndarray, split_dim_id: int, split_n: int) -> dict:
    """
    Compute bin indices for each point along split_dim_id into split_n equal spans.
    Mimics:
        min_value = data[:, split_dim_id].min()
        max_value = data[:, split_dim_id].max()
        step_size = (max-min) / split_n
        bin_indices = ((x - min) // step_size)
        clamp to [0, split_n-1]
    """
    coord = data[:, split_dim_id]
    min_value = float(coord.min())
    max_value = float(coord.max())
    span = max_value - min_value

    if split_n <= 0:
        raise ValueError("split_n must be > 0")

    # Handle degenerate span to avoid division by zero
    if np.isclose(span, 0.0):
        step_size = 0.0
        bin_indices = np.zeros((data.shape[0],), dtype=np.int64)
    else:
        step_size = span / float(split_n)
        # floor((x - min) / step_size) is equivalent to integer // for positive step_size
        raw = np.floor((coord - min_value) / step_size).astype(np.int64)
        bin_indices = np.clip(raw, 0, split_n - 1)

    return {
        "min_value": min_value,
        "max_value": max_value,
        "span": span,
        "step_size": float(step_size),
        "bin_indices": bin_indices,
    }


def split_train_test(
    bin_indices: np.ndarray,
    split_n: int,
    dataset: str,
    obj: str,
    vali_train_ratio: int,
    mid_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce your conditional rules.
    Returns (train_mask, test_mask)
    """
    if not (0 <= vali_train_ratio <= split_n):
        raise ValueError(f"vali_train_ratio must be in [0, {split_n}], got {vali_train_ratio}")

    dataset = (dataset or "").lower()
    obj = (obj or "").lower()

    if dataset == "david":
        # 后 x 份作为训练集，前 (split_n - x) 份作为测试集（按你原代码的写法）
        train_mask = bin_indices >= vali_train_ratio
        test_mask = bin_indices < vali_train_ratio

    elif obj == "head":
        # 头段作为测试集：前 x 份训练，后 (split_n-x) 测试
        train_mask = bin_indices < vali_train_ratio
        test_mask = bin_indices >= vali_train_ratio

    elif obj == "tail":
        # 尾段作为测试集：后 x 份训练，前 (split_n-x) 测试（与你原逻辑一致）
        train_mask = bin_indices >= (split_n - vali_train_ratio)
        test_mask = bin_indices < (split_n - vali_train_ratio)

    elif obj == "mid":
        # 中段作为测试集：默认 mid_start=3，测试 [mid_start, vali_train_ratio)
        train_mask = (bin_indices < mid_start) | (bin_indices >= vali_train_ratio)
        test_mask = (bin_indices >= mid_start) & (bin_indices < vali_train_ratio)

    else:
        raise ValueError(
            "No split rule matched. Please set either --dataset david OR --object head/tail/mid."
        )

    return train_mask, test_mask


def save_txt(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # format for C columns
    fmt = ["%.6f"] * arr.shape[1]
    np.savetxt(str(path), arr, fmt=fmt)


def main():
    ap = argparse.ArgumentParser(
        description="Split point cloud txt by equal-span bins along one dimension; save bins + train/test."
    )
    ap.add_argument("--input", type=str, default='./04_gt.txt', help="Input point cloud txt path.")
    ap.add_argument("--out_dir", type=str, default='./spliteye/04', help="Output directory.")
    ap.add_argument("--split_dim_id", type=int, default=1, choices=[0, 1, 2], help="0:x, 1:y, 2:z")
    ap.add_argument("--split_n", type=int, default=10, help="Number of bins (segments).")
    ap.add_argument(
        "--vali_train_ratio",
        type=int,
        default=7,
        help="Integer threshold used in rules (not a float ratio).",
    )
    ap.add_argument("--dataset", type=str, default=None, help='Use rule for dataset, e.g. "david".')
    ap.add_argument("--object", type=str, default="head", help='Use rule for object: "head"|"tail"|"mid".')
    ap.add_argument(
        "--mid_start",
        type=int,
        default=3,
        help='For object="mid": test bins are [mid_start, vali_train_ratio). Default 3 (matches your code).',
    )
    ap.add_argument(
        "--save_empty_bins",
        action="store_true",
        help="If set, also create empty bin files (usually not needed).",
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_point_cloud_txt(in_path)

    bins_info = compute_bins(data, args.split_dim_id, args.split_n)
    bin_indices = bins_info["bin_indices"]

    # 1) save each segment
    base = in_path.stem
    parts_dir = out_dir / f"{base}_parts_dim{args.split_dim_id}_n{args.split_n}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    for i in range(args.split_n):
        part = data[bin_indices == i]
        counts.append(int(part.shape[0]))
        if part.shape[0] == 0 and not args.save_empty_bins:
            continue
        save_txt(parts_dir / f"{base}_part_{i:02d}.txt", part)

    # 2) train/test split according to rules
    train_mask, test_mask = split_train_test(
        bin_indices=bin_indices,
        split_n=args.split_n,
        dataset=args.dataset,
        obj=args.object,
        vali_train_ratio=args.vali_train_ratio,
        mid_start=args.mid_start,
    )
    train_data = data[train_mask]
    test_data = data[test_mask]

    save_txt(out_dir / f"{base}_train.txt", train_data)
    save_txt(out_dir / f"{base}_test.txt", test_data)

    # meta info
    meta = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "split_dim_id": args.split_dim_id,
        "split_n": args.split_n,
        "vali_train_ratio": args.vali_train_ratio,
        "dataset": args.dataset,
        "object": args.object,
        "mid_start": args.mid_start,
        "min_value": bins_info["min_value"],
        "max_value": bins_info["max_value"],
        "span": bins_info["span"],
        "step_size": bins_info["step_size"],
        "bin_counts": counts,
        "train_count": int(train_data.shape[0]),
        "test_count": int(test_data.shape[0]),
        "total_count": int(data.shape[0]),
        "num_columns": int(data.shape[1]),
    }
    with open(out_dir / f"{base}_split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"- Saved parts to: {parts_dir}")
    print(f"- Saved train: {out_dir / (base + '_train.txt')}")
    print(f"- Saved test : {out_dir / (base + '_test.txt')}")
    print(f"- Meta      : {out_dir / (base + '_split_meta.json')}")


if __name__ == "__main__":
    main()