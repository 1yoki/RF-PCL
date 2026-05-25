#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np


def load_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError(f"Need >=3 columns (x y z). Got {arr.shape}")
    return arr


def compute_bins_from_ref(ref: np.ndarray, split_dim_id: int, split_n: int):
    coord = ref[:, split_dim_id]
    minv = float(coord.min())
    maxv = float(coord.max())
    span = maxv - minv
    if split_n <= 0:
        raise ValueError("split_n must be > 0")

    if np.isclose(span, 0.0):
        step = 0.0
    else:
        step = span / float(split_n)

    return minv, maxv, span, step


def assign_bins(data: np.ndarray, split_dim_id: int, split_n: int, minv: float, step: float):
    """
    Assign points to bins using reference minv & step.
    Points outside range will be clipped to [0, split_n-1].
    """
    x = data[:, split_dim_id]
    if np.isclose(step, 0.0):
        bins = np.zeros((data.shape[0],), dtype=np.int64)
        return bins

    raw = np.floor((x - minv) / step).astype(np.int64)
    bins = np.clip(raw, 0, split_n - 1)
    return bins


def split_train_test_masks(bin_indices: np.ndarray, split_n: int, dataset: str, obj: str,
                          vali_train_ratio: int, mid_start: int):
    if not (0 <= vali_train_ratio <= split_n):
        raise ValueError(f"vali_train_ratio must be in [0, {split_n}], got {vali_train_ratio}")

    dataset = (dataset or "").lower()
    obj = (obj or "").lower()

    if dataset == "david":
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
        raise ValueError('Set either --dataset david OR --object head/tail/mid.')
    return train_mask, test_mask


def random_sample(arr: np.ndarray, k: int, rng: np.random.Generator, replace: bool):
    if k < 0:
        raise ValueError("k must be >=0")
    n = arr.shape[0]
    if n == 0:
        return arr
    idx = rng.choice(n, size=k, replace=replace)
    return arr[idx]


def match_counts(orig_local: np.ndarray,
                 recon_local: np.ndarray,
                 mode: str,
                 target_n: int,
                 rng: np.random.Generator,
                 upsample: str,
                 jitter_sigma: float):
    """
    mode:
      - "min": downsample both to min(n1,n2)
      - "orig": make recon match orig count (upsample recon if needed)
      - "fixed": make both equal to target_n
    upsample (used when need increase points):
      - "repeat": sample with replacement
      - "jitter": sample with replacement then add gaussian noise to xyz
    """
    n1 = orig_local.shape[0]
    n2 = recon_local.shape[0]

    if mode == "min":
        k = min(n1, n2)
        o = random_sample(orig_local, k, rng, replace=False) if n1 > k else orig_local
        r = random_sample(recon_local, k, rng, replace=False) if n2 > k else recon_local
        return o, r

    if mode == "orig":
        k = n1
        # orig: keep as-is (or downsample if you want exactly k but n1==k anyway)
        o = orig_local
        # recon: downsample or upsample to k
        if n2 >= k:
            r = random_sample(recon_local, k, rng, replace=False)
        else:
            if upsample not in ("repeat", "jitter"):
                raise ValueError("--upsample must be repeat|jitter")
            r = random_sample(recon_local, k, rng, replace=True) if n2 > 0 else recon_local
            if upsample == "jitter" and r.shape[0] > 0 and jitter_sigma > 0:
                r = r.copy()
                r[:, :3] += rng.normal(0.0, jitter_sigma, size=(r.shape[0], 3))
        return o, r

    if mode == "fixed":
        if target_n is None or target_n <= 0:
            raise ValueError("mode=fixed requires --target_n > 0")
        k = target_n

        # orig
        if n1 >= k:
            o = random_sample(orig_local, k, rng, replace=False)
        else:
            o = random_sample(orig_local, k, rng, replace=True) if n1 > 0 else orig_local

        # recon
        if n2 >= k:
            r = random_sample(recon_local, k, rng, replace=False)
        else:
            r = random_sample(recon_local, k, rng, replace=True) if n2 > 0 else recon_local

        # optional jitter (if you want it, apply when upsample happened)
        if upsample == "jitter" and jitter_sigma > 0:
            if n1 < k and o.shape[0] > 0:
                o = o.copy()
                o[:, :3] += rng.normal(0.0, jitter_sigma, size=(o.shape[0], 3))
            if n2 < k and r.shape[0] > 0:
                r = r.copy()
                r[:, :3] += rng.normal(0.0, jitter_sigma, size=(r.shape[0], 3))
        return o, r

    raise ValueError("mode must be min|orig|fixed")


def save_txt(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ["%.6f"] * arr.shape[1]
    np.savetxt(str(path), arr, fmt=fmt)


def main():
    ap = argparse.ArgumentParser("Extract same local region from orig/recon and match point counts.")
    ap.add_argument("--orig", default="./04_gt.txt", type=str, help="Original point cloud txt")
    ap.add_argument("--recon", default="./04_rie.txt", type=str, help="Reconstructed point cloud txt")
    ap.add_argument("--out_dir",default="./spliteye/04", type=str)

    ap.add_argument("--split_dim_id", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--split_n", type=int, default=10)
    ap.add_argument("--vali_train_ratio", type=int, default=7)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--object", type=str, default="head")
    ap.add_argument("--mid_start", type=int, default=3)

    ap.add_argument("--region", type=str, default="test", choices=["train", "test"],
                    help="Which local region to extract (train or test mask).")

    ap.add_argument("--mode", type=str, default="min", choices=["min", "orig", "fixed"],
                    help="How to match counts. min is best for EMD.")
    ap.add_argument("--target_n", type=int, default=None,
                    help="Used when mode=fixed.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--upsample", type=str, default="repeat", choices=["repeat", "jitter"],
                    help="Upsample strategy when needed (mode=orig/fixed and cloud has fewer points).")
    ap.add_argument("--jitter_sigma", type=float, default=1e-4,
                    help="Std of Gaussian noise on xyz when upsample=jitter.")

    args = ap.parse_args()

    orig = load_txt(Path(args.orig))
    recon = load_txt(Path(args.recon))

    # 1) define bin edges from orig (reference)
    minv, maxv, span, step = compute_bins_from_ref(orig, args.split_dim_id, args.split_n)

    # 2) assign bins for both using SAME edges
    orig_bins = assign_bins(orig, args.split_dim_id, args.split_n, minv, step)
    recon_bins = assign_bins(recon, args.split_dim_id, args.split_n, minv, step)

    # 3) build masks using the SAME rule but different bin assignments
    orig_train_mask, orig_test_mask = split_train_test_masks(
        orig_bins, args.split_n, args.dataset, args.object, args.vali_train_ratio, args.mid_start
    )
    recon_train_mask, recon_test_mask = split_train_test_masks(
        recon_bins, args.split_n, args.dataset, args.object, args.vali_train_ratio, args.mid_start
    )

    if args.region == "train":
        orig_local = orig[orig_train_mask]
        recon_local = recon[recon_train_mask]
    else:
        orig_local = orig[orig_test_mask]
        recon_local = recon[recon_test_mask]

    rng = np.random.default_rng(args.seed)

    # 4) match counts
    orig_matched, recon_matched = match_counts(
        orig_local, recon_local,
        mode=args.mode,
        target_n=args.target_n,
        rng=rng,
        upsample=args.upsample,
        jitter_sigma=args.jitter_sigma
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_o = Path(args.orig).stem
    base_r = Path(args.recon).stem
    tag = f"dim{args.split_dim_id}_n{args.split_n}_{args.region}_{args.mode}"

    save_txt(out_dir / f"{base_o}_{tag}.txt", orig_matched)
    save_txt(out_dir / f"{base_r}_{tag}.txt", recon_matched)

    meta = {
        "orig": args.orig,
        "recon": args.recon,
        "split_dim_id": args.split_dim_id,
        "split_n": args.split_n,
        "vali_train_ratio": args.vali_train_ratio,
        "dataset": args.dataset,
        "object": args.object,
        "mid_start": args.mid_start,
        "region": args.region,
        "mode": args.mode,
        "target_n": args.target_n,
        "upsample": args.upsample,
        "jitter_sigma": args.jitter_sigma,
        "ref_min": minv,
        "ref_max": maxv,
        "ref_span": span,
        "ref_step": step,
        "orig_local_before": int(orig_local.shape[0]),
        "recon_local_before": int(recon_local.shape[0]),
        "orig_local_after": int(orig_matched.shape[0]),
        "recon_local_after": int(recon_matched.shape[0]),
        "num_cols_orig": int(orig.shape[1]),
        "num_cols_recon": int(recon.shape[1]),
    }
    with open(out_dir / f"meta_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("orig matched :", out_dir / f"{base_o}_{tag}.txt")
    print("recon matched:", out_dir / f"{base_r}_{tag}.txt")
    print("meta         :", out_dir / f"meta_{tag}.json")


if __name__ == "__main__":
    main()
