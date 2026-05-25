import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from match_cost import match_cost


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_pointcloud_txt(txt_path: Path) -> torch.Tensor:
    arr = np.loadtxt(str(txt_path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(f"点云列数不足3：{txt_path}，shape={arr.shape}")
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    return torch.from_numpy(arr)  # (N, 3)


def resample_point_cloud(pc: torch.Tensor, target_N: int) -> torch.Tensor:
    """
    重新采样点云到 target_N 个点。
    若点数过多则随机下采样，若点数不足则重复采样。
    """
    N, _ = pc.shape
    if N == target_N:
        return pc
    elif N > target_N:
        idx = torch.randperm(N, device=pc.device)[:target_N]
    else:
        idx = torch.randint(0, N, (target_N,), device=pc.device)
    return pc[idx]


def pick_txt_in_dir(d: Path) -> Path:
    txts = sorted(d.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"目录下没有txt：{d}")
    refmeta = [p for p in txts if "refmeta" in p.name.lower()]
    return refmeta[0] if refmeta else txts[0]


def find_pc_file(sample_dir: Path, region: str, tag: str) -> Path:
    """
    默认结构（与你截图/原脚本一致）：
      {sample}/{region}/{sample}_{tag}/{sample}_{tag}_test_refmeta.txt

    若不完全匹配，会回退到：
      1) 在 region 下找名字包含 tag 的子目录
      2) 在该子目录内挑一个 txt（优先 refmeta）
    """
    sid = sample_dir.name
    region_dir = sample_dir / region
    if not region_dir.exists():
        raise FileNotFoundError(f"缺少区域目录：{region_dir}")

    # 优先严格匹配：{sid}_{tag}
    cand_dir = region_dir / f"{sid}_{tag}"
    if cand_dir.exists() and cand_dir.is_dir():
        preferred = cand_dir / f"{sid}_{tag}_test_refmeta.txt"
        if preferred.exists():
            return preferred
        return pick_txt_in_dir(cand_dir)

    # 回退：region 下找任何包含 tag 的子目录
    subdirs = [p for p in region_dir.iterdir() if p.is_dir()]
    hit = [p for p in subdirs if p.name.lower().endswith(f"_{tag}") or (tag.lower() in p.name.lower())]
    if hit:
        return pick_txt_in_dir(hit[0])

    raise FileNotFoundError(f"找不到点云文件：sample={sid}, region={region}, tag={tag}（期望目录 {cand_dir}）")


@torch.no_grad()
def compute_emd_single_pair(x_tensor: torch.Tensor, recon_tensor: torch.Tensor, target_n: int = 0) -> float:
    """
    计算单对点云的 approximate EMD（match_cost / Sinkhorn）。
    输入:
      x_tensor: (N,3) 参考点云
      recon_tensor: (M,3) 重建点云
    target_n:
      - 0 表示用 min(N,M)
      - >0 表示都重采样到 target_n（更利于跨样本公平对比，也更稳定）
    """
    if target_n and target_n > 0:
        N = target_n
    else:
        N = min(x_tensor.shape[0], recon_tensor.shape[0])

    x_sampled = resample_point_cloud(x_tensor, N).unsqueeze(0)         # (1,N,3)
    recon_sampled = resample_point_cloud(recon_tensor, N).unsqueeze(0) # (1,N,3)

    emd = match_cost(x_sampled, recon_sampled)  # (1,)
    return float(emd.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="包含样本子目录的根目录（默认当前目录）")
    parser.add_argument("--regions", type=str, default="head,mid,tail", help="区域列表")
    parser.add_argument("--methods", type=str, default="rie,shape,diff,snow,Shape2vecset", help="方法列表（不含gt）")
    parser.add_argument("--device", type=str, default="cuda", help="auto/cuda/cpu/cuda:0 ...")
    parser.add_argument("--target_n", type=int, default=0, help="EMD重采样点数：0用min(N,M)，>0固定点数")
    parser.add_argument("--out_results", type=str, default="emd_results.csv")
    parser.add_argument("--out_summary", type=str, default="emd_summary_region_mean.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    device = get_device(args.device)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # 样本目录：名字为数字（包含 04 这种）
    sample_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    sample_dirs.sort(key=lambda p: int(p.name))

    if not sample_dirs:
        raise RuntimeError(f"在 {root} 下未找到数字命名的样本目录")

    rows = []
    sum_emd = defaultdict(float)
    cnt_emd = defaultdict(int)

    print(f"[INFO] root={root}")
    print(f"[INFO] device={device}")
    print(f"[INFO] samples={len(sample_dirs)} | regions={regions} | methods={methods} | target_n={args.target_n}")

    for sdir in sample_dirs:
        sid = sdir.name
        for region in regions:
            try:
                gt_path = find_pc_file(sdir, region, "gt")
            except Exception as e:
                print(f"[WARN] 跳过：sample={sid} region={region}（GT缺失）: {e}")
                continue

            gt = load_pointcloud_txt(gt_path).to(device)

            for method in methods:
                try:
                    recon_path = find_pc_file(sdir, region, method)
                except Exception as e:
                    print(f"[WARN] 跳过：sample={sid} region={region} method={method}（重构缺失）: {e}")
                    continue

                recon = load_pointcloud_txt(recon_path).to(device)

                try:
                    emd = compute_emd_single_pair(gt, recon, target_n=args.target_n)
                except Exception as e:
                    print(f"[WARN] 计算失败：sample={sid} region={region} method={method}: {e}")
                    continue

                rows.append({
                    "sample": sid,
                    "region": region,
                    "method": method,
                    "emd": emd,
                    "gt_file": str(gt_path.relative_to(root)),
                    "recon_file": str(recon_path.relative_to(root)),
                })

                key = (region, method)
                sum_emd[key] += emd
                cnt_emd[key] += 1

    # 写明细
    out_results = root / args.out_results
    with out_results.open("w", encoding="utf-8") as f:
        f.write("sample,region,method,emd,gt_file,recon_file\n")
        for r in rows:
            f.write(f"{r['sample']},{r['region']},{r['method']},{r['emd']},{r['gt_file']},{r['recon_file']}\n")

    # 写区域均值汇总
    out_summary = root / args.out_summary
    with out_summary.open("w", encoding="utf-8") as f:
        f.write("region,method,mean_emd,n\n")
        for region in regions:
            for method in methods:
                key = (region, method)
                n = cnt_emd.get(key, 0)
                mean_emd = (sum_emd[key] / n) if n > 0 else ""
                f.write(f"{region},{method},{mean_emd},{n}\n")

    print(f"[DONE] 结果已保存：{out_results}")
    print(f"[DONE] 汇总已保存：{out_summary}")

    print("\n[SUMMARY] mean EMD by (region, method):")
    for region in regions:
        for method in methods:
            key = (region, method)
            n = cnt_emd.get(key, 0)
            if n == 0:
                print(f"  {region:>4s} | {method:<6s} : n=0 (missing)")
            else:
                print(f"  {region:>4s} | {method:<6s} : mean={sum_emd[key]/n:.6f} (n={n})")


if __name__ == "__main__":
    main()
