import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def load_pointcloud_txt(txt_path: Path) -> torch.Tensor:
    arr = np.loadtxt(str(txt_path), dtype=np.float32)
    if arr.ndim == 1:
        # 可能是单行数据
        arr = arr.reshape(1, -1)

    # 如果有多列（比如 xyz+其他），默认取前 3 列作为点坐标
    if arr.shape[1] < 3:
        raise ValueError(f"点云列数不足3：{txt_path}，shape={arr.shape}")
    if arr.shape[1] > 3:
        arr = arr[:, :3]

    return torch.from_numpy(arr)  # (N, 3)

def pick_txt_in_dir(d: Path) -> Path:
    # 优先选择包含 refmeta 的 txt；否则选第一个 txt
    txts = sorted(d.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"目录下没有txt：{d}")
    refmeta = [p for p in txts if "refmeta" in p.name.lower()]
    return refmeta[0] if refmeta else txts[0]

def find_pc_file(sample_dir: Path, region: str, tag: str) -> Path:
    """
    默认你的结构是：
      {sample}/{region}/{sample}_{tag}/{sample}_{tag}_test_refmeta.txt
    若文件名不完全匹配，则回退到该目录下任意 *.txt（优先 refmeta）。
    """
    sid = sample_dir.name
    region_dir = sample_dir / region
    if not region_dir.exists():
        raise FileNotFoundError(f"缺少区域目录：{region_dir}")

    # 优先按你现在的命名规则找
    cand_dir = region_dir / f"{sid}_{tag}"
    if cand_dir.exists() and cand_dir.is_dir():
        preferred = cand_dir / f"{sid}_{tag}_test_refmeta.txt"
        if preferred.exists():
            return preferred
        return pick_txt_in_dir(cand_dir)

    # 回退：在 region 里找任何包含 tag 的子目录
    subdirs = [p for p in region_dir.iterdir() if p.is_dir()]
    hit = [p for p in subdirs if p.name.lower().endswith(f"_{tag}") or (tag.lower() in p.name.lower())]
    if hit:
        return pick_txt_in_dir(hit[0])

    raise FileNotFoundError(f"找不到点云文件：sample={sid}, region={region}, tag={tag}（期望目录 {cand_dir}）")

def chamfer_distance_chunked(x: torch.Tensor, y: torch.Tensor, chunk: int = 4096) -> float:
    """
    对称 Chamfer Distance，使用 chunk 降低 torch.cdist 的显存/内存峰值。
    x: (N, 3), y: (M, 3)
    返回 float
    """
    assert x.ndim == 2 and y.ndim == 2 and x.shape[1] == y.shape[1] == 3

    def one_way_mean_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # mean_i min_j ||a_i - b_j||
        mins = []
        for i in range(0, a.shape[0], chunk):
            a_chunk = a[i:i+chunk]  # (c, 3)
            dist = torch.cdist(a_chunk, b, p=2)  # (c, M)
            #dist = dist ** 2
            mins.append(dist.min(dim=1).values)  # (c,)
        mins = torch.cat(mins, dim=0)
        return mins.mean()

    cd_f = one_way_mean_min(x, y)
    cd_b = one_way_mean_min(y, x)
    return (cd_f + cd_b).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="包含样本子目录的根目录（默认当前目录）")
    parser.add_argument("--regions", type=str, default="head,mid,tail", help="区域列表")
    parser.add_argument("--methods", type=str, default="rie,shape,diff,snow,shape2vecset", help="方法列表（不含gt）")
    parser.add_argument("--device", type=str, default="cuda", help="auto/cuda/cpu/cuda:0 ...")
    parser.add_argument("--chunk", type=int, default=2048, help="cdist 分块大小，点数大时可调小")
    parser.add_argument("--out_results", type=str, default="cd_results.csv")
    parser.add_argument("--out_summary", type=str, default="cd_summary_region_mean.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    device = get_device(args.device)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # 遍历样本目录：名字为纯数字（包括 04 这种）
    sample_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    sample_dirs.sort(key=lambda p: int(p.name))  # 04 -> 4，排序更自然

    if not sample_dirs:
        raise RuntimeError(f"在 {root} 下未找到数字命名的样本目录")

    rows = []
    sum_cd = defaultdict(float)
    cnt_cd = defaultdict(int)

    print(f"[INFO] root={root}")
    print(f"[INFO] device={device}")
    print(f"[INFO] samples={len(sample_dirs)} | regions={regions} | methods={methods}")

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
                cd = chamfer_distance_chunked(recon, gt, chunk=args.chunk)

                rows.append({
                    "sample": sid,
                    "region": region,
                    "method": method,
                    "cd": cd,
                    "gt_file": str(gt_path.relative_to(root)),
                    "recon_file": str(recon_path.relative_to(root)),
                })

                key = (region, method)
                sum_cd[key] += cd
                cnt_cd[key] += 1

    # 写明细结果
    out_results = root / args.out_results
    with out_results.open("w", encoding="utf-8") as f:
        f.write("sample,region,method,cd,gt_file,recon_file\n")
        for r in rows:
            f.write(f"{r['sample']},{r['region']},{r['method']},{r['cd']},{r['gt_file']},{r['recon_file']}\n")

    # 写区域均值汇总
    out_summary = root / args.out_summary
    with out_summary.open("w", encoding="utf-8") as f:
        f.write("region,method,mean_cd,n\n")
        for region in regions:
            for method in methods:
                key = (region, method)
                n = cnt_cd.get(key, 0)
                mean_cd = (sum_cd[key] / n) if n > 0 else ""
                f.write(f"{region},{method},{mean_cd},{n}\n")

    print(f"[DONE] 结果已保存：{out_results}")
    print(f"[DONE] 汇总已保存：{out_summary}")

    # 同时在终端打印一下汇总
    print("\n[SUMMARY] mean CD by (region, method):")
    for region in regions:
        for method in methods:
            key = (region, method)
            n = cnt_cd.get(key, 0)
            if n == 0:
                print(f"  {region:>4s} | {method:<6s} : n=0 (missing)")
            else:
                print(f"  {region:>4s} | {method:<6s} : mean={sum_cd[key]/n:.6f} (n={n})")

if __name__ == "__main__":
    main()
