"""
Parse all metrics.pkl files under a given directory and print aggregate stats.

Each metrics.pkl is produced by rebot-net/eval.py and contains:
    {
        "per_object": {obj_idx: {"MSE": ..., "PSNR": ..., "SSIM": ..., "LPIPS": ...}, ...},
        "average":    {"MSE": ..., "PSNR": ..., "SSIM": ..., "LPIPS": ...}
    }

Usage:
    python parse_metrics_rebotnet.py --dir log/rebot_eval_M
    python parse_metrics_rebotnet.py --dir log/rebot_eval_M --verbose
"""

import argparse
import os
import pickle
from os import path as osp


METRIC_KEYS = ["MSE", "PSNR", "SSIM", "LPIPS"]


def find_metrics_pkls(root):
    """Recursively find all metrics.pkl files under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics.pkl" in filenames:
            paths.append(osp.join(dirpath, "metrics.pkl"))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate rebot-net metrics.pkl files and print average metrics."
    )
    parser.add_argument("--dir", required=True, type=str,
                        help="Root directory to search for metrics.pkl files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print per-object averages.")
    args = parser.parse_args()

    pkl_paths = find_metrics_pkls(args.dir)
    if not pkl_paths:
        print(f"No metrics.pkl files found under: {args.dir}")
        return

    print(f"Found {len(pkl_paths)} metrics.pkl file(s) under: {args.dir}\n")

    all_object_metrics = {}  # (pkl_path, obj_idx) -> metric dict

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        per_object = data.get("per_object", {})
        pkl_label = osp.relpath(osp.dirname(pkl_path), args.dir)

        if args.verbose:
            obj_avg = data.get("average", {})
            if obj_avg:
                print(f"  [{pkl_label}]  MSE: {obj_avg['MSE']:.5f} | "
                      f"PSNR: {obj_avg['PSNR']:.2f} | "
                      f"SSIM: {obj_avg['SSIM']:.4f} | "
                      f"LPIPS: {obj_avg['LPIPS']:.4f}  "
                      f"({len(per_object)} object(s))")
            for obj_idx, m in sorted(per_object.items()):
                print(f"    obj {obj_idx:4d}:  MSE: {m['MSE']:.5f} | "
                      f"PSNR: {m['PSNR']:.2f} | "
                      f"SSIM: {m['SSIM']:.4f} | "
                      f"LPIPS: {m['LPIPS']:.4f}")

        for obj_idx, m in per_object.items():
            all_object_metrics[(pkl_label, obj_idx)] = m

    if not all_object_metrics:
        print("No per-object metrics found in any pkl file.")
        return

    n = len(all_object_metrics)
    global_avg = {k: sum(m[k] for m in all_object_metrics.values()) / n
                  for k in METRIC_KEYS}

    if args.verbose:
        print()
    print(f"{'='*60}")
    print(f"Global average over {n} object(s) "
          f"({len(pkl_paths)} pkl file(s))")
    print(f"{'='*60}")
    print(f"  MSE  : {global_avg['MSE']:.5f}")
    print(f"  PSNR : {global_avg['PSNR']:.2f}")
    print(f"  SSIM : {global_avg['SSIM']:.4f}")
    print(f"  LPIPS: {global_avg['LPIPS']:.4f}")
    print(f"{'='*60}")
    print("\nTSV:")
    print("MSE\tPSNR\tSSIM\tLPIPS")
    print(f"{global_avg['MSE']:.5f}\t{global_avg['PSNR']:.2f}\t"
          f"{global_avg['SSIM']:.4f}\t{global_avg['LPIPS']:.4f}")


if __name__ == "__main__":
    main()
