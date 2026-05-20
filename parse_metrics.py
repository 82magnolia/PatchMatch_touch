"""
Parse all metrics.pkl files under a given directory and print aggregate stats.

Each metrics.pkl is produced by main_retrieval_transfer_accel.py --eval and
contains:
    {
        "per_touch": {query_idx: {"MSE": ..., "PSNR": ..., "SSIM": ..., "LPIPS": ...}, ...},
        "average":   {"MSE": ..., "PSNR": ..., "SSIM": ..., "LPIPS": ...}
    }

Usage:
    python parse_metrics.py --dir log/transfer
    python parse_metrics.py --dir log/transfer --verbose
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
        description="Aggregate metrics.pkl files and print average metrics."
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

    all_touch_metrics = {}  # (pkl_path, query_idx) -> metric dict

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        per_touch = data.get("per_touch", {})
        obj_label = osp.relpath(osp.dirname(pkl_path), args.dir)

        if args.verbose:
            obj_avg = data.get("average", {})
            if obj_avg:
                print(f"  [{obj_label}]  MSE: {obj_avg['MSE']:.5f} | "
                      f"PSNR: {obj_avg['PSNR']:.2f} | "
                      f"SSIM: {obj_avg['SSIM']:.4f} | "
                      f"LPIPS: {obj_avg['LPIPS']:.4f}  "
                      f"({len(per_touch)} touch locations)")

        for q_idx, m in per_touch.items():
            all_touch_metrics[(obj_label, q_idx)] = m

    if not all_touch_metrics:
        print("No per-touch metrics found in any pkl file.")
        return

    n = len(all_touch_metrics)
    global_avg = {k: sum(m[k] for m in all_touch_metrics.values()) / n
                  for k in METRIC_KEYS}

    if args.verbose:
        print()
    print(f"{'='*60}")
    print(f"Global average over {n} touch locations "
          f"({len(pkl_paths)} object(s))")
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
