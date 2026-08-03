"""Aggregate the already-computed coarse-transfer metrics over the 50 eval objects.

The ground-truth-retrieval benchmark coarse transfer lives in
log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue/{obj}/metrics.pkl
(written by main_retrieval_transfer_feat_match.py --eval). The eval split is the
last 50 object ids, matching rebot_net/eval.py's all_ids[950:].
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
TRANSFER = os.path.join(
    ROOT, "log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue")


def main():
    all_ids = sorted(int(d) for d in os.listdir(TRANSFER)
                     if os.path.isdir(os.path.join(TRANSFER, d)))
    test_ids = all_ids[950:]
    print(f"Eval objects: {len(test_ids)}  ({test_ids[0]}..{test_ids[-1]})")

    per_object = {}
    missing = []
    for oid in test_ids:
        p = os.path.join(TRANSFER, str(oid), "metrics.pkl")
        if not os.path.exists(p):
            missing.append(oid)
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        per_object[oid] = {k: float(v) for k, v in d["average"].items()}

    if missing:
        print(f"WARNING: {len(missing)} objects missing metrics.pkl: {missing}")

    keys = ["MSE", "PSNR", "SSIM", "LPIPS"]
    avg = {k: float(np.mean([m[k] for m in per_object.values()])) for k in keys}
    std = {k: float(np.std([m[k] for m in per_object.values()])) for k in keys}

    print(f"\nOurs (coarse transfer), {len(per_object)} objects")
    print("MSE\t\tPSNR\tSSIM\tLPIPS")
    print(f"{avg['MSE']:.5f}\t{avg['PSNR']:.2f}\t{avg['SSIM']:.4f}\t{avg['LPIPS']:.4f}")

    out = {"method": "ours_coarse", "n_objects": len(per_object),
           "average": avg, "std": std, "per_object": per_object,
           "missing": missing}
    out_path = os.path.join(ROOT, "paper_experiments/job1_gt_retrieval/ours_coarse.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
