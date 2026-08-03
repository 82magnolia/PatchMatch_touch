"""Score the press-sequence quilting predictions on the Job 1 benchmark.

quilting_video.py only writes videos; this computes MSE / PSNR / SSIM / LPIPS
against the same ground truth and writes {obj}/transfer/metrics.pkl in exactly
the layout the Job 1 aggregator already reads (a per_touch dict plus an average),
so adding the row is a one-line source path.

Metrics are computed identically to every other method in the table -- per frame,
averaged per touch, then per object -- so the new row is directly comparable.
Prints the tiled-image numbers alongside for the before/after.
"""
import argparse
import os
import pickle
import sys

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
NEW = os.path.join(ROOT, "log/paper_job1_baselines/quilting_video")
OLD = os.path.join(ROOT, "log/paper_job1_baselines/quilting")
# Score against the ORIGINAL benchmark videos, which is what the other baselines
# were scored against (they shutil.copy2 them). The transfer directory's copy is
# re-encoded with lossy mp4v and differs by ~2.2/255, which is worth ~0.2 dB --
# enough to make rows non-comparable if mixed.
GT = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")


def read_video(p):
    if not os.path.exists(p):
        return None
    c = cv2.VideoCapture(p)
    fs = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    c.release()
    return fs or None


def score(pred, gt, lp, device):
    n = min(len(pred), len(gt))
    v = {k: [] for k in ("MSE", "PSNR", "SSIM", "LPIPS")}
    for i in range(n):
        p, g = pred[i], gt[i]
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]))
        mse = compute_mse(g, p)
        v["MSE"].append(mse)
        v["PSNR"].append(compute_psnr(g, p, data_range=1.0) if mse > 0 else 100.0)
        v["SSIM"].append(compute_ssim(g, p, data_range=1.0, channel_axis=-1))
        gt_t = torch.from_numpy(g).permute(2, 0, 1)[None].to(device) * 2 - 1
        pr_t = torch.from_numpy(p).permute(2, 0, 1)[None].to(device) * 2 - 1
        v["LPIPS"].append(lp(gt_t, pr_t).item())
    return {k: float(np.mean(x)) for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", type=int, default=list(range(951, 1001)))
    ap.add_argument("--touches", type=int, default=8)
    ap.add_argument("--recompute_old", action="store_true",
                    help="Also re-score the tiled runs (they already have stored "
                         "metrics against the same original GT, so usually redundant)")
    ap.add_argument("--benchmark", choices=["job1", "job2"], default="job1")
    args = ap.parse_args()

    global NEW, OLD, GT
    if args.benchmark == "job2":
        NEW = os.path.join(ROOT, "log/paper_job2_baselines/quilting_video")
        OLD = os.path.join(ROOT, "log/paper_job2_baselines/quilting")
        GT = os.path.join(ROOT, "Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = lpips.LPIPS(net="alex").to(device)
    for p in lp.parameters():
        p.requires_grad_(False)

    new_obj, old_obj = [], []
    for oi, obj in enumerate(args.objects):
        per_touch, olds = {}, []
        for t in range(args.touches):
            gt = read_video(os.path.join(GT, str(obj), f"{t}_tactile_normal.mp4"))
            pr = read_video(os.path.join(NEW, str(obj), "transfer", f"{t}_transferred.mp4"))
            if gt is None or pr is None:
                continue
            with torch.no_grad():
                per_touch[t] = score(pr, gt, lp, device)
            if args.recompute_old:
                op = read_video(os.path.join(OLD, str(obj), "transfer", f"{t}_transferred.mp4"))
                if op is not None:
                    with torch.no_grad():
                        olds.append(score(op, gt, lp, device))
        if not per_touch:
            continue
        avg = {k: float(np.mean([m[k] for m in per_touch.values()])) for k in per_touch[0]}
        d = os.path.join(NEW, str(obj), "transfer")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.pkl"), "wb") as f:
            pickle.dump({"per_touch": per_touch, "average": avg}, f)
        new_obj.append(avg)
        if olds:
            old_obj.append({k: float(np.mean([m[k] for m in olds])) for k in olds[0]})
        if (oi + 1) % 10 == 0:
            print(f"  scored {oi + 1}/{len(args.objects)} objects", flush=True)

    agg = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    n, o = agg(new_obj), agg(old_obj) if old_obj else None
    print(f"\nTactile Normal Quilting, {len(new_obj)} objects")
    print(f"{'variant':28s} {'PSNR':>7s} {'SSIM':>8s} {'LPIPS':>8s} {'MSE':>9s}")
    print("-" * 64)
    if o:
        print(f"{'tiled still (before)':28s} {o['PSNR']:7.2f} {o['SSIM']:8.4f} "
              f"{o['LPIPS']:8.4f} {o['MSE']:9.5f}")
    print(f"{'press sequence (after)':28s} {n['PSNR']:7.2f} {n['SSIM']:8.4f} "
          f"{n['LPIPS']:8.4f} {n['MSE']:9.5f}")
    if o:
        print(f"{'delta':28s} {n['PSNR'] - o['PSNR']:+7.2f} {n['SSIM'] - o['SSIM']:+8.4f} "
              f"{n['LPIPS'] - o['LPIPS']:+8.4f} {n['MSE'] - o['MSE']:+9.5f}")


if __name__ == "__main__":
    sys.exit(main())
