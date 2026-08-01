"""Re-score predictions after projecting them onto unit-length surface normals.

Tactile-normal videos encode a surface normal n (|n| = 1) as
    rgb = uint8((n + 1) / 2 * 255)
A generative model has no constraint forcing its output back onto the unit
sphere, so part of its error can be pure magnitude error rather than a wrong
normal *direction*. This script decodes each predicted frame, renormalises

    n = rgb / 255 * 2 - 1        ->        n / max(|n|, eps)

re-encodes, and recomputes MSE / PSNR / SSIM / LPIPS against the same ground
truth, so the comparison is about direction only.

IMPORTANT: the stored ground truth is itself NOT unit-norm (mean |n| ~ 0.96),
because the encoding is uint8-quantised and the videos are compressed. Projecting
only the prediction onto the sphere therefore moves it AWAY from the GT and makes
every method look worse. Use --renormalise_gt to put BOTH sides on the sphere,
which is the meaningful "direction error only" comparison.

It reports the metrics BEFORE and AFTER for every method, plus how far each
method's raw output already sits from unit norm -- renormalising a method that
is already on the sphere is a no-op, which is what makes applying the same
transform to everyone fair rather than a thumb on the scale for one baseline.
"""
import argparse
import json
import os
import pickle

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
EPS = 1e-6

# method -> (directory template holding {q}_transferred.mp4, gt template)
JOB1_SOURCES = {
    "TaRF": ("log/paper_job1_baselines/tarf/{obj}/transfer", "{q}_transferred.mp4"),
    "Tactile Normal Quilting": ("log/paper_job1_baselines/quilting/{obj}/transfer", "{q}_transferred.mp4"),
    "ObjectFolder INR": ("log/paper_job1_baselines/inr/{obj}/transfer", "{q}_transferred.mp4"),
    "Ours (coarse transfer, normals)": ("log/paper_job1_transfer_normal/{obj}", "{q}_transferred.mp4"),
    "Ours (refined, normals)": ("log/paper_job1_refine_ours_normal/videos", "{obj}_{q}_enhanced.mp4"),
}
GT_DIR = "log/paper_job1_transfer_normal/{obj}"
GT_NAME = "{q}_query_tactile_normal.mp4"


def read_video(path):
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return frames or None


def to_normal(rgb):
    """[0,1] RGB -> normal vector field in [-1,1]."""
    return rgb * 2.0 - 1.0


def to_rgb(n):
    return np.clip((n + 1.0) / 2.0, 0.0, 1.0)


def renormalise(rgb):
    """Project an encoded frame back onto unit-length normals."""
    n = to_normal(rgb)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(mag < EPS, np.array([0.0, 0.0, 1.0], np.float32), n / np.maximum(mag, EPS))
    return to_rgb(n)


def norm_stats(rgb):
    mag = np.linalg.norm(to_normal(rgb), axis=-1)
    return float(mag.mean()), float(np.abs(mag - 1.0).mean())


def score(pred_frames, gt_frames, lpips_model, device):
    n = min(len(pred_frames), len(gt_frames))
    vals = {k: [] for k in ("MSE", "PSNR", "SSIM", "LPIPS")}
    for i in range(n):
        p, g = pred_frames[i], gt_frames[i]
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]))
        mse = compute_mse(g, p)
        vals["MSE"].append(mse)
        vals["PSNR"].append(compute_psnr(g, p, data_range=1.0) if mse > 0 else 100.0)
        vals["SSIM"].append(compute_ssim(g, p, data_range=1.0, channel_axis=-1))
        gt_t = torch.from_numpy(g).permute(2, 0, 1)[None].to(device) * 2 - 1
        pr_t = torch.from_numpy(p).permute(2, 0, 1)[None].to(device) * 2 - 1
        vals["LPIPS"].append(lpips_model(gt_t, pr_t).item())
    return {k: float(np.mean(v)) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", type=int,
                    default=list(range(951, 1001)))
    ap.add_argument("--touches", type=int, default=8)
    ap.add_argument("--methods", nargs="+", default=list(JOB1_SOURCES))
    ap.add_argument("--renormalise_gt", action="store_true",
                    help="Also project the ground truth onto unit normals, so the "
                         "comparison measures direction error only (recommended: "
                         "the stored GT is not unit-norm).")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "paper_experiments/job1_gt_retrieval/renormalised.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = lpips.LPIPS(net="alex").to(device)
    for p in lp.parameters():
        p.requires_grad_(False)

    results = {}
    for method in args.methods:
        dir_tmpl, name_tmpl = JOB1_SOURCES[method]
        raw_obj, norm_obj, mags, devs = [], [], [], []
        for obj in args.objects:
            raw_t, norm_t = [], []
            for q in range(args.touches):
                pred_path = os.path.join(ROOT, dir_tmpl.format(obj=obj),
                                         name_tmpl.format(obj=obj, q=q))
                gt_path = os.path.join(ROOT, GT_DIR.format(obj=obj),
                                       GT_NAME.format(q=q))
                pred = read_video(pred_path)
                gt = read_video(gt_path)
                if pred is None or gt is None:
                    continue
                m, d = norm_stats(np.stack(pred))
                mags.append(m)
                devs.append(d)
                gt_for_norm = [renormalise(f) for f in gt] if args.renormalise_gt else gt
                with torch.no_grad():
                    raw_t.append(score(pred, gt, lp, device))
                    norm_t.append(score([renormalise(f) for f in pred],
                                        gt_for_norm, lp, device))
            if raw_t:
                raw_obj.append({k: float(np.mean([t[k] for t in raw_t])) for k in raw_t[0]})
                norm_obj.append({k: float(np.mean([t[k] for t in norm_t])) for k in norm_t[0]})

        if not raw_obj:
            print(f"{method:34s}  no data")
            continue
        agg = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[method] = {
            "n_objects": len(raw_obj),
            "raw": agg(raw_obj),
            "renormalised": agg(norm_obj),
            "mean_normal_magnitude": float(np.mean(mags)),
            "mean_abs_deviation_from_unit": float(np.mean(devs)),
        }
        r, nrm = results[method]["raw"], results[method]["renormalised"]
        print(f"{method:34s} |n|={results[method]['mean_normal_magnitude']:.3f} "
              f"(dev {results[method]['mean_abs_deviation_from_unit']:.3f})  "
              f"PSNR {r['PSNR']:.2f} -> {nrm['PSNR']:.2f}   "
              f"SSIM {r['SSIM']:.4f} -> {nrm['SSIM']:.4f}   "
              f"LPIPS {r['LPIPS']:.4f} -> {nrm['LPIPS']:.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
