"""
Is the render mask's problem *when* it fires, or *where* it lands?

sweep_render_mask.py shows that even an oracle that picks the ideal height
threshold per frame reaches only ~0.21 IoU against the tactile pseudo-GT on
contact frames. A depth schedule can only ever move along that threshold axis,
so if the ceiling is low because the projected height map sits in the wrong
place on the sensor, no amount of work on the pressing-depth estimate will
help -- the fix would belong in the ARuCO->gel calibration instead.

This measures the ceiling's sensitivity to a rigid 2D shift of the predicted
mask: for each touch's peak-contact frame, sweep (dx, dy) and re-take the
oracle over thresholds. If the best shift is consistently non-zero and lifts
IoU substantially, the dominant error is spatial calibration, not timing.
"""

import glob
import os
import sys
from os import path as osp

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from eval_render_mask import load_cache  # noqa: E402

SHIFTS = np.arange(-40, 41, 4)


def best_iou_over_thresholds(hm, valid, gt, thr_grid):
    """Oracle IoU for one frame over a threshold grid."""
    best = 0.0
    gsz = gt.sum()
    if gsz == 0:
        return np.nan
    for t in thr_grid:
        pred = (hm < t) & valid
        inter = np.count_nonzero(pred & gt)
        union = np.count_nonzero(pred | gt)
        if union:
            best = max(best, inter / union)
    return best


def main():
    cache_dir = os.environ.get(
        "RM_CACHE", osp.join(os.environ.get("SCRATCH", "/tmp"), "rm_cache"))
    files = sorted(glob.glob(osp.join(cache_dir, "*.npz")))
    limit = int(os.environ.get("LIMIT", "120"))
    files = files[:limit]
    print(f"{len(files)} touches, shifts {SHIFTS[0]}..{SHIFTS[-1]} px\n")

    base_scores, best_scores, best_dx, best_dy = [], [], [], []

    for f in files:
        c = load_cache(f)
        hm = c["height_map_0"]
        valid = ~c["invalid"]
        gt = c["gt"]
        areas = gt.mean(axis=(1, 2))
        if areas.max() <= 0:
            continue
        t = int(np.argmax(areas))          # peak-contact frame
        g = gt[t]

        # coarse threshold grid from this touch's own height distribution
        hv = hm[valid]
        if hv.size < 100:
            continue
        thr_grid = np.percentile(hv, np.linspace(1, 99, 25))

        base = best_iou_over_thresholds(hm, valid, g, thr_grid)
        if not np.isfinite(base):
            continue

        bi, bdx, bdy = base, 0, 0
        for dy in SHIFTS:
            for dx in SHIFTS:
                gs = np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)
                v = best_iou_over_thresholds(hm, valid, gs, thr_grid)
                if np.isfinite(v) and v > bi:
                    bi, bdx, bdy = v, dx, dy
        base_scores.append(base); best_scores.append(bi)
        best_dx.append(bdx); best_dy.append(bdy)

    b = np.array(base_scores); s = np.array(best_scores)
    dx = np.array(best_dx); dy = np.array(best_dy)
    print(f"n = {len(b)}")
    print(f"oracle IoU, no shift        : {b.mean():.4f}")
    print(f"oracle IoU, best rigid shift: {s.mean():.4f}  "
          f"(+{100*(s.mean()-b.mean())/max(b.mean(),1e-9):.1f}%)")
    print()
    print(f"best dx: mean {dx.mean():+.1f} px  median {np.median(dx):+.1f}  "
          f"std {dx.std():.1f}")
    print(f"best dy: mean {dy.mean():+.1f} px  median {np.median(dy):+.1f}  "
          f"std {dy.std():.1f}")
    print(f"fraction needing |shift| > 8 px: "
          f"{float((np.hypot(dx, dy) > 8).mean()):.3f}")
    print()
    print("A large, *consistent* mean shift implies a fixable calibration bias")
    print("(ARUCO_TO_CONTACT_X_M / _Y_M); a large but *scattered* shift implies")
    print("per-touch pose noise, which calibration cannot remove.")


if __name__ == "__main__":
    main()
