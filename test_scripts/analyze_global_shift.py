"""
Does a single global 2D shift of the render mask improve it?

analyze_spatial_align.py finds that per-touch best shifts lift the oracle IoU a
lot, but those are fitted per touch and so partly fit pose noise. This asks the
actionable question instead: is there ONE (dx, dy) -- i.e. a correction to
ARUCO_TO_CONTACT_X_M / _Y_M -- that helps across the whole dataset?

Fitted on a train split and reported on held-out touches, so the gain quoted is
not the shift memorizing the data it was chosen on.
"""

import glob
import os
import sys
from os import path as osp

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from eval_render_mask import load_cache  # noqa: E402
from _gelsight_processing import GELSIGHT_FOV_W_MM, GELSIGHT_W  # noqa: E402

SHIFTS = np.arange(-32, 33, 4)
MM_PER_PX = GELSIGHT_FOV_W_MM / GELSIGHT_W


def touch_oracle(hm, valid, g, thr_grid):
    best = 0.0
    if g.sum() == 0:
        return np.nan
    for t in thr_grid:
        pred = (hm < t) & valid
        u = np.count_nonzero(pred | g)
        if u:
            best = max(best, np.count_nonzero(pred & g) / u)
    return best


def main():
    cache_dir = os.environ.get(
        "RM_CACHE", osp.join(os.environ.get("SCRATCH", "/tmp"), "rm_cache"))
    files = sorted(glob.glob(osp.join(cache_dir, "*.npz")))
    files = files[:int(os.environ.get("LIMIT", "200"))]

    items = []
    for f in files:
        c = load_cache(f)
        gt = c["gt"]
        areas = gt.mean(axis=(1, 2))
        if areas.max() <= 0:
            continue
        valid = ~c["invalid"]
        hv = c["height_map_0"][valid]
        if hv.size < 100:
            continue
        items.append((c["height_map_0"], valid, gt[int(np.argmax(areas))],
                      np.percentile(hv, np.linspace(1, 99, 25))))
    print(f"{len(items)} usable touches")

    rng = np.random.RandomState(0)
    order = rng.permutation(len(items))
    tr = [items[i] for i in order[:len(items) // 2]]
    te = [items[i] for i in order[len(items) // 2:]]

    def mean_oracle(subset, dx, dy):
        vals = []
        for hm, valid, g, thr in subset:
            gs = np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)
            v = touch_oracle(hm, valid, gs, thr)
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals))

    base_tr = mean_oracle(tr, 0, 0)
    best = (base_tr, 0, 0)
    for dy in SHIFTS:
        for dx in SHIFTS:
            v = mean_oracle(tr, dx, dy)
            if v > best[0]:
                best = (v, dx, dy)
    _, bdx, bdy = best

    base_te = mean_oracle(te, 0, 0)
    shift_te = mean_oracle(te, bdx, bdy)

    print(f"\nbest global shift on train: dx={bdx:+d} px  dy={bdy:+d} px "
          f"({bdx*MM_PER_PX:+.2f} mm, {bdy*MM_PER_PX:+.2f} mm)")
    print(f"train oracle IoU: {base_tr:.4f} -> {best[0]:.4f}")
    print(f"TEST  oracle IoU: {base_te:.4f} -> {shift_te:.4f}  "
          f"({100*(shift_te-base_te)/max(base_te,1e-9):+.1f}%)")
    print("\nCompare with the per-touch best shift (~+28%): the gap between")
    print("that and this held-out global gain is per-touch pose noise, which a")
    print("calibration constant cannot remove.")


if __name__ == "__main__":
    main()
