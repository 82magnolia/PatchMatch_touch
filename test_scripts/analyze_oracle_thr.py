"""
What does the per-frame optimal render-mask threshold actually track?

sweep_render_mask.py's oracle shows a ceiling well above every hand-built depth
schedule. This extracts the oracle's chosen threshold trajectory and regresses
it against the signals actually available at capture time (ARuCO pressing
depth, diff-from-blank, frame position), to find out which of them -- if any --
carries the information the schedules are missing.

Reports both per-frame correlations pooled across touches and, separately,
within-touch correlations (a signal can predict the between-touch level well
while carrying no temporal information, or vice versa).
"""

import glob
import os
import sys
from os import path as osp

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import rm_response as R  # noqa: E402
from sweep_render_mask import (  # noqa: E402
    load_all, resample, d_rbf, gt_area_in_valid, CACHE_DIR,
)


def oracle_threshold(resp):
    """Per-frame argmax-IoU threshold and the IoU it achieves."""
    pred = resp["n_valid_cum"].astype(np.float64)[None, :]
    inter = resp["inter_cum"].astype(np.float64)
    gtsz = resp["gt_size"].astype(np.float64)[:, None]
    union = gtsz + pred - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
    k = iou.argmax(axis=1)
    return R.THR_GRID[k], iou.max(axis=1)


def main():
    cache_dir = os.environ.get("RM_CACHE", CACHE_DIR)
    data = load_all(cache_dir)
    print(f"loaded {len(data)} touches\n")

    rows_t, rows_d, rows_s, rows_f, rows_a = [], [], [], [], []
    within = {"aruco": [], "diff": [], "frac": []}
    per_touch_level = []

    for m, resp, _ in data:
        thr, best_iou = oracle_threshold(resp)
        F = len(thr)
        s_i = resample(m["diffs"], m, aligned=True, num_frames=F)
        d_i = np.asarray(d_rbf(m, 0.1, aligned=True, num_frames=F)).ravel()
        frac = np.linspace(0, 1, F)
        area = gt_area_in_valid(resp)[:F]

        rows_t.append(thr); rows_d.append(d_i); rows_s.append(s_i)
        rows_f.append(frac); rows_a.append(area)
        per_touch_level.append((thr.mean(), d_i.mean(), s_i.mean()))

        for key, sig in (("aruco", d_i), ("diff", s_i), ("frac", frac)):
            if np.std(thr) > 1e-9 and np.std(sig) > 1e-9:
                within[key].append(np.corrcoef(thr, sig)[0, 1])

    thr = np.concatenate(rows_t); d = np.concatenate(rows_d)
    s = np.concatenate(rows_s); f = np.concatenate(rows_f)
    a = np.concatenate(rows_a)

    print("oracle threshold stats (m):")
    print(f"  mean {thr.mean():+.4f}  std {thr.std():.4f}  "
          f"p10 {np.percentile(thr,10):+.4f}  p90 {np.percentile(thr,90):+.4f}")

    print("\npooled correlation with oracle threshold:")
    for name, sig in (("aruco depth (RBF)", d), ("diff-from-blank", s),
                      ("frame fraction", f), ("gt area frac", a)):
        print(f"  {name:<20} r = {np.corrcoef(thr, sig)[0,1]:+.3f}")

    print("\nwithin-touch correlation (temporal information only):")
    for k, v in within.items():
        print(f"  {k:<20} mean r = {np.mean(v):+.3f}  "
              f"median {np.median(v):+.3f}  (n={len(v)})")

    lvl = np.array(per_touch_level)
    print("\nbetween-touch correlation (per-touch mean level):")
    print(f"  aruco  r = {np.corrcoef(lvl[:,0], lvl[:,1])[0,1]:+.3f}")
    print(f"  diff   r = {np.corrcoef(lvl[:,0], lvl[:,2])[0,1]:+.3f}")

    # How much of the oracle threshold is a per-touch constant vs temporal?
    tot_var = float(np.var(thr))
    within_var = float(np.mean([np.var(t) for t in rows_t]))
    print(f"\nvariance decomposition of oracle threshold:")
    print(f"  total          {tot_var:.3e}")
    print(f"  within-touch   {within_var:.3e}  "
          f"({100*within_var/tot_var:.1f}% -- what temporal modulation can win)")
    print(f"  between-touch  {tot_var-within_var:.3e}  "
          f"({100*(1-within_var/tot_var):.1f}% -- per-touch level)")


if __name__ == "__main__":
    main()
