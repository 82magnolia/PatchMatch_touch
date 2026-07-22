"""
Threshold-response tables for fast render-mask sweeps.

The hard render mask is `height_map_0 < scalar` intersected with a fixed valid
region, so the IoU of frame t against its pseudo-GT mask is a pure function of
that scalar. Precomputing cumulative histograms of height_map_0 -- once over
the valid region, and once over (GT_t AND valid) for each frame -- yields
|pred|, |GT_t AND pred| and hence IoU/Dice at *every* threshold on a grid, with
no per-variant mask rendering.

That turns a depth-schedule evaluation into an array lookup, which is what
makes the wide parameter sweeps in sweep_render_mask.py feasible.

Morphological opening is deliberately omitted here (it is not expressible as a
threshold lookup); it is a small denoise step, and the winning configuration is
re-verified with morph enabled by eval_render_mask.py.
"""

import glob
import os
from os import path as osp

import numpy as np

# Threshold grid, metres along the sensor normal. Spans the observed range of
# height_map_0 with margin on both sides.
THR_MIN, THR_MAX, THR_STEP = -0.080, 0.120, 0.00025
THR_GRID = np.arange(THR_MIN, THR_MAX + 1e-9, THR_STEP)


def build_response(c):
    """Per-touch threshold response.

    Returns dict with:
      n_valid_cum : (T,)      |pred| at each grid threshold
      inter_cum   : (F, T)    |GT_t AND pred| at each grid threshold
      gt_size     : (F,)      |GT_t| (including pixels outside the valid region)
    """
    hm = c["height_map_0"]
    valid = ~c["invalid"]
    gt = c["gt"]

    hv = hm[valid]
    # searchsorted on a sorted copy gives the cumulative count below each
    # threshold in one pass per pixel set.
    hv_sorted = np.sort(hv)
    n_valid_cum = np.searchsorted(hv_sorted, THR_GRID, side="left")

    F = len(gt)
    inter_cum = np.zeros((F, len(THR_GRID)), dtype=np.int32)
    gt_size = np.zeros(F, dtype=np.int64)
    for t in range(F):
        g = gt[t]
        gt_size[t] = g.sum()
        sel = hm[g & valid]
        if sel.size:
            inter_cum[t] = np.searchsorted(np.sort(sel), THR_GRID, side="left")
    return dict(n_valid_cum=n_valid_cum.astype(np.int32),
                inter_cum=inter_cum, gt_size=gt_size,
                n_pixels=np.int64(hm.size))


def score_from_response(resp, thresholds):
    """Per-frame scores for a length-F array of scalar thresholds.

    Returns (iou, dice, mse, pred_area, has_gt), all length F.

    IoU/Dice are left as NaN on frames whose pseudo-GT is empty, so callers can
    aggregate two ways. Both are needed, because each alone has a degenerate
    optimum:

      * Scoring every frame with the IoU(empty, empty) = 1 convention hands a
        free 28% of frames (this dataset's no-contact fraction) to a schedule
        that predicts nothing -- an always-empty predictor scores 0.283 that
        way, beating the real production baseline.
      * Scoring only contact frames leaves false positives on the other 28%
        entirely unpenalized, so the best threshold saturates and every
        schedule ties while covering 60% of each no-contact frame.

    Callers should report all-frames IoU alongside the always-empty and
    always-full reference rows, and watch pred_area on the no-contact frames.
    """
    idx = np.clip(np.searchsorted(THR_GRID, thresholds, side="left"),
                  0, len(THR_GRID) - 1)
    pred = resp["n_valid_cum"][idx].astype(np.float64)          # (F,)
    inter = resp["inter_cum"][np.arange(len(idx)), idx].astype(np.float64)
    gtsz = resp["gt_size"].astype(np.float64)
    has_gt = gtsz > 0

    union = gtsz + pred - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), np.nan)
    denom = gtsz + pred
    dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1e-9), np.nan)
    iou = np.where(has_gt, iou, np.nan)
    dice = np.where(has_gt, dice, np.nan)
    # symmetric difference / total pixels == MSE of two binary images
    mse = (gtsz + pred - 2 * inter) / float(resp["n_pixels"])
    pred_area = pred / float(resp["n_pixels"])
    return iou, dice, mse, pred_area, has_gt


def response_path(cache_dir, key):
    return osp.join(cache_dir, "_resp", key)


def build_all(cache_dir, jobs=12, limit=None):
    """Build (and cache to disk) response tables for every touch cache."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(osp.join(cache_dir, "_resp"), exist_ok=True)
    files = sorted(glob.glob(osp.join(cache_dir, "*.npz")))
    if limit:
        files = files[:limit]
    todo = [f for f in files
            if not osp.exists(response_path(cache_dir, osp.basename(f)))]
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(_worker, [(f, cache_dir) for f in todo], chunksize=4))
    return files


def _worker(arg):
    import sys
    f, cache_dir = arg
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from eval_render_mask import load_cache
    c = load_cache(f)
    r = build_response(c)
    np.savez_compressed(response_path(cache_dir, osp.basename(f)), **r)


def load_response(cache_dir, key):
    z = np.load(response_path(cache_dir, key))
    return dict(n_valid_cum=z["n_valid_cum"], inter_cum=z["inter_cum"],
                gt_size=z["gt_size"], n_pixels=z["n_pixels"])
