"""
Evaluation harness for GelSight "hard" render-mask generation.

Scores geometry-derived render masks (height_map_0 thresholded by a per-frame
pressing depth) against an image-derived pseudo-ground-truth contact mask
computed from the tactile shadow video via
_gelsight_processing.compute_contact_mask.

Runs entirely off already-processed touch outputs -- no ZED, no GelSight, no
ortho re-projection. Each touch directory supplies:
  {i}_contact_data.npz   height_map_0, valid_depth_remap, mask_crop,
                         sensor_z_0, tvec_0   (frozen at contact start)
  {i}_pose_contact.npz   per-frame ARuCO rvecs/tvecs over the segment
  {i}_diffs.npz          per-frame diff-from-blank over the segment
  {i}_meta.json          cs_idx / ce_idx / peak_idx within the segment
  {i}_shadow.mp4         resampled tactile video (pseudo-GT source)

Phase A caches the expensive per-touch invariants (GT masks, height map,
static geometry) to an .npz per touch; phase B replays cheap variants against
that cache. Use --rebuild to force phase A.

Example:
    python test_scripts/eval_render_mask.py --variants baseline,baseline_aligned \\
        --limit 300
"""

import argparse
import glob
import json
import os
import sys
from os import path as osp

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)),
                            "..", "real_data_transfer"))

from _gelsight_processing import (  # noqa: E402
    compute_contact_mask, GELSIGHT_W, GELSIGHT_H, MASK_OPEN_PX,
    RENDER_MASK_THRES_M,
)

DEFAULT_ROOT = "log/real_data_real_retrieval"
NUM_FRAMES = 50

_MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * MASK_OPEN_PX + 1, 2 * MASK_OPEN_PX + 1))


# ── Touch discovery / loading ────────────────────────────────────────────────

def find_touches(root):
    """(dir, touch_idx) for every touch with the full file set."""
    out = []
    for meta in sorted(glob.glob(osp.join(root, "*", "*_meta.json"))):
        if meta.endswith("_seg_meta.json"):
            continue
        d = osp.dirname(meta)
        i = osp.basename(meta).split("_")[0]
        need = [f"{i}_contact_data.npz", f"{i}_pose_contact.npz",
                f"{i}_diffs.npz", f"{i}_shadow.mp4"]
        if all(osp.exists(osp.join(d, n)) for n in need):
            out.append((d, i))
    return out


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def aruco_depths(tvecs, sensor_z_0, tvec_0):
    """Per-segment-frame sensor advance along its own normal, NaN where the
    marker was missing. This is the quantity make_render_mask_video's RBF fits."""
    d = np.full(len(tvecs), np.nan, dtype=np.float64)
    for i, tv in enumerate(tvecs):
        if np.isfinite(tv).all():
            d[i] = float(np.dot(sensor_z_0, tv - tvec_0))
    return d


def build_cache(d, i, contact_threshold, blur_sigma, morph_radius):
    """Per-touch invariants: GT masks (bit-packed) + geometry + curves."""
    cd = np.load(osp.join(d, f"{i}_contact_data.npz"))
    pc = np.load(osp.join(d, f"{i}_pose_contact.npz"))
    df = np.load(osp.join(d, f"{i}_diffs.npz"))
    with open(osp.join(d, f"{i}_meta.json")) as f:
        meta = json.load(f)

    shadow = read_video(osp.join(d, f"{i}_shadow.mp4"))
    if len(shadow) < 2:
        return None

    base = shadow[0].astype(np.float32) / 255.0
    gt = np.zeros((len(shadow), GELSIGHT_H, GELSIGHT_W), dtype=bool)
    for t, fr in enumerate(shadow):
        m = compute_contact_mask(fr.astype(np.float32) / 255.0, base,
                                 contact_threshold, blur_sigma, morph_radius)
        gt[t] = m[..., 0] > 0.5

    d_aruco = aruco_depths(pc["tvecs"], cd["sensor_z_0"], cd["tvec_0"])
    return dict(
        gt_packed=np.packbits(gt, axis=None),
        gt_shape=np.array(gt.shape),
        height_map_0=cd["height_map_0"].astype(np.float32),
        invalid=(~cd["valid_depth_remap"]) | (cd["mask_crop"] == 0),
        d_aruco=d_aruco,
        diffs=df["diffs"].astype(np.float64),
        smooth_diffs=df["smooth_diffs"].astype(np.float64),
        cs_idx=meta["cs_idx"], ce_idx=meta["ce_idx"],
        peak_idx=meta["peak_idx"], n_raw=meta["n_raw_frames"],
    )


def load_cache(path):
    z = np.load(path)
    shape = tuple(z["gt_shape"])
    n = int(np.prod(shape))
    gt = np.unpackbits(z["gt_packed"])[:n].astype(bool).reshape(shape)
    return dict(
        gt=gt,
        height_map_0=z["height_map_0"],
        invalid=z["invalid"],
        d_aruco=z["d_aruco"],
        diffs=z["diffs"],
        smooth_diffs=z["smooth_diffs"],
        cs_idx=int(z["cs_idx"]), ce_idx=int(z["ce_idx"]),
        peak_idx=int(z["peak_idx"]), n_raw=int(z["n_raw"]),
    )


# ── Depth schedules ──────────────────────────────────────────────────────────
#
# Every schedule returns a length-NUM_FRAMES array of pressing depths (metres,
# signed along the sensor normal) to be added to render_mask_thres. They differ
# in (a) what signal drives the shape and (b) which time axis they sample.

def _eval_positions(c, aligned, num_frames=NUM_FRAMES):
    """Normalized segment positions at which to sample the depth curve.

    The render-mask video has always spanned the whole segment uniformly, while
    the shadow video it is paired with is trim_and_resample'd to the contact
    window [cs_idx, ce_idx]. `aligned=True` samples the same window the shadow
    frames actually came from, putting the two videos on one time axis.
    """
    n = max(c["n_raw"] - 1, 1)
    if not aligned:
        return np.linspace(0.0, 1.0, num_frames)
    cs, ce = c["cs_idx"], c["ce_idx"]
    return np.linspace(cs / n, ce / n, num_frames)


def _rbf_curve(c, smoothing, aligned, num_frames=NUM_FRAMES):
    from scipy.interpolate import RBFInterpolator
    d = c["d_aruco"]
    ok = np.isfinite(d)
    n = max(len(d) - 1, 1)
    raw_x = np.nonzero(ok)[0] / n
    raw_y = d[ok]
    ex = _eval_positions(c, aligned, num_frames)
    if len(raw_x) < 2:
        return np.zeros(num_frames)
    rbf = RBFInterpolator(raw_x[:, None], raw_y,
                          kernel="thin_plate_spline", smoothing=smoothing)
    return rbf(ex[:, None])


def _diff_shape(c, aligned, num_frames=NUM_FRAMES, use_smooth=True,
                anchor="start"):
    """Diff-from-blank curve, resampled onto the eval axis and normalized to
    [0, 1] with 0 at the no-contact anchor and 1 at peak."""
    s = c["smooth_diffs"] if use_smooth else c["diffs"]
    n = max(len(s) - 1, 1)
    ex = _eval_positions(c, aligned, num_frames)
    s_i = np.interp(ex, np.arange(len(s)) / n, s)
    if anchor == "min":
        s0 = float(np.min(s))
    elif anchor == "cs":
        s0 = float(s[min(c["cs_idx"], len(s) - 1)])
    else:
        s0 = float(s[0])
    smax = float(np.max(s))
    if smax - s0 <= 1e-12:
        return np.zeros(num_frames)
    return np.clip((s_i - s0) / (smax - s0), 0.0, None)


def sched_baseline(c, **kw):
    """Current production behaviour: RBF(smoothing=0.1) over the full segment."""
    return _rbf_curve(c, 0.1, aligned=False)


def sched_baseline_aligned(c, **kw):
    """Baseline curve, sampled on the shadow video's trimmed time axis."""
    return _rbf_curve(c, 0.1, aligned=True)


def make_rbf_sched(smoothing, aligned=True):
    def f(c, **kw):
        return _rbf_curve(c, smoothing, aligned=aligned)
    return f


def make_diff_max_sched(aligned=True, anchor="start", depth_stat="max",
                        use_smooth=True):
    """d_t = d_peak * normalized_diff_t.

    Takes only a single scalar (the peak pressing depth) from the noisy ARuCO
    track and gets the entire temporal shape from the far more stable
    diff-from-blank curve.
    """
    def f(c, **kw):
        shape = _diff_shape(c, aligned, use_smooth=use_smooth, anchor=anchor)
        d = c["d_aruco"]
        ok = np.isfinite(d)
        if not ok.any():
            return np.zeros(NUM_FRAMES)
        dv = d[ok]
        if depth_stat == "max":
            d_peak = float(np.max(dv))
        elif depth_stat == "p90":
            d_peak = float(np.percentile(dv, 90))
        elif depth_stat == "at_peak":
            pk = min(c["peak_idx"], len(d) - 1)
            d_peak = float(d[pk]) if np.isfinite(d[pk]) else float(np.max(dv))
        else:
            raise ValueError(depth_stat)
        return d_peak * shape
    return f


def make_diff_const_sched(coeff, aligned=True, anchor="start"):
    """d_t = coeff * (diff_t - diff_anchor): pure image-driven depth, no ARuCO
    magnitude at all. `coeff` carries metres per unit diff."""
    def f(c, **kw):
        s = c["smooth_diffs"]
        n = max(len(s) - 1, 1)
        ex = _eval_positions(c, aligned)
        s_i = np.interp(ex, np.arange(len(s)) / n, s)
        s0 = float(np.min(s)) if anchor == "min" else float(s[0])
        return coeff * np.clip(s_i - s0, 0.0, None)
    return f


def make_diff_affine_sched(slope, intercept, aligned=True, use_smooth=False,
                           thres=RENDER_MASK_THRES_M):
    """Absolute threshold affine in diff-from-blank, expressed as a depth so it
    composes with the caller's `thres` (which it cancels out)."""
    def f(c, **kw):
        s = c["smooth_diffs"] if use_smooth else c["diffs"]
        n = max(len(s) - 1, 1)
        ex = _eval_positions(c, aligned)
        s_i = np.interp(ex, np.arange(len(s)) / n, s)
        return intercept + slope * s_i - thres
    return f


VARIANTS = {
    "baseline":          sched_baseline,
    "baseline_aligned":  sched_baseline_aligned,
    "diff_max":          make_diff_max_sched(),
    "diff_max_unaligned": make_diff_max_sched(aligned=False),
    "diff_max_raw":      make_diff_max_sched(use_smooth=False),
    # slope/intercept from sweep_render_mask.py --sweep affine (fitted on a
    # train split, verified held out). Performance sits on a flat ridge from
    # slope ~2.5 upward, so these are an interior point, not a tuned optimum.
    "diff_affine":       make_diff_affine_sched(2.5, -0.060),
}


# ── Scoring ──────────────────────────────────────────────────────────────────

def render_masks(c, depths, thres, morph=True):
    """Hard-threshold render masks for one touch, given a depth schedule."""
    hm = c["height_map_0"]
    invalid = c["invalid"]
    out = np.zeros((len(depths), GELSIGHT_H, GELSIGHT_W), dtype=bool)
    for t, dv in enumerate(depths):
        m = (hm < float(dv) + thres) & ~invalid
        if morph:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                                 _MORPH_KERNEL).astype(bool)
        out[t] = m
    return out


def score_touch(c, sched, thres, **kw):
    """Per-frame scores; IoU/Dice are NaN on frames with empty pseudo-GT.

    Those frames must not be scored as a perfect match when the prediction is
    also empty -- 28% of frames in this dataset have no contact, so that
    convention rewards a schedule for degenerating to an all-empty mask (see
    rm_response.score_from_response).
    """
    gt = c["gt"]
    depths = sched(c, **kw)
    pred = render_masks(c, depths, thres)
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    gs = gt.sum(axis=(1, 2)).astype(np.float64)
    ps = pred.sum(axis=(1, 2)).astype(np.float64)
    inter = (gt & pred).sum(axis=(1, 2)).astype(np.float64)
    union = (gt | pred).sum(axis=(1, 2)).astype(np.float64)
    has_gt = gs > 0
    iou = np.where(has_gt & (union > 0), inter / np.maximum(union, 1), np.nan)
    dice = np.where(has_gt & (gs + ps > 0), 2 * inter / np.maximum(gs + ps, 1),
                    np.nan)
    mse = (gt ^ pred).mean(axis=(1, 2)).astype(np.float64)
    fp = pred[~has_gt].mean() if (~has_gt).any() else 0.0
    return dict(iou=iou, dice=dice, mse=mse, has_gt=has_gt,
                pred_area=pred.mean(axis=(1, 2)).astype(np.float64),
                gt_area=gt.mean(axis=(1, 2)).astype(np.float64),
                fp_area=float(fp))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache_dir", default=None,
                   help="Where per-touch caches live (default: scratchpad)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--variants", default="baseline,baseline_aligned,diff_max")
    p.add_argument("--thres", type=float, default=RENDER_MASK_THRES_M)
    # Match production: process_single_shot.NORMAL_CONTACT_THRESHOLD gates the
    # real pipeline's contact footprint at 0.025. compute_contact_mask's own
    # default (0.05) is too tight here (ragged holes) and, scored as GT, biases
    # the whole sweep toward "predict nothing" -- see log/render_mask_sync_study.html.
    p.add_argument("--contact_threshold", type=float, default=0.025)
    p.add_argument("--blur_sigma", type=float, default=3.0)
    p.add_argument("--morph_radius", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    cache_dir = args.cache_dir or osp.join(
        os.environ.get("SCRATCH", "/tmp"), "rm_cache")
    os.makedirs(cache_dir, exist_ok=True)

    touches = find_touches(args.root)
    if args.limit:
        touches = touches[:args.limit]
    print(f"{len(touches)} touches under {args.root}")

    # ── Phase A: cache ──────────────────────────────────────────────────────
    todo = []
    for d, i in touches:
        key = f"{osp.basename(d)}_{i}.npz"
        cp = osp.join(cache_dir, key)
        if args.rebuild or not osp.exists(cp):
            todo.append((d, i, cp))
    if todo:
        print(f"Building {len(todo)} caches...")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_cache_worker, d, i, cp, args.contact_threshold,
                              args.blur_sigma, args.morph_radius)
                    for d, i, cp in todo]
            for n, f in enumerate(futs):
                f.result()
                if (n + 1) % 100 == 0:
                    print(f"  {n + 1}/{len(todo)}")

    # ── Phase B: score ──────────────────────────────────────────────────────
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = {n: [] for n in names}
    for k, (d, i) in enumerate(touches):
        cp = osp.join(cache_dir, f"{osp.basename(d)}_{i}.npz")
        if not osp.exists(cp):
            continue
        c = load_cache(cp)
        for n in names:
            results[n].append(score_touch(c, VARIANTS[n], args.thres))
        if (k + 1) % 200 == 0:
            print(f"  scored {k + 1}/{len(touches)}")

    print(f"\n{'variant':<22} {'IoU_all':>8} {'IoU_cont':>9} {'Dice':>8} "
          f"{'MSE':>8} {'gt_area':>8} {'pred_area':>10} {'FP_area':>8}")
    print("-" * 92)
    for n in names:
        rs = results[n]
        if not rs:
            continue
        cat = lambda k: np.concatenate([r[k] for r in rs])  # noqa: E731
        has = cat("has_gt")
        parea = cat("pred_area")
        # all-frames convention: empty GT scores 1.0 only if nothing predicted
        iou_all = np.where(has, cat("iou"), np.where(parea > 0, 0.0, 1.0))
        print(f"{n:<22} {np.nanmean(iou_all):>8.4f} "
              f"{np.nanmean(cat('iou')):>9.4f} "
              f"{np.nanmean(cat('dice')):>8.4f} {cat('mse').mean():>8.4f} "
              f"{cat('gt_area').mean():>8.4f} {parea.mean():>10.4f} "
              f"{float(np.mean([r['fp_area'] for r in rs])):>8.4f}")


def _cache_worker(d, i, cp, ct, bs, mr):
    c = build_cache(d, i, ct, bs, mr)
    if c is not None:
        np.savez_compressed(cp, **c)


if __name__ == "__main__":
    main()
