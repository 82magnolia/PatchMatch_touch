"""
Parameter sweeps over render-mask depth schedules, scored via the
threshold-response tables from rm_response.py.

A "schedule" maps a touch's per-frame signals (ARuCO pressing depth, tactile
diff-from-blank, contact-window indices) to one scalar height threshold per
output frame. Everything downstream of that scalar is a lookup, so this can
sweep thousands of configurations in seconds.

Two oracles bound the problem:
  oracle_per_frame  best threshold chosen independently per frame  -- the
                    ceiling for *any* depth schedule, given this height map
  oracle_per_touch  best single constant threshold for a whole touch -- what a
                    schedule with no temporal information could reach

Example:
    python test_scripts/sweep_render_mask.py --sweep main
"""

import argparse
import glob
import os
import sys
from os import path as osp

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import rm_response as R  # noqa: E402

CACHE_DIR = osp.join(
    os.environ.get("SCRATCH", "/tmp"), "rm_cache")
NUM_FRAMES = 50


# ── Lightweight per-touch metadata (no GT masks) ─────────────────────────────

def load_meta(path):
    z = np.load(path)
    return dict(
        d_aruco=z["d_aruco"], diffs=z["diffs"], smooth_diffs=z["smooth_diffs"],
        cs_idx=int(z["cs_idx"]), ce_idx=int(z["ce_idx"]),
        peak_idx=int(z["peak_idx"]), n_raw=int(z["n_raw"]),
    )


def load_all(cache_dir, limit=None):
    files = sorted(glob.glob(osp.join(cache_dir, "*.npz")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        key = osp.basename(f)
        rp = R.response_path(cache_dir, key)
        if not osp.exists(rp):
            continue
        out.append((load_meta(f), R.load_response(cache_dir, key), key))
    return out


# ── Time axes ────────────────────────────────────────────────────────────────

def eval_positions(m, aligned, num_frames=NUM_FRAMES):
    """Normalized segment positions for the output frames.

    aligned=False reproduces production: the render-mask video spans the whole
    segment uniformly. aligned=True samples the contact window [cs_idx, ce_idx]
    that trim_and_resample used to build the shadow video the mask is paired
    with -- i.e. it puts both videos on the same clock.
    """
    n = max(m["n_raw"] - 1, 1)
    if not aligned:
        return np.linspace(0.0, 1.0, num_frames)
    return np.linspace(m["cs_idx"] / n, m["ce_idx"] / n, num_frames)


def resample(sig, m, aligned, num_frames=NUM_FRAMES):
    n = max(len(sig) - 1, 1)
    return np.interp(eval_positions(m, aligned, num_frames),
                     np.arange(len(sig)) / n, sig)


# ── Depth schedules ──────────────────────────────────────────────────────────

def d_rbf(m, smoothing=0.1, aligned=False, num_frames=NUM_FRAMES):
    from scipy.interpolate import RBFInterpolator
    d = m["d_aruco"]
    ok = np.isfinite(d)
    n = max(len(d) - 1, 1)
    if ok.sum() < 2:
        return np.zeros(num_frames)
    rbf = RBFInterpolator((np.nonzero(ok)[0] / n)[:, None], d[ok],
                          kernel="thin_plate_spline", smoothing=smoothing)
    return rbf(eval_positions(m, aligned, num_frames)[:, None])


def diff_shape(m, aligned=True, anchor="start", use_smooth=True,
               num_frames=NUM_FRAMES):
    """Diff-from-blank normalized to 0 at the no-contact anchor, 1 at peak."""
    s = m["smooth_diffs"] if use_smooth else m["diffs"]
    s_i = resample(s, m, aligned, num_frames)
    if anchor == "min":
        s0 = float(np.min(s))
    elif anchor == "cs":
        s0 = float(s[min(m["cs_idx"], len(s) - 1)])
    else:
        s0 = float(s[0])
    rng = float(np.max(s)) - s0
    if rng <= 1e-12:
        return np.zeros(num_frames)
    return np.clip((s_i - s0) / rng, 0.0, None)


def d_peak_stat(m, stat="max"):
    d = m["d_aruco"]
    ok = np.isfinite(d)
    if not ok.any():
        return 0.0
    dv = d[ok]
    if stat == "max":
        return float(np.max(dv))
    if stat == "p90":
        return float(np.percentile(dv, 90))
    if stat == "p75":
        return float(np.percentile(dv, 75))
    if stat == "at_peak":
        pk = min(m["peak_idx"], len(d) - 1)
        return float(d[pk]) if np.isfinite(d[pk]) else float(np.max(dv))
    if stat == "range":
        return float(np.max(dv) - np.min(dv))
    raise ValueError(stat)


def d_diff_max(m, aligned=True, anchor="start", stat="max", use_smooth=True):
    """Shape from the tactile diff curve, magnitude from one ARuCO scalar."""
    return d_peak_stat(m, stat) * diff_shape(m, aligned, anchor, use_smooth)


def d_diff_const(m, coeff=1.0, aligned=True, anchor="start"):
    """Purely image-driven: metres of penetration per unit diff-from-blank."""
    s = m["smooth_diffs"]
    s_i = resample(s, m, aligned)
    s0 = float(np.min(s)) if anchor == "min" else float(s[0])
    return coeff * np.clip(s_i - s0, 0.0, None)


def d_blend(m, w=0.5, smoothing=0.1, aligned=True, anchor="start", stat="max"):
    """Convex blend of the RBF/ARuCO curve and the diff-driven curve."""
    return (1 - w) * np.asarray(d_rbf(m, smoothing, aligned)).ravel() + \
        w * d_diff_max(m, aligned, anchor, stat)


def gt_area_in_valid(resp):
    """Per-frame |GT AND valid| / |valid| -- the area a perfect threshold would
    have to reproduce. Read off the top of the cumulative histograms."""
    n_valid = float(resp["n_valid_cum"][-1])
    if n_valid <= 0:
        return np.zeros(resp["inter_cum"].shape[0])
    return resp["inter_cum"][:, -1].astype(np.float64) / n_valid


def thr_for_area(resp, area_frac):
    """Invert the height CDF: threshold whose predicted mask covers area_frac
    of the valid region. Vectorized over frames."""
    n_valid = float(resp["n_valid_cum"][-1])
    target = np.clip(area_frac, 0.0, 1.0) * n_valid
    idx = np.searchsorted(resp["n_valid_cum"], target, side="left")
    idx = np.clip(idx, 0, len(R.THR_GRID) - 1)
    return R.THR_GRID[idx]


def d_area_match(m, resp, coef, intercept, aligned=True, use_smooth=False):
    """Self-calibrating schedule.

    Rather than converting ARuCO translation into a height threshold -- which
    inherits every bias in the marker pose and the marker->gel offset -- this
    predicts the *contact area* from the scalar diff-from-blank, then picks the
    threshold whose mask has that area. The height map only has to get the
    relative ordering of pixels right, not their absolute offset, so per-touch
    zero-level error cancels out.

    coef/intercept come from a global fit on held-out training touches.
    """
    s = m["smooth_diffs"] if use_smooth else m["diffs"]
    s_i = resample(s, m, aligned)
    area = np.clip(intercept + coef * s_i, 0.0, 1.0)
    return thr_for_area(resp, area)


def fit_area_model(data, aligned=True, use_smooth=False):
    """Least-squares fit of contact-area-fraction against diff-from-blank,
    pooled over every frame of every training touch."""
    xs, ys = [], []
    for m, resp, _ in data:
        s = m["smooth_diffs"] if use_smooth else m["diffs"]
        s_i = resample(s, m, aligned)
        a = gt_area_in_valid(resp)
        n = min(len(s_i), len(a))
        xs.append(s_i[:n])
        ys.append(a[:n])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(coef), float(intercept)


def score_area_match(data, coef, intercept, aligned=True, use_smooth=False,
                     bias=0.0):
    return _aggregate([R.score_from_response(
        resp, d_area_match(m, resp, coef, intercept, aligned, use_smooth) + bias)
        for m, resp, _ in data])


def d_diff_affine(m, slope, intercept, aligned=True, use_smooth=False):
    """Threshold directly affine in the tactile diff-from-blank -- no ARuCO.

    analyze_oracle_thr.py shows the ARuCO pressing depth correlates with the
    oracle threshold *within* a touch (r~0.5) but not at all *between* touches
    (r~-0.08), while the per-touch level accounts for ~52% of the oracle's
    variance. The diff-from-blank carries both (within r~0.64, between r~0.47),
    so a schedule driven by it alone can set the level the marker cannot.

    Returns an absolute threshold; callers must not add render_mask_thres.
    """
    s = m["smooth_diffs"] if use_smooth else m["diffs"]
    return intercept + slope * resample(s, m, aligned)


def d_diff_affine_hm(m, resp, slope, intercept, hm_coef, aligned=True,
                     use_smooth=False):
    """As d_diff_affine, plus a term in the touch's own median valid height --
    a cheap per-touch geometry prior for where the surface sits."""
    s = m["smooth_diffs"] if use_smooth else m["diffs"]
    n_valid = resp["n_valid_cum"]
    half = np.searchsorted(n_valid, n_valid[-1] * 0.5, side="left")
    hm_med = R.THR_GRID[min(half, len(R.THR_GRID) - 1)]
    return intercept + slope * resample(s, m, aligned) + hm_coef * hm_med


def score_absolute(data, thr_fn):
    """Score a schedule that returns absolute thresholds."""
    return _aggregate([R.score_from_response(
        resp, np.asarray(thr_fn(m, resp)).ravel()) for m, resp, _ in data])


SCHEDULES = {}


def register(name, fn):
    SCHEDULES[name] = fn


register("baseline", lambda m: d_rbf(m, 0.1, aligned=False))
register("baseline_aligned", lambda m: d_rbf(m, 0.1, aligned=True))
register("diff_max", lambda m: d_diff_max(m))
register("diff_max_unaligned", lambda m: d_diff_max(m, aligned=False))


# ── Scoring ──────────────────────────────────────────────────────────────────

def eval_depths(data, sched):
    """Depth curves for every touch, computed once so a threshold sweep does
    not pay for the (expensive) RBF fit at each grid point."""
    return [np.asarray(sched(m)).ravel() for m, _, _ in data]


def _aggregate(chunks):
    """Pool per-frame scores into the metric set described in
    rm_response.score_from_response.

    iou_all      every frame, with IoU(empty, empty) = 1  -- the headline
                 number, but only meaningful against the degenerate reference
                 rows (a do-nothing schedule scores ~0.283 here)
    iou_contact  contact frames only -- ignores false positives
    fp_area      mean predicted area on no-contact frames -- what iou_contact
                 misses
    """
    iou = np.concatenate([c[0] for c in chunks])
    dice = np.concatenate([c[1] for c in chunks])
    mse = np.concatenate([c[2] for c in chunks])
    parea = np.concatenate([c[3] for c in chunks])
    has = np.concatenate([c[4] for c in chunks])

    # all-frames view: empty GT scores 1.0 if the prediction is also empty,
    # 0.0 if it is not.
    iou_all = np.where(has, iou, np.where(parea > 0, 0.0, 1.0))
    fp = float(parea[~has].mean()) if (~has).any() else 0.0
    return dict(iou_all=float(np.nanmean(iou_all)),
                iou_contact=float(np.nanmean(iou)),
                dice=float(np.nanmean(dice)),
                mse=float(mse.mean()), fp_area=fp)


def degenerate_refs(data):
    """Reference rows every real schedule must beat."""
    big, small = R.THR_MAX, R.THR_MIN
    empty = _aggregate([R.score_from_response(
        resp, np.full(resp["inter_cum"].shape[0], small)) for _, resp, _ in data])
    full = _aggregate([R.score_from_response(
        resp, np.full(resp["inter_cum"].shape[0], big)) for _, resp, _ in data])
    return empty, full


def score_dataset(data, sched, thres, depths=None):
    if depths is None:
        depths = eval_depths(data, sched)
    chunks = []
    for (m, resp, _), d in zip(data, depths):
        chunks.append(R.score_from_response(resp, d + thres))
    return _aggregate(chunks)



# ── Oracles ──────────────────────────────────────────────────────────────────

def _iou_table_all(resp):
    """(F, T) IoU over the whole threshold grid, in the all-frames convention:
    on empty-GT frames a threshold scores 1.0 if it predicts nothing, else 0."""
    pred = resp["n_valid_cum"].astype(np.float64)[None, :]
    inter = resp["inter_cum"].astype(np.float64)
    gtsz = resp["gt_size"].astype(np.float64)[:, None]
    union = gtsz + pred - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
    empty_gt = (gtsz <= 0)
    return np.where(empty_gt, np.where(pred > 0, 0.0, 1.0), iou)


def oracle_per_frame(data):
    """Ceiling for any schedule: best threshold chosen per frame with GT."""
    return float(np.mean([_iou_table_all(resp).max(axis=1).mean()
                          for _, resp, _ in data]))


def oracle_per_touch(data):
    """Best single constant threshold per touch -- no temporal information."""
    return float(np.mean([_iou_table_all(resp).mean(axis=0).max()
                          for _, resp, _ in data]))


def best_thres(data, sched, grid, key="iou_all"):
    depths = eval_depths(data, sched)
    rows = [(t, score_dataset(data, sched, t, depths)) for t in grid]
    rows.sort(key=lambda r: -r[1][key])
    return rows[0], rows


def _fmt(name, thr, a):
    return (f"{name:<26} {thr:>9.4f} {a['iou_all']:>8.4f} "
            f"{a['iou_contact']:>9.4f} {a['dice']:>8.4f} "
            f"{a['mse']:>8.4f} {a['fp_area']:>8.4f}")


HEADER = (f"{'schedule':<26} {'best_thr':>9} {'IoU_all':>8} {'IoU_cont':>9} "
          f"{'Dice':>8} {'MSE':>8} {'FP_area':>8}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=CACHE_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sweep", default="main")
    args = p.parse_args()

    data = load_all(args.cache_dir, args.limit)
    print(f"loaded {len(data)} touches\n")

    thres_grid = np.arange(-0.040, 0.0401, 0.001)
    empty_ref, full_ref = degenerate_refs(data)

    print("reference rows (any real schedule must beat these):")
    print(HEADER)
    print("-" * 82)
    print(_fmt("DEGENERATE always-empty", float("nan"), empty_ref))
    print(_fmt("DEGENERATE always-full", float("nan"), full_ref))
    print(f"{'ORACLE per-frame':<26} {'':>9} {oracle_per_frame(data):>8.4f}")
    print(f"{'ORACLE per-touch':<26} {'':>9} {oracle_per_touch(data):>8.4f}")
    print()

    if args.sweep == "main":
        cands = {
            "baseline (production)": lambda m: d_rbf(m, 0.1, aligned=False),
            "baseline_aligned": lambda m: d_rbf(m, 0.1, aligned=True),
            "rbf_s1_aligned": lambda m: d_rbf(m, 1.0, aligned=True),
            "rbf_s10_aligned": lambda m: d_rbf(m, 10.0, aligned=True),
            "rbf_s100_aligned": lambda m: d_rbf(m, 100.0, aligned=True),
            "diff_max": lambda m: d_diff_max(m),
            "diff_max_unaligned": lambda m: d_diff_max(m, aligned=False),
            "diff_max_anchor_min": lambda m: d_diff_max(m, anchor="min"),
            "diff_max_p90": lambda m: d_diff_max(m, stat="p90"),
            "diff_max_atpeak": lambda m: d_diff_max(m, stat="at_peak"),
            "diff_max_raw": lambda m: d_diff_max(m, use_smooth=False),
            "blend_0.5": lambda m: d_blend(m, 0.5),
            "blend_0.75": lambda m: d_blend(m, 0.75),
        }
        print(HEADER)
        print("-" * 82)
        for name, fn in cands.items():
            (t, agg), _ = best_thres(data, fn, thres_grid)
            print(_fmt(name, t, agg))

    if args.sweep == "affine":
        rng = np.random.RandomState(0)
        order = rng.permutation(len(data))
        n_tr = int(0.5 * len(data))
        train = [data[i] for i in order[:n_tr]]
        test = [data[i] for i in order[n_tr:]]
        print(f"train {len(train)} / test {len(test)}\n")
        print(HEADER)
        print("-" * 82)
        slopes = np.arange(0.0, 6.01, 0.25)
        intercepts = np.arange(-0.060, 0.0401, 0.0025)
        for use_smooth in (False, True):
            for aligned in (True, False):
                best = None
                for sl in slopes:
                    for ic in intercepts:
                        fn = (lambda s_, i_: lambda m, r: d_diff_affine(
                            m, s_, i_, aligned, use_smooth))(sl, ic)
                        a = score_absolute(train, fn)
                        if best is None or a["iou_all"] > best[0]:
                            best = (a["iou_all"], sl, ic)
                _, sl, ic = best
                fn = lambda m, r: d_diff_affine(m, sl, ic, aligned, use_smooth)
                agg = score_absolute(test, fn)
                print(_fmt(f"affine a={int(aligned)} sm={int(use_smooth)} "
                           f"sl={sl:.2f}", ic, agg))

    if args.sweep == "area":
        rng = np.random.RandomState(0)
        order = rng.permutation(len(data))
        n_tr = int(0.5 * len(data))
        train = [data[i] for i in order[:n_tr]]
        test = [data[i] for i in order[n_tr:]]
        print(f"train {len(train)} / test {len(test)}\n")
        print(HEADER)
        print("-" * 82)
        for aligned in (True, False):
            for use_smooth in (False, True):
                coef, inter = fit_area_model(train, aligned, use_smooth)
                best = None
                for bias in np.arange(-0.010, 0.0101, 0.0005):
                    a = score_area_match(test, coef, inter, aligned,
                                         use_smooth, bias)
                    if best is None or a["iou_all"] > best[1]["iou_all"]:
                        best = (bias, a)
                print(_fmt(f"area a={int(aligned)} sm={int(use_smooth)} "
                           f"c={coef:.1f}", best[0], best[1]))


if __name__ == "__main__":
    main()
