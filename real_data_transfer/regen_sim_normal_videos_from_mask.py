"""Regenerate the *simulated* tactile normal videos (*_tactile_normal.mp4) in
Taxim/results/*_tactile_normal_* so they match the *real* pipeline's look: a
contact mask selects where the object's normals are kept, everything outside is
the flat (0, 0, 1) background, and the seam between the two is Poisson-blended
(poisson_blend_normals) exactly as the real captures are.

Why this is needed: the sim video written by gen_contact_video.py stores
height_map_to_normals over the WHOLE gel, so the deformation bleeds into the
"background" and there is a hard object boundary -- unlike the real videos, which
sit on a clean flat background with a soft, Poisson-blended contact halo.

Contact mask: instead of the geometry-derived *_mask.mp4 (which is the tight
object footprint and clips off the soft gel-deformation halo around it), the mask
is derived per frame from the normal image itself with compute_contact_mask --
thresholding each frame's L2 difference against frame 0 (the blank, no-contact
reference). This is the same image-diff contact mask the real pipeline uses, and
it captures the full deformed region (bulge included), so the blend preserves it.

Source of the in-mask normals: the existing *_tactile_normal.mp4 itself. Its RGB
already encodes the sim normals as (n + 1) / 2, so we decode it back, keep only
the tangential (nx, ny) inside the mask, blend against the flat background, and
re-encode. No re-simulation and no other artifact is touched.

Usage:
  conda activate pm_real
  python regen_sim_normal_videos_from_mask.py \
      ../Taxim/results/gen_contact_full_tactile_normal_pseudo_mini \
      ../Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
"""
import argparse
import glob
import multiprocessing as mp
import os
import re
import sys
import time
from os import path as osp

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _gelsight_processing import (  # noqa: E402
    compute_contact_mask, normals_to_colormap, read_video_frames, write_video,
    VIDEO_FPS)
from _tactile_normal_net import (  # noqa: E402
    poisson_blend_normals, unit_normals)

# Image-diff contact-mask settings. Chosen by the sweep in
# log/sim_normal_mask_sweep: threshold is the only knob that meaningfully moves
# the result (blur/morph barely change the sharp sim signal, though morph is kept
# non-zero to fill interior holes). 0.05 is higher than the real pipeline's 0.025
# to contain the deformation halo, while staying well above the ~0.005 blank-frame
# noise floor so the background is perfectly clean.
CONTACT_THRESHOLD = 0.05
BLUR_SIGMA = 3.0
MORPH_RADIUS = 5


def decode_normal_frame(frame_bgr):
    """*_tactile_normal.mp4 BGR frame -> (H, W, 3) float32 normal map (nx,ny,nz).

    Inverse of the (n + 1) / 2 * 255 RGB encoding used by both gen_contact_video
    and normals_to_colormap.
    """
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return rgb * 2.0 - 1.0


def diff_contact_mask(frame_bgr, base_bgr):
    """Boolean contact mask from the normal image's L2 diff against frame 0."""
    return compute_contact_mask(
        frame_bgr.astype(np.float32) / 255.0,
        base_bgr.astype(np.float32) / 255.0,
        threshold=CONTACT_THRESHOLD, blur_sigma=BLUR_SIGMA,
        morph_radius=MORPH_RADIUS)[..., 0] > 0.5


def regen_frame(normal_frame_bgr, base_bgr):
    """One frame: diff mask -> keep in-mask normals, flat bg, Poisson seam."""
    m = diff_contact_mask(normal_frame_bgr, base_bgr)
    nrm = decode_normal_frame(normal_frame_bgr)
    nx = nrm[..., 0].copy()
    ny = nrm[..., 1].copy()
    # Background is the flat gel: zero tangential slope == (0, 0, 1).
    nx[~m] = 0.0
    ny[~m] = 0.0
    nx, ny = poisson_blend_normals(nx, ny, m)
    return normals_to_colormap(unit_normals(nx, ny))


def regen_touch(tn_path, out_path=None):
    """Rewrite one *_tactile_normal.mp4 in place (or to out_path).

    The contact mask is derived from each frame's difference against frame 0
    (the blank no-contact reference), not from the geometry *_mask.mp4.
    """
    tn = read_video_frames(tn_path)
    n = len(tn)
    if n == 0:
        return 0
    base = tn[0]
    frames = [regen_frame(tn[i], base) for i in range(n)]
    write_video(out_path or tn_path, frames, VIDEO_FPS)
    return n


def _work(args):
    tn_path, out_path = args
    if out_path:
        os.makedirs(osp.dirname(out_path), exist_ok=True)
    return tn_path, regen_touch(tn_path, out_path)


def _numkey(pair):
    p = pair[0]
    m = re.findall(r"\d+", osp.basename(osp.dirname(p)))
    return (int(m[0]) if m else 0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+",
                    help="Taxim results dir(s) holding per-example subdirs")
    ap.add_argument("--out-suffix", default=None,
                    help="If set, write to a sibling dir named <root><suffix> "
                         "(mirroring the per-example/touch layout) instead of "
                         "overwriting the videos in place. e.g. '_maskblend'.")
    ap.add_argument("--workers", type=int,
                    default=min(24, os.cpu_count() or 1))
    args = ap.parse_args()

    jobs = []  # (tn_path, out_path or None)
    for root in args.roots:
        root = osp.normpath(root)
        found = glob.glob(osp.join(root, "*", "*_tactile_normal.mp4"))
        base_root = root
        if not found:  # root may itself be a single example dir
            found = glob.glob(osp.join(root, "*_tactile_normal.mp4"))
            base_root = osp.dirname(root)
        if args.out_suffix:
            out_root = root + args.out_suffix
        for tn in found:
            out = osp.join(out_root, osp.relpath(tn, root)) \
                if args.out_suffix else None
            jobs.append((tn, out))
    jobs = sorted(set(jobs), key=_numkey)
    total = len(jobs)
    dest = f"sibling '*{args.out_suffix}' dirs" if args.out_suffix else "in place"
    print(f"Found {total} sim tactile-normal video(s) across {len(args.roots)} "
          f"root(s); writing {dest}; {args.workers} workers.", flush=True)

    t0 = time.time()
    nframes = 0
    with mp.Pool(args.workers) as pool:
        for k, (tp, nf) in enumerate(pool.imap_unordered(_work, jobs), 1):
            nframes += nf
            if k % 50 == 0 or k == total:
                dt = time.time() - t0
                rate = k / dt if dt else 0
                eta = (total - k) / rate if rate else 0
                print(f"[{k}/{total}] {rate:.1f} touch/s, {nframes} frames, "
                      f"ETA {eta/60:.1f} min", flush=True)
    print(f"Done. Regenerated {total} sim normal video(s) in "
          f"{(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
