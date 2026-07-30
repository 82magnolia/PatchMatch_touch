"""Regenerate the *simulated* tactile normal videos (*_tactile_normal.mp4) in
Taxim/results/*_tactile_normal_* so they match the *real* pipeline's look: the
per-touch contact mask (*_mask.mp4) selects where the object's normals are kept,
everything outside is the flat (0, 0, 1) background, and the seam between the two
is Poisson-blended (poisson_blend_normals) exactly as process_single_shot does
for the real captures.

Why this is needed: the sim video written by gen_contact_video.py stores
height_map_to_normals over the WHOLE gel, so the deformation bleeds into the
"background" and there is a hard object boundary -- unlike the real videos, which
sit on a clean flat background with a soft, Poisson-blended contact halo.

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
    normals_to_colormap, read_video_frames, write_video, VIDEO_FPS)
from _tactile_normal_net import (  # noqa: E402
    poisson_blend_normals, unit_normals)

MASK_THRESHOLD = 127  # mask.mp4 is a soft (anti-aliased) 0..255 map; binarize it


def decode_normal_frame(frame_bgr):
    """*_tactile_normal.mp4 BGR frame -> (H, W, 3) float32 normal map (nx,ny,nz).

    Inverse of the (n + 1) / 2 * 255 RGB encoding used by both gen_contact_video
    and normals_to_colormap.
    """
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return rgb * 2.0 - 1.0


def regen_frame(normal_frame_bgr, mask_frame):
    """One frame: mask -> keep in-mask normals, flat background, Poisson seam."""
    m = mask_frame[..., 0] > MASK_THRESHOLD if mask_frame.ndim == 3 \
        else mask_frame > MASK_THRESHOLD
    nrm = decode_normal_frame(normal_frame_bgr)
    nx = nrm[..., 0].copy()
    ny = nrm[..., 1].copy()
    # Background is the flat gel: zero tangential slope == (0, 0, 1).
    nx[~m] = 0.0
    ny[~m] = 0.0
    nx, ny = poisson_blend_normals(nx, ny, m)
    return normals_to_colormap(unit_normals(nx, ny))


def regen_touch(tn_path, out_path=None):
    """Rewrite one *_tactile_normal.mp4 in place (or to out_path) from its mask."""
    mask_path = tn_path[:-len("_tactile_normal.mp4")] + "_mask.mp4"
    if not osp.exists(mask_path):
        return 0
    tn = read_video_frames(tn_path)
    mk = read_video_frames(mask_path)
    n = min(len(tn), len(mk))
    if n == 0:
        return 0
    frames = [regen_frame(tn[i], mk[i]) for i in range(n)]
    write_video(out_path or tn_path, frames, VIDEO_FPS)
    return n


def _work(tn_path):
    return tn_path, regen_touch(tn_path)


def _numkey(p):
    m = re.findall(r"\d+", osp.basename(osp.dirname(p)))
    return (int(m[0]) if m else 0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+",
                    help="Taxim results dir(s) holding per-example subdirs")
    ap.add_argument("--workers", type=int,
                    default=min(24, os.cpu_count() or 1))
    args = ap.parse_args()

    tns = []
    for root in args.roots:
        found = glob.glob(osp.join(root, "*", "*_tactile_normal.mp4"))
        if not found:  # root may itself be a single example dir
            found = glob.glob(osp.join(root, "*_tactile_normal.mp4"))
        tns += found
    tns = sorted(set(tns), key=_numkey)
    total = len(tns)
    print(f"Found {total} sim tactile-normal video(s) across {len(args.roots)} "
          f"root(s); {args.workers} workers.", flush=True)

    t0 = time.time()
    nframes = 0
    with mp.Pool(args.workers) as pool:
        for k, (tp, nf) in enumerate(pool.imap_unordered(_work, tns), 1):
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
