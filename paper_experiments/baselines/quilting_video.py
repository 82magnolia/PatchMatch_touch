"""Turn the Tactile Normal Quilting baseline into a real video prediction.

As run, the quilting baseline synthesises ONE tactile image and calls
write_repeated_video to tile it across the press (run_baseline.py:446) -- so it
appears in the comparison figure as "N/A (image only)". That is a property of how
the baseline was implemented, not of the underlying Tactile DreamFusion idea: the
quilted output is a surface relief, and a relief plus a press depth is exactly
what the simulator turns into a tactile frame.

This script upgrades the baseline to emit a press sequence:

  quilted normal map -> Poisson integration -> surface relief H
  for each frame, a press depth d(t) reveals the part of the relief that has
  penetrated the gel:  z = max(H - (H_max - f(t) * range), 0),  f = d/d_max
  the soft gel is then approximated with Taxim's own pyramid-Gaussian scheme
  (Basics/params.py: pyramid_kernel_size, contact_scale, kernel_size, mirroring
  simOptical.deformApprox), and the deformed height is encoded back to a tactile
  normal frame with Taxim's height_map_to_normals -- the same encoding the
  ground-truth videos use.

The press profile is the benchmark's own canonical one (back_forth_press,
0 -> 10 mm over 50 frames, from gen_contact_*/_run.sh). That is a fixed dataset
constant, identical for every touch, so this uses no per-touch privileged
information: the baseline is told "presses look like this", not "this press was
this deep".

Writes a new prediction alongside the original so both can be compared; the
tiled-image runs are left untouched.

--benchmark job2 reads the full-pipeline runs instead, where the quilt cache is
keyed by the RETRIEVED reference index rather than the query index, so the
query->reference pairing is read back from each run's resolved_config.json.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, os.path.join(ROOT, "Taxim/OpticalSimulation"))
sys.path.insert(0, os.path.join(ROOT, "baselines/RandomQuiltingTactile/TactileDreamFusion"))
sys.path.insert(0, ROOT)

from poisson_solver import poisson_dct_neumann          # noqa: E402

# Taxim gel-deformation constants (Taxim/Basics/params.py)
PYRAMID_KERNELS = [201, 101, 51, 21, 11, 5]
KERNEL_SIZE = 5
CONTACT_SCALE = 0.4
# Benchmark press profile (train_refine_scripts/gen_contact_*/_run.sh)
PRESS_MIN, PRESS_MAX, N_STEP = 0.0, 10.0, 50
MODE = "back_forth_press"

QUILT_SRC = os.path.join(ROOT, "log/paper_job1_baselines/quilting")
GT_SRC = os.path.join(ROOT, "log/paper_job1_transfer_normal")
OUT = os.path.join(ROOT, "log/paper_job1_baselines/quilting_video")


def press_depths():
    if MODE == "continuous_press":
        return np.linspace(PRESS_MIN, PRESS_MAX, N_STEP)
    fwd = np.linspace(PRESS_MIN, PRESS_MAX, N_STEP // 2)
    bwd = np.linspace(PRESS_MAX, PRESS_MIN, N_STEP - N_STEP // 2 + 1)
    return np.concatenate([fwd, bwd[1:]])


def normal_to_height(rgb01):
    """Tactile-normal encoding -> relief height via Poisson integration."""
    n = 2.0 * rgb01 - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    nz = np.clip(n[..., 2], 0.05, 1.0)
    return poisson_dct_neumann(-n[..., 0] / nz, -n[..., 1] / nz)


def height_to_normal_rgb(H):
    """Deformed height -> tactile-normal RGB, Taxim's own convention.

    Mirrors simOptical.height_map_to_normals: normal = [-dzdx, -dzdy, 1]/norm,
    encoded as uint8((n+1)/2*255), which is what the GT videos store.
    """
    h, w = H.shape
    dzdx = np.zeros_like(H)
    dzdy = np.zeros_like(H)
    dzdx[1:h - 1, 1:w - 1] = (H[2:h, 1:w - 1] - H[0:h - 2, 1:w - 1]) / 2.0
    dzdy[1:h - 1, 1:w - 1] = (H[1:h - 1, 2:w] - H[1:h - 1, 0:w - 2]) / 2.0
    n = np.dstack([-dzdx, -dzdy, np.ones_like(H)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return np.clip((n + 1.0) / 2.0, 0.0, 1.0)


def deform(H_rel, frac):
    """Gel deformation at press fraction `frac` in [0,1] over the relief H_rel.

    frac scales how far the relief has penetrated the gel; the pyramid-Gaussian
    cascade approximates the soft body exactly as simOptical.deformApprox does,
    keeping the in-contact region sharp and smoothing the surround.
    """
    rng = float(H_rel.max() - H_rel.min())
    if rng < 1e-9 or frac <= 0:
        return np.zeros_like(H_rel)
    cut = H_rel.max() - frac * rng
    zq = np.maximum(H_rel - cut, 0.0).astype(np.float32)
    zq_back = zq.copy()
    mask = zq > (frac * rng) * CONTACT_SCALE
    for k in PYRAMID_KERNELS:
        zq = cv2.GaussianBlur(zq, (k, k), 0)
        zq[mask] = zq_back[mask]
    return cv2.GaussianBlur(zq, (KERNEL_SIZE, KERNEL_SIZE), 0)


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    tmp = path + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    os.replace(tmp, path)


def gt_info(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", type=int, default=list(range(951, 1001)))
    ap.add_argument("--touches", type=int, default=8)
    ap.add_argument("--benchmark", choices=["job1", "job2"], default="job1")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Explicit obj:touch list, e.g. 81:0 83:0 -- figure candidates only")
    args = ap.parse_args()

    global QUILT_SRC, GT_SRC, OUT
    if args.benchmark == "job2":
        QUILT_SRC = os.path.join(ROOT, "log/paper_job2_baselines/quilting")
        GT_SRC = os.path.join(ROOT, "log/paper_job2_pipeline_normal")
        OUT = os.path.join(ROOT, "log/paper_job2_baselines/quilting_video")

    def gt_path_for(obj, t):
        if args.benchmark == "job2":
            return os.path.join(GT_SRC, str(obj), "transfer", f"{t}_query_tactile_normal.mp4")
        return os.path.join(GT_SRC, str(obj), f"{t}_query_tactile_normal.mp4")

    def quilt_png_for(obj, t):
        cache = os.path.join(QUILT_SRC, str(obj), "cache")
        if args.benchmark == "job1":
            return os.path.join(cache, f"ref_{t}_quilted.png")
        cfg = os.path.join(QUILT_SRC, str(obj), "resolved_config.json")
        if not os.path.exists(cfg):
            return None
        with open(cfg) as f:
            pairs = json.load(f).get("pairs", [])
        ref = dict((int(q), int(r)) for q, r in pairs).get(int(t))
        return None if ref is None else os.path.join(cache, f"ref_{ref}_quilted.png")

    todo = None
    if args.pairs:
        todo = [tuple(int(x) for x in p.split(":")) for p in args.pairs]

    depths = press_depths()
    frac_all = (depths - PRESS_MIN) / (PRESS_MAX - PRESS_MIN)
    made = 0
    iter_pairs = todo if todo else [(o, t) for o in args.objects for t in range(args.touches)]
    for obj, t in iter_pairs:
        if True:
            quilt_png = quilt_png_for(obj, t)
            gt_path = gt_path_for(obj, t)
            if quilt_png is None or not os.path.exists(quilt_png) or not os.path.exists(gt_path):
                continue
            n_frames, fps, w, h = gt_info(gt_path)
            if n_frames <= 0:
                continue

            q = cv2.imread(quilt_png)
            q = cv2.cvtColor(cv2.resize(q, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H_rel = normal_to_height(q)

            fr = (np.interp(np.linspace(0, len(frac_all) - 1, n_frames),
                            np.arange(len(frac_all)), frac_all))
            frames = [height_to_normal_rgb(deform(H_rel, f)) for f in fr]

            d = os.path.join(OUT, str(obj), "transfer")
            os.makedirs(d, exist_ok=True)
            write_video(os.path.join(d, f"{t}_transferred.mp4"), frames, fps)
            made += 1
            print(f"  {obj}_{t} -> press sequence ({n_frames} frames)", flush=True)

    print(f"\nwrote {made} press-sequence videos -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
