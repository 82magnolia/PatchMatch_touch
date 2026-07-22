"""Regenerate ONLY the tactile normal videos (*_tactile_normal.mp4) in place,
from each touch's existing *_shadow.mp4, using the current production settings
(compute_contact_mask @ NORMAL_CONTACT_THRESHOLD, Poisson boundary blend @
NORMAL_POISSON_BAND_PX). No other artifact is touched and no re-segmentation
happens -- the shadow video IS the resampled sequence the normal video is built
from, so this exactly reproduces process_single_shot's normal-video block.

Usage:
  conda activate pm_real
  python regen_normal_videos_from_shadow.py ../log/real_data_gt_retrieval_sync \
                                            ../log/real_data_real_retrieval_sync
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from os import path as osp

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _gelsight_processing import (  # noqa: E402
    compute_contact_mask, normals_to_colormap, write_video, VIDEO_FPS)
from _tactile_normal_net import load_normal_net, frame_to_normals  # noqa: E402
from process_single_shot import (  # noqa: E402
    NORMAL_CONTACT_THRESHOLD, NORMAL_POISSON_BAND_PX, DEFAULT_NORMAL_NN_MODEL_PATH)


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


def regen_touch(shadow_path, net, device):
    shadow = read_video(shadow_path)
    if len(shadow) < 1:
        return 0
    base = shadow[0].astype(np.float32) / 255.0
    frames = []
    for f in shadow:
        cmask = compute_contact_mask(
            f.astype(np.float32) / 255.0, base,
            threshold=NORMAL_CONTACT_THRESHOLD)[..., 0] > 0.5
        frames.append(normals_to_colormap(frame_to_normals(
            f, net, device, contact_mask=cmask,
            poisson_band_px=NORMAL_POISSON_BAND_PX)))
    out = shadow_path[:-len("_shadow.mp4")] + "_tactile_normal.mp4"
    write_video(out, frames, VIDEO_FPS)
    return len(frames)


_NET = None  # per-worker globals (Poisson solve is CPU-bound; net is a tiny MLP)
_DEV = "cpu"


def _init(net_path):
    global _NET
    import torch
    torch.set_num_threads(1)  # avoid thread oversubscription across workers
    _NET = load_normal_net(net_path, _DEV)


def _work(shadow_path):
    return shadow_path, regen_touch(shadow_path, _NET, _DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Root dir(s) holding session subdirs")
    ap.add_argument("--net", default=DEFAULT_NORMAL_NN_MODEL_PATH)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = ap.parse_args()

    shadows = []
    for root in args.roots:
        found = glob.glob(osp.join(root, "*", "*_shadow.mp4"))
        if not found:  # root may itself be a single session dir
            found = glob.glob(osp.join(root, "*_shadow.mp4"))
        shadows += sorted(found)
    total = len(shadows)
    print(f"Found {total} shadow video(s) across {len(args.roots)} root(s); "
          f"{args.workers} workers.", flush=True)

    t0 = time.time()
    nframes = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.net,)) as pool:
        for k, (sp, nf) in enumerate(pool.imap_unordered(_work, shadows), 1):
            nframes += nf
            if k % 50 == 0 or k == total:
                dt = time.time() - t0
                rate = k / dt
                eta = (total - k) / rate if rate else 0
                print(f"[{k}/{total}] {rate:.1f} touch/s, {nframes} frames, "
                      f"ETA {eta/60:.1f} min", flush=True)
    print(f"Done. Regenerated {total} normal video(s) in "
          f"{(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
