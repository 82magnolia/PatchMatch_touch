"""
Builds the filmstrip figures for log/render_mask_study.html.

For a handful of representative touches, renders a row per method at three
time points (early / peak / late) so the failure modes are visible rather than
only quoted as numbers: the ARuCO baseline firing at the wrong time, and
diff_affine flooding the whole footprint.
"""

import base64
import glob
import io
import os
import sys
from os import path as osp

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)),
                            "..", "real_data_transfer"))

from eval_render_mask import load_cache, render_masks, VARIANTS  # noqa: E402
from _gelsight_processing import RENDER_MASK_THRES_M  # noqa: E402

OUT = osp.join("log", "render_mask_assets")


def colorize(gt, pred):
    """Green = both, red = predicted only, blue = missed."""
    h, w = gt.shape
    img = np.zeros((h, w, 3), np.uint8)
    img[gt & pred] = (110, 200, 110)
    img[~gt & pred] = (90, 90, 235)     # BGR -> red
    img[gt & ~pred] = (235, 160, 90)    # BGR -> blue
    return img


def png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def main():
    cache_dir = os.environ.get(
        "RM_CACHE", osp.join(os.environ.get("SCRATCH", "/tmp"), "rm_cache"))
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(osp.join(cache_dir, "*.npz")))[:400]

    # pick touches with a decent contact area so the figures are legible
    picks = []
    for f in files:
        c = load_cache(f)
        a = c["gt"].mean(axis=(1, 2))
        if a.max() > 0.12:
            picks.append((f, c))
        if len(picks) >= 4:
            break

    methods = [("baseline", "baseline"), ("diff_scaled", "diff_max_raw"),
               ("diff_affine", "diff_affine")]
    rows = []
    for f, c in picks:
        areas = c["gt"].mean(axis=(1, 2))
        pk = int(np.argmax(areas))
        idxs = [max(0, pk // 2), pk, min(len(areas) - 1, pk + (len(areas) - pk) // 2)]
        entry = {"name": osp.basename(f).replace(".npz", ""), "cols": []}
        preds = {}
        for label, key in methods:
            preds[label] = render_masks(c, VARIANTS[key](c), RENDER_MASK_THRES_M)
        for t in idxs:
            col = {"t": t, "gt_area": float(c["gt"][t].mean()), "imgs": {}}
            for label, _ in methods:
                col["imgs"][label] = png_b64(colorize(c["gt"][t], preds[label][t]))
            entry["cols"].append(col)
        rows.append(entry)

    np.save(osp.join(OUT, "rows.npy"), np.array(rows, dtype=object),
            allow_pickle=True)
    print(f"wrote {len(rows)} filmstrip rows to {OUT}/rows.npy")


if __name__ == "__main__":
    main()
