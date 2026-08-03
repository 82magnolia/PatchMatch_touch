"""The ground-truth-retrieval qualitative figure: touches down, streams across.

Six columns, following paper_source/figures/fig_gt_retrieval.tex:

    reference touch | reference normals | query normals |
    coarse transfer | refined (ours) | ground truth

Rows are touch locations, chosen by hand from log/paper_figure_candidates.html.

Drawn on a white page with black text, a white gap between every frame, and a
black outline on the two geometry columns only -- the tactile frames fill their
own cell edge to edge, so a box around them would only add clutter, while the
normal renders are mostly pale background and need a boundary to read as images.
Those two columns show the render at four times the sensor footprint, with the
sensor's own footprint marked in red, and their empty background repainted white
to match the page.

    python make_gt_figure.py --touches 974_5 978_4 969_3 970_7 995_5
"""
import argparse
import os
import pickle

import cv2
import numpy as np

import sys
ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, f"{ROOT}/paper_experiments/04_paper_figures")
from figlib import Page, load, white_bg               # noqa: E402

ASSETS = f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch/assets"
METRICS = f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch/per_touch_metrics.pkl"
OUT = f"{ROOT}/log/paper_job06_gt_retrieval_figure"

# (asset suffix, column header, is it a geometry render?)
COLS = [("01_ref_touch", "Reference touch", False),
        ("02_ref_normal", "Reference normals", True),
        ("03_query_normal", "Query normals", True),
        ("04_coarse", "Coarse transfer", False),
        ("05_refined", "Refined (ours)", False),
        ("06_gt_query", "Ground truth", False)]


def cell(tag, suffix, is_geom, scale, boxed, normal_bg):
    """One cell of the figure, already in the form it will be drawn."""
    if is_geom:
        stem = f"{ASSETS}/{tag}_{suffix}_scale{scale}"
        path = f"{stem}_box.png" if boxed and os.path.exists(f"{stem}_box.png") else f"{stem}.png"
        im = load(path)
        return white_bg(im) if normal_bg == "white" else im
    return load(f"{ASSETS}/{tag}_{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", nargs="+",
                    default=["974_5", "978_4", "969_3", "970_7", "995_5"],
                    help="'object_touch' rows, top to bottom")
    ap.add_argument("--normal_scale", default="25", choices=["100", "50", "25"],
                    help="field of view of the two geometry columns: 100 = 1x the sensor "
                         "footprint, 50 = 2x, 25 = 4x (the scale the matching runs at)")
    ap.add_argument("--no_box", action="store_true",
                    help="do not mark the 1x sensor footprint inside the wider render")
    ap.add_argument("--normal_bg", default="white", choices=["white", "black"])
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--gap", type=float, default=0.05, help="white gap between frames")
    ap.add_argument("--row_gap_mult", type=float, default=2.0,
                    help="row separation, in multiples of the gap between columns")
    ap.add_argument("--border", type=float, default=0.8,
                    help="line width of the black outline on the geometry columns")
    ap.add_argument("--tag", default="gt_retrieval")
    ap.add_argument("--out_dir", default=OUT)
    args = ap.parse_args()

    cells_dir = f"{args.out_dir}/assets_pdf/{args.tag}"
    os.makedirs(cells_dir, exist_ok=True)

    scores = {}
    if os.path.exists(METRICS):
        for r in pickle.load(open(METRICS, "rb")):
            scores[f"{r['obj']}_{r['pair']}"] = r

    n_rows, n_cols = len(args.touches), len(COLS)
    left = right = 0.09
    iw = (args.width - left - right - (n_cols - 1) * args.gap) / n_cols
    ih = iw * 0.75
    row_gap = args.gap * args.row_gap_mult
    head_h, bottom = 0.20, 0.14
    height = head_h + n_rows * ih + (n_rows - 1) * row_gap + bottom

    p = Page(args.width, height)
    for c, (_, name, _) in enumerate(COLS):
        p.text(left + c * (iw + args.gap) + iw / 2, head_h - 0.09, name,
               size=6.4, ha="center", color="black")

    for r, tag in enumerate(args.touches):
        y = head_h + r * (ih + row_gap)
        for c, (suffix, _, is_geom) in enumerate(COLS):
            im = cell(tag, suffix, is_geom, args.normal_scale, not args.no_box,
                      args.normal_bg)
            cv2.imwrite(f"{cells_dir}/row{r + 1}_{tag}_col{c + 1}_{suffix}.png",
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            p.img(left + c * (iw + args.gap), y, iw, ih, im,
                  edge="black" if is_geom else None,
                  lw=args.border if is_geom else 0)
        s = scores.get(tag)
        if s:
            print(f"  row {r + 1}: object {s['obj']}, touch {s['pair']}  "
                  f"coarse {s['coarse']['PSNR']:.1f} dB -> refined {s['refined']['PSNR']:.1f} dB")

    p.save(f"{args.out_dir}/{args.tag}")
    print(f"cells: {cells_dir}")


if __name__ == "__main__":
    main()
