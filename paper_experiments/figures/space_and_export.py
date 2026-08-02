"""Add white gutters to the figure-candidate previews, export the raw panels, and write PDFs.

The candidate previews written by make_candidates.py / make_candidates_job2.py are
tight grids: every panel touches its neighbours, so it is hard to tell where one
frame stops and the next starts. This script re-composes each preview with white
space between the frames, and at the same time collects everything needed to lay
the figure out in the paper:

  log/paper_figure_assets/<figure>/<object>_<touch>/panels/*.png
      the individual frames at their native 320x240, with the caption strip
      removed -- these are the raw images to drop into LaTeX
  log/paper_figure_assets/<figure>/<object>_<touch>/figure.png / figure.pdf
      the spaced composite, as a picture and as a vector-wrapped PDF
  log/paper_figure_assets/<figure>/<figure>_all_candidates.pdf
      all five candidates for that figure, one per page, for flipping through

Nothing is re-predicted: the panels are cut back out of the previews that are
already on disk, so this is cheap and always consistent with the report.
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
CAND_ROOT = os.path.join(ROOT, "log/paper_figure_candidates")
ASSET_ROOT = os.path.join(ROOT, "log/paper_figure_assets")

CELL_W, CELL_H = 320, 240      # the frame itself
CAP_H = 26                     # the white caption strip make_candidates.py puts on top
GAP = 18                       # white space between frames side by side
ROW_GAP_SCALE = 1.2 * 1.4      # rows were widened a fifth, then a further 40%
ROW_GAP = round(GAP * ROW_GAP_SCALE)   # so the rows read as separate bands
MARGIN = 18                    # white space around the whole grid
PDF_DPI = 200

# Panel names for the figures whose panels each mean something different. Order
# matches the order the builders stack them in; names become the filenames of the
# extracted raw panels.
LAYOUTS = {
    "teaser": [["A_reference_geometry", "A_reference_touch"],
               ["B_query_geometry", "B_our_prediction"]],
    "gt_retrieval": [["1_reference_touch", "2_reference_normal", "3_query_normal",
                      "4_coarse_transfer", "5_refined_ours", "6_ground_truth"]],
    "ablation": [["1_without_refinement", "2_without_temporal_film",
                  "3_without_normal_concat", "4_ours_full", "5_ground_truth"]],
}

# Figures where a row is a method and the columns are frames of one press. How
# many frames is read off the preview itself, so changing it in the builder needs
# no edit here.
FRAME_ROWS = {
    "recon": ["1_reference", "2_prediction", "3_relief_3d", "4_visuo_tactile_rgb"],
    "full_pipeline": ["1_reference", "2_quilting", "3_objectfolder_inr",
                      "4_ours_coarse", "5_ours_refined", "6_ground_truth"],
}


def layout_for(fig, preview):
    if fig in LAYOUTS:
        return LAYOUTS[fig]
    n_cols = preview.shape[1] // CELL_W
    return [[f"{r}_f{c}" for c in range(n_cols)] for r in FRAME_ROWS[fig]]

FIGURE_TEX = {
    "teaser": "fig_teaser.tex",
    "gt_retrieval": "fig_gt_retrieval.tex",
    "full_pipeline": "fig_full_pipeline.tex",
    "ablation": "fig_ablation.tex",
    "recon": "fig_recon.tex",
    "method": "fig_method.tex",
}


def imread(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def imwrite(p, im):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def split_grid(preview, layout):
    """Cut a tight preview back into its (caption strip + frame) cells."""
    n_rows, n_cols = len(layout), len(layout[0])
    exp_h, exp_w = n_rows * (CAP_H + CELL_H), n_cols * CELL_W
    if preview.shape[:2] != (exp_h, exp_w):
        raise ValueError(f"expected {exp_h}x{exp_w}, got {preview.shape[1::-1]}")
    cells = []
    for r in range(n_rows):
        y = r * (CAP_H + CELL_H)
        cells.append([preview[y:y + CAP_H + CELL_H, c * CELL_W:(c + 1) * CELL_W]
                      for c in range(n_cols)])
    return cells


def compose_spaced(cells):
    """Re-stack the cells with white space between them."""
    n_rows, n_cols = len(cells), len(cells[0])
    ch, cw = CAP_H + CELL_H, CELL_W
    h = 2 * MARGIN + n_rows * ch + (n_rows - 1) * ROW_GAP
    w = 2 * MARGIN + n_cols * cw + (n_cols - 1) * GAP
    canvas = np.full((h, w, 3), 255, np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y = MARGIN + r * (ch + ROW_GAP)
            x = MARGIN + c * (cw + GAP)
            canvas[y:y + ch, x:x + cw] = cells[r][c]
    return canvas


def to_pdf(images, path, dpi=PDF_DPI):
    """Write one page per image. Sizing by dpi keeps the frames at print scale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pages = [Image.fromarray(im).convert("RGB") for im in images]
    pages[0].save(path, "PDF", resolution=dpi, save_all=True,
                  append_images=pages[1:])


def do_method(cands, manifest):
    """The method panel is a single SuperPoint/SuperGlue match image, not a grid."""
    pages, names = [], []
    for c in cands:
        o, t = c["object"], c["touch"]
        src = os.path.join(ROOT, c["dir"], f"{t}_matches.png")
        if not os.path.exists(src):
            print(f"  method {o}_{t}: no match image, skipped")
            continue
        dst_dir = os.path.join(ASSET_ROOT, "method", f"{o}_{t}")
        os.makedirs(os.path.join(dst_dir, "panels"), exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, "panels", "1_superglue_matches.png"))
        # the clips the correspondences were computed between, for redrawing the panel
        for suffix in ("ref_tactile_normal.mp4", "query_tactile_normal.mp4",
                       "transferred.mp4"):
            v = os.path.join(ROOT, c["dir"], f"{t}_{suffix}")
            if os.path.exists(v):
                shutil.copy2(v, os.path.join(dst_dir, "panels", f"0_{suffix}"))
        im = imread(src)
        pad = np.full((im.shape[0] + 2 * MARGIN, im.shape[1] + 2 * MARGIN, 3), 255, np.uint8)
        pad[MARGIN:MARGIN + im.shape[0], MARGIN:MARGIN + im.shape[1]] = im
        imwrite(os.path.join(dst_dir, "figure.png"), pad)
        to_pdf([pad], os.path.join(dst_dir, "figure.pdf"))
        pages.append(pad)
        names.append(f"{o}_{t}")
        manifest.setdefault("method", []).append(
            {"object": o, "touch": t, "panels": 1,
             "assets": os.path.relpath(dst_dir, ROOT)})
        print(f"  method {o}_{t}: 1 panel")
    if pages:
        h = max(p.shape[0] for p in pages)
        w = max(p.shape[1] for p in pages)
        pages = [np.pad(p, ((0, h - p.shape[0]), (0, w - p.shape[1]), (0, 0)),
                        constant_values=255) for p in pages]
        to_pdf(pages, os.path.join(ASSET_ROOT, "method", "method_all_candidates.pdf"))
    return names


def do_grid_figure(fig, cands, manifest):
    pages = []
    for c in cands:
        o, t = c["object"], c["touch"]
        src = os.path.join(ROOT, c["preview"])
        preview = imread(src)
        if preview is None:
            print(f"  {fig} {o}_{t}: preview missing, skipped")
            continue
        layout = layout_for(fig, preview)
        try:
            cells = split_grid(preview, layout)
        except ValueError as e:
            print(f"  {fig} {o}_{t}: {e}, skipped")
            continue

        spaced = compose_spaced(cells)
        # the spaced preview lands next to the original so the report can use it
        imwrite(os.path.join(os.path.dirname(src), "preview_spaced.png"), spaced)

        dst_dir = os.path.join(ASSET_ROOT, fig, f"{o}_{t}")
        n = 0
        for r, row in enumerate(cells):
            for cc, cellimg in enumerate(row):
                raw = cellimg[CAP_H:]           # drop the caption strip
                imwrite(os.path.join(dst_dir, "panels", f"{layout[r][cc]}.png"), raw)
                n += 1
        imwrite(os.path.join(dst_dir, "figure.png"), spaced)
        to_pdf([spaced], os.path.join(dst_dir, "figure.pdf"))
        pages.append(spaced)
        manifest.setdefault(fig, []).append(
            {"object": o, "touch": t, "panels": n,
             "assets": os.path.relpath(dst_dir, ROOT)})
        print(f"  {fig} {o}_{t}: {n} panels -> {spaced.shape[1]}x{spaced.shape[0]}")
    if pages:
        to_pdf(pages, os.path.join(ASSET_ROOT, fig, f"{fig}_all_candidates.pdf"))


README = """# Raw assets for the qualitative figures

Everything here was cut out of the candidate previews under
`log/paper_figure_candidates/`, so it matches the report
`log/paper_figure_candidates_report.html` exactly.

Layout, per figure and per candidate (candidates are named `<object>_<touch>`):

    <figure>/<object>_<touch>/panels/*.png   individual frames, 320x240, no caption
    <figure>/<object>_<touch>/figure.png     the panels laid out with white space between them
    <figure>/<object>_<touch>/figure.pdf     the same layout as a PDF page
    <figure>/<figure>_all_candidates.pdf     all candidates for that figure, one per page

The files in `panels/` are the ones to use in LaTeX: they carry no caption text
and no borders, so labels can be set in the document instead of being baked into
the picture. `figure.png` and `figure.pdf` keep the captions, and are meant for
looking at rather than for the paper.

`manifest.json` lists every candidate, how many panels it has, and where they are.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+",
                    default=list(LAYOUTS) + list(FRAME_ROWS) + ["method"])
    args = ap.parse_args()

    with open(os.path.join(CAND_ROOT, "candidates.json")) as f:
        cands = json.load(f)

    manifest = {}
    for fig in args.figures:
        items = cands.get(fig, [])
        print(f"\n=== {fig}  ({len(items)} candidates)")
        if not items:
            continue
        if fig == "method":
            do_method(items, manifest)
        else:
            do_grid_figure(fig, items, manifest)

    os.makedirs(ASSET_ROOT, exist_ok=True)
    # keep entries for figures this run did not touch, so --figures does not wipe them
    mpath = os.path.join(ASSET_ROOT, "manifest.json")
    figures = json.load(open(mpath)).get("figures", {}) if os.path.exists(mpath) else {}
    figures.update(manifest)
    with open(mpath, "w") as f:
        json.dump({"gap_px": GAP, "row_gap_px": ROW_GAP, "margin_px": MARGIN,
                   "panel_size": [CELL_W, CELL_H],
                   "figures": figures, "tex": FIGURE_TEX}, f, indent=2)
    with open(os.path.join(ASSET_ROOT, "README.md"), "w") as f:
        f.write(README)
    print(f"\nAssets -> {ASSET_ROOT}")


if __name__ == "__main__":
    main()
