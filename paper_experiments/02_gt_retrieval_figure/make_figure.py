"""Assemble the 3 x 6 ground-truth-retrieval qualitative figure.

Columns, following the paper outline:
  1  reference touch, middle frame          (the tactile video we transfer from)
  2  reference surface normal render        (geometry at the reference pose)
  3  query surface normal render            (geometry at the query pose)
  4  coarse transfer, middle frame          (feature-matching alignment only)
  5  refined transfer, middle frame         (after the refinement network)
  6  ground-truth query touch, middle frame

Assets are the per-cell PNGs already cached by run_refine_eval.py; this script
picks which touches to show and stitches the preview grid.
"""
import argparse
import os
import pickle

import cv2
import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job02_gt_retrieval_figure"

COLS = [
    ("01_ref_touch", "Reference touch"),
    ("02_ref_normal", "Reference normal"),
    ("03_query_normal", "Query normal"),
    ("04_coarse", "Coarse transfer"),
    ("05_refined", "Refined transfer (ours)"),
    ("06_gt_query", "Ground-truth query"),
]
CH, CW = 210, 280
BOX_BGR = (0, 0, 255)


def cell(path, whiten=False):
    if not os.path.exists(path):
        return np.zeros((CH, CW, 3), np.uint8)
    im = cv2.imread(path)
    if whiten:
        im = white_bg(im)
    return cv2.resize(im, (CW, CH))


def white_bg(img, thr=96):
    """Repaint a normal render's empty background from black to white.

    Taxim renders empty space as exactly (0, 0, 0), while surface pixels encode
    a unit normal and so are never dark: across the benchmark renders the
    darkest surface pixel has a maximum channel of about 122, and no pixel at
    all falls between 64 and 120. A single threshold on the brightest channel
    therefore separates the two cleanly, JPEG ringing included. The red sensor
    box drawn by draw_sensor_box is well above the threshold and survives.
    """
    out = img.copy()
    out[img.max(axis=2) < thr] = 255
    return out


def draw_sensor_box(img, scale, thickness=2):
    """Outline the 1x sensor footprint inside a wider-field-of-view render.

    Taxim's obj_scale_factor convention: the rendered window is always the same
    240x320 pixels, and a *smaller* scale value shrinks the object so the window
    covers proportionally more of it. So the region a scale100 (1x) render shows
    full-frame appears in a scale S render as a centred box of linear size
    S/100 -- one quarter for scale25 (4x), one half for scale50 (2x). Verified
    empirically by template-matching the downscaled 1x render into the 4x one:
    the best offset lands on the centre to within a pixel or two.
    """
    ratio = 100.0 / float(scale)
    if ratio <= 1.0:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    bh, bw = int(round(h / ratio)), int(round(w / ratio))
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    cv2.rectangle(out, (x0, y0), (x0 + bw - 1, y0 + bh - 1), BOX_BGR, thickness)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", nargs="*", default=None,
                    help="Explicit 'obj_pair' selections; otherwise picked automatically.")
    ap.add_argument("--n_rows", type=int, default=3)
    ap.add_argument("--normal_scale", default="25", choices=["100", "50", "25"],
                    help="Which field of view to show in the two normal-render columns: "
                         "100 = 1x the sensor footprint, 50 = 2x, 25 = 4x (the scale the "
                         "feature matching actually runs at).")
    ap.add_argument("--base", default=OUT,
                    help="Job-2 output directory holding per_touch_metrics.pkl and assets/.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base_dir = args.base
    if args.out is None:
        args.out = f"{base_dir}/figure_gt_retrieval_3x6.png"

    recs = pickle.load(open(f"{base_dir}/per_touch_metrics.pkl", "rb"))
    by_key = {f"{r['obj']}_{r['pair']}": r for r in recs}

    if args.touches:
        chosen = [by_key[k] for k in args.touches if k in by_key]
    else:
        # Rather than showing only the touches the refinement helps most (which
        # would flatter the method), take a spread through the distribution:
        # a strong case, a median case, and a weak-but-typical case, each on a
        # different object.
        by_gain = sorted(recs, key=lambda r: r["refined"]["PSNR"] - r["coarse"]["PSNR"])
        pick_at = [int(0.95 * (len(by_gain) - 1)),
                   int(0.50 * (len(by_gain) - 1)),
                   int(0.20 * (len(by_gain) - 1))][:args.n_rows]
        chosen, seen = [], set()
        for p in pick_at:
            # walk outward from the quantile until we hit an unused object
            for d in range(len(by_gain)):
                for q in (p - d, p + d):
                    if 0 <= q < len(by_gain) and by_gain[q]["obj"] not in seen:
                        chosen.append(by_gain[q])
                        seen.add(by_gain[q]["obj"])
                        break
                else:
                    continue
                break

    # Make both fields of view available as assets for every shown touch, so the
    # figure can be re-made at a different scale without re-running the network.
    render_src = {
        "02_ref_normal": f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini",
        "03_query_normal": f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini",
    }
    for r in chosen:
        for tag, render_dir in render_src.items():
            idx = r["ref_idx"] if tag.startswith("02") else r["pair"]
            for sc in ("100", "50", "25"):
                src = f"{render_dir}/{r['obj']}/{idx}_scale{sc}_normal.jpg"
                stem = f"{base_dir}/assets/{r['obj']}_{r['pair']}_{tag}_scale{sc}"
                if not os.path.exists(src):
                    continue
                im = cv2.imread(src)
                cv2.imwrite(f"{stem}.png", im)
                if sc != "100":
                    # same render with the 1x sensor footprint outlined in red
                    cv2.imwrite(f"{stem}_box.png", draw_sensor_box(im, sc))

    rows, meta = [], []
    for r in chosen:
        base = f"{base_dir}/assets/{r['obj']}_{r['pair']}"
        cells = []
        for tag, _ in COLS:
            p = f"{base}_{tag}.png"
            if tag in render_src:
                # prefer the variant with the 1x sensor footprint boxed in red
                for cand in (f"{base}_{tag}_scale{args.normal_scale}_box.png",
                             f"{base}_{tag}_scale{args.normal_scale}.png"):
                    if os.path.exists(cand):
                        p = cand
                        break
            # the cached assets stay as the renderer produced them; the empty
            # background is repainted white only for what goes on the page
            cells.append(cell(p, whiten=tag in render_src))
        rows.append(np.hstack(cells))
        meta.append(r)

    hdr = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    for i, (_, name) in enumerate(COLS):
        cv2.putText(hdr, name, (i * CW + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    labelled = []
    for row, r in zip(rows, meta):
        bar = np.zeros((24, row.shape[1], 3), np.uint8)
        cv2.putText(bar, f"object {r['obj']}, touch {r['pair']}  |  coarse PSNR "
                         f"{r['coarse']['PSNR']:.1f} dB -> refined {r['refined']['PSNR']:.1f} dB"
                         f"  (frame {r['mid_frame']}/{r['n_frames']})",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        labelled += [bar, row]

    grid = np.vstack([hdr] + labelled)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, grid)
    print("wrote", args.out)
    for r in meta:
        print(f"  row: object {r['obj']} touch {r['pair']}  "
              f"coarse {r['coarse']['PSNR']:.2f} -> refined {r['refined']['PSNR']:.2f}")


if __name__ == "__main__":
    main()
