"""Compare ways to soften ONLY the contact-mask boundary of the tactile normal
video, while leaving the interior normals untouched (preserve local detail).

Methods (all recompute nz from the unit constraint, so output stays unit-norm):
  hard      : current behaviour, sharp step at the mask edge.
  gauss     : band-limited Gaussian -- blur (nx,ny), but keep the blurred values
              only inside a band straddling the boundary; interior kept as-is.
  poisson   : gradient-domain solve in the same band. Deep interior (net) and
              deep exterior (0) are Dirichlet-fixed; the band is solved so its
              Laplacian matches the net's (imports rim detail) while smoothly
              meeting the background -> smooth seam, interior untouched.

Usage:
  conda activate pm_real
  python test_scripts/compare_boundary_blend.py --session ../log/real_data_gt_retrieval/1
"""
import argparse
import glob
import os
import sys
from os import path as osp

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)),
                            "..", "real_data_transfer"))

from _gelsight_processing import compute_contact_mask, normals_to_colormap  # noqa: E402
from _tactile_normal_net import (  # noqa: E402
    load_normal_net, frame_to_normals,
    unit_normals, boundary_band, poisson_blend_normals)
from process_single_shot import NORMAL_CONTACT_THRESHOLD  # noqa: E402

BAND_PX = 8
GAUSS_SIGMA = 4.0
TIME_FRAC = 0.6


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


def zoom_crop(img, band, up=4, size=80):
    """Upscaled crop centred on the band centroid, for eyeballing the seam."""
    ys, xs = np.where(band)
    if len(ys) == 0:
        return cv2.resize(img, (size * up, size * up),
                          interpolation=cv2.INTER_NEAREST)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = img.shape[:2]
    y0 = np.clip(cy - size // 2, 0, h - size)
    x0 = np.clip(cx - size // 2, 0, w - size)
    crop = img[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, (size * up, size * up),
                      interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--net", default=osp.join(
        osp.dirname(osp.abspath(__file__)), "..", "real_data_transfer",
        "gsnormal_models", "nnmini.pt"))
    ap.add_argument("--touches", nargs="*", default=None)
    ap.add_argument("--band_px", type=int, default=BAND_PX)
    ap.add_argument("--outdir", default="log/boundary_blend")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_normal_net(args.net, device)

    all_touches = [osp.basename(p).split("_")[0]
                   for p in sorted(glob.glob(osp.join(args.session, "*_shadow.mp4")))]
    touches = args.touches or all_touches[:4]
    session_name = osp.basename(osp.normpath(args.session))
    img_dir = osp.join(args.outdir, session_name)
    os.makedirs(img_dir, exist_ok=True)

    col_labels = ["shadow", "hard", f"gauss (band {args.band_px}, σ{GAUSS_SIGMA:g})",
                  f"poisson (band {args.band_px})"]
    rows_html = []
    max_unit_err = 0.0

    for ti in touches:
        shadow = read_video(osp.join(args.session, f"{ti}_shadow.mp4"))
        if len(shadow) < 2:
            continue
        base = shadow[0].astype(np.float32) / 255.0
        n = len(shadow)
        fi = int(round(TIME_FRAC * (n - 1)))
        frame = shadow[fi]
        cmask = compute_contact_mask(
            frame.astype(np.float32) / 255.0, base,
            threshold=NORMAL_CONTACT_THRESHOLD)[..., 0] > 0.5
        if not cmask.any():
            continue

        hard = frame_to_normals(frame, net, device, contact_mask=cmask)
        nx, ny = hard[..., 0].copy(), hard[..., 1].copy()
        band = boundary_band(cmask, args.band_px)

        # Gaussian band: blur, keep blurred only inside the band.
        gx = cv2.GaussianBlur(nx, (0, 0), GAUSS_SIGMA)
        gy = cv2.GaussianBlur(ny, (0, 0), GAUSS_SIGMA)
        gnx = np.where(band, gx, nx)
        gny = np.where(band, gy, ny)
        gauss = unit_normals(gnx, gny)

        # Poisson band.
        pnx, pny = poisson_blend_normals(nx, ny, cmask, args.band_px)
        poisson = unit_normals(pnx, pny)

        variants = [hard, gauss, poisson]
        for v in variants:
            err = float(np.abs(np.linalg.norm(v, axis=-1) - 1.0).max())
            max_unit_err = max(max_unit_err, err)

        color = [normals_to_colormap(v) for v in variants]
        full = np.hstack([frame] + color)
        zoom = np.hstack([zoom_crop(frame, band)] +
                         [zoom_crop(c, band) for c in color])

        fn_full = f"{ti}_full.png"
        fn_zoom = f"{ti}_zoom.png"
        cv2.imwrite(osp.join(img_dir, fn_full), full)
        cv2.imwrite(osp.join(img_dir, fn_zoom), zoom)
        rows_html.append(
            f'<h3>touch {ti} (frame {fi}/{n-1})</h3>'
            f'<figure><img src="{session_name}/{fn_full}">'
            f'<figcaption>full &mdash; {" | ".join(col_labels)}</figcaption></figure>'
            f'<figure><img src="{session_name}/{fn_zoom}">'
            f'<figcaption>boundary zoom (×4) &mdash; {" | ".join(col_labels)}'
            f'</figcaption></figure>')

    ok = "PASS" if max_unit_err < 1e-4 else "FAIL"
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Boundary blend comparison &mdash; {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background:#111; color:#eee; }}
  h1 {{ font-size: 20px; }} h3 {{ margin: 26px 0 6px; color:#ddd; }}
  figure {{ margin: 0 0 14px; }}
  img {{ image-rendering: pixelated; width: 100%; max-width: 1500px; border:1px solid #333; }}
  figcaption {{ font-size: 13px; color:#aaa; margin-top:4px; }}
  code {{ color:#8cf; }} .ok {{ color:#7d7; }}
</style>
<h1>Contact-boundary blend &mdash; session {session_name}</h1>
<p>Only a {args.band_px}px band around the mask boundary is modified; the deep
interior is left untouched (local detail preserved). All variants recompute
nz from the unit constraint. Mask = <code>compute_contact_mask</code> @
{NORMAL_CONTACT_THRESHOLD}.</p>
<p>Unit-norm invariant: <span class="ok">{ok}</span>
(max |&#8214;n&#8214;&minus;1| = {max_unit_err:.2e}).</p>
{''.join(rows_html)}
"""
    out_html = osp.join(args.outdir, f"{session_name}.html")
    with open(out_html, "w") as f:
        f.write(html)
    print("Wrote", out_html)
    print(f"unit-norm {ok}, max err {max_unit_err:.2e}")


if __name__ == "__main__":
    main()
