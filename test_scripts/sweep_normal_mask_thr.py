"""Sweep compute_contact_mask thresholds for the tactile normal video and
emit an HTML report so we can eyeball how much to "loosen" the mask.

Reuses each touch's already-generated *_shadow.mp4 (== the resampled frames the
normal video is built from); base = frame 0, exactly as the production path and
the render-mask GT do. For each threshold we recompute the per-frame contact
mask + gated normals, and render a few time samples side by side.

Usage:
  conda activate pm_real
  python test_scripts/sweep_normal_mask_thr.py --session ../log/real_data_gt_retrieval/1
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
from _tactile_normal_net import load_normal_net, frame_to_normals  # noqa: E402

# Current production threshold is 0.05 (tight). Loosen downward.
DEFAULT_THRESHOLDS = [0.05, 0.035, 0.025, 0.015]
BLUR_SIGMA = 3.0
MORPH_RADIUS = 5
TIME_FRACS = [0.3, 0.6, 0.9]  # where in the (trimmed) segment to sample


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


def touch_indices(session_dir):
    out = []
    for p in sorted(glob.glob(osp.join(session_dir, "*_shadow.mp4"))):
        out.append(osp.basename(p).split("_")[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="Session dir containing *_shadow.mp4")
    ap.add_argument("--net", default=osp.join(
        osp.dirname(osp.abspath(__file__)), "..", "real_data_transfer",
        "gsnormal_models", "nnmini.pt"))
    ap.add_argument("--touches", nargs="*", default=None,
                    help="Touch indices to include (default: first 4)")
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--outdir", default="log/normal_thr_sweep")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_normal_net(args.net, device)

    all_touches = touch_indices(args.session)
    touches = args.touches or all_touches[:4]
    session_name = osp.basename(osp.normpath(args.session))

    img_dir = osp.join(args.outdir, session_name)
    os.makedirs(img_dir, exist_ok=True)

    # coverage[thr] -> list of mean mask-coverage fractions over all sampled frames
    coverage = {t: [] for t in args.thresholds}
    rows_html = []

    for ti in touches:
        shadow = read_video(osp.join(args.session, f"{ti}_shadow.mp4"))
        if len(shadow) < 2:
            continue
        base = shadow[0].astype(np.float32) / 255.0
        n = len(shadow)

        for frac in TIME_FRACS:
            fi = int(round(frac * (n - 1)))
            frame = shadow[fi]
            cells = [frame]  # first cell: raw shadow
            captions = ["shadow"]
            for thr in args.thresholds:
                cmask = compute_contact_mask(
                    frame.astype(np.float32) / 255.0, base,
                    thr, BLUR_SIGMA, MORPH_RADIUS)[..., 0] > 0.5
                coverage[thr].append(float(cmask.mean()))
                normals = frame_to_normals(frame, net, device, contact_mask=cmask)
                cells.append(normals_to_colormap(normals))
                captions.append(f"thr={thr:g} ({100*cmask.mean():.1f}%)")

            montage = np.hstack(cells)
            fname = f"{ti}_f{fi:02d}.png"
            cv2.imwrite(osp.join(img_dir, fname), montage)
            rows_html.append(
                f'<figure><img src="{session_name}/{fname}">'
                f'<figcaption>touch {ti}, frame {fi}/{n-1} &mdash; '
                f'columns: {" | ".join(captions)}</figcaption></figure>')

    # Summary table of mean coverage per threshold
    thr_cells = "".join(f"<th>thr={t:g}</th>" for t in args.thresholds)
    cov_cells = "".join(
        f"<td>{100*np.mean(coverage[t]):.1f}%</td>" for t in args.thresholds)

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Normal-mask threshold sweep &mdash; {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background:#111; color:#eee; }}
  h1 {{ font-size: 20px; }}
  figure {{ margin: 0 0 28px; }}
  img {{ image-rendering: pixelated; width: 100%; max-width: 1400px; border:1px solid #333; }}
  figcaption {{ font-size: 13px; color:#aaa; margin-top:6px; }}
  table {{ border-collapse: collapse; margin: 12px 0 28px; }}
  th, td {{ border:1px solid #444; padding:6px 12px; text-align:center; }}
  code {{ color:#8cf; }}
</style>
<h1>Normal-mask threshold sweep &mdash; session {session_name}</h1>
<p>Gating uses <code>compute_contact_mask(frame, base, threshold,
blur_sigma={BLUR_SIGMA}, morph_radius={MORPH_RADIUS})</code>, base = shadow frame 0.
Lower threshold = looser (larger) mask. Production default is <code>0.05</code>.</p>
<h2>Mean mask coverage across sampled frames</h2>
<table><tr><th>metric</th>{thr_cells}</tr>
<tr><td>mean contact-mask coverage</td>{cov_cells}</tr></table>
<h2>Per-touch comparison (columns = shadow, then each threshold)</h2>
{''.join(rows_html)}
"""

    out_html = osp.join(args.outdir, f"{session_name}.html")
    with open(out_html, "w") as f:
        f.write(html)
    print("Wrote", out_html)
    print("Coverage means:",
          {f"{t:g}": round(100*np.mean(coverage[t]), 1) for t in args.thresholds})


if __name__ == "__main__":
    main()
