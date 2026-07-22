"""Render tactile normal videos with Poisson contact-boundary blending, as
side-by-side [shadow | hard | poisson] mp4s, and an HTML index to view them.

Reuses the band/Poisson helpers from compare_boundary_blend.py. Runs the full
resampled sequence (== the shadow video) per touch.

Usage:
  conda activate pm_real
  python test_scripts/make_poisson_normal_videos.py --session ../log/real_data_gt_retrieval/1
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

from _gelsight_processing import (  # noqa: E402
    compute_contact_mask, normals_to_colormap, write_video, VIDEO_FPS)
from _tactile_normal_net import (  # noqa: E402
    load_normal_net, frame_to_normals, unit_normals, poisson_blend_normals)
from process_single_shot import NORMAL_CONTACT_THRESHOLD  # noqa: E402
from compare_boundary_blend import read_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--net", default=osp.join(
        osp.dirname(osp.abspath(__file__)), "..", "real_data_transfer",
        "gsnormal_models", "nnmini.pt"))
    ap.add_argument("--touches", nargs="*", default=None)
    ap.add_argument("--band_px", type=int, default=8)
    ap.add_argument("--outdir", default="log/boundary_blend_video")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_normal_net(args.net, device)

    all_touches = [osp.basename(p).split("_")[0]
                   for p in sorted(glob.glob(osp.join(args.session, "*_shadow.mp4")))]
    touches = args.touches or all_touches[:4]
    session_name = osp.basename(osp.normpath(args.session))
    out_dir = osp.join(args.outdir, session_name)
    os.makedirs(out_dir, exist_ok=True)

    vids_html = []
    max_unit_err = 0.0

    for ti in touches:
        shadow = read_video(osp.join(args.session, f"{ti}_shadow.mp4"))
        if len(shadow) < 2:
            continue
        base = shadow[0].astype(np.float32) / 255.0
        sbs_frames = []
        for frame in shadow:
            cmask = compute_contact_mask(
                frame.astype(np.float32) / 255.0, base,
                threshold=NORMAL_CONTACT_THRESHOLD)[..., 0] > 0.5

            hard = frame_to_normals(frame, net, device, contact_mask=cmask)
            if cmask.any():
                pnx, pny = poisson_blend_normals(
                    hard[..., 0], hard[..., 1], cmask, args.band_px)
                poisson = unit_normals(pnx, pny)
            else:
                poisson = hard
            max_unit_err = max(max_unit_err, float(
                np.abs(np.linalg.norm(poisson, axis=-1) - 1.0).max()))

            sbs_frames.append(np.hstack([
                frame, normals_to_colormap(hard), normals_to_colormap(poisson)]))

        fn = f"{ti}_poisson_sbs.mp4"
        write_video(osp.join(out_dir, fn), sbs_frames, VIDEO_FPS)
        vids_html.append(
            f'<figure><video src="{session_name}/{fn}" controls loop muted '
            f'playsinline width="960"></video>'
            f'<figcaption>touch {ti} &mdash; shadow | hard | poisson (band '
            f'{args.band_px})</figcaption></figure>')
        print(f"  touch {ti}: wrote {fn} ({len(sbs_frames)} frames)")

    ok = "PASS" if max_unit_err < 1e-4 else "FAIL"
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Poisson boundary-blend videos &mdash; {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background:#111; color:#eee; }}
  h1 {{ font-size: 20px; }}
  figure {{ margin: 0 0 26px; }}
  video {{ width: 100%; max-width: 960px; border:1px solid #333; background:#000; }}
  figcaption {{ font-size: 13px; color:#aaa; margin-top:6px; }}
  code {{ color:#8cf; }} .ok {{ color:#7d7; }}
</style>
<h1>Poisson contact-boundary blend &mdash; session {session_name}</h1>
<p>Each video: raw shadow | hard mask boundary | Poisson band-blended
(band {args.band_px}px). Mask = <code>compute_contact_mask</code> @
{NORMAL_CONTACT_THRESHOLD}. Unit-norm: <span class="ok">{ok}</span>
(max |&#8214;n&#8214;&minus;1| = {max_unit_err:.2e}).</p>
{''.join(vids_html)}
"""
    out_html = osp.join(args.outdir, f"{session_name}.html")
    with open(out_html, "w") as f:
        f.write(html)
    print("Wrote", out_html)
    print(f"unit-norm {ok}, max err {max_unit_err:.2e}")


if __name__ == "__main__":
    main()
