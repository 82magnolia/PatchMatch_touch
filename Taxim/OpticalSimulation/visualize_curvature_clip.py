"""Visualize height2laplacian's curvature encoding: old (pre-fix) per-image
min-max vs. the current per-image zero-anchored (0 -> 0.5) encoding.

Loads existing raw_height .npz files (already on disk; no re-simulation
needed) and calls simOptical.height2laplacian() directly, so this always
reflects whatever encoding is actually in production. Output is a single
self-contained HTML gallery (base64-embedded images, no external files) so
rows (touches) can be compared side by side.

Usage:
    python visualize_curvature_clip.py --out_html /path/to/gallery.html
"""
import argparse
import base64
import io
import random
import sys
from os import path as osp

import numpy as np
from PIL import Image

sys.path.append("..")
from simOptical import height2laplacian, raw_laplacian

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
REF_BASE = f"{PROJECT_ROOT}/Taxim/results/gen_contact_full_pseudo_mini"
QUERY_BASE = f"{PROJECT_ROOT}/Taxim/results/gen_contact_full_query_pseudo_mini"


def old_per_image_normalize(L):
    """The pre-fix encoding: per-image min-max stretch to uint8. L=0 lands at
    a different pixel intensity in every image depending on that image's own
    min/max skew -- the bug height2laplacian's current encoding fixes.
    """
    return (255 * (L - L.min()) / (L.max() - L.min())).astype(np.uint8)


def to_data_uri(gray_u8):
    img = Image.fromarray(gray_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def find_sample_touches(n_objects, touches_per_object, seed):
    rng = random.Random(seed)
    all_ids = list(range(1, 1001))
    obj_ids = sorted(rng.sample(all_ids, n_objects))

    samples = []  # (label, height_npz_path)
    for obj_id in obj_ids:
        for base_name, base_dir in [("ref", REF_BASE), ("query", QUERY_BASE)]:
            touch_ids = rng.sample(range(8), min(touches_per_object, 8))
            for t in touch_ids:
                path = f"{base_dir}/{obj_id}/{t}_scale100_height.npz"
                if osp.exists(path):
                    samples.append((f"obj {obj_id} / {base_name} #{t}", path))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=8)
    parser.add_argument("--touches_per_object", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_html", type=str, required=True)
    args = parser.parse_args()

    samples = find_sample_touches(args.n_objects, args.touches_per_object, args.seed)
    print(f"Visualizing {len(samples)} sample touches")

    rows_html = []
    for label, path in samples:
        H = np.load(path)["height"]
        L = raw_laplacian(H)
        abs_max = float(np.abs(L).max())

        cells = [
            ("old (per-image min/max, pre-fix)", to_data_uri(old_per_image_normalize(L))),
            ("current (height2laplacian, 0→0.5)", to_data_uri(height2laplacian(H))),
        ]

        cell_html = "".join(
            f'''<div class="cell">
                  <div class="cell-label">{name}</div>
                  <img src="{uri}" />
                </div>'''
            for name, uri in cells
        )
        rows_html.append(f'''
          <div class="row">
            <div class="row-label">{label}<br><span class="muted">raw |L| max = {abs_max:.1f}</span></div>
            <div class="row-cells">{cell_html}</div>
          </div>''')

    html = f'''<title>Curvature encoding: old vs current</title>
<style>
:root {{
  --surface-1: #fcfcfb; --page-plane: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #898781; --border: rgba(11,11,11,0.10);
  --gridline: #e1e0d9;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --border: rgba(255,255,255,0.10);
    --gridline: #2c2c2a;
  }}
}}
:root[data-theme="dark"] {{
  --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --text-muted: #898781; --border: rgba(255,255,255,0.10);
  --gridline: #2c2c2a;
}}
:root[data-theme="light"] {{
  --surface-1: #fcfcfb; --page-plane: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #898781; --border: rgba(11,11,11,0.10);
  --gridline: #e1e0d9;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background: var(--page-plane); color: var(--text-primary);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 900px; margin: 0 auto; padding: 32px 20px 80px; }}
h1 {{ font-size: 20px; margin: 0 0 6px; }}
p.sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 24px; line-height: 1.5; }}
.row {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
       padding: 14px 16px; margin-bottom: 16px; }}
.row-label {{ font-size: 13px; font-weight: 600; margin-bottom: 10px; }}
.row-label .muted {{ font-weight: 400; color: var(--text-muted); font-size: 12px; }}
.row-cells {{ display: flex; gap: 10px; }}
.cell {{ flex: none; width: 220px; }}
.cell img {{ width: 100%; border-radius: 6px; border: 1px solid var(--gridline); display: block; }}
.cell-label {{ font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;
               text-align: center; font-variant-numeric: tabular-nums; }}
</style>
<div class="wrap">
  <h1>Curvature encoding: old vs current</h1>
  <p class="sub">
    Each row is one touch's Laplacian curvature field, rendered at the old (pre-fix) per-image
    min-max encoding and the current simOptical.height2laplacian() encoding (per-image, zero
    anchored to pixel 128).
  </p>
  {"".join(rows_html)}
</div>
'''
    with open(args.out_html, "w") as f:
        f.write(html)
    print(f"Wrote gallery to {args.out_html}")


if __name__ == "__main__":
    main()
