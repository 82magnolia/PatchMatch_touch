"""Builds an HTML report for compare_matcher_scale.py's results.pkl:
per-case side-by-side query/reference/warped images, plus an interactive
alpha-blend overlay (a slider crossfades the warped reference over the
query image) for visually judging alignment quality.

Example usage:
    python test_scripts/make_warp_overlay_report.py \
        log/output_photometric_sweep/matcher_scale_sweep
"""

import html
import os
import pickle
import sys
from os import path as osp

METRICS = ["MSE", "PSNR", "SSIM", "LPIPS"]
METRIC_DIRECTIONS = {"MSE": "min", "PSNR": "max", "SSIM": "max", "LPIPS": "min"}

# dataviz skill's validated categorical palette (first 5 slots) used as fixed
# matcher identity accents.
MATCHER_COLORS = {
    "sift_lightglue":        ("#2a78d6", "#3987e5"),  # blue
    "disk_lightglue":        ("#008300", "#008300"),  # green
    "superpoint_lightglue":  ("#e87ba4", "#d55181"),  # magenta
    "superpoint_superglue":  ("#eda100", "#c98500"),  # yellow
    "loftr":                 ("#1baf7a", "#199e70"),  # aqua
    "dinov3":                ("#eb6834", "#d95926"),  # orange
}


def rel(path, out_dir):
    return osp.relpath(path, out_dir) if path and osp.exists(path) else None


def metrics_str(r):
    if r["status"] != "ok":
        return html.escape(r["reason"])
    return (f"MSE {r['MSE']:.5f} · PSNR {r['PSNR']:.2f} · "
           f"SSIM {r['SSIM']:.4f} · LPIPS {r['LPIPS']:.4f}")


def render_case(row, out_dir, idx):
    color = MATCHER_COLORS.get(row["matcher"], ("#898781", "#898781"))[0]
    title = (f"{html.escape(row['matcher'])} / scale={row['scale']:g} "
            f"(session {html.escape(str(row['session']))}, "
            f"query {row['query_idx']}&rarr;{row['ref_idx']})")

    if row["status"] != "ok" or not row["warped_png"]:
        return f"""
        <h3><span class="swatch" style="background:{color}"></span>{title}</h3>
        <p class="failed-cell">{html.escape(row['reason'])}</p>
        """

    query_src = rel(row["query_png"], out_dir)
    ref_src = rel(row["ref_png"], out_dir)
    warped_src = rel(row["warped_png"], out_dir)
    slider_id = f"slider_{idx}"

    return f"""
    <h3><span class="swatch" style="background:{color}"></span>{title}</h3>
    <div class="grid grid-3">
      <div class="card">
        <div class="card-title">Query (ground truth)</div>
        <img class="card-img" src="{query_src}">
      </div>
      <div class="card">
        <div class="card-title">Reference (source)</div>
        <img class="card-img" src="{ref_src}">
      </div>
      <div class="card">
        <div class="card-title">Warped reference</div>
        <img class="card-img" src="{warped_src}">
        <div class="card-metrics">{metrics_str(row)}</div>
      </div>
    </div>
    <div class="overlay-wrap">
      <div class="overlay-title">Alignment overlay -- drag to blend query &harr; warped reference</div>
      <div class="overlay-stage">
        <img class="overlay-img overlay-base" src="{query_src}">
        <img class="overlay-img overlay-top" id="{slider_id}_img" src="{warped_src}" style="opacity:0.5">
      </div>
      <input type="range" min="0" max="100" value="50" class="overlay-slider"
             id="{slider_id}"
             oninput="document.getElementById('{slider_id}_img').style.opacity = this.value / 100;
                      document.getElementById('{slider_id}_label').textContent = this.value + '% warped';">
      <div class="overlay-label" id="{slider_id}_label">50% warped</div>
    </div>
    """


CSS = """
:root { color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; } }
:root[data-theme="dark"] { color-scheme: dark; }

body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #f9f9f7; color: #0b0b0b;
  margin: 0; padding: 24px 32px 64px;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) body { background: #0d0d0d; color: #ffffff; } }
:root[data-theme="dark"] body { background: #0d0d0d; color: #ffffff; }

h1 { font-size: 1.5rem; margin-bottom: 4px; }
h3 { font-size: 0.95rem; margin-top: 36px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px;
     display: flex; align-items: center; gap: 6px; }
:root[data-theme="dark"] h3 { border-bottom-color: #2c2c2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) h3 { border-bottom-color: #2c2c2a; } }
.subtitle { color: #52514e; margin-top: 0; }
:root[data-theme="dark"] .subtitle { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .subtitle { color: #c3c2b7; } }

.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.failed-cell { color: #d03b3b; font-style: italic; }

.grid { display: grid; gap: 12px; margin-bottom: 10px; }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.card { border: 2px solid #c3c2b7; border-radius: 8px; overflow: hidden; background: #fcfcfb; }
:root[data-theme="dark"] .card { background: #1a1a19; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card { background: #1a1a19; } }
.card-title { font-size: 0.8rem; padding: 6px 8px; font-weight: 600; }
.card-img { width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: contain; background: #0d0d0d; }
.card-metrics { font-size: 0.72rem; padding: 4px 8px 8px; color: #52514e; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .card-metrics { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card-metrics { color: #c3c2b7; } }

.overlay-wrap { max-width: 420px; border: 1px solid #c3c2b7; border-radius: 8px; padding: 10px; margin-bottom: 8px; }
:root[data-theme="dark"] .overlay-wrap { border-color: #383835; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .overlay-wrap { border-color: #383835; } }
.overlay-title { font-size: 0.75rem; color: #52514e; margin-bottom: 6px; }
:root[data-theme="dark"] .overlay-title { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .overlay-title { color: #c3c2b7; } }
.overlay-stage { position: relative; aspect-ratio: 1 / 1; background: #0d0d0d; border-radius: 4px; overflow: hidden; }
.overlay-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
.overlay-slider { width: 100%; margin-top: 8px; }
.overlay-label { font-size: 0.72rem; text-align: center; color: #52514e; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .overlay-label { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .overlay-label { color: #c3c2b7; } }
"""


def write_report(rows, out_dir):
    legend = "".join(
        f'<span style="margin-right:16px"><span class="swatch" style="background:{c[0]}"></span> {m}</span>'
        for m, c in MATCHER_COLORS.items() if any(r["matcher"] == m for r in rows)
    )
    cases_html = "".join(render_case(r, out_dir, i) for i, r in enumerate(rows))

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    body = f"""
    <h1>Matcher &times; Scale Warp Comparison (no refinement)</h1>
    <p class="subtitle">{len(rows)} touch-location pairs, {n_ok} succeeded. Each pair uses a
    distinct (matcher, scale) combination and a distinct session, modality=curvature,
    transform_type=rbf_homography, --photometric_refine disabled.</p>
    <div>{legend}</div>
    {cases_html}
    """
    out_html = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>Matcher x Scale Warp Comparison</title>"
               f"<style>{CSS}</style></head><body>{body}</body></html>")
    report_path = osp.join(out_dir, "warp_overlay_report.html")
    with open(report_path, "w") as f:
        f.write(out_html)
    return report_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "log/output_photometric_sweep/matcher_scale_sweep"
    with open(osp.join(out_dir, "results.pkl"), "rb") as f:
        data = pickle.load(f)
    path = write_report(data["rows"], out_dir)
    print(f"Report written to: {path}")
