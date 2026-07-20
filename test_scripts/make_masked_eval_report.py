"""Builds an HTML report comparing ReBotNet eval runs on masked
(render-mask-blended, via postprocess_mask_transfer.py) transfer data across
matchers -- e.g. the two ReBotNet S / masked runs for superpoint_superglue
and disk_lightglue.

Expects each run dir to be an eval_rebot_pseudo_mini/eval_rebot_*.sh
--save_dir, i.e. containing metrics.pkl (written by rebot_net/eval.py) and
videos/{obj_id}_{contact}_grid.mp4 (--video_save's 2x2 grid: Reference |
Query(GT) / Transferred | Refined).

The grid videos are written by cv2.VideoWriter with the mpeg4/mp4v fourcc,
which is not browser-playable, so representative videos are re-encoded to
H.264 on the fly (skipped if already done) into <out_dir>/_h264/.

Example usage:
    python test_scripts/make_masked_eval_report.py \
        --run superpoint_superglue log/rebot_eval_S_pseudo_mini_superpoint_superglue_masked \
        --run disk_lightglue log/rebot_eval_S_pseudo_mini_disk_lightglue_masked \
        --out_html log/rebot_eval_S_pseudo_mini_masked_report.html
"""

import argparse
import html
import os
import pickle
import subprocess
from os import path as osp

METRIC_DIRECTIONS = {"MSE": "min", "PSNR": "max", "SSIM": "max", "LPIPS": "min"}
METRICS = list(METRIC_DIRECTIONS.keys())

# dataviz skill's validated categorical palette, keyed by matcher name --
# reused verbatim from test_scripts/make_concat_video_report.py so colors
# stay consistent across reports.
MATCHER_COLORS = {
    "sift_lightglue":        ("#2a78d6", "#3987e5"),
    "disk_lightglue":        ("#008300", "#008300"),
    "superpoint_lightglue":  ("#e87ba4", "#d55181"),
    "superpoint_superglue":  ("#eda100", "#c98500"),
    "loftr":                 ("#1baf7a", "#199e70"),
    "dinov3":                ("#eb6834", "#d95926"),
}
N_REPRESENTATIVE = 3


def load_run(label, out_dir):
    with open(osp.join(out_dir, "metrics.pkl"), "rb") as f:
        data = pickle.load(f)
    return {"label": label, "dir": out_dir, "average": data["average"],
            "per_object": data["per_object"]}


def transcode_h264(src, dst):
    if osp.exists(dst):
        return dst
    os.makedirs(osp.dirname(dst), exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "23", dst],
            check=True,
        )
        return dst
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [transcode] Failed for {src} ({e}).")
        return None


def rel(path, out_dir):
    return osp.relpath(path, out_dir) if path and osp.exists(path) else None


def render_summary_table(runs):
    rows_html = []
    for r in runs:
        color = MATCHER_COLORS.get(r["label"], ("#898781", "#898781"))[0]
        cells = "".join(f"<td>{r['average'][m]:.5f}</td>" for m in METRICS)
        rows_html.append(
            f'<tr><td><span class="swatch" style="background:{color}"></span>'
            f'{html.escape(r["label"])}</td>{cells}</tr>'
        )
    header = "".join(f"<th>{m}</th>" for m in METRICS)
    return f"""
    <table class="summary-table">
      <thead><tr><th>run</th>{header}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def render_object_table(runs):
    object_ids = sorted(runs[0]["per_object"].keys())
    header_cells = "".join(
        f'<th colspan="{len(METRICS)}">{html.escape(r["label"])}</th>' for r in runs
    )
    sub_header = "".join(f"<th>{m}</th>" for _ in runs for m in METRICS)
    rows_html = []
    for obj_id in object_ids:
        best = {}
        for m in METRICS:
            reverse = METRIC_DIRECTIONS[m] == "max"
            vals = [(i, r["per_object"][obj_id][m]) for i, r in enumerate(runs)
                    if obj_id in r["per_object"]]
            if vals:
                best[m] = sorted(vals, key=lambda v: v[1], reverse=reverse)[0][0]
        cells = []
        for i, r in enumerate(runs):
            po = r["per_object"].get(obj_id)
            if po is None:
                cells.append(f'<td colspan="{len(METRICS)}">&mdash;</td>')
                continue
            for m in METRICS:
                cls = "best-cell" if best.get(m) == i else ""
                cells.append(f'<td class="{cls}">{po[m]:.5f}</td>')
        rows_html.append(f"<tr><td>{obj_id}</td>{''.join(cells)}</tr>")
    return f"""
    <table class="object-table">
      <thead>
        <tr><th rowspan="2">object</th>{header_cells}</tr>
        <tr>{sub_header}</tr>
      </thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def pick_representative_objects(runs):
    """Best / median / worst object by mean PSNR averaged across runs."""
    object_ids = sorted(runs[0]["per_object"].keys())
    scored = []
    for obj_id in object_ids:
        psnrs = [r["per_object"][obj_id]["PSNR"] for r in runs if obj_id in r["per_object"]]
        if psnrs:
            scored.append((obj_id, sum(psnrs) / len(psnrs)))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []
    n = len(scored)
    picks = [("best", scored[0]), ("median", scored[n // 2]), ("worst", scored[-1])]
    seen = set()
    out = []
    for tag, (obj_id, mean_psnr) in picks:
        if obj_id in seen:
            continue
        seen.add(obj_id)
        out.append((tag, obj_id, mean_psnr))
    return out


def render_video_section(runs, obj_id, tag, mean_psnr, out_dir, contact=0):
    cards = []
    for r in runs:
        color = MATCHER_COLORS.get(r["label"], ("#898781", "#898781"))[0]
        src_path = osp.join(r["dir"], "videos", f"{obj_id}_{contact}_grid.mp4")
        if not osp.exists(src_path):
            cards.append(
                f'<div class="card failed"><div class="card-title">{html.escape(r["label"])}</div>'
                f'<div class="card-body failed-body">missing {html.escape(src_path)}</div></div>'
            )
            continue
        dst_path = osp.join(r["dir"], "_h264", f"{obj_id}_{contact}_grid.mp4")
        transcoded = transcode_h264(src_path, dst_path)
        po = r["per_object"].get(obj_id, {})
        metrics_str = (f"MSE {po['MSE']:.5f} · PSNR {po['PSNR']:.2f} · "
                       f"SSIM {po['SSIM']:.4f} · LPIPS {po['LPIPS']:.4f}") if po else ""
        if not transcoded:
            cards.append(
                f'<div class="card failed"><div class="card-title">{html.escape(r["label"])}</div>'
                f'<div class="card-body failed-body">ffmpeg transcode failed</div></div>'
            )
            continue
        cards.append(f"""
        <div class="card" style="border-color:{color}">
          <div class="card-title"><span class="swatch" style="background:{color}"></span>{html.escape(r["label"])}</div>
          <video class="card-video" src="{rel(transcoded, out_dir)}" controls muted loop playsinline></video>
          <div class="card-metrics">{metrics_str}</div>
        </div>
        """)
    return f"""
    <h3>{tag.capitalize()}: object {obj_id} (mean PSNR {mean_psnr:.2f}), contact {contact}</h3>
    <p class="subtitle">Grid layout per video: Reference | Query (ground truth) — top row;
    Transferred | Refined (network output) — bottom row.</p>
    <div class="video-grid">{"".join(cards)}</div>
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
h2 { font-size: 1.15rem; margin-top: 40px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px; }
h3 { font-size: 0.95rem; margin-top: 32px; }
:root[data-theme="dark"] h2 { border-bottom-color: #2c2c2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) h2 { border-bottom-color: #2c2c2a; } }
.subtitle { color: #52514e; margin-top: 0; }
:root[data-theme="dark"] .subtitle { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .subtitle { color: #c3c2b7; } }

table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: #fcfcfb; }
:root[data-theme="dark"] table { background: #1a1a19; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) table { background: #1a1a19; } }

th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e1e0d9; font-variant-numeric: tabular-nums; font-size: 0.9rem; }
:root[data-theme="dark"] th, :root[data-theme="dark"] td { border-bottom-color: #2c2c2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) th, :root:not([data-theme="light"]) td { border-bottom-color: #2c2c2a; } }
th { color: #52514e; font-weight: 600; }
:root[data-theme="dark"] th { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) th { color: #c3c2b7; } }

.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.best-cell { background: rgba(12,163,12,0.12); font-weight: 700; }
.object-table { max-height: 480px; display: block; overflow-y: auto; }

.video-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.card { border: 2px solid #c3c2b7; border-radius: 8px; overflow: hidden; background: #fcfcfb; }
:root[data-theme="dark"] .card { background: #1a1a19; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card { background: #1a1a19; } }
.card-title { font-size: 0.85rem; padding: 6px 8px; font-weight: 600; display: flex; align-items: center; }
.card-video { width: 100%; display: block; background: #0d0d0d; }
.card-metrics { font-size: 0.75rem; padding: 4px 8px 8px; color: #52514e; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .card-metrics { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card-metrics { color: #c3c2b7; } }
.card-body.failed-body { padding: 24px 8px; text-align: center; color: #d03b3b; font-size: 0.8rem; }
"""


def write_report(runs, out_html_path):
    out_dir = osp.dirname(out_html_path)
    representative = pick_representative_objects(runs)
    video_sections = "".join(
        render_video_section(runs, obj_id, tag, mean_psnr, out_dir)
        for tag, obj_id, mean_psnr in representative
    )
    body = f"""
    <h1>ReBotNet S &mdash; Masked Transfer Eval Comparison</h1>
    <p class="subtitle">Evaluated on log/transfer_feat_match_pseudo_mini_&lt;matcher&gt;_masked
    (render-mask-blended via postprocess_mask_transfer.py), 50 held-out test objects,
    8 contacts each.</p>

    <h2>Average metrics</h2>
    {render_summary_table(runs)}

    <h2>Representative videos</h2>
    {video_sections}

    <h2>Per-object metrics (all 50 objects)</h2>
    {render_object_table(runs)}
    """
    out_html = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>ReBotNet S Masked Transfer Eval Comparison</title>"
               f"<style>{CSS}</style></head><body>{body}</body></html>")
    with open(out_html_path, "w") as f:
        f.write(out_html)
    return out_html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"),
                        required=True, help="Repeatable: a matcher label and its eval --save_dir.")
    parser.add_argument("--out_html", required=True)
    args = parser.parse_args()

    runs = [load_run(label, out_dir) for label, out_dir in args.run]
    path = write_report(runs, args.out_html)
    print(f"Report written to: {path}")
