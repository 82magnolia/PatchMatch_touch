"""Builds an HTML report for the matcher x scale video sweep, but instead of
embedding 3 separate <video> elements per case (query, reference,
transferred), spatially concatenates them into ONE synced video per case via
ffmpeg's hstack filter (query | reference | transferred, left to right) --
one player to scrub instead of juggling three.

Expects <video_root>/<case_id>/transfer/{query_idx}_{query,ref}_shadow.mp4
and {query_idx}_transferred.mp4, as produced by transfer_pipeline.py
(--transfer_backend dinov3_feat_match).

Example usage:
    python test_scripts/make_concat_video_report.py \
        log/output_photometric_sweep/matcher_scale_sweep
"""

import html
import os
import pickle
import subprocess
import sys
from os import path as osp

MATCHER_COLORS = {
    "sift_lightglue":        ("#2a78d6", "#3987e5"),
    "disk_lightglue":        ("#008300", "#008300"),
    "superpoint_lightglue":  ("#e87ba4", "#d55181"),
    "superpoint_superglue":  ("#eda100", "#c98500"),
    "loftr":                 ("#1baf7a", "#199e70"),
    "dinov3":                ("#eb6834", "#d95926"),
}


def rel(path, out_dir):
    return osp.relpath(path, out_dir) if path and osp.exists(path) else None


def concat_hstack(query_path, ref_path, transferred_path, out_dir, case_id):
    """Horizontally stack query | reference | transferred into one H.264 mp4
    under out_dir/_concat/<case_id>.mp4. Skips if already built."""
    concat_dir = osp.join(out_dir, "_concat")
    os.makedirs(concat_dir, exist_ok=True)
    out_path = osp.join(concat_dir, f"{case_id}.mp4")
    if osp.exists(out_path):
        return out_path
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", query_path, "-i", ref_path, "-i", transferred_path,
             "-filter_complex", "hstack=inputs=3",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "23", out_path],
            check=True,
        )
        return out_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [concat] Failed for {case_id} ({e}).")
        return None


def metrics_str(m):
    if not m:
        return ""
    return (f"MSE {m['MSE']:.5f} · PSNR {m['PSNR']:.2f} · "
           f"SSIM {m['SSIM']:.4f} · LPIPS {m['LPIPS']:.4f}")


def load_metrics(video_root, case_id, query_idx):
    p = osp.join(video_root, case_id, "transfer", "metrics.pkl")
    if not osp.exists(p):
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data.get("per_touch", {}).get(query_idx)


def render_case(row, video_root, out_dir):
    case_id = row["case_id"]
    query_idx, ref_idx = row["query_idx"], row["ref_idx"]
    color = MATCHER_COLORS.get(row["matcher"], ("#898781", "#898781"))[0]
    title = (f"{html.escape(row['matcher'])} / scale={row['scale']:g} "
            f"(session {html.escape(str(row['session']))}, "
            f"query {query_idx}&rarr;{ref_idx})")

    transfer_dir = osp.join(video_root, case_id, "transfer")
    query_video = osp.join(transfer_dir, f"{query_idx}_query_shadow.mp4")
    ref_video = osp.join(transfer_dir, f"{query_idx}_ref_shadow.mp4")
    transferred_video = osp.join(transfer_dir, f"{query_idx}_transferred.mp4")

    if not all(osp.exists(p) for p in (query_video, ref_video, transferred_video)):
        return f"""
        <h3><span class="swatch" style="background:{color}"></span>{title}</h3>
        <p class="failed-cell">missing video(s) under {html.escape(transfer_dir)}</p>
        """

    concat_path = concat_hstack(query_video, ref_video, transferred_video, out_dir, case_id)
    m = load_metrics(video_root, case_id, query_idx)
    if not concat_path:
        return f"""
        <h3><span class="swatch" style="background:{color}"></span>{title}</h3>
        <p class="failed-cell">ffmpeg concat failed</p>
        """

    src = rel(concat_path, out_dir)
    return f"""
    <h3><span class="swatch" style="background:{color}"></span>{title}</h3>
    <div class="concat-wrap">
      <video class="concat-video" src="{src}" controls muted loop playsinline></video>
      <div class="concat-caption">Query (ground truth) &nbsp;|&nbsp; Reference (source) &nbsp;|&nbsp; Transferred
      &mdash; {metrics_str(m)}</div>
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

.concat-wrap { max-width: 900px; }
.concat-video { width: 100%; display: block; background: #0d0d0d; border-radius: 6px; }
.concat-caption { font-size: 0.75rem; color: #52514e; padding: 6px 2px; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .concat-caption { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .concat-caption { color: #c3c2b7; } }
"""


def write_report(rows, video_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    legend = "".join(
        f'<span style="margin-right:16px"><span class="swatch" style="background:{c[0]}"></span> {m}</span>'
        for m, c in MATCHER_COLORS.items() if any(r["matcher"] == m for r in rows)
    )
    cases_html = "".join(render_case(r, video_root, out_dir) for r in rows)
    body = f"""
    <h1>Matcher &times; Scale Tactile Video Comparison (no refinement)</h1>
    <p class="subtitle">{len(rows)} touch-location pairs. Each video is query | reference |
    transferred concatenated side by side into a single clip (ffmpeg hstack) for easier
    scrubbing/comparison.</p>
    <div>{legend}</div>
    {cases_html}
    """
    out_html = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>Matcher x Scale Video Comparison</title>"
               f"<style>{CSS}</style></head><body>{body}</body></html>")
    report_path = osp.join(out_dir, "concat_video_report.html")
    with open(report_path, "w") as f:
        f.write(out_html)
    return report_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "log/output_photometric_sweep/matcher_scale_sweep"
    video_root = osp.join(out_dir, "videos")
    with open(osp.join(out_dir, "results.pkl"), "rb") as f:
        data = pickle.load(f)
    path = write_report(data["rows"], video_root, out_dir)
    print(f"Report written to: {path}")
