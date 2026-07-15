"""Builds an HTML report embedding the actual transferred tactile videos,
comparing a --photometric_refine transfer_pipeline.py run against a baseline
(no-refine) run, side by side.

Expects two directory trees produced by transfer_pipeline.py
(--transfer_backend dinov3_feat_match), one per session under each of
--baseline_dir/--refined_dir:
    <dir>/<session>/transfer/{query_idx}_query_shadow.mp4
    <dir>/<session>/transfer/{query_idx}_ref_shadow.mp4
    <dir>/<session>/transfer/{query_idx}_transferred.mp4
    <dir>/<session>/transfer/metrics.pkl   (if run without --skip_eval)

Sessions/queries are auto-discovered from --baseline_dir's *_transferred.mp4
files; --refined_dir is expected to have run the same sessions.

Example usage (after running both variants, e.g. via
train_refine_scripts/transfer_all_real_data_gt_retrieval/run_refined_*.sh
and its non-refined counterpart, or the smaller manual runs used to spot-check):
    python test_scripts/compare_refine_videos.py \
        --baseline_dir log/output_photometric_sweep/video_demo/baseline \
        --refined_dir  log/output_photometric_sweep/video_demo/refined \
        --out_dir      log/output_photometric_sweep/video_demo
"""

import argparse
import html
import os
import pickle
import re
import subprocess
from os import path as osp

TRANSFERRED_RE = re.compile(r"^(\d+)_transferred\.mp4$")


def transcode_for_web(src_path, out_dir, web_name):
    """Re-encode src_path to H.264/yuv420p under out_dir/_web/<web_name>.mp4.

    main_retrieval_transfer_feat_match.py's write_video uses OpenCV's "mp4v"
    fourcc, which produces MPEG-4 Part 2 video in an .mp4 container -- a
    valid file, but a codec essentially no browser's <video> tag can decode,
    so it silently fails to play despite the src resolving fine. Skips
    re-encoding if the target already exists (idempotent across report
    regenerations). Falls back to the original path if ffmpeg fails/is
    missing, so the report degrades to "doesn't play" rather than crashing.
    """
    web_dir = osp.join(out_dir, "_web")
    os.makedirs(web_dir, exist_ok=True)
    web_path = osp.join(web_dir, f"{web_name}.mp4")
    if osp.exists(web_path):
        return web_path
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "23", web_path],
            check=True,
        )
        return web_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [transcode] Failed for {src_path} ({e}); falling back to original (may not play in-browser).")
        return src_path


def discover_cases(baseline_dir):
    """(session, query_idx) pairs discovered from baseline_dir's *_transferred.mp4 files."""
    cases = []
    if not osp.isdir(baseline_dir):
        return cases
    for session in sorted((s for s in os.listdir(baseline_dir)), key=lambda s: (len(s), s)):
        transfer_dir = osp.join(baseline_dir, session, "transfer")
        if not osp.isdir(transfer_dir):
            continue
        for fname in sorted(os.listdir(transfer_dir)):
            m = TRANSFERRED_RE.match(fname)
            if m:
                cases.append((session, int(m.group(1))))
    return cases


def load_metrics(dir_, session, query_idx):
    metrics_path = osp.join(dir_, session, "transfer", "metrics.pkl")
    if not osp.exists(metrics_path):
        return None
    with open(metrics_path, "rb") as f:
        data = pickle.load(f)
    return data.get("per_touch", {}).get(query_idx)


def rel(path, out_dir):
    return osp.relpath(path, out_dir) if path and osp.exists(path) else None


def metrics_str(m):
    if not m:
        return ""
    return (f"MSE {m['MSE']:.5f} · PSNR {m['PSNR']:.2f} · "
           f"SSIM {m['SSIM']:.4f} · LPIPS {m['LPIPS']:.4f}")


def video_card(title, video_path, out_dir, web_name, caption=""):
    if not video_path or not osp.exists(video_path):
        return (f'<div class="card failed"><div class="card-title">{html.escape(title)}</div>'
               f'<div class="card-body failed-body">missing</div></div>')
    web_path = transcode_for_web(video_path, out_dir, web_name)
    src = rel(web_path, out_dir)
    return (f'<div class="card"><div class="card-title">{html.escape(title)}</div>'
           f'<video class="card-video" src="{src}" controls muted loop playsinline></video>'
           f'<div class="card-metrics">{caption}</div></div>')


def render_case(session, query_idx, baseline_dir, refined_dir, out_dir):
    b_transfer = osp.join(baseline_dir, session, "transfer")
    r_transfer = osp.join(refined_dir, session, "transfer")

    query_video = osp.join(b_transfer, f"{query_idx}_query_shadow.mp4")
    ref_video = osp.join(b_transfer, f"{query_idx}_ref_shadow.mp4")
    baseline_video = osp.join(b_transfer, f"{query_idx}_transferred.mp4")
    refined_video = osp.join(r_transfer, f"{query_idx}_transferred.mp4")

    baseline_m = load_metrics(baseline_dir, session, query_idx)
    refined_m = load_metrics(refined_dir, session, query_idx)

    prefix = f"session{session}_q{query_idx}"
    cards = "".join([
        video_card("Query (ground truth)", query_video, out_dir, f"{prefix}_query"),
        video_card("Reference (source)", ref_video, out_dir, f"{prefix}_ref"),
        video_card("Transferred (no refine)", baseline_video, out_dir, f"{prefix}_baseline", metrics_str(baseline_m)),
        video_card("Transferred (refined)", refined_video, out_dir, f"{prefix}_refined", metrics_str(refined_m)),
    ])
    return f"""
    <h3>Session {html.escape(session)}, Query {query_idx}</h3>
    <div class="grid grid-4">{cards}</div>
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
h3 { font-size: 1rem; margin-top: 32px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px; }
:root[data-theme="dark"] h3 { border-bottom-color: #2c2c2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) h3 { border-bottom-color: #2c2c2a; } }
.subtitle { color: #52514e; margin-top: 0; }
:root[data-theme="dark"] .subtitle { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .subtitle { color: #c3c2b7; } }

.grid { display: grid; gap: 12px; }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.card { border: 2px solid #c3c2b7; border-radius: 8px; overflow: hidden; background: #fcfcfb; }
:root[data-theme="dark"] .card { background: #1a1a19; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card { background: #1a1a19; } }
.card-title { font-size: 0.8rem; padding: 6px 8px; font-weight: 600; }
.card-video { width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: contain; background: #0d0d0d; }
.card-metrics { font-size: 0.72rem; padding: 4px 8px 8px; color: #52514e; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .card-metrics { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card-metrics { color: #c3c2b7; } }
.card-body.failed-body { padding: 24px 8px; text-align: center; color: #d03b3b; }
"""


def write_report(cases, baseline_dir, refined_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    case_sections = "".join(
        render_case(session, query_idx, baseline_dir, refined_dir, out_dir)
        for session, query_idx in cases
    )
    body = f"""
    <h1>Tactile Video Transfer: baseline vs. photometric-refined</h1>
    <p class="subtitle">{len(cases)} query/session pairs. Each row: query touch video (ground
    truth), reference touch video (source), transferred without refinement, transferred with
    --photometric_refine. baseline_dir={html.escape(baseline_dir)},
    refined_dir={html.escape(refined_dir)}.</p>
    {case_sections}
    """
    out_html = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>Tactile Video Transfer Comparison</title>"
               f"<style>{CSS}</style></head><body>{body}</body></html>")
    report_path = osp.join(out_dir, "video_report.html")
    with open(report_path, "w") as f:
        f.write(out_html)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an HTML report comparing transferred tactile videos with/without "
                    "photometric refinement.")
    parser.add_argument("--baseline_dir", required=True, type=str)
    parser.add_argument("--refined_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cases = discover_cases(args.baseline_dir)
    print(f"Discovered {len(cases)} query/session pairs from {args.baseline_dir}")
    report_path = write_report(cases, args.baseline_dir, args.refined_dir, args.out_dir)
    print(f"Report written to: {report_path}")
