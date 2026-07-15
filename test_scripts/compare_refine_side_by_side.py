"""Focused rbf_homography baseline-vs-refined comparison at a fixed
(scale, modality) setting, across several sessions.

Follow-up to compare_photometric_refine.py's full sweep: that sweep found
scale=8/modality=curvature to be the best-performing (scale, modality)
setting overall, with rbf_homography + gradient-loss refinement winning
there specifically (even though gradient underperforms on average across the
whole sweep). This script drills into exactly that setting on more samples
-- pulled from several different real_data_gt_retrieval sessions rather than
deeper into one, for genuine variety -- and produces a compact side-by-side
HTML report: query | reference | warped (no refine) | warped (refined),
with MSE/PSNR/SSIM/LPIPS for both and the delta between them.

Example usage:
    python test_scripts/compare_refine_side_by_side.py \
        --sessions_base log/real_data_gt_retrieval --num_sessions 10 \
        --scale 8 --modality curvature --transform_type rbf_homography \
        --refine_loss gradient --matcher sift_lightglue
"""

import argparse
import html
import os
import pickle
import sys
from os import path as osp

import numpy as np
import torch
import lpips

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from main_retrieval_transfer_feat_match import (
    _load_query_ref_static_for_matching, reconstruct_avg, evaluate_video_metrics,
)
from dinov3.dense_match import _fit_dense_field
from test_scripts.compare_photometric_refine import (
    discover_odd_even_pairs, compute_sparse_matches, save_png, build_nnf,
)
from test_scripts.make_report import CSS, rel

METRICS = ["MSE", "PSNR", "SSIM", "LPIPS"]
METRIC_DIRECTIONS = {"MSE": "min", "PSNR": "max", "SSIM": "max", "LPIPS": "min"}


def fit_and_warp(pts_l, pts_r, r, q, transform_type, reproj_threshold, refine, refine_loss,
                 refine_iters, refine_lr, refine_huber_delta):
    h2, w2 = q.shape[:2]
    src_row, src_col, inlier_count, total = _fit_dense_field(
        pts_l, pts_r, h2, w2, transform_type, reproj_threshold,
        image_left=r, image_right=q, refine=refine, refine_loss=refine_loss,
        refine_iters=refine_iters, refine_lr=refine_lr, refine_huber_delta=refine_huber_delta)
    nnf = build_nnf(src_row, src_col, r.shape)
    return reconstruct_avg(nnf, r[..., :3], patch_size=1)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS model on {device}...")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    sessions = sorted(
        (s for s in os.listdir(args.sessions_base) if s.isdigit()), key=int
    )[:args.num_sessions]

    rows = []
    for session in sessions:
        session_dir = osp.join(args.sessions_base, session)
        pairs = discover_odd_even_pairs(session_dir, args.scale, args.modality)[:args.queries_per_session]
        for query_idx, ref_idx in pairs:
            print(f"\nSession {session}: Query {query_idx} -> Reference {ref_idx}")
            try:
                q, r = _load_query_ref_static_for_matching(
                    session_dir, session_dir, query_idx, ref_idx, [args.modality],
                    args.scale, None, None, args.matcher)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping (static image load failed): {e}")
                continue

            case_id = f"session{session}_q{query_idx}"
            c_dir = osp.join(args.out_dir, case_id)
            os.makedirs(c_dir, exist_ok=True)
            query_png = osp.join(c_dir, "query.png")
            ref_png = osp.join(c_dir, "ref.png")
            save_png(query_png, q)
            save_png(ref_png, r)

            row = {
                "session": session, "query_idx": query_idx, "ref_idx": ref_idx,
                "case_id": case_id, "query_png": query_png, "ref_png": ref_png,
                "baseline": None, "refined": None,
                "baseline_png": None, "refined_png": None,
                "baseline_status": "ok", "refined_status": "ok",
                "baseline_reason": "", "refined_reason": "",
            }

            try:
                pts_l, pts_r = compute_sparse_matches(
                    args.matcher, r, q, args.dinov3_model, args.dinov3_weights,
                    args.dinov3_num_points, args.dinov3_stratify_threshold)
            except RuntimeError as e:
                print(f"  Sparse matching failed, skipping: {e}")
                row["baseline_status"] = row["refined_status"] = "failed"
                row["baseline_reason"] = row["refined_reason"] = f"sparse matching failed: {e}"
                rows.append(row)
                continue

            for label, refine in [("baseline", False), ("refined", True)]:
                try:
                    warped = fit_and_warp(
                        pts_l, pts_r, r, q, args.transform_type, args.reproj_threshold,
                        refine, args.refine_loss, args.photometric_refine_iters,
                        args.photometric_refine_lr, args.photometric_refine_huber_delta)
                    metrics = evaluate_video_metrics([q[..., :3]], [warped], lpips_model, device)
                    row[label] = metrics
                    png_path = osp.join(c_dir, f"{label}.png")
                    save_png(png_path, warped)
                    row[f"{label}_png"] = png_path
                    print(f"  {label:9s} MSE={metrics['MSE']:.5f} PSNR={metrics['PSNR']:.2f} "
                         f"SSIM={metrics['SSIM']:.4f} LPIPS={metrics['LPIPS']:.4f}")
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                    row[f"{label}_status"] = "failed"
                    row[f"{label}_reason"] = f"{type(e).__name__}: {e}"
                    print(f"  {label:9s} FAILED: {e}")

            rows.append(row)

    results_pkl = osp.join(args.out_dir, "results.pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({"rows": rows, "args": vars(args)}, f)
    print(f"\nSaved {len(rows)} sample results to: {results_pkl}")
    return rows


def render_summary(rows, args):
    ok_rows = [r for r in rows if r["baseline_status"] == "ok" and r["refined_status"] == "ok"]
    if not ok_rows:
        return "<p>No samples succeeded for both baseline and refined.</p>", {}

    avg = {}
    for label in ("baseline", "refined"):
        avg[label] = {m: sum(r[label][m] for r in ok_rows) / len(ok_rows) for m in METRICS}

    cells = []
    for m in METRICS:
        b, ref = avg["baseline"][m], avg["refined"][m]
        better = (ref < b) if METRIC_DIRECTIONS[m] == "min" else (ref > b)
        delta = ref - b
        cls = "delta-good" if better else "delta-bad"
        sign = "+" if delta >= 0 else ""
        cells.append(f"<td>{b:.5f}</td><td>{ref:.5f}</td>"
                     f'<td class="{cls}">{sign}{delta:.5f}</td>')
    header = "".join(f"<th>{m} (baseline)</th><th>{m} (refined)</th><th>{m} &Delta;</th>" for m in METRICS)
    table = f"""
    <table class="summary-table">
      <thead><tr>{header}</tr></thead>
      <tbody><tr>{"".join(cells)}</tr></tbody>
    </table>
    """
    n_failed_refine = sum(1 for r in rows if r["refined_status"] != "ok")
    return (table + f'<p class="subtitle">Averaged over {len(ok_rows)}/{len(rows)} samples where both '
            f"baseline and refined succeeded ({n_failed_refine} sample(s) had refinement fail "
            f"outright, e.g. too-few-inliers).</p>"), avg


def render_case_row(row, out_dir):
    def metric_str(m):
        if row["baseline_status"] != "ok" or row["refined_status"] != "ok":
            return ""
        b, ref = row["baseline"][m], row["refined"][m]
        better = (ref < b) if METRIC_DIRECTIONS[m] == "min" else (ref > b)
        cls = "delta-good" if better else "delta-bad"
        return f'<span class="{cls}">{m} {b:.4f}&rarr;{ref:.4f}</span>'

    def img_card(title, png_path, metrics_line=""):
        if not png_path:
            return f'<div class="card failed"><div class="card-title">{html.escape(title)}</div><div class="card-body failed-body">failed</div></div>'
        return (f'<div class="card"><div class="card-title">{html.escape(title)}</div>'
               f'<img class="card-img" src="{rel(png_path, out_dir)}" loading="lazy">'
               f'<div class="card-metrics">{metrics_line}</div></div>')

    baseline_metrics = " · ".join(metric_str(m) for m in METRICS) if row["baseline_status"] == "ok" else html.escape(row["baseline_reason"])
    refined_metrics = " · ".join(metric_str(m) for m in METRICS) if row["refined_status"] == "ok" else html.escape(row["refined_reason"])

    cards = "".join([
        img_card("Query (ground truth)", row["query_png"]),
        img_card("Reference (source)", row["ref_png"]),
        img_card("Warped (no refine)", row["baseline_png"], baseline_metrics),
        img_card("Warped (refined)", row["refined_png"], refined_metrics),
    ])
    return f"""
    <h3>Session {row['session']}: Query {row['query_idx']} &rarr; Reference {row['ref_idx']}</h3>
    <div class="grid grid-4">{cards}</div>
    """


def write_report(rows, args):
    summary_html, avg = render_summary(rows, args)
    case_rows = "".join(render_case_row(r, args.out_dir) for r in rows)

    extra_css = """
    .grid-4 { grid-template-columns: repeat(4, 1fr); }
    .delta-good { color: #0ca30c; font-weight: 600; }
    .delta-bad { color: #d03b3b; font-weight: 600; }
    """

    body = f"""
    <h1>rbf_homography: baseline vs. {html.escape(args.refine_loss)}-refined</h1>
    <p class="subtitle">scale={args.scale:g}, modality={html.escape(args.modality)}, matcher={html.escape(args.matcher)},
    {len(rows)} samples across {len(set(r['session'] for r in rows))} sessions
    ({args.sessions_base}).</p>

    <h2>Average metrics</h2>
    {summary_html}

    {case_rows}
    """

    out_html = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>rbf_homography refine comparison</title>"
               f"<style>{CSS}{extra_css}</style></head><body>{body}</body></html>")
    report_path = osp.join(args.out_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(out_html)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare rbf_homography with/without photometric refinement at a fixed "
                    "scale/modality, across several sessions.")
    parser.add_argument("--sessions_base", default="log/real_data_gt_retrieval", type=str)
    parser.add_argument("--num_sessions", default=10, type=int)
    parser.add_argument("--queries_per_session", default=1, type=int)
    parser.add_argument("--scale", default=8.0, type=float)
    parser.add_argument("--modality", default="curvature",
                        choices=["color", "normal", "curvature", "height",
                                 "raw_normal", "raw_height", "shapeindex"])
    parser.add_argument("--transform_type", default="rbf_homography",
                        choices=["affine", "homography", "rbf_affine", "rbf_homography"])
    parser.add_argument("--refine_loss", default="gradient",
                        choices=["l1", "l2", "huber", "gradient", "ncc"])
    parser.add_argument("--matcher", default="sift_lightglue",
                        choices=["dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                                 "superpoint_lightglue", "sift_lightglue"])
    parser.add_argument("--dinov3_model", default="dinov3_vitb16",
                        choices=["dinov3_vits16", "dinov3_vits16plus",
                                 "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"])
    parser.add_argument("--dinov3_weights", default=None, type=str)
    parser.add_argument("--dinov3_num_points", default=100, type=int)
    parser.add_argument("--dinov3_stratify_threshold", default=20.0, type=float)
    parser.add_argument("--reproj_threshold", default=8.0, type=float)
    parser.add_argument("--photometric_refine_iters", default=100, type=int)
    parser.add_argument("--photometric_refine_lr", default=1e-2, type=float)
    parser.add_argument("--photometric_refine_huber_delta", default=1.0, type=float)
    parser.add_argument("--out_dir", default=None, type=str,
                        help="Defaults to log/output_photometric_sweep/"
                             "<matcher>_<transform_type>_<modality>_scale<scale>_side_by_side.")
    args = parser.parse_args()
    if args.matcher == "dinov3" and args.dinov3_weights is None:
        parser.error("--dinov3_weights is required when --matcher dinov3.")
    if args.out_dir is None:
        name = f"{args.matcher}_{args.transform_type}_{args.modality}_scale{args.scale:g}_side_by_side"
        args.out_dir = osp.join("log", "output_photometric_sweep", name)
    return args


if __name__ == "__main__":
    args = parse_args()
    rows = run(args)
    report_path = write_report(rows, args)
    print(f"\nReport written to: {report_path}")
