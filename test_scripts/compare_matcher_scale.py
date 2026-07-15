"""No-refinement matcher x scale sweep: ~20 distinct touch-location pairs,
each testing one (matcher, scale) combination, modality fixed at curvature
(the best-performing modality found by the earlier full sweep) and
transform_type fixed at rbf_homography (this dataset's default/best
transform). Follow-up to compare_photometric_refine.py's sweep and the
photometric-refine video demos -- after visually inspecting those,
refinement was judged not to help, so this sweep tests matcher/scale choice
alone, without --photometric_refine.

Produces per-case: query.png, reference.png, warped.png (feeds
make_warp_overlay_report.py's side-by-side + alpha-blend-slider HTML).

Example usage:
    python test_scripts/compare_matcher_scale.py \
        --sessions_base log/real_data_gt_retrieval --num_pairs 20 \
        --scales 1 2 4 8 \
        --matchers sift_lightglue disk_lightglue superpoint_lightglue superpoint_superglue loftr \
        --modality curvature --transform_type rbf_homography \
        --out_dir log/output_photometric_sweep/matcher_scale_sweep
"""

import argparse
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


def build_combos(matchers, scales, num_pairs):
    """Cartesian product of matchers x scales, truncated/cycled to num_pairs,
    each combo assigned one distinct session (round-robin session index)."""
    combos = [(m, s) for m in matchers for s in scales]
    if num_pairs < len(combos):
        combos = combos[:num_pairs]
    elif num_pairs > len(combos):
        # cycle through the cartesian product again with a different session offset
        reps = (num_pairs + len(combos) - 1) // len(combos)
        combos = (combos * reps)[:num_pairs]
    return combos


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS model on {device}...")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    sessions = sorted(
        (s for s in os.listdir(args.sessions_base) if s.isdigit()), key=int
    )
    combos = build_combos(args.matchers, args.scales, args.num_pairs)

    rows = []
    for i, (matcher, scale) in enumerate(combos):
        session = sessions[i % len(sessions)]
        session_dir = osp.join(args.sessions_base, session)
        pairs = discover_odd_even_pairs(session_dir, scale, args.modality)
        if not pairs:
            print(f"  [skip] session {session}: no pairs for scale={scale} modality={args.modality}")
            continue
        query_idx, ref_idx = pairs[0]
        case_id = f"{matcher}_scale{scale:g}_session{session}"
        print(f"\n[{i+1}/{len(combos)}] {case_id}: Query {query_idx} -> Reference {ref_idx}")

        try:
            q, r = _load_query_ref_static_for_matching(
                session_dir, session_dir, query_idx, ref_idx, [args.modality],
                scale, None, None, matcher)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping (static image load failed): {e}")
            continue

        c_dir = osp.join(args.out_dir, case_id)
        os.makedirs(c_dir, exist_ok=True)
        query_png = osp.join(c_dir, "query.png")
        ref_png = osp.join(c_dir, "ref.png")
        save_png(query_png, q)
        save_png(ref_png, r)

        row = {
            "case_id": case_id, "matcher": matcher, "scale": scale, "session": session,
            "query_idx": query_idx, "ref_idx": ref_idx,
            "query_png": query_png, "ref_png": ref_png,
            "warped_png": None, "status": "ok", "reason": "",
            "MSE": None, "PSNR": None, "SSIM": None, "LPIPS": None,
        }

        try:
            pts_l, pts_r = compute_sparse_matches(
                matcher, r, q, args.dinov3_model, args.dinov3_weights,
                args.dinov3_num_points, args.dinov3_stratify_threshold)
            h2, w2 = q.shape[:2]
            src_row, src_col, inlier_count, total = _fit_dense_field(
                pts_l, pts_r, h2, w2, args.transform_type, args.reproj_threshold,
                image_left=r, image_right=q, refine=False)
            nnf = build_nnf(src_row, src_col, r.shape)
            warped = reconstruct_avg(nnf, r[..., :3], patch_size=1)
            metrics = evaluate_video_metrics([q[..., :3]], [warped], lpips_model, device)
            row.update(metrics)
            warped_png = osp.join(c_dir, "warped.png")
            save_png(warped_png, warped)
            row["warped_png"] = warped_png
            print(f"  MSE={metrics['MSE']:.5f} PSNR={metrics['PSNR']:.2f} "
                 f"SSIM={metrics['SSIM']:.4f} LPIPS={metrics['LPIPS']:.4f}")
        except RuntimeError as e:
            row["status"] = "failed"
            row["reason"] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {e}")

        rows.append(row)

    results_pkl = osp.join(args.out_dir, "results.pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({"rows": rows, "args": vars(args)}, f)
    print(f"\nSaved {len(rows)} case results to: {results_pkl}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-refinement matcher x scale sweep across ~20 distinct touch-location pairs.")
    parser.add_argument("--sessions_base", default="log/real_data_gt_retrieval", type=str)
    parser.add_argument("--num_pairs", default=20, type=int)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--matchers", nargs="+",
                        default=["sift_lightglue", "disk_lightglue", "superpoint_lightglue",
                                "superpoint_superglue", "loftr"],
                        choices=["dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                                "superpoint_lightglue", "sift_lightglue"])
    parser.add_argument("--modality", default="curvature",
                        choices=["color", "normal", "curvature", "height",
                                "raw_normal", "raw_height", "shapeindex"])
    parser.add_argument("--transform_type", default="rbf_homography",
                        choices=["affine", "homography", "rbf_affine", "rbf_homography"])
    parser.add_argument("--dinov3_model", default="dinov3_vitb16",
                        choices=["dinov3_vits16", "dinov3_vits16plus",
                                "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"])
    parser.add_argument("--dinov3_weights", default=None, type=str)
    parser.add_argument("--dinov3_num_points", default=100, type=int)
    parser.add_argument("--dinov3_stratify_threshold", default=20.0, type=float)
    parser.add_argument("--reproj_threshold", default=8.0, type=float)
    parser.add_argument("--out_dir", default=None, type=str)
    args = parser.parse_args()
    if "dinov3" in args.matchers and args.dinov3_weights is None:
        parser.error("--dinov3_weights is required when 'dinov3' is in --matchers.")
    if args.out_dir is None:
        args.out_dir = osp.join("log", "output_photometric_sweep", "matcher_scale_sweep")
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
