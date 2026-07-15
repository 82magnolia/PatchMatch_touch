"""Sweeps transform_type x photometric-refinement-loss x scale x modality
combinations on a few query/ref pairs, scores each combination's warped
static image against the query static image (whole-image, unmasked
MSE/PSNR/SSIM/LPIPS), and writes an HTML comparison report.

Computes sparse matches once per (query/ref pair, scale, modality) -- the
expensive step, a DINOv3/IMCUI forward pass -- and reuses them across every
(transform_type, refine_loss) combo, calling dinov3.dense_match._fit_dense_field
directly -- mirrors the pattern main_retrieval_transfer_feat_match.py's
_compute_sparse_matches_and_inliers already uses to recompute matches for its
own diagnostics.

Two retrieval sources (--retrieval_mode):
  pkl               (default) load query/ref pairs (top-1) from a
                     retrieve_touch.py results.pkl via --retrieval_pkl.
  real_gt_retrieval  auto odd->even pairing (odd=query, even=ref=odd-1) within
                     a single --query_dir/--ref_dir, no pkl needed -- mirrors
                     transfer_pipeline.py's --retrieval_mode real_gt_retrieval
                     (see train_refine_scripts/transfer_all_real_data_gt_retrieval/).
                     Indices are (re)discovered per (scale, modality) combo via
                     retrieve_touch.discover_files, since availability can vary.

--out_dir defaults to log/output_photometric_sweep/<matcher> if not given --
a single umbrella folder (log/output_photometric_sweep/) holding one
descriptively-named subfolder per sweep run.

Example usage (pkl retrieval, DINOv3):
    python test_scripts/compare_photometric_refine.py \
        --query_dir log/RealData/gear --ref_dir log/RealData/gear \
        --retrieval_pkl log/pipeline_gear/retrieval/results.pkl \
        --modalities raw_normal --scales 1 \
        --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --num_queries 3

Example usage (real_gt_retrieval, scale x modality sweep, an IMCUI backend --
no gated weights required):
    python test_scripts/compare_photometric_refine.py \
        --query_dir log/real_data_gt_retrieval/1 --ref_dir log/real_data_gt_retrieval/1 \
        --retrieval_mode real_gt_retrieval \
        --scales 1 2 4 8 --modalities raw_normal curvature normal height color \
        --matcher sift_lightglue --num_queries 3
"""

import argparse
import html
import os
import pickle
import sys
from os import path as osp

import cv2
import numpy as np
import torch
import lpips
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from main_retrieval_transfer_feat_match import (
    _load_query_ref_static_for_matching, reconstruct_avg, evaluate_video_metrics,
)
from dinov3.dense_match import (
    TRANSFORM_TYPES, REFINE_LOSS_TYPES, _fit_dense_field,
    _load_model, _find_sparse_matches, MODEL_N_LAYERS,
)

REFINE_CONFIGS = [None] + list(REFINE_LOSS_TYPES)  # None = no refinement ("baseline")


def discover_odd_even_pairs(dir_, scale, modality):
    """Odd query idx -> even (idx - 1) ref idx pairs present in dir_ for one
    (scale, modality) combo -- mirrors transfer_pipeline.py's
    _auto_odd_to_even_tsv (real_gt_retrieval mode), reusing the same
    retrieve_touch.discover_files index scan. raw_normal/raw_height are
    npz-backed but colocated under the same idx numbering as their plain
    (jpg) counterpart, so we discover indices via the base modality name.
    """
    from retrieve_touch import discover_files
    base_modality = modality[len("raw_"):] if modality.startswith("raw_") else modality
    entries = discover_files(dir_, base_modality, scale)
    all_idxs = [idx for idx, _ in entries]
    even_idxs = set(idx for idx in all_idxs if idx % 2 == 0)
    odd_idxs = sorted(idx for idx in all_idxs if idx % 2 == 1)
    return [(q, q - 1) for q in odd_idxs if (q - 1) in even_idxs]


def refine_label(refine_cfg):
    return "none" if refine_cfg is None else refine_cfg


def save_png(path, img_rgb_float):
    img_u8 = (np.clip(img_rgb_float[..., :3], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))


def build_nnf(src_row, src_col, ref_shape):
    h2, w2 = src_row.shape
    nnf = np.zeros((h2, w2, 2), dtype=np.int32)
    nnf[..., 0] = np.clip(np.round(src_col), 0, ref_shape[1] - 1)
    nnf[..., 1] = np.clip(np.round(src_row), 0, ref_shape[0] - 1)
    return nnf


def compute_sparse_matches(matcher, r, q, dinov3_model, dinov3_weights,
                           num_points, stratify_threshold):
    """Compute sparse (pts_l, pts_r) matches once for a query/ref pair,
    shared across every (transform_type, refine_loss) combo swept below."""
    if matcher == "dinov3":
        model, device = _load_model(dinov3_model, dinov3_weights)
        n_layers = MODEL_N_LAYERS[dinov3_model]
        pil_left = Image.fromarray((np.clip(r, 0, 1) * 255).astype(np.uint8))
        pil_right = Image.fromarray((np.clip(q, 0, 1) * 255).astype(np.uint8))
        return _find_sparse_matches(
            pil_left, pil_right, model, n_layers, device, num_points, stratify_threshold)
    from imcui_match import compute_imcui_sparse_matches
    return compute_imcui_sparse_matches(r, q, matcher)


def resolve_entries(args, scale, modality):
    """(query_idx, ref_idx) pairs to sweep for one (scale, modality) combo."""
    if args.retrieval_mode == "real_gt_retrieval":
        pairs = discover_odd_even_pairs(args.query_dir, scale, modality)
        return pairs[:args.num_queries]
    with open(args.retrieval_pkl, "rb") as f:
        retrieval_results = pickle.load(f)
    return [(e["query_idx"], e["topk_ref_indices"][0])
            for e in retrieval_results[:args.num_queries]]


def run_sweep(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS model on {device}...")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    rows = []
    cases_info = []

    for scale in args.scales:
        for modality in args.modalities:
            entries = resolve_entries(args, scale, modality)
            print(f"\n=== scale={scale} modality={modality}: "
                 f"{len(entries)} query/ref pair(s) ===")

            for query_idx, ref_idx in entries:
                print(f"\nQuery {query_idx} -> Reference {ref_idx} "
                     f"(scale={scale}, modality={modality})")

                try:
                    q, r = _load_query_ref_static_for_matching(
                        args.query_dir, args.ref_dir, query_idx, ref_idx, [modality],
                        scale, None, None, args.matcher)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  Skipping (static image load failed): {e}")
                    continue

                case_id = f"{query_idx}_scale{scale:g}_{modality}"
                c_dir = osp.join(args.out_dir, case_id)
                os.makedirs(c_dir, exist_ok=True)
                query_png = osp.join(c_dir, "query.png")
                ref_png = osp.join(c_dir, "ref.png")
                save_png(query_png, q)
                save_png(ref_png, r)
                cases_info.append({
                    "case_id": case_id, "query_idx": query_idx, "ref_idx": ref_idx,
                    "scale": scale, "modality": modality,
                    "query_png": query_png, "ref_png": ref_png,
                })

                try:
                    pts_l, pts_r = compute_sparse_matches(
                        args.matcher, r, q, args.dinov3_model, args.dinov3_weights,
                        args.dinov3_num_points, args.dinov3_stratify_threshold)
                except RuntimeError as e:
                    print(f"  Sparse matching failed, skipping all combos for this case: {e}")
                    continue

                h2, w2 = q.shape[:2]
                for transform_type in TRANSFORM_TYPES:
                    for refine_cfg in REFINE_CONFIGS:
                        label = refine_label(refine_cfg)
                        row = {
                            "case_id": case_id, "query_idx": query_idx, "ref_idx": ref_idx,
                            "scale": scale, "modality": modality,
                            "transform_type": transform_type, "refine": label,
                            "status": "ok", "reason": "",
                            "MSE": None, "PSNR": None, "SSIM": None, "LPIPS": None,
                            "warped_png": None,
                        }
                        try:
                            src_row, src_col, inlier_count, total = _fit_dense_field(
                                pts_l, pts_r, h2, w2, transform_type, args.reproj_threshold,
                                image_left=r, image_right=q,
                                refine=(refine_cfg is not None), refine_loss=refine_cfg or "l1",
                                refine_iters=args.photometric_refine_iters,
                                refine_lr=args.photometric_refine_lr,
                                refine_huber_delta=args.photometric_refine_huber_delta)
                            nnf = build_nnf(src_row, src_col, r.shape)
                            warped = reconstruct_avg(nnf, r[..., :3], patch_size=1)
                            metrics = evaluate_video_metrics([q[..., :3]], [warped], lpips_model, device)
                            row.update(metrics)
                            warped_png = osp.join(c_dir, f"{transform_type}_{label}.png")
                            save_png(warped_png, warped)
                            row["warped_png"] = warped_png
                            print(f"  {transform_type:16s} refine={label:9s} "
                                 f"MSE={metrics['MSE']:.5f} PSNR={metrics['PSNR']:.2f} "
                                 f"SSIM={metrics['SSIM']:.4f} LPIPS={metrics['LPIPS']:.4f}")
                        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                            row["status"] = "failed"
                            row["reason"] = f"{type(e).__name__}: {e}"
                            print(f"  {transform_type:16s} refine={label:9s} FAILED: {e}")
                        rows.append(row)

    results_pkl = osp.join(args.out_dir, "results.pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({"rows": rows, "cases": cases_info}, f)
    print(f"\nSaved {len(rows)} combo results to: {results_pkl}")
    return rows, cases_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare transform_type x photometric-refinement-loss x scale x "
                    "modality combinations on static-image reconstruction quality.")
    parser.add_argument("--query_dir", required=True, type=str)
    parser.add_argument("--ref_dir", required=True, type=str)
    parser.add_argument("--retrieval_mode", default="pkl", choices=["pkl", "real_gt_retrieval"],
                        help="'pkl' (default): load query/ref pairs from --retrieval_pkl. "
                             "'real_gt_retrieval': auto odd->even pairing within --query_dir "
                             "(odd=query, even=ref=odd-1), no pkl needed -- mirrors "
                             "transfer_pipeline.py's --retrieval_mode real_gt_retrieval.")
    parser.add_argument("--retrieval_pkl", default=None, type=str,
                        help="Required when --retrieval_mode pkl (the default).")
    parser.add_argument("--modalities", required=True, nargs="+",
                        choices=["color", "normal", "curvature", "height",
                                 "raw_normal", "raw_height", "shapeindex"],
                        help="Modalities to sweep -- each is tried independently as its own "
                             "single-modality combo (matching requires exactly 3 channels, "
                             "so these aren't concatenated together).")
    parser.add_argument("--scales", required=True, nargs="+", type=float,
                        help="Scale values to sweep (e.g. --scales 1 2 4 8).")
    parser.add_argument("--matcher", default="dinov3",
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
    parser.add_argument("--num_queries", default=3, type=int,
                        help="Number of query/ref pairs to sweep per (scale, modality) combo "
                             "(default: 3).")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="Output directory for results.pkl/report.html/warped PNGs. "
                             "Defaults to log/output_photometric_sweep/<matcher> -- a single "
                             "umbrella folder (log/output_photometric_sweep/) holding one "
                             "descriptively-named subfolder per sweep run, matching this "
                             "repo's log/<pipeline>_<matcher> convention (e.g. "
                             "log/transfer_pipeline_real_data_gt_retrieval_sift_lightglue).")
    args = parser.parse_args()
    if args.matcher == "dinov3" and args.dinov3_weights is None:
        parser.error("--dinov3_weights is required when --matcher dinov3 (the default).")
    if args.retrieval_mode == "pkl" and args.retrieval_pkl is None:
        parser.error("--retrieval_pkl is required when --retrieval_mode pkl (the default).")
    if args.out_dir is None:
        args.out_dir = osp.join("log", "output_photometric_sweep", args.matcher)
    return args


if __name__ == "__main__":
    args = parse_args()
    rows, cases_info = run_sweep(args)
    from make_report import write_report
    report_path = write_report(rows, cases_info, args.out_dir)
    print(f"\nReport written to: {report_path}")
