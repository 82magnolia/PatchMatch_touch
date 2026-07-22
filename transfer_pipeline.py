"""
End-to-end tactile transfer pipeline: retrieve → transfer → (optional) ReBotNet.

Given a flat directory of N reference touches and M query touch locations, runs:
  1. retrieve_touch.py      — finds top-K reference matches per query
  2. Stage 2 transfer, backend selected via --transfer_backend:
       'patchmatch' (default) — main_retrieval_transfer_accel.py, PatchMatch/EM loop
       'dinov3_feat_match'    — main_retrieval_transfer_feat_match.py, decomposed
                                 (offset + linear) feature-match, no PatchMatch/CUDA
  3. rebot_net/infer.py     — (optional) neural refinement of transferred videos

Works with both Taxim-generated data and real GelSight captures.

Output layout under --save_dir:
  identity.tsv        (auto-generated when --retrieval_mode tsv and no --tsv given)
  retrieval/
    results.pkl
  transfer/
    {query_idx}_transferred_em.mp4    ('patchmatch' backend)
    {query_idx}_transferred.mp4       ('dinov3_feat_match' backend)
    {query_idx}_ref_{video_type}.mp4
    {query_idx}_query_{video_type}.mp4
    ...
  enhanced/
    {query_idx}_transferred_em_enhanced.mp4  ('patchmatch' backend)
    {query_idx}_transferred_enhanced.mp4     ('dinov3_feat_match' backend)

Examples:

  # Taxim single object — identity TSV retrieval
  python transfer_pipeline.py \\
      --ref_dir Taxim/results/gen_contact_full/52 \\
      --query_dir Taxim/results/gen_contact_full_query/52 \\
      --scale 100 --retrieval_mode tsv \\
      --save_dir log/pipeline/52

  # Taxim — DINOv3 multi-modality retrieval + ReBotNet
  python transfer_pipeline.py \\
      --ref_dir Taxim/results/gen_contact_full/52 \\
      --query_dir Taxim/results/gen_contact_full_query/52 \\
      --scale 100 --retrieval_mode dinov3 \\
      --retrieval_modality normal curvature \\
      --dino_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \\
      --use_keyframe --use_accel --use_downsample_em \\
      --checkpoint log/rebot_checkpoints/best.pth \\
      --save_dir log/pipeline/52_dinov3

  # Real GelSight — multi-scale DINOv3 + residual ReBotNet
  python transfer_pipeline.py \\
      --ref_dir log/gelsight_captures/session_01 \\
      --query_dir log/gelsight_captures/session_01 \\
      --scale 0.5 1 2 --retrieval_mode dinov3 \\
      --dino_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \\
      --use_keyframe --use_accel --use_downsample_em \\
      --checkpoint log/rebot_checkpoints/best.pth --residual \\
      --save_dir log/pipeline/session_01

  # Real GelSight — DINOv3 feature-match transfer backend (no PatchMatch/CUDA)
  python transfer_pipeline.py \\
      --ref_dir log/gelsight_captures/session_01 \\
      --query_dir log/gelsight_captures/session_01 \\
      --scale 1 --retrieval_mode real_gt_retrieval \\
      --transfer_backend dinov3_feat_match \\
      --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \\
      --save_dir log/pipeline/session_01_dinov3
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, label):
    print(f"\n{'=' * 70}")
    print(f"[{label}]")
    print("  " + " ".join(str(c) for c in cmd))
    print("=" * 70, flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def _discover_or_exit(ref_dir, scale, modality):
    from retrieve_touch import discover_files
    entries = discover_files(ref_dir, modality, scale)
    if not entries:
        scale_str = f"_scale{scale:g}_" if scale is not None else "_"
        raise SystemExit(
            f"[auto-TSV] No files matching '{{idx}}{scale_str}{modality}.jpg' "
            f"found in {ref_dir}"
        )
    return entries


def _make_viz(transferred_path, query_path, ref_path, enhanced_path,
              query_normal_path, ref_normal_path, out_path, fps=5.0):
    """Create a 2×3 grid video.

    Layout:
      | Query Normal   | GT Query    | Ref Normal  |
      | PM Transferred | Enhanced    | Reference   |
    """
    import cv2
    sys.path.insert(0, str(PROJECT_ROOT / "rebot_net"))
    from trainer import _read_video_frames  # lazy import — avoids torch at startup

    def _load_video(path):
        if path and os.path.exists(path):
            frames = _read_video_frames(path)
            return frames or None
        return None

    def _load_still(path, n, h, w):
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w, h))
                return [img.astype(np.float32) / 255.0] * n
        return None

    xfer = _load_video(transferred_path)
    if not xfer:
        print(f"  [viz] Cannot read {transferred_path} — skipping.")
        return

    n = len(xfer)
    h, w = xfer[0].shape[:2]
    blank = [np.zeros((h, w, 3), dtype=np.float32)] * n

    query    = _load_video(query_path)    or blank
    ref      = _load_video(ref_path)      or blank
    enhanced = _load_video(enhanced_path) or blank
    q_normal = _load_still(query_normal_path, n, h, w) or blank
    r_normal = _load_still(ref_normal_path,   n, h, w) or blank

    def _label(path, name):
        return name if path and os.path.exists(path) else f"{name} (N/A)"

    panels = [
        (q_normal, _label(query_normal_path, "Query Normal")),
        (query,    _label(query_path,        "GT Query")),
        (r_normal, _label(ref_normal_path,   "Ref Normal")),
        (xfer,     "PM Transferred"),
        (enhanced, _label(enhanced_path,     "Enhanced")),
        (ref,      _label(ref_path,          "Reference")),
    ]

    n_frames = min(len(p) for p, _ in panels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, pad = 0.7, 2, 4

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h * 2))
    for i in range(n_frames):
        cells = []
        for frames, label in panels:
            cell = (np.clip(frames[i], 0, 1) * 255).astype(np.uint8)
            cell = cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(cell, (pad, pad), (pad + tw + pad, pad + th + pad), (0, 0, 0), -1)
            cv2.putText(cell, label, (pad * 2, pad + th), font, font_scale, (255, 255, 255), thickness)
            cells.append(cell)
        out.write(np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
    out.release()
    print(f"  [viz] {os.path.basename(out_path)}")


def _evaluate_videos(pred_dir, query_dir, video_type, pred_glob, query_stem_fn,
                     out_pkl):
    """Compute MSE/PSNR/SSIM/LPIPS for a set of predicted videos vs GT query videos.

    pred_glob      — glob pattern under pred_dir for predicted MP4s
    query_stem_fn  — callable(stem) → query video filename stem
    out_pkl        — where to save metrics.pkl

    Saves {per_touch: {query_idx: metrics}, average: metrics} — same format as
    main_retrieval_transfer_accel.py --eval, readable by parse_metrics.py.
    """
    import cv2
    import pickle
    import torch
    import lpips
    from skimage.metrics import mean_squared_error as compute_mse
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)

    def _read(path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        cap.release()
        return frames

    pred_paths = sorted(glob.glob(os.path.join(pred_dir, pred_glob)))
    if not pred_paths:
        print(f"  [eval] No videos matching '{pred_glob}' in {pred_dir} — skipping.")
        return

    METRIC_KEYS = ["MSE", "PSNR", "SSIM", "LPIPS"]
    per_touch = {}

    for pred_path in pred_paths:
        stem = os.path.basename(pred_path)
        # strip everything after the first underscore-delimited numeric prefix
        parts = stem.split("_")
        try:
            query_idx = int(parts[0])
        except ValueError:
            print(f"  [eval] Cannot parse query_idx from '{stem}' — skipping.")
            continue

        q_stem = query_stem_fn(query_idx)
        q_path = os.path.join(query_dir, q_stem)
        if not os.path.exists(q_path):
            print(f"  [eval] GT not found: {q_path} — skipping.")
            continue

        pred_frames = _read(pred_path)
        gt_frames   = _read(q_path)
        if not pred_frames or not gt_frames:
            continue

        n = min(len(pred_frames), len(gt_frames))
        mse_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        for i in range(n):
            gt, pred = gt_frames[i], pred_frames[i]
            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            mse = compute_mse(gt, pred)
            mse_sum  += mse
            psnr_sum += compute_psnr(gt, pred, data_range=1.0) if mse > 0 else 100.0
            ssim_sum += compute_ssim(gt, pred, data_range=1.0, channel_axis=-1)
            gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad():
                lpips_sum += loss_fn(gt_t, pred_t).item()

        per_touch[query_idx] = {k: v / n for k, v in zip(
            METRIC_KEYS, [mse_sum, psnr_sum, ssim_sum, lpips_sum])}
        m = per_touch[query_idx]
        print(f"  [eval] idx={query_idx}  MSE={m['MSE']:.5f}  PSNR={m['PSNR']:.2f}"
              f"  SSIM={m['SSIM']:.4f}  LPIPS={m['LPIPS']:.4f}")

    if not per_touch:
        return

    avg = {k: sum(m[k] for m in per_touch.values()) / len(per_touch) for k in METRIC_KEYS}
    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump({"per_touch": per_touch, "average": avg}, f)
    print(f"  [eval] Saved metrics ({len(per_touch)} entries) → {out_pkl}")
    print(f"  [eval] Average — MSE={avg['MSE']:.5f}  PSNR={avg['PSNR']:.2f}"
          f"  SSIM={avg['SSIM']:.4f}  LPIPS={avg['LPIPS']:.4f}")


def _auto_identity_tsv(ref_dir, scale, modality, save_path):
    """Scan ref_dir for touch indices and write an identity TSV (query idx = ref idx)."""
    entries = _discover_or_exit(ref_dir, scale, modality)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("query\tref\n")
        for idx, _ in entries:
            f.write(f"{idx}\t{idx}\n")
    print(f"[auto-TSV] Wrote identity mapping ({len(entries)} entries) → {save_path}")
    return save_path


def _auto_odd_to_even_tsv(ref_dir, scale, modality, save_path):
    """Scan ref_dir, split indices into even (ref) and odd (query), write odd→even TSV."""
    entries = _discover_or_exit(ref_dir, scale, modality)
    all_idxs = [idx for idx, _ in entries]
    even_idxs = set(idx for idx in all_idxs if idx % 2 == 0)
    odd_idxs  = sorted(idx for idx in all_idxs if idx % 2 == 1)
    if not even_idxs:
        raise SystemExit("[auto-TSV] No even-indexed touches found in ref_dir (need even=ref, odd=query).")
    if not odd_idxs:
        raise SystemExit("[auto-TSV] No odd-indexed touches found in ref_dir (need even=ref, odd=query).")

    rows = []
    for q in odd_idxs:
        if q - 1 not in even_idxs:
            print(f"[auto-TSV] Warning: odd index {q} has no paired even index {q - 1} — skipping pair.")
            continue
        rows.append((q, q - 1))

    if not rows:
        raise SystemExit("[auto-TSV] No valid odd→even pairs found in ref_dir after filtering.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("query\tref\n")
        for q, r in rows:
            f.write(f"{q}\t{r}\n")
    print(f"[auto-TSV] Wrote odd→even mapping ({len(rows)} pairs) → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="End-to-end tactile transfer: retrieve → PatchMatch → ReBotNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--ref_dir", required=True,
                   help="Flat directory with N reference touches.")
    p.add_argument("--query_dir", required=True,
                   help="Flat directory with M query touches.")
    p.add_argument("--save_dir", default="./log/tactile_transfer",
                   help="Root output directory (default: ./log/tactile_transfer).")
    p.add_argument("--scale", type=float, nargs="+", default=None,
                   help="Scale suffix(es) for file matching, e.g. 100 for Taxim or "
                        "0.5 1 2 for GelSight multi-scale. First value is used for "
                        "PatchMatch transfer; all values are used for retrieval features.")
    p.add_argument("--video_type", default="shadow", choices=["shadow", "sim"],
                   help="Touch video variant to transfer (default: shadow).")

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    g_ret = p.add_argument_group("Stage 1 — Retrieval")
    g_ret.add_argument("--retrieval_modality", nargs="+", default=["normal"],
                       choices=["color", "normal", "curvature", "height", "shapeindex"],
                       help="Modality(ies) for DINOv3 feature extraction (default: normal).")
    g_ret.add_argument("--retrieval_mode", default="dinov3",
                       choices=["dinov3", "tsv", "sim_gt_retrieval", "real_gt_retrieval"],
                       help=(
                           "'dinov3' (default) — DINOv3 feature retrieval; "
                           "'tsv' — explicit TSV file via --tsv; "
                           "'sim_gt_retrieval' — auto identity TSV (query idx = ref idx), "
                           "for Taxim synthetic data; "
                           "'real_gt_retrieval' — auto odd→even TSV (odd=query, even=ref), "
                           "for real GelSight captures in a single directory."
                       ))
    g_ret.add_argument("--tsv", default=None,
                       help="Path to retrieval TSV ('tsv' mode only).")
    g_ret.add_argument("--top_k", type=int, default=5,
                       help="Top-K retrievals per query (dinov3 mode, default: 5).")
    g_ret.add_argument("--dino_model", default="dinov3_vitb16",
                       choices=["dinov3_vits16", "dinov3_vits16plus",
                                "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"],
                       help="DINOv3 variant (dinov3 mode, default: dinov3_vitb16).")
    g_ret.add_argument("--dino_weights", default=None,
                       help="Path to gated DINOv3 .pth weights. Required when "
                            "--retrieval_mode dinov3 is set. Separate from Stage 2's "
                            "--dinov3_weights so retrieval and transfer can use different "
                            "model sizes.")
    g_ret.add_argument("--mask_mode", default="none",
                       choices=["black_pixels", "white_pixels", "none"],
                       help="Patch masking mode for DINOv3 (default: none).")

    # ── Stage 2: Transfer ─────────────────────────────────────────────────────
    g_tr = p.add_argument_group("Stage 2 — PatchMatch Transfer")
    g_tr.add_argument("--transfer_modality", nargs="+", default=["raw_normal"],
                      choices=["color", "normal", "curvature", "height", "shapeindex",
                               "raw_normal", "raw_height"],
                      help="Modality(ies) for NNF computation (default: raw_normal).")
    g_tr.add_argument("--patch_size", type=int, default=3,
                      help="PatchMatch patch size (default: 3).")
    g_tr.add_argument("--pm_iters", type=int, default=10,
                      help="PatchMatch propagation iterations (default: 10).")
    g_tr.add_argument("--em_iters", type=int, default=10,
                      help="EM iterations for the first / keyframe (default: 10).")
    g_tr.add_argument("--em_iters_subseq", type=int, default=1,
                      help="EM iterations for subsequent frames (default: 1).")
    g_tr.add_argument("--downsample_res", type=int, default=4,
                      help="Downsampling factor for low-res NNF seed (default: 4).")
    g_tr.add_argument("--use_downsample_em", action="store_true",
                      help="Run all EM at downsampled resolution + one final full-res pass.")
    g_tr.add_argument("--use_keyframe", action="store_true",
                      help="Find max-contact frame, run full EM on it, propagate ±.")
    g_tr.add_argument("--use_accel", action="store_true",
                      help="Warm-start PatchMatch with the previous frame's NNF.")
    g_tr.add_argument("--use_mask", action="store_true",
                      help="Composite with query render_mask video.")
    g_tr.add_argument("--use_ref_static_mask", action="store_true",
                      help="Keep background pixels unchanged (zero ref_static regions).")
    g_tr.add_argument("--init_scale", type=float, default=None,
                      help="Additional scale suffix used to compute a high-resolution seed NNF "
                           "between static images only, warm-starting the first EM PatchMatch "
                           "call. Requires --scale and --init_scale_convention.")
    g_tr.add_argument("--init_scale_convention", default=None,
                      choices=["render_scale", "obj_scale_factor"],
                      help="How to relate --init_scale back to --scale's field of view: "
                           "'render_scale' for real GelSight captures, 'obj_scale_factor' for "
                           "Taxim synthetic data. Required when --init_scale is set.")
    g_tr.add_argument("--init_dinov3_match_scale", type=float, default=None,
                      help="Alternative to --init_scale: seed NNF via DINOv3 patch-feature "
                           "matching instead of PatchMatch. Mutually exclusive with --init_scale. "
                           "Requires --scale, --init_dinov3_match_scale_convention, and "
                           "--dinov3_weights.")
    g_tr.add_argument("--init_dinov3_match_scale_convention", default=None,
                      choices=["render_scale", "obj_scale_factor"],
                      help="Same semantics as --init_scale_convention, applied to "
                           "--init_dinov3_match_scale. Required when --init_dinov3_match_scale "
                           "is set.")
    g_tr.add_argument("--dinov3_model", default="dinov3_vitb16",
                      choices=["dinov3_vits16", "dinov3_vits16plus",
                               "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"],
                      help="DINOv3 model variant, used by --init_dinov3_match_scale (patchmatch "
                           "backend) and --transfer_backend dinov3_feat_match (default: dinov3_vitb16).")
    g_tr.add_argument("--dinov3_weights", default=None,
                      help="Path to gated DINOv3 .pth weights. Required when "
                           "--init_dinov3_match_scale is set, or when "
                           "--transfer_backend dinov3_feat_match is used.")
    g_tr.add_argument("--transfer_backend", default="patchmatch",
                      choices=["patchmatch", "dinov3_feat_match"],
                      help="Stage 2 backend: 'patchmatch' (default) runs main_retrieval_transfer_accel.py "
                           "with the EM/PatchMatch loop; 'dinov3_feat_match' runs "
                           "main_retrieval_transfer_feat_match.py, the decomposed (offset + linear) "
                           "feature-match pipeline with no PatchMatch/CUDA dependency (any of the "
                           "supported matchers, not just DINOv3).")
    g_tr.add_argument("--transfer_matcher", default="disk_lightglue",
                      choices=["dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                               "superpoint_lightglue", "sift_lightglue"],
                      help="dinov3_feat_match backend only: matcher for the LINEAR stage, run at "
                           "--dinov3_match_scale (default: disk_lightglue). --transfer_offset_matcher "
                           "selects the OFFSET stage's matcher separately. Only 'dinov3' requires "
                           "--dinov3_weights. See main_retrieval_transfer_feat_match.py and README.md.")
    g_tr.add_argument("--transfer_offset_matcher", default="dinov3",
                      choices=["dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                               "superpoint_lightglue", "sift_lightglue"],
                      help="dinov3_feat_match backend only: matcher for the OFFSET stage, run at "
                           "--scale between the zero-offset-warped reference and the query "
                           "(default: dinov3). --transfer_matcher selects the LINEAR stage's "
                           "matcher instead; the two are independent.")
    g_tr.add_argument("--transfer_offset_method", default="median",
                      choices=["none", "median", "ransac"],
                      help="dinov3_feat_match backend only: how the OFFSET stage reduces match "
                           "displacements to a translation (default: median). 'ransac' is a "
                           "translation-only RANSAC; 'none' disables the offset stage, leaving "
                           "the centred zero-offset linear warp.")
    g_tr.add_argument("--dinov3_match_scale", type=float, default=None,
                      help="dinov3_feat_match backend only: scale at which the LINEAR stage is "
                           "fit -- a wider physical footprint than --scale (the video scale), "
                           "giving the matcher more object structure. The fitted transform is "
                           "conjugated back into --scale's coordinate space (no image cropping); "
                           "the offset is then re-estimated at --scale. Forwarded to "
                           "main_retrieval_transfer_feat_match.py as --match_scale. Requires "
                           "--dinov3_match_scale_convention. (Applies to whichever "
                           "--transfer_matcher is selected, despite the historical name.)")
    g_tr.add_argument("--dinov3_match_scale_convention", default=None,
                      choices=["render_scale", "obj_scale_factor"],
                      help="How to read --scale/--dinov3_match_scale as a physical footprint "
                           "ratio. 'render_scale' (GelSight): FOV proportional to scale. "
                           "'obj_scale_factor' (Taxim): FOV fixed, larger scale = finer detail. "
                           "Required when --dinov3_match_scale is set.")
    g_tr.add_argument("--dinov3_num_points", type=int, default=100,
                      help="dinov3_feat_match backend only: max sparse DINOv3 keypoints (default: 100).")
    g_tr.add_argument("--dinov3_stratify_threshold", type=float, default=20.0,
                      help="dinov3_feat_match backend only: spatial stratification threshold in px "
                           "(default: 20.0).")
    g_tr.add_argument("--dinov3_reproj_threshold", type=float, default=8.0,
                      help="dinov3_feat_match backend only: RANSAC reprojection threshold in px, "
                           "used to fit/select inliers for --dinov3_transform_type (default: 8.0, "
                           "found to beat the previous 3.0 default on both synthetic and real "
                           "data when warp quality is measured directly on the static image).")
    g_tr.add_argument("--dinov3_transform_type", default="homography",
                      choices=["affine", "homography"],
                      help="dinov3_feat_match backend only: the LINEAR component of the "
                           "decomposed (offset + linear) warp, RANSAC-fit at --dinov3_match_scale "
                           "(default: homography).")
    # (photometric refinement was removed along with the RBF transform types
    # when the dinov3_feat_match backend moved to the decomposed offset+linear
    # approach; main_retrieval_transfer_feat_match.py no longer accepts those
    # flags. dinov3/dense_match.py still provides them for the patchmatch backend.)

    # ── Stage 3: Refine ───────────────────────────────────────────────────────
    g_ref = p.add_argument_group("Stage 3 — ReBotNet Refinement")
    g_ref.add_argument("--checkpoint", default=None,
                       help="ReBotNet .pth checkpoint. Stage 3 is skipped if omitted.")
    g_ref.add_argument("--model_size", default="rebot_S",
                       choices=["rebot_XS", "rebot_S", "rebot_M", "rebot_L"],
                       help="ReBotNet model variant (default: rebot_S).")
    g_ref.add_argument("--residual", action="store_true",
                       help="Residual mode: subtract blank frame, predict refined residual.")

    # ── Stage control ─────────────────────────────────────────────────────────
    p.add_argument("--skip_retrieval", action="store_true")
    p.add_argument("--skip_transfer", action="store_true")
    p.add_argument("--skip_refine", action="store_true")
    p.add_argument("--skip_viz", action="store_true",
                   help="Skip Stage 4 grid visualization.")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip metric evaluation (MSE/PSNR/SSIM/LPIPS).")
    p.add_argument("--save_nnf_figures", action="store_true",
                   help="Save per-query NNF diagnostic figures in the transfer dir "
                        "(disabled by default).")
    p.add_argument("--save_match_figures", action="store_true",
                   help="dinov3_feat_match backend only: save a per-query ref|query panel of "
                        "the raw sparse feature matches, colored by RANSAC inlier/outlier/"
                        "fit-failed status (disabled by default; adds runtime since it "
                        "recomputes the sparse-matching stage).")

    args = p.parse_args()

    if args.init_scale is not None and args.init_dinov3_match_scale is not None:
        p.error("--init_scale and --init_dinov3_match_scale are mutually exclusive; "
               "pick one NNF-seeding strategy.")

    if args.init_scale is not None:
        if not args.scale:
            p.error("--init_scale requires --scale to also be set.")
        if args.init_scale_convention is None:
            p.error("--init_scale requires --init_scale_convention to be set.")

    if args.init_dinov3_match_scale is not None:
        if not args.scale:
            p.error("--init_dinov3_match_scale requires --scale to also be set.")
        if args.init_dinov3_match_scale_convention is None:
            p.error("--init_dinov3_match_scale requires "
                    "--init_dinov3_match_scale_convention to be set.")
        if not args.dinov3_weights:
            p.error("--init_dinov3_match_scale requires --dinov3_weights to be set.")

    if args.transfer_backend == "dinov3_feat_match":
        _active = [args.transfer_matcher]
        if args.transfer_offset_method != "none":
            _active.append(args.transfer_offset_matcher)
        if "dinov3" in _active and not args.dinov3_weights:
            p.error("--transfer_backend dinov3_feat_match requires --dinov3_weights "
                    "when --transfer_matcher or --transfer_offset_matcher is dinov3 "
                    "(--transfer_offset_matcher defaults to dinov3).")
        if args.dinov3_match_scale is not None and args.dinov3_match_scale_convention is None:
            p.error("--dinov3_match_scale requires --dinov3_match_scale_convention to be set.")

    if args.retrieval_mode == "dinov3" and not args.dino_weights:
        p.error("--retrieval_mode dinov3 requires --dino_weights to be set.")

    transfer_suffix = "_em" if args.transfer_backend == "patchmatch" else ""

    # Derived output paths
    save_dir      = os.path.abspath(args.save_dir)
    retrieval_dir = os.path.join(save_dir, "retrieval")
    transfer_dir  = os.path.join(save_dir, "transfer")
    enhanced_dir  = os.path.join(save_dir, "enhanced")
    retrieval_pkl = os.path.join(retrieval_dir, "results.pkl")

    for d in (retrieval_dir, transfer_dir):
        os.makedirs(d, exist_ok=True)

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    if not args.skip_retrieval:
        scale0 = args.scale[0] if args.scale else None
        tsv_path = args.tsv
        retrieval_mode_actual = args.retrieval_mode  # what retrieve_touch.py sees

        if args.retrieval_mode == "sim_gt_retrieval":
            retrieval_mode_actual = "tsv"
            tsv_path = os.path.join(save_dir, "identity.tsv")
            _auto_identity_tsv(
                ref_dir=args.ref_dir,
                scale=scale0,
                modality=args.retrieval_modality[0],
                save_path=tsv_path,
            )
        elif args.retrieval_mode == "real_gt_retrieval":
            retrieval_mode_actual = "tsv"
            tsv_path = os.path.join(save_dir, "odd_to_even.tsv")
            _auto_odd_to_even_tsv(
                ref_dir=args.ref_dir,
                scale=scale0,
                modality=args.retrieval_modality[0],
                save_path=tsv_path,
            )

        cmd = [
            sys.executable, PROJECT_ROOT / "retrieve_touch.py",
            "--ref_dir", args.ref_dir,
            "--query_dir", args.query_dir,
            "--modality", *args.retrieval_modality,
            "--retrieval_mode", retrieval_mode_actual,
            "--save_dir", retrieval_dir,
            "--no_figures",
        ]
        if args.scale is not None:
            cmd += ["--scale"] + [f"{s:g}" for s in args.scale]
        if retrieval_mode_actual == "dinov3":
            cmd += [
                "--top_k", str(args.top_k),
                "--dino_model", args.dino_model,
                "--dinov3_weights", args.dino_weights,
                "--mask_mode", args.mask_mode,
            ]
        else:
            cmd += ["--tsv", tsv_path]
        _run(cmd, "Stage 1: Retrieval")
    else:
        print("[Stage 1] Skipped (--skip_retrieval).")

    # ── Stage 2: Transfer ──────────────────────────────────────────────────────
    if not args.skip_transfer:
        transfer_scale = args.scale[0] if args.scale else None

        if args.transfer_backend == "patchmatch":
            cmd = [
                sys.executable, PROJECT_ROOT / "main_retrieval_transfer_accel.py",
                "--query_dir", args.query_dir,
                "--ref_dir",   args.ref_dir,
                "--retrieval_pkl", retrieval_pkl,
                "--modality",  *args.transfer_modality,
                "--video_type", args.video_type,
                "--save_dir",  transfer_dir,
                "--em",
                "--iters",         str(args.pm_iters),
                "--em_iters",      str(args.em_iters),
                "--em_iters_subseq", str(args.em_iters_subseq),
                "--patch_size",    str(args.patch_size),
                "--downsample_res", str(args.downsample_res),
            ]
            if not args.save_nnf_figures:
                cmd.append("--no_nnf_figures")
            if transfer_scale is not None:
                cmd += ["--scale", f"{transfer_scale:g}"]
            if args.init_scale is not None:
                cmd += ["--init_scale", f"{args.init_scale:g}",
                        "--init_scale_convention", args.init_scale_convention]
            if args.init_dinov3_match_scale is not None:
                cmd += ["--init_dinov3_match_scale", f"{args.init_dinov3_match_scale:g}",
                        "--init_dinov3_match_scale_convention", args.init_dinov3_match_scale_convention,
                        "--dinov3_model", args.dinov3_model,
                        "--dinov3_weights", args.dinov3_weights]
            if args.use_keyframe:
                cmd.append("--use_keyframe")
            if args.use_accel:
                cmd.append("--use_accel")
            if args.use_mask:
                cmd.append("--use_mask")
            if args.use_ref_static_mask:
                cmd.append("--use_ref_static_mask")
            if args.use_downsample_em:
                cmd.append("--use_downsample_em")
            if not args.skip_eval:
                cmd.append("--eval")
            _run(cmd, "Stage 2: PatchMatch Transfer")
        else:
            cmd = [
                sys.executable, PROJECT_ROOT / "main_retrieval_transfer_feat_match.py",
                "--query_dir", args.query_dir,
                "--ref_dir",   args.ref_dir,
                "--retrieval_pkl", retrieval_pkl,
                "--modality",  *args.transfer_modality,
                "--video_type", args.video_type,
                "--save_dir",  transfer_dir,
                "--matcher", args.transfer_matcher,
                "--dinov3_num_points", str(args.dinov3_num_points),
                "--dinov3_stratify_threshold", str(args.dinov3_stratify_threshold),
                "--reproj_threshold", str(args.dinov3_reproj_threshold),
                "--transform_type", args.dinov3_transform_type,
                "--offset_matcher", args.transfer_offset_matcher,
                "--offset_method", args.transfer_offset_method,
            ]
            if "dinov3" in (args.transfer_matcher, args.transfer_offset_matcher):
                cmd += ["--dinov3_model", args.dinov3_model,
                        "--dinov3_weights", args.dinov3_weights]
            if not args.save_nnf_figures:
                cmd.append("--no_nnf_figures")
            if args.save_match_figures:
                cmd.append("--save_match_figures")
            if transfer_scale is not None:
                cmd += ["--video_scale", f"{transfer_scale:g}"]
            if args.dinov3_match_scale is not None:
                cmd += ["--match_scale", f"{args.dinov3_match_scale:g}",
                        "--match_scale_convention", args.dinov3_match_scale_convention]
            if args.use_mask:
                cmd.append("--use_mask")
            if not args.skip_eval:
                cmd.append("--eval")
            _run(cmd, f"Stage 2: {args.transfer_matcher} Feature-Match Transfer")
    else:
        print("[Stage 2] Skipped (--skip_transfer).")

    # ── Stage 3: ReBotNet Refinement ──────────────────────────────────────────
    if args.skip_refine or args.checkpoint is None:
        if args.checkpoint is None and not args.skip_refine:
            print("[Stage 3] Skipped (no --checkpoint provided).")
        else:
            print("[Stage 3] Skipped (--skip_refine).")
    else:
        os.makedirs(enhanced_dir, exist_ok=True)
        transferred_glob = f"*_transferred{transfer_suffix}.mp4"
        transferred_videos = sorted(
            glob.glob(os.path.join(transfer_dir, transferred_glob))
        )
        if not transferred_videos:
            print(f"[Stage 3] No {transferred_glob} found in transfer dir — skipping.")
        else:
            print(f"\n[Stage 3] Refining {len(transferred_videos)} video(s) with ReBotNet...")
            for vid_path in transferred_videos:
                cmd = [
                    sys.executable, PROJECT_ROOT / "rebot_net" / "infer.py",
                    "--input_video", vid_path,
                    "--checkpoint",  args.checkpoint,
                    "--model_size",  args.model_size,
                    "--save_dir",    enhanced_dir,
                ]
                if args.residual:
                    cmd.append("--residual")
                _run(cmd, f"Stage 3: Refine {os.path.basename(vid_path)}")

    # ── Stage 4: Grid Visualization ──────────────────────────────────────────────
    if not args.skip_viz:
        import pickle

        viz_dir = os.path.join(save_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)

        # Load retrieval pkl to map query_idx → top-1 ref_idx
        query_to_ref = {}
        if os.path.exists(retrieval_pkl):
            with open(retrieval_pkl, "rb") as f:
                for row in pickle.load(f):
                    query_to_ref[row["query_idx"]] = row["topk_ref_indices"][0]

        scale0 = args.scale[0] if args.scale else None

        def _normal_path(folder, idx):
            if scale0 is not None:
                return os.path.join(folder, f"{idx}_scale{scale0:g}_normal.jpg")
            return os.path.join(folder, f"{idx}_normal.jpg")

        transferred_glob = f"*_transferred{transfer_suffix}.mp4"
        transferred_videos = sorted(
            glob.glob(os.path.join(transfer_dir, transferred_glob))
        )
        if not transferred_videos:
            print(f"[Stage 4] No {transferred_glob} found — skipping visualization.")
        else:
            print(f"\n[Stage 4] Creating {len(transferred_videos)} grid visualization(s)...")
            for xfer_path in transferred_videos:
                stem = os.path.basename(xfer_path).replace(f"_transferred{transfer_suffix}.mp4", "")
                try:
                    query_idx = int(stem)
                except ValueError:
                    query_idx = None
                ref_idx = query_to_ref.get(query_idx)

                _make_viz(
                    transferred_path=xfer_path,
                    query_path=os.path.join(transfer_dir, f"{stem}_query_{args.video_type}.mp4"),
                    ref_path=os.path.join(transfer_dir,   f"{stem}_ref_{args.video_type}.mp4"),
                    enhanced_path=os.path.join(enhanced_dir, f"{stem}_transferred{transfer_suffix}_enhanced.mp4"),
                    query_normal_path=_normal_path(args.query_dir, query_idx) if query_idx is not None else None,
                    ref_normal_path=_normal_path(args.ref_dir, ref_idx) if ref_idx is not None else None,
                    out_path=os.path.join(viz_dir, f"{stem}_grid.mp4"),
                )
    else:
        print("[Stage 4] Skipped (--skip_viz).")

    # ── Stage 5: Evaluate Enhanced Output ────────────────────────────────────
    if not args.skip_eval and args.checkpoint is not None and not args.skip_refine:
        print(f"\n[Stage 5] Computing enhanced-output metrics...")
        _evaluate_videos(
            pred_dir=enhanced_dir,
            query_dir=args.query_dir,
            video_type=args.video_type,
            pred_glob=f"*_transferred{transfer_suffix}_enhanced.mp4",
            query_stem_fn=lambda idx: f"{idx}_{args.video_type}.mp4",
            out_pkl=os.path.join(enhanced_dir, "metrics.pkl"),
        )
    elif args.skip_eval:
        print("[Stage 5] Skipped (--skip_eval).")
    else:
        print("[Stage 5] Skipped (no Stage 3 output to evaluate).")

    print(f"\nDone. All outputs under: {save_dir}")


if __name__ == "__main__":
    main()
