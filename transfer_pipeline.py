"""
End-to-end tactile transfer pipeline: retrieve → PatchMatch → (optional) ReBotNet.

Given a flat directory of N reference touches and M query touch locations, runs:
  1. retrieve_touch.py      — finds top-K reference matches per query
  2. main_retrieval_transfer_accel.py — warps reference videos to query layout via PatchMatch
  3. rebot_net/infer.py     — (optional) neural refinement of transferred videos

Works with both Taxim-generated data and real GelSight captures.

Output layout under --save_dir:
  identity.tsv        (auto-generated when --retrieval_mode tsv and no --tsv given)
  retrieval/
    results.pkl
  transfer/
    {query_idx}_transferred_em.mp4
    {query_idx}_ref_{video_type}.mp4
    {query_idx}_query_{video_type}.mp4
    ...
  enhanced/
    {query_idx}_transferred_em_enhanced.mp4

Examples:

  # Taxim single object — identity TSV retrieval
  python transfer_pipeline.py \\
      --ref_dir Taxim/results/gen_contact_full/52 \\
      --query_dir Taxim/results/gen_contact_full_query/52 \\
      --scale 100 --retrieval_mode tsv \\
      --save_dir log/pipeline/52

  # Taxim — DINOv2 multi-modality retrieval + ReBotNet
  python transfer_pipeline.py \\
      --ref_dir Taxim/results/gen_contact_full/52 \\
      --query_dir Taxim/results/gen_contact_full_query/52 \\
      --scale 100 --retrieval_mode dinov2 \\
      --retrieval_modality normal curvature \\
      --use_keyframe --use_accel --use_downsample_em \\
      --checkpoint log/rebot_checkpoints/best.pth \\
      --save_dir log/pipeline/52_dinov2

  # Real GelSight — multi-scale DINOv2 + residual ReBotNet
  python transfer_pipeline.py \\
      --ref_dir log/gelsight_captures/session_01 \\
      --query_dir log/gelsight_captures/session_01 \\
      --scale 0.5 1 2 --retrieval_mode dinov2 \\
      --use_keyframe --use_accel --use_downsample_em \\
      --checkpoint log/rebot_checkpoints/best.pth --residual \\
      --save_dir log/pipeline/session_01
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
                       help="Modality(ies) for DINOv2 feature extraction (default: normal).")
    g_ret.add_argument("--retrieval_mode", default="dinov2",
                       choices=["dinov2", "tsv", "sim_gt_retrieval", "real_gt_retrieval"],
                       help=(
                           "'dinov2' (default) — DINOv2 feature retrieval; "
                           "'tsv' — explicit TSV file via --tsv; "
                           "'sim_gt_retrieval' — auto identity TSV (query idx = ref idx), "
                           "for Taxim synthetic data; "
                           "'real_gt_retrieval' — auto odd→even TSV (odd=query, even=ref), "
                           "for real GelSight captures in a single directory."
                       ))
    g_ret.add_argument("--tsv", default=None,
                       help="Path to retrieval TSV ('tsv' mode only).")
    g_ret.add_argument("--top_k", type=int, default=5,
                       help="Top-K retrievals per query (dinov2 mode, default: 5).")
    g_ret.add_argument("--dino_model", default="dinov2_vits14",
                       choices=["dinov2_vits14", "dinov2_vitb14",
                                "dinov2_vitl14", "dinov2_vitg14"],
                       help="DINOv2 variant (default: dinov2_vits14).")
    g_ret.add_argument("--mask_mode", default="none",
                       choices=["black_pixels", "white_pixels", "none"],
                       help="Patch masking mode for DINOv2 (default: none).")

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

    args = p.parse_args()

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
        if retrieval_mode_actual == "dinov2":
            cmd += [
                "--top_k", str(args.top_k),
                "--dino_model", args.dino_model,
                "--mask_mode", args.mask_mode,
            ]
        else:
            cmd += ["--tsv", tsv_path]
        _run(cmd, "Stage 1: Retrieval")
    else:
        print("[Stage 1] Skipped (--skip_retrieval).")

    # ── Stage 2: PatchMatch Transfer ──────────────────────────────────────────
    if not args.skip_transfer:
        transfer_scale = args.scale[0] if args.scale else None

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
        print("[Stage 2] Skipped (--skip_transfer).")

    # ── Stage 3: ReBotNet Refinement ──────────────────────────────────────────
    if args.skip_refine or args.checkpoint is None:
        if args.checkpoint is None and not args.skip_refine:
            print("[Stage 3] Skipped (no --checkpoint provided).")
        else:
            print("[Stage 3] Skipped (--skip_refine).")
    else:
        os.makedirs(enhanced_dir, exist_ok=True)
        transferred_videos = sorted(
            glob.glob(os.path.join(transfer_dir, "*_transferred_em.mp4"))
        )
        if not transferred_videos:
            print("[Stage 3] No *_transferred_em.mp4 found in transfer dir — skipping.")
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

        transferred_videos = sorted(
            glob.glob(os.path.join(transfer_dir, "*_transferred_em.mp4"))
        )
        if not transferred_videos:
            print("[Stage 4] No *_transferred_em.mp4 found — skipping visualization.")
        else:
            print(f"\n[Stage 4] Creating {len(transferred_videos)} grid visualization(s)...")
            for xfer_path in transferred_videos:
                stem = os.path.basename(xfer_path).replace("_transferred_em.mp4", "")
                try:
                    query_idx = int(stem)
                except ValueError:
                    query_idx = None
                ref_idx = query_to_ref.get(query_idx)

                _make_viz(
                    transferred_path=xfer_path,
                    query_path=os.path.join(transfer_dir, f"{stem}_query_{args.video_type}.mp4"),
                    ref_path=os.path.join(transfer_dir,   f"{stem}_ref_{args.video_type}.mp4"),
                    enhanced_path=os.path.join(enhanced_dir, f"{stem}_transferred_em_enhanced.mp4"),
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
            pred_glob="*_transferred_em_enhanced.mp4",
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
