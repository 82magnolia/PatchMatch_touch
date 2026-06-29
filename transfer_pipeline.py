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


def _auto_identity_tsv(ref_dir, scale, modality, save_path):
    """Scan ref_dir for touch indices and write an identity TSV (query idx = ref idx)."""
    from retrieve_touch import discover_files
    entries = discover_files(ref_dir, modality, scale)
    if not entries:
        scale_str = f"_scale{scale:g}_" if scale is not None else "_"
        sys.exit(
            f"[auto-TSV] No files matching '{{idx}}{scale_str}{modality}.jpg' "
            f"found in {ref_dir}"
        )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("query\tref\n")
        for idx, _ in entries:
            f.write(f"{idx}\t{idx}\n")
    print(f"[auto-TSV] Wrote identity mapping ({len(entries)} entries) → {save_path}")
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
    g_ret.add_argument("--retrieval_mode", default="dinov2", choices=["dinov2", "tsv"],
                       help="'dinov2' (default) or 'tsv' (identity/pre-specified mapping).")
    g_ret.add_argument("--tsv", default=None,
                       help="Path to retrieval TSV (tsv mode). Auto-generated as "
                            "identity mapping if omitted.")
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
    g_tr.add_argument("--eval", action="store_true",
                      help="Compute PSNR/SSIM/LPIPS against the query video.")

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
        # Resolve TSV path for tsv mode
        tsv_path = args.tsv
        if args.retrieval_mode == "tsv" and tsv_path is None:
            tsv_path = os.path.join(save_dir, "identity.tsv")
            _auto_identity_tsv(
                ref_dir=args.ref_dir,
                scale=args.scale[0] if args.scale else None,
                modality=args.retrieval_modality[0],
                save_path=tsv_path,
            )

        cmd = [
            sys.executable, PROJECT_ROOT / "retrieve_touch.py",
            "--ref_dir", args.ref_dir,
            "--query_dir", args.query_dir,
            "--modality", *args.retrieval_modality,
            "--retrieval_mode", args.retrieval_mode,
            "--save_dir", retrieval_dir,
            "--no_figures",
        ]
        if args.scale is not None:
            cmd += ["--scale"] + [f"{s:g}" for s in args.scale]
        if args.retrieval_mode == "dinov2":
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
            "--no_nnf_figures",
        ]
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
        if args.eval:
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

    print(f"\nDone. All outputs under: {save_dir}")


if __name__ == "__main__":
    main()
