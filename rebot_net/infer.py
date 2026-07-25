"""
Inference script for ReBotNet tactile video enhancement.

Runs enhancement on transferred videos and saves the results.
Does not require ground-truth videos.

Two modes:
  --input_video   Process a single MP4 file.
  --transfer_dir  Process all (or selected) objects from the transfer directory.

Usage:
    # Single video
    python rebot_net/infer.py \
        --input_video log/transfer/52/0_transferred_em.mp4 \
        --checkpoint  log/rebot_checkpoints/best.pth \
        --model_size  rebot_S \
        --save_dir    log/rebot_infer

    # All objects in transfer_dir
    python rebot_net/infer.py \
        --transfer_dir log/transfer \
        --checkpoint   log/rebot_checkpoints/best.pth \
        --model_size   rebot_S \
        --save_dir     log/rebot_infer

    # Specific objects only
    python rebot_net/infer.py \
        --transfer_dir log/transfer \
        --object_ids   52 597 381 \
        --checkpoint   log/rebot_checkpoints/best.pth \
        --model_size   rebot_S \
        --save_dir     log/rebot_infer

Residual mode (subtract blank from input, predict refined residual, add blank back):
    python rebot_net/infer.py \
        --input_video log/transfer/52/0_transferred_em.mp4 \
        --checkpoint  log/rebot_checkpoints_residual/best.pth \
        --model_size  rebot_S \
        --save_dir    log/rebot_infer_residual \
        --residual
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import MODEL_CONFIGS, build_model
from trainer import _write_video, _read_video_frames, _make_grid_video, _residual_to_vis
from dataset import _resolve_lq_path, _normal_blank


def _strip_transferred_suffix(stem):
    """Strip a '_transferred[_em]' suffix to recover the shared '{query_idx}' stem.

    '_transferred_em' is main_retrieval_transfer_accel.py's (patchmatch backend)
    output naming; plain '_transferred' is main_retrieval_transfer_feat_match.py's
    (dinov3_feat_match backend). Longer suffix must be tried first so the
    '_em'-suffixed case doesn't fall through to the shorter match.
    """
    for suffix in ("_transferred_em", "_transferred"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _to_tensor(frame_rgb):
    return torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def enhance_video(model, frames_rgb, device, blank_tensor=None):
    """Enhance a list of (H,W,3) uint8 frames; returns float32 [0,1] absolute frames.

    When blank_tensor is provided (residual mode), subtracts it before inference
    and adds it back afterward. blank_tensor is a (3,H,W) float32 CPU tensor.
    Returns also a list of residual visualization frames (or None if not residual).
    """
    enhanced = []
    residual_vis = [] if blank_tensor is not None else None
    prev_tensor = None

    for frame in frames_rgb:
        curr_tensor = _to_tensor(frame)
        if prev_tensor is None:
            prev_tensor = curr_tensor

        if blank_tensor is not None:
            lq_curr = curr_tensor - blank_tensor
            lq_prev = prev_tensor - blank_tensor
        else:
            lq_curr = curr_tensor
            lq_prev = prev_tensor

        lq = torch.stack([lq_prev, lq_curr], dim=0).unsqueeze(0).to(device)
        pred = model(lq).squeeze(0).cpu()

        if blank_tensor is not None:
            pred_res_np = pred.clamp(-1, 1).permute(1, 2, 0).numpy()
            pred_abs = np.clip(pred_res_np + blank_tensor.permute(1, 2, 0).numpy(), 0, 1)
            enhanced.append(pred_abs)
            residual_vis.append(_residual_to_vis(pred_res_np))
        else:
            enhanced.append(pred.clamp(0, 1).permute(1, 2, 0).numpy())

        prev_tensor = curr_tensor

    return enhanced, residual_vis


def process_single_video(model, input_path, save_dir, device, save_gt=False,
                         residual=False, video_type='shadow', normal_blank=False):
    frames_uint8, fps = _read_video(input_path)
    if not frames_uint8:
        print(f"  Warning: no frames read from {input_path}, skipping.")
        return

    blank_tensor = None
    if residual:
        if normal_blank:
            # Fixed flat-surface-normal (0,0,1) encoding: the physically
            # correct no-contact reading for tactile_normal videos,
            # independent of any particular frame.
            h, w = frames_uint8[0].shape[:2]
            blank_tensor = _normal_blank(h, w)
        else:
            # Blank is frame 0 of the transferred video itself: always
            # available at real deployment (unlike the query's own touch
            # video, which only exists for paired train/eval data) and
            # already in query coordinate space (unlike the raw reference
            # video).
            blank_tensor = _to_tensor(frames_uint8[0])

    print(f"  Enhancing {input_path}  ({len(frames_uint8)} frames @ {fps:.1f} fps) ...",
          flush=True)
    enhanced, res_vis = enhance_video(model, frames_uint8, device, blank_tensor)

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]

    out_path = os.path.join(save_dir, f"{stem}_enhanced.mp4")
    _write_video(out_path, enhanced, fps=fps)
    print(f"  Saved: {out_path}")

    if res_vis is not None:
        res_path = os.path.join(save_dir, f"{stem}_pred_residual.mp4")
        _write_video(res_path, res_vis, fps=fps)
        print(f"  Saved: {res_path}")

    if save_gt:
        src_dir = os.path.dirname(input_path)
        base = _strip_transferred_suffix(stem)

        # Save the transferred (input) video
        transferred = [f.astype(np.float32) / 255.0 for f in frames_uint8]
        _write_video(os.path.join(save_dir, f"{stem}.mp4"), transferred, fps=fps)

        # Load reference and query videos
        ref_path = os.path.join(src_dir, f"{base}_ref_{video_type}.mp4")
        query_path = os.path.join(src_dir, f"{base}_query_{video_type}.mp4")
        ref_frames = _read_video_frames(ref_path) if os.path.exists(ref_path) else []
        query_frames = _read_video_frames(query_path) if os.path.exists(query_path) else []

        if ref_frames and query_frames:
            _make_grid_video(
                os.path.join(save_dir, f"{base}_grid.mp4"),
                tl=ref_frames, tr=query_frames,
                bl=transferred, br=enhanced, fps=fps)

        if res_vis is not None and query_frames:
            # GT residual: query - blank (blank already loaded above)
            blank_np = blank_tensor.permute(1, 2, 0).numpy()
            gt_res_vis = [_residual_to_vis(np.clip(f - blank_np, -1, 1))
                          for f in query_frames]
            _write_video(os.path.join(save_dir, f"{base}_gt_residual.mp4"),
                         gt_res_vis, fps=fps)


def parse_args():
    p = argparse.ArgumentParser(description="ReBotNet inference — enhance transferred tactile videos")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input_video', help="Path to a single transferred video (.mp4)")
    mode.add_argument('--transfer_dir', help="Root of log/transfer/ (process all objects)")

    p.add_argument('--object_ids', nargs='+', type=int, default=None,
                   help="Subset of object IDs to process (only with --transfer_dir)")
    p.add_argument('--checkpoint', required=True, help="Path to model checkpoint (.pth)")
    p.add_argument('--model_size', default='rebot_S', choices=list(MODEL_CONFIGS),
                   help="Model variant (must match checkpoint)")
    p.add_argument('--save_dir', default='log/rebot_infer',
                   help="Directory to write enhanced videos")
    p.add_argument('--save_gt', action='store_true',
                   help="Also copy the ground-truth query and reference videos alongside enhanced output")
    p.add_argument('--residual', action='store_true',
                   help="Residual mode: subtract blank (frame 0 of the input transferred "
                        "video) before inference, predict refined residual, add blank back "
                        "for absolute output. Also saves *_pred_residual.mp4 visualization.")
    p.add_argument('--video_type', default='shadow',
                   help="Appearance domain of the GT/ref videos loaded with --save_gt: "
                        "{base}_query_{video_type}.mp4 / {base}_ref_{video_type}.mp4. Use "
                        "'tactile_normal' for the surface-normal-encoded domain.")
    p.add_argument('--normal_blank', action='store_true',
                   help="With --residual: use the fixed flat-surface-normal (0,0,1) encoding "
                        "as the blank instead of frame 0 of the input video. Physically "
                        "correct for --video_type tactile_normal.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}  |  Residual mode: {args.residual}")

    # Load model
    model = build_model(args.model_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded {args.model_size} from epoch {ckpt.get('epoch', '?')}"
          f"  (best val PSNR: {ckpt.get('best_psnr', 0):.2f})")

    if args.input_video:
        process_single_video(model, args.input_video, args.save_dir, device,
                             save_gt=args.save_gt, residual=args.residual,
                             video_type=args.video_type, normal_blank=args.normal_blank)

    else:
        # Collect object IDs
        if args.object_ids:
            object_ids = sorted(args.object_ids)
        else:
            object_ids = sorted(
                int(d) for d in os.listdir(args.transfer_dir)
                if os.path.isdir(os.path.join(args.transfer_dir, d))
            )

        print(f"Processing {len(object_ids)} objects from {args.transfer_dir}")

        total = sum(
            1 for obj_id in object_ids
            for pair_idx in range(8)
            if _resolve_lq_path(os.path.join(args.transfer_dir, str(obj_id)), pair_idx)
        )
        done = 0

        for obj_id in object_ids:
            obj_dir = os.path.join(args.transfer_dir, str(obj_id))
            out_dir = os.path.join(args.save_dir, str(obj_id))
            for pair_idx in range(8):
                vid_path = _resolve_lq_path(obj_dir, pair_idx)
                if vid_path is None:
                    continue
                done += 1
                print(f"[{done}/{total}] obj {obj_id} pair {pair_idx}", end='  ', flush=True)
                process_single_video(model, vid_path, out_dir, device,
                                     save_gt=args.save_gt, residual=args.residual,
                                     video_type=args.video_type, normal_blank=args.normal_blank)

    print("\nDone.")


if __name__ == '__main__':
    main()
