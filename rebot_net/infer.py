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
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import MODEL_CONFIGS, build_model
from trainer import _write_video


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
def enhance_video(model, frames_rgb, device):
    """Enhance a list of (H,W,3) uint8 frames; returns float32 [0,1] frames."""
    enhanced = []
    prev_tensor = None
    for frame in frames_rgb:
        curr_tensor = _to_tensor(frame)
        if prev_tensor is None:
            prev_tensor = curr_tensor
        lq = torch.stack([prev_tensor, curr_tensor], dim=0).unsqueeze(0).to(device)
        pred = model(lq).squeeze(0).cpu().clamp(0, 1)
        enhanced.append(pred.permute(1, 2, 0).numpy())
        prev_tensor = curr_tensor
    return enhanced


def process_single_video(model, input_path, save_dir, device, save_gt=False):
    frames, fps = _read_video(input_path)
    if not frames:
        print(f"  Warning: no frames read from {input_path}, skipping.")
        return

    print(f"  Enhancing {input_path}  ({len(frames)} frames @ {fps:.1f} fps) ...", flush=True)
    enhanced = enhance_video(model, frames, device)

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(save_dir, f"{stem}_enhanced.mp4")
    _write_video(out_path, enhanced, fps=fps)
    print(f"  Saved: {out_path}")

    if save_gt:
        src_dir = os.path.dirname(input_path)
        base = stem.replace('_transferred_em', '')
        for suffix in ('query_shadow', 'ref_shadow'):
            src = os.path.join(src_dir, f"{base}_{suffix}.mp4")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_dir, f"{base}_{suffix}.mp4"))


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
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = build_model(args.model_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded {args.model_size} from epoch {ckpt.get('epoch', '?')}"
          f"  (best val PSNR: {ckpt.get('best_psnr', 0):.2f})")

    if args.input_video:
        process_single_video(model, args.input_video, args.save_dir, device,
                             save_gt=args.save_gt)

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
            if os.path.exists(os.path.join(args.transfer_dir, str(obj_id),
                                           f"{pair_idx}_transferred_em.mp4"))
        )
        done = 0

        for obj_id in object_ids:
            obj_dir = os.path.join(args.transfer_dir, str(obj_id))
            out_dir = os.path.join(args.save_dir, str(obj_id))
            for pair_idx in range(8):
                vid_path = os.path.join(obj_dir, f"{pair_idx}_transferred_em.mp4")
                if not os.path.exists(vid_path):
                    continue
                done += 1
                print(f"[{done}/{total}] obj {obj_id} pair {pair_idx}", end='  ', flush=True)
                process_single_video(model, vid_path, out_dir, device,
                                     save_gt=args.save_gt)

    print("\nDone.")


if __name__ == '__main__':
    main()
