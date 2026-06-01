"""
Evaluation script for ReBotNet normal-image baseline.

Runs inference on the test split using static surface normal images as input
and reports MSE, PSNR, SSIM, LPIPS.

Usage:
    python rebot_net/eval_normal.py \
        --transfer_dir log/transfer \
        --query_normal_dir Taxim/results/gen_contact_full_query \
        --checkpoint log/rebot_checkpoints_normal/best.pth \
        --model_size rebot_S \
        --save_dir log/rebot_eval_normal \
        --video_save \
        --save_gt
"""

import argparse
import os
import pickle
import sys

import lpips
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

sys.path.insert(0, os.path.dirname(__file__))
from dataset_normal import NormalBaselineDataset
from train import MODEL_CONFIGS, build_model
from trainer import _write_video, _make_grid_video


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ReBotNet normal-image baseline on the test split")
    p.add_argument('--transfer_dir', required=True,
                   help="Root directory of log/transfer (contains per-object GT videos)")
    p.add_argument('--query_normal_dir', required=True,
                   help="Root of gen_contact_full_query (contains per-object normal images)")
    p.add_argument('--checkpoint', required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument('--model_size', default='rebot_S',
                   choices=list(MODEL_CONFIGS), help="Model variant (must match checkpoint)")
    p.add_argument('--save_dir', default='log/rebot_eval_normal',
                   help="Directory to save metrics.pkl and optional videos")
    p.add_argument('--video_save', action='store_true',
                   help="Save enhanced test videos to --save_dir/videos/")
    p.add_argument('--save_gt', action='store_true',
                   help="Also save normal-image and GT videos alongside enhanced output")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data split ---
    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    test_ids = all_ids[950:]
    print(f"Test objects: {len(test_ids)}")

    test_dataset = NormalBaselineDataset(
        args.query_normal_dir, args.transfer_dir, test_ids, split='test')

    # --- Model ---
    model = build_model(args.model_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(best val PSNR: {ckpt.get('best_psnr', 0):.2f})")

    lpips_model = lpips.LPIPS(net='alex').to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    video_save_dir = os.path.join(args.save_dir, 'videos') if args.video_save else None
    if video_save_dir:
        os.makedirs(video_save_dir, exist_ok=True)

    per_object = {}
    all_mse, all_psnr, all_ssim, all_lpips_vals = [], [], [], []

    model.eval()
    print("\nEvaluating test objects...")
    print("-" * 60)

    with torch.no_grad():
        for obj_id in test_ids:
            obj_mse, obj_psnr, obj_ssim, obj_lpips = [], [], [], []

            for pair_idx in range(test_dataset.NUM_PAIRS):
                if not test_dataset.lq_video_exists(obj_id, pair_idx):
                    continue

                print(f"  Object {obj_id}  contact {pair_idx}", flush=True)

                pred_frames, gt_frames, normal_frames = [], [], []
                for lq_pair, gt_frame in test_dataset.iter_video_pairs(obj_id, pair_idx):
                    lq_in = lq_pair.unsqueeze(0).to(device)
                    pred = model(lq_in).squeeze(0)
                    pred_np = pred.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    gt_np = gt_frame.permute(1, 2, 0).numpy()
                    normal_np = lq_pair[1].permute(1, 2, 0).numpy()
                    pred_frames.append(pred_np)
                    gt_frames.append(gt_np)
                    normal_frames.append(normal_np)

                if not pred_frames:
                    continue

                for pred_np, gt_np in zip(pred_frames, gt_frames):
                    mse = compute_mse(gt_np, pred_np)
                    obj_mse.append(mse)
                    obj_psnr.append(compute_psnr(gt_np, pred_np, data_range=1.0)
                                    if mse > 0 else 100.0)
                    obj_ssim.append(compute_ssim(gt_np, pred_np, data_range=1.0,
                                                 channel_axis=-1))
                    gt_t = (torch.from_numpy(gt_np).permute(2, 0, 1)
                            .unsqueeze(0).to(device) * 2 - 1)
                    pr_t = (torch.from_numpy(pred_np).permute(2, 0, 1)
                            .unsqueeze(0).to(device) * 2 - 1)
                    obj_lpips.append(lpips_model(gt_t, pr_t).item())

                if video_save_dir:
                    out_path = os.path.join(video_save_dir,
                                            f"{obj_id}_{pair_idx}_enhanced.mp4")
                    _write_video(out_path, pred_frames)

                    if args.save_gt:
                        _write_video(
                            os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_normal.mp4"),
                            normal_frames)
                        _make_grid_video(
                            os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_grid.mp4"),
                            tl=normal_frames, tr=gt_frames,
                            bl=normal_frames, br=pred_frames,
                            labels=("Normal", "GT", "Normal", "Predicted"))

            if not obj_mse:
                continue

            m = {
                'MSE':   float(np.mean(obj_mse)),
                'PSNR':  float(np.mean(obj_psnr)),
                'SSIM':  float(np.mean(obj_ssim)),
                'LPIPS': float(np.mean(obj_lpips)),
            }
            per_object[obj_id] = m
            all_mse.append(m['MSE'])
            all_psnr.append(m['PSNR'])
            all_ssim.append(m['SSIM'])
            all_lpips_vals.append(m['LPIPS'])

            print(f"  Object {obj_id:4d} — "
                  f"MSE: {m['MSE']:.5f}  PSNR: {m['PSNR']:.2f}  "
                  f"SSIM: {m['SSIM']:.4f}  LPIPS: {m['LPIPS']:.4f}")

    # --- Aggregate ---
    avg = {
        'MSE':   float(np.mean(all_mse)),
        'PSNR':  float(np.mean(all_psnr)),
        'SSIM':  float(np.mean(all_ssim)),
        'LPIPS': float(np.mean(all_lpips_vals)),
    }

    print("\n" + "=" * 60)
    print(f"Average ({len(per_object)} objects)")
    print("MSE\t\tPSNR\tSSIM\tLPIPS")
    print(f"{avg['MSE']:.5f}\t{avg['PSNR']:.2f}\t{avg['SSIM']:.4f}\t{avg['LPIPS']:.4f}")

    # --- Save ---
    metrics_out = {'per_object': per_object, 'average': avg}
    metrics_path = os.path.join(args.save_dir, 'metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_out, f)
    print(f"\nMetrics saved to: {metrics_path}")
    if video_save_dir:
        print(f"Enhanced videos saved to: {video_save_dir}")


if __name__ == '__main__':
    main()
