"""
Evaluation script for ReBotNet tactile video enhancement.

Runs inference on the test split (objects 951-1000 by default) and reports
MSE, PSNR, SSIM, LPIPS in the same format as main_retrieval_transfer_accel.py.

Usage:
    python rebot-net/eval.py \
        --transfer_dir log/transfer \
        --checkpoint log/rebot_checkpoints/best.pth \
        --model_size rebot_S \
        --save_dir log/rebot_eval \
        --video_save

Residual mode:
    python rebot-net/eval.py \
        --transfer_dir log/transfer \
        --checkpoint log/rebot_checkpoints_residual/best.pth \
        --model_size rebot_S \
        --save_dir log/rebot_eval_residual \
        --video_save --residual
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
from dataset import TactileTransferDataset
from train import MODEL_CONFIGS, build_model
from trainer import Trainer, _write_video, _read_video_frames, _make_grid_video, _residual_to_vis
import cond_utils


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ReBotNet on the test split")
    p.add_argument('--transfer_dir', required=True,
                   help="Root directory of log/transfer")
    p.add_argument('--checkpoint', required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument('--model_size', default='rebot_S',
                   choices=list(MODEL_CONFIGS), help="Model variant (must match checkpoint)")
    p.add_argument('--save_dir', default='log/rebot_eval',
                   help="Directory to save metrics.pkl and optional videos")
    p.add_argument('--video_save', action='store_true',
                   help="Save enhanced test videos to --save_dir/videos/")
    p.add_argument('--save_gt', action='store_true',
                   help="Also copy the ground-truth query and reference videos alongside enhanced output")
    p.add_argument('--residual', action='store_true',
                   help="Evaluate in residual space: LQ and GT are (video - blank), "
                        "blank from frame 0 of the transferred video. "
                        "Metrics are computed on absolute reconstructions.")
    p.add_argument('--video_type', default='shadow',
                   help="Appearance domain of the videos to load: {pair}_query_{video_type}.mp4 "
                        "/ {pair}_ref_{video_type}.mp4, matching "
                        "main_retrieval_transfer_feat_match.py's output naming. Use "
                        "'tactile_normal' for the surface-normal-encoded domain.")
    p.add_argument('--normal_blank', action='store_true',
                   help="With --residual: use the fixed flat-surface-normal (0,0,1) encoding "
                        "as the blank instead of frame 0 of the transferred video. Physically "
                        "correct for --video_type tactile_normal.")
    p.add_argument('--bottleneck_hw', type=int, default=24,
                   help="Must match the checkpoint's training-time --bottleneck_hw (default 24).")
    cond_utils.add_cond_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cond_utils.check_cond_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data split (same logic as train.py) ---
    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    test_ids = all_ids[950:]
    print(f"Test objects: {len(test_ids)}  |  Residual mode: {args.residual}")

    test_dataset = TactileTransferDataset(args.transfer_dir, test_ids, split='test',
                                          residual=args.residual,
                                          video_type=args.video_type,
                                          normal_blank=args.normal_blank,
                                          **cond_utils.dataset_cond_kwargs(args))

    # --- Model ---
    cond_chans, film_chans = cond_utils.cond_dims(args)
    model = build_model(args.model_size, cond_chans, film_chans,
                        bottleneck_hw=args.bottleneck_hw,
                        time_cond=cond_utils.time_cond_mode(args)).to(device)
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

                pred_frames, gt_frames, transferred_frames = [], [], []
                pred_residual_frames, gt_residual_frames = [], []

                for lq_pair, gt_frame, blank, film, t_norm in test_dataset.iter_video_pairs(obj_id, pair_idx):
                    lq_in = lq_pair.unsqueeze(0).to(device)
                    film_in = film.unsqueeze(0).to(device) if film is not None else None
                    t_in = torch.tensor([t_norm], device=device)   # ignored unless model has a time head
                    pred = model(lq_in, film=film_in, t=t_in).squeeze(0)
                    lq_rgb = lq_pair[1, :3]                 # transferred RGB only

                    if args.residual:
                        blank_np = blank.permute(1, 2, 0).numpy()
                        pred_res_np = pred.cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
                        gt_res_np = gt_frame.permute(1, 2, 0).numpy()
                        pred_np = np.clip(pred_res_np + blank_np, 0, 1)
                        gt_np = np.clip(gt_res_np + blank_np, 0, 1)
                        transferred_np = np.clip(
                            lq_rgb.permute(1, 2, 0).numpy() + blank_np, 0, 1)
                        pred_residual_frames.append(_residual_to_vis(pred_res_np))
                        gt_residual_frames.append(_residual_to_vis(gt_res_np))
                    else:
                        pred_np = pred.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        gt_np = gt_frame.permute(1, 2, 0).numpy()
                        transferred_np = lq_rgb.permute(1, 2, 0).numpy()

                    pred_frames.append(pred_np)
                    gt_frames.append(gt_np)
                    transferred_frames.append(transferred_np)

                if not pred_frames:
                    continue

                for pred_np, gt_np in zip(pred_frames, gt_frames):
                    mse = compute_mse(gt_np, pred_np)
                    obj_mse.append(mse)
                    obj_psnr.append(compute_psnr(gt_np, pred_np, data_range=1.0)
                                    if mse > 0 else 100.0)
                    obj_ssim.append(compute_ssim(gt_np, pred_np, data_range=1.0,
                                                 channel_axis=-1))
                    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
                    pr_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
                    obj_lpips.append(lpips_model(gt_t, pr_t).item())

                if video_save_dir:
                    out_path = os.path.join(video_save_dir,
                                            f"{obj_id}_{pair_idx}_enhanced.mp4")
                    _write_video(out_path, pred_frames)

                    if args.residual:
                        _write_video(
                            os.path.join(video_save_dir,
                                         f"{obj_id}_{pair_idx}_pred_residual.mp4"),
                            pred_residual_frames)
                        _write_video(
                            os.path.join(video_save_dir,
                                         f"{obj_id}_{pair_idx}_gt_residual.mp4"),
                            gt_residual_frames)

                    if args.save_gt:
                        _write_video(
                            os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_transferred.mp4"),
                            transferred_frames)
                        ref_path = os.path.join(args.transfer_dir, str(obj_id),
                                                f"{pair_idx}_ref_{args.video_type}.mp4")
                        ref_frames = _read_video_frames(ref_path) if os.path.exists(ref_path) else []
                        if ref_frames:
                            _make_grid_video(
                                os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_grid.mp4"),
                                tl=ref_frames, tr=gt_frames,
                                bl=transferred_frames, br=pred_frames)

            if not obj_mse:
                continue

            m = {
                'MSE': float(np.mean(obj_mse)),
                'PSNR': float(np.mean(obj_psnr)),
                'SSIM': float(np.mean(obj_ssim)),
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
        'MSE': float(np.mean(all_mse)),
        'PSNR': float(np.mean(all_psnr)),
        'SSIM': float(np.mean(all_ssim)),
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
