"""
Evaluate a (fine-tuned) ReBotNet on the real-data eval split, recording BOTH the
before (transferred, pre-enhancement) and after (enhanced) metrics against the
ground-truth query video, so the improvement is directly quantifiable.

Split matches rebot_net/finetune.py: objects are sorted numerically and the first
--num_eval (default 20) are the held-out eval set.

Usage:
    python rebot_net/eval_real.py \
        --transfer_dir log/transfer_pipeline_real_data_gt_retrieval_superpoint_superglue \
        --checkpoint   log/rebot_finetune_S_real_data_gt_retrieval_superpoint_superglue_full/best.pth \
        --model_size   rebot_S \
        --save_dir     log/rebot_finetune_eval_superpoint_superglue/nonresidual \
        --video_save --save_gt

Add --residual for a residual-mode checkpoint.
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
from dataset_real import RealTactileTransferDataset
from train import MODEL_CONFIGS, build_model
from trainer import _write_video, _read_video_frames, _make_grid_video, _residual_to_vis
import cond_utils


def _frame_metrics(gt_np, pred_np, lpips_model, device):
    mse = compute_mse(gt_np, pred_np)
    psnr = compute_psnr(gt_np, pred_np, data_range=1.0) if mse > 0 else 100.0
    ssim = compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=-1)
    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    pr_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    lp = lpips_model(gt_t, pr_t).item()
    return mse, psnr, ssim, lp


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ReBotNet on the real-data eval split (before/after)")
    p.add_argument('--transfer_dir', required=True,
                   help="Root of the real-data transfer pipeline output")
    p.add_argument('--checkpoint', required=True, help="Path to model checkpoint (.pth)")
    p.add_argument('--model_size', default='rebot_S', choices=list(MODEL_CONFIGS),
                   help="Model variant (must match checkpoint)")
    p.add_argument('--save_dir', default='log/rebot_finetune_eval',
                   help="Directory for metrics.pkl and optional videos")
    p.add_argument('--num_eval', type=int, default=20,
                   help="Number of leading sorted objects used as the eval set")
    p.add_argument('--video_save', action='store_true',
                   help="Save enhanced eval videos to --save_dir/videos/")
    p.add_argument('--save_gt', action='store_true',
                   help="Also save transferred/GT and a 2x2 grid video")
    p.add_argument('--residual', action='store_true',
                   help="Residual-mode checkpoint (metrics still on absolute reconstructions)")
    p.add_argument('--video_type', default='shadow',
                   help="Appearance domain of the videos to load: {pair}_query_{video_type}.mp4 "
                        "/ {pair}_ref_{video_type}.mp4. Use 'tactile_normal' for the "
                        "surface-normal-encoded domain (must match the checkpoint's domain).")
    p.add_argument('--normal_blank', action='store_true',
                   help="With --residual: use the fixed flat-surface-normal (0,0,1) encoding "
                        "as the blank instead of frame 0 of the transferred video (physically "
                        "correct for --video_type tactile_normal). Must match training.")
    cond_utils.add_cond_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cond_utils.check_cond_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    eval_ids = all_ids[:args.num_eval]
    print(f"Eval objects: {len(eval_ids)}  |  Residual mode: {args.residual}")

    dataset = RealTactileTransferDataset(args.transfer_dir, eval_ids, split='test',
                                         residual=args.residual,
                                         video_type=args.video_type,
                                         normal_blank=args.normal_blank,
                                         **cond_utils.dataset_cond_kwargs(args))

    cond_chans, film_chans = cond_utils.cond_dims(args)
    model = build_model(args.model_size, cond_chans, film_chans).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(best val PSNR: {ckpt.get('best_psnr', 0):.2f})")

    lpips_model = lpips.LPIPS(net='alex').to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    video_save_dir = os.path.join(args.save_dir, 'videos') if args.video_save else None
    if video_save_dir:
        os.makedirs(video_save_dir, exist_ok=True)

    per_object = {}          # obj_id -> {'before': {...}, 'after': {...}}
    agg = {'before': {k: [] for k in ('MSE', 'PSNR', 'SSIM', 'LPIPS')},
           'after':  {k: [] for k in ('MSE', 'PSNR', 'SSIM', 'LPIPS')}}

    print("\nEvaluating...\n" + "-" * 60)
    with torch.no_grad():
        for obj_id in eval_ids:
            obj = {'before': {k: [] for k in agg['before']},
                   'after':  {k: [] for k in agg['after']}}
            for pair_idx in range(dataset.NUM_PAIRS):
                if not dataset.lq_video_exists(obj_id, pair_idx):
                    continue
                print(f"  Object {obj_id}  contact {pair_idx}", flush=True)

                pred_frames, gt_frames, transferred_frames = [], [], []
                for lq_pair, gt_frame, blank, film, t_norm in dataset.iter_video_pairs(obj_id, pair_idx):
                    film_in = film.unsqueeze(0).to(device) if film is not None else None
                    t_in = torch.tensor([t_norm], device=device)   # ignored unless model has a time head
                    pred = model(lq_pair.unsqueeze(0).to(device), film=film_in, t=t_in).squeeze(0)
                    lq_rgb = lq_pair[1, :3]                 # transferred RGB only
                    if args.residual:
                        blank_np = blank.permute(1, 2, 0).numpy()
                        pred_np = np.clip(pred.cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + blank_np, 0, 1)
                        gt_np = np.clip(gt_frame.permute(1, 2, 0).numpy() + blank_np, 0, 1)
                        transferred_np = np.clip(lq_rgb.permute(1, 2, 0).numpy() + blank_np, 0, 1)
                    else:
                        pred_np = pred.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        gt_np = gt_frame.permute(1, 2, 0).numpy()
                        transferred_np = lq_rgb.permute(1, 2, 0).numpy()
                    pred_frames.append(pred_np)
                    gt_frames.append(gt_np)
                    transferred_frames.append(transferred_np)

                if not pred_frames:
                    continue

                for pred_np, tr_np, gt_np in zip(pred_frames, transferred_frames, gt_frames):
                    for tag, src in (('after', pred_np), ('before', tr_np)):
                        mse, psnr, ssim, lp = _frame_metrics(gt_np, src, lpips_model, device)
                        obj[tag]['MSE'].append(mse); obj[tag]['PSNR'].append(psnr)
                        obj[tag]['SSIM'].append(ssim); obj[tag]['LPIPS'].append(lp)

                if video_save_dir:
                    _write_video(os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_enhanced.mp4"), pred_frames)
                    if args.save_gt:
                        _write_video(os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_transferred.mp4"), transferred_frames)
                        _write_video(os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_query.mp4"), gt_frames)
                        ref_path = os.path.join(dataset._obj_dir(obj_id),
                                                f"{pair_idx}_ref_{args.video_type}.mp4")
                        ref_frames = _read_video_frames(ref_path) if os.path.exists(ref_path) else []
                        if ref_frames:
                            _make_grid_video(
                                os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_grid.mp4"),
                                tl=ref_frames, tr=gt_frames, bl=transferred_frames, br=pred_frames)

            if not obj['after']['MSE']:
                continue
            summ = {tag: {k: float(np.mean(v)) for k, v in obj[tag].items()} for tag in obj}
            per_object[obj_id] = summ
            for tag in agg:
                for k in agg[tag]:
                    agg[tag][k].append(summ[tag][k])
            b, a = summ['before'], summ['after']
            print(f"  Object {obj_id:4d} — PSNR {b['PSNR']:.2f} -> {a['PSNR']:.2f}   "
                  f"SSIM {b['SSIM']:.4f} -> {a['SSIM']:.4f}   LPIPS {b['LPIPS']:.4f} -> {a['LPIPS']:.4f}")

    average = {tag: {k: float(np.mean(v)) for k, v in agg[tag].items()} for tag in agg}

    print("\n" + "=" * 60)
    print(f"Average over {len(per_object)} objects  (before -> after)")
    for k in ('MSE', 'PSNR', 'SSIM', 'LPIPS'):
        print(f"  {k:6s} {average['before'][k]:.5f} -> {average['after'][k]:.5f}")

    metrics_out = {'per_object': per_object, 'average': average,
                   'residual': args.residual, 'checkpoint': args.checkpoint,
                   'model_size': args.model_size}
    metrics_path = os.path.join(args.save_dir, 'metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_out, f)
    print(f"\nMetrics saved to: {metrics_path}")
    if video_save_dir:
        print(f"Videos saved to: {video_save_dir}")


if __name__ == '__main__':
    main()
