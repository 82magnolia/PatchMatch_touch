"""
Training script for ReBotNet normal-image baseline.

Uses a static surface normal image at the query touch location as input
(tiled into a video) instead of PatchMatch-transferred frames.

Usage:
    python rebot_net/train_normal.py \
        --transfer_dir log/transfer \
        --query_normal_dir Taxim/results/gen_contact_full_query \
        --save_dir log/rebot_checkpoints_normal \
        --model_size rebot_S \
        --epochs 100 \
        --batch_size 4 \
        --lr 2e-4 \
        --num_workers 4 \
        --wandb_project tactile_enhance \
        --wandb_run_name normal_baseline_S
"""

import argparse
import os
import sys

import lpips
import torch
import torch.utils.data as data
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from dataset_normal import NormalBaselineDataset
from train import MODEL_CONFIGS, build_model
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(
        description="Train ReBotNet normal-image baseline for tactile video enhancement")
    p.add_argument('--transfer_dir', required=True,
                   help="Root directory of log/transfer (contains per-object GT videos)")
    p.add_argument('--query_normal_dir', required=True,
                   help="Root of gen_contact_full_query (contains per-object normal images)")
    p.add_argument('--save_dir', default='log/rebot_checkpoints_normal',
                   help="Directory to save checkpoints")
    p.add_argument('--model_size', default='rebot_S',
                   choices=list(MODEL_CONFIGS), help="Model variant")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--log_interval', type=int, default=10,
                   help="Log training loss every log_interval steps")
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', default=None, help="Path to checkpoint to resume from")
    p.add_argument('--video_save', action='store_true',
                   help="Save enhanced validation videos each epoch")
    p.add_argument('--wandb_project', default='tactile_enhance')
    p.add_argument('--wandb_run_name', default=None)
    p.add_argument('--wandb_offline', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data split (keyed on transfer_dir object IDs) ---
    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    train_ids = all_ids[:930]
    val_ids   = all_ids[930:950]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val objects")

    train_dataset = NormalBaselineDataset(
        args.query_normal_dir, args.transfer_dir, train_ids, split='train')
    val_dataset = NormalBaselineDataset(
        args.query_normal_dir, args.transfer_dir, val_ids, split='val')
    print(f"Train samples: {len(train_dataset)}, Val objects: {len(val_ids)}")

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- Model ---
    model = build_model(args.model_size).to(device)
    print(f"Model: {args.model_size}  |  Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    trainer = Trainer(model, train_loader, val_dataset, val_ids,
                      optimizer, scheduler, lpips_model, device, args)

    # --- Resume ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_psnr = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch-1}, best PSNR={best_psnr:.2f}")

    # --- Wandb ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"normal_{args.model_size}_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
        mode='offline' if args.wandb_offline else 'online',
    )

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss = trainer.train_epoch(epoch)
        print(f"Train loss: {train_loss:.5f}")

        video_save_dir = None
        if args.video_save:
            video_save_dir = os.path.join(args.save_dir, 'videos', f'epoch_{epoch+1:04d}')

        val_metrics = trainer.evaluate(val_ids, val_dataset, epoch, 'val',
                                       video_save_dir=video_save_dir)
        trainer.log_metrics(val_metrics, epoch, 'val')
        trainer.print_metrics(val_metrics, 'val')

        trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pth'), epoch, best_psnr)

        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best.pth'), epoch, best_psnr)
            print(f"  ** New best PSNR: {best_psnr:.2f} — checkpoint saved **")

    wandb.finish()
    print(f"\nTraining complete. Best val PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
