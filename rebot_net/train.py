"""
Training script for ReBotNet tactile video enhancement.

Usage:
    python rebot-net/train.py \
        --transfer_dir log/transfer \
        --save_dir log/rebot_checkpoints \
        --model_size rebot_S \
        --epochs 100 \
        --batch_size 4 \
        --lr 2e-4 \
        --wandb_project tactile_enhance
"""

import argparse
import os
import sys

import lpips
import torch
import torch.utils.data as data
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from models.archs import rebotnet as ReBotNet
from dataset import TactileTransferDataset
from dataset_real import RealTactileTransferDataset
from trainer import Trainer
import cond_utils


MODEL_CONFIGS = {
    'rebot_XS': dict(upscale=1, img_size=[2,384,384], window_size=[6,8,8],
                     depths=[4,3,3,4], indep_reconsts=[9,10],
                     embed_dims=[8,12,20,32],
                     num_heads=[6]*13, pa_frames=2, deformable_groups=12,
                     mlp_dim=64, bottle_depth=4, bottle_dim=32,
                     dropout=0.1, patch_size=1),
    'rebot_S':  dict(upscale=1, img_size=[2,384,384], window_size=[6,8,8],
                     depths=[4,4,4,5], indep_reconsts=[9,10],
                     embed_dims=[28,36,48,64],
                     num_heads=[6]*13, pa_frames=2, deformable_groups=12,
                     mlp_dim=256, bottle_depth=4, bottle_dim=64,
                     dropout=0.1, patch_size=1),
    'rebot_M':  dict(upscale=1, img_size=[2,384,384], window_size=[6,8,8],
                     depths=[3,3,3,3], indep_reconsts=[9,10],
                     embed_dims=[64,80,108,116],
                     num_heads=[6]*13, pa_frames=2, deformable_groups=12,
                     mlp_dim=256, bottle_depth=4, bottle_dim=116,
                     dropout=0.1, patch_size=1),
    'rebot_L':  dict(upscale=1, img_size=[2,384,384], window_size=[6,8,8],
                     depths=[5,5,5,5], indep_reconsts=[9,10],
                     embed_dims=[128,224,280,320],
                     num_heads=[6]*13, pa_frames=2, deformable_groups=12,
                     mlp_dim=512, bottle_depth=4, bottle_dim=320,
                     dropout=0.1, patch_size=1),
}


def build_model(model_size, cond_chans=0, film_chans=0):
    cfg = MODEL_CONFIGS[model_size]
    return ReBotNet(**cfg, cond_chans=cond_chans, film_chans=film_chans)


def parse_args():
    p = argparse.ArgumentParser(description="Train ReBotNet for tactile video enhancement")
    p.add_argument('--transfer_dir', required=True,
                   help="Root directory of log/transfer (contains per-object folders)")
    p.add_argument('--save_dir', default='log/rebot_checkpoints',
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
    p.add_argument('--residual', action='store_true',
                   help="Train in residual space: subtract blank (frame 0 of the transferred "
                        "video) from LQ and GT; model predicts refined residuals, added back "
                        "to blank for absolute output")
    p.add_argument('--real_data', action='store_true',
                   help="Train from scratch on the real-data transfer tree "
                        "(RealTactileTransferDataset: nested {obj}/transfer/ layout, odd query "
                        "indices). Uses the same first-num_eval / rest split as finetune.py so a "
                        "from-scratch run is directly comparable to a fine-tune on identical data.")
    p.add_argument('--num_eval', type=int, default=20,
                   help="With --real_data: number of leading (sorted) objects held out for "
                        "validation; the rest are used for training.")
    cond_utils.add_cond_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cond_utils.check_cond_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data split ---
    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    # Real-data from-scratch runs mirror finetune.py's split (first num_eval held
    # out) so they are a controlled comparison; the sim path keeps its 930/20 split.
    Dataset = RealTactileTransferDataset if args.real_data else TactileTransferDataset
    if args.real_data:
        val_ids   = all_ids[:args.num_eval]
        train_ids = all_ids[args.num_eval:]
    else:
        train_ids = all_ids[:930]
        val_ids   = all_ids[930:950]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val objects")

    cond_kw = cond_utils.dataset_cond_kwargs(args)
    train_dataset = Dataset(args.transfer_dir, train_ids, split='train',
                            residual=args.residual, **cond_kw)
    val_dataset   = Dataset(args.transfer_dir, val_ids,   split='val',
                            residual=args.residual, **cond_kw)
    print(f"Train samples: {len(train_dataset)}, Val objects: {len(val_ids)}")

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- Model ---
    cond_chans, film_chans = cond_utils.cond_dims(args)
    model = build_model(args.model_size, cond_chans, film_chans).to(device)
    print(f"Model: {args.model_size}  |  Device: {device}  |  "
          f"cond_chans={cond_chans} film_chans={film_chans}")

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
        name=args.wandb_run_name or f"{args.model_size}_bs{args.batch_size}_lr{args.lr}",
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

        # Save latest
        trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pth'), epoch, best_psnr)

        # Save best
        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best.pth'), epoch, best_psnr)
            print(f"  ** New best PSNR: {best_psnr:.2f} — checkpoint saved **")

    wandb.finish()
    print(f"\nTraining complete. Best val PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
