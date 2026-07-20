"""
Fine-tune a pre-trained ReBotNet on real-data (transferred → query) pairs.

Loads synthetic-pretrained weights (e.g. produced by train.py and stored under
log/rebot_checkpoints_S_pseudo_mini_residual_superpoint_superglue) and adapts
them to the real GelSight domain using the paired videos emitted by the
real-data transfer pipeline (e.g.
log/transfer_pipeline_real_data_gt_retrieval_superpoint_superglue).

Objects are sorted numerically; the first --num_eval objects are held out for
evaluation and the remaining objects are used for fine-tuning (default: eval on
the first 20, fine-tune on the last 80 of the 100 objects).

Which layers to fine-tune (--finetune_mode):
    full                every parameter (largest capacity, most overfit risk).
    decoder_bottleneck  (default) freeze the ConvNeXt encoder; train the
                        bottleneck transformers + reconstruction decoder. The
                        encoder's low-level tactile features transfer well
                        across the sim→real gap, while the real sensor's
                        appearance/color statistics are absorbed by the global
                        bottleneck and the decoder that writes the RGB output.
    decoder             train only the reconstruction decoder (upsample/chchange
                        /conv_last) — the lightest adaptation.
    last                train only the final conv_last output layer.

Usage:
    conda activate pm_touch
    python rebot_net/finetune.py \
        --pretrained   log/rebot_checkpoints_S_pseudo_mini_residual_superpoint_superglue \
        --transfer_dir log/transfer_pipeline_real_data_gt_retrieval_superpoint_superglue \
        --save_dir     log/rebot_finetune_S_real_superpoint_superglue \
        --model_size   rebot_S \
        --residual \
        --finetune_mode decoder_bottleneck \
        --epochs 8 --batch_size 4 --lr 5e-5 \
        --wandb_run_name finetune_real_S
"""

import argparse
import os
import random
import sys

import lpips
import torch
import torch.utils.data as data
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from train import MODEL_CONFIGS, build_model
from dataset_real import RealTactileTransferDataset
from trainer import Trainer
import cond_utils


# Parameter-name prefixes for each functional block of the network.
ENCODER_PREFIXES = ('downsample_layers', 'stages',
                    'norm0', 'norm1', 'norm2', 'norm3')
BOTTLENECK_PREFIXES = ('to_patch_embedding', 'big_embedding1', 'big_embedding2',
                       'bottleneck', 'temporal_transformer', 'norm.',
                       'conv_after_body')
DECODER_PREFIXES = ('upsample1', 'upsample2', 'upsample3', 'upsample4',
                    'upsamplef1', 'upsamplef2', 'chchange1', 'chchange2',
                    'chchange3', 'conv_last')
# The FiLM conditioning encoder is always fine-tuned (it is the whole point of
# conditioning); it is frozen at identity from pretraining otherwise.
FILM_PREFIXES = ('film_encoder',)

FINETUNE_TRAINABLE = {
    'full':               ENCODER_PREFIXES + BOTTLENECK_PREFIXES + DECODER_PREFIXES + FILM_PREFIXES,
    'decoder_bottleneck': BOTTLENECK_PREFIXES + DECODER_PREFIXES + FILM_PREFIXES,
    'decoder':            DECODER_PREFIXES + FILM_PREFIXES,
    'last':               ('conv_last',) + FILM_PREFIXES,
}


def set_trainable(model, mode):
    """Freeze/unfreeze parameters per finetune mode; return list of trainable params."""
    trainable_prefixes = FINETUNE_TRAINABLE[mode]
    trainable = []
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if name.startswith(trainable_prefixes):
            p.requires_grad_(True)
            trainable.append(p)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
    print(f"Finetune mode '{mode}': {n_train:,}/{n_total:,} params trainable "
          f"({100.0 * n_train / n_total:.1f}%)")
    return trainable


def load_pretrained(model, pretrained, device):
    """Load model weights from a checkpoint file or a directory containing best.pth."""
    path = pretrained
    if os.path.isdir(path):
        path = os.path.join(path, 'best.pth')
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    print(f"Loaded pretrained weights from {path} "
          f"(epoch {ckpt.get('epoch', '?')}, best PSNR {ckpt.get('best_psnr', 0):.2f})")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ReBotNet on real-data tactile pairs")
    p.add_argument('--pretrained', required=True,
                   help="Pretrained checkpoint (.pth) or a directory containing best.pth")
    p.add_argument('--transfer_dir', required=True,
                   help="Root of the real-data transfer pipeline output "
                        "(contains {obj_id}/transfer/*.mp4)")
    p.add_argument('--save_dir', default='log/rebot_finetune',
                   help="Directory to save fine-tuned checkpoints")
    p.add_argument('--model_size', default='rebot_S',
                   choices=list(MODEL_CONFIGS), help="Model variant (must match --pretrained)")
    p.add_argument('--finetune_mode', default='decoder_bottleneck',
                   choices=list(FINETUNE_TRAINABLE),
                   help="Which layers to fine-tune")
    p.add_argument('--num_eval', type=int, default=20,
                   help="Number of leading (sorted) objects held out for evaluation; "
                        "the rest are used for fine-tuning")
    p.add_argument('--max_train_objects', type=int, default=None,
                   help="Cap the number of fine-tuning objects (for short runs)")
    p.add_argument('--max_eval_objects', type=int, default=None,
                   help="Cap the number of evaluation objects (for short runs)")
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--log_interval', type=int, default=10,
                   help="Log training loss every log_interval steps")
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    p.add_argument('--resume', default=None,
                   help="Path to a fine-tuning checkpoint to resume from")
    p.add_argument('--video_save', action='store_true',
                   help="Save enhanced eval videos each epoch")
    p.add_argument('--save_gt', action='store_true',
                   help="With --video_save, also save transferred/GT/grid videos")
    p.add_argument('--residual', action='store_true',
                   help="Residual space (match the pretrained checkpoint's training mode)")
    p.add_argument('--wandb_project', default='tactile_enhance')
    p.add_argument('--wandb_run_name', default=None)
    p.add_argument('--wandb_offline', action='store_true')
    cond_utils.add_cond_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cond_utils.check_cond_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data split: first num_eval objects for eval, remainder for fine-tuning ---
    all_ids = sorted(int(d) for d in os.listdir(args.transfer_dir)
                     if os.path.isdir(os.path.join(args.transfer_dir, d)))
    eval_ids = all_ids[:args.num_eval]
    train_ids = all_ids[args.num_eval:]
    if args.max_eval_objects is not None:
        eval_ids = eval_ids[:args.max_eval_objects]
    if args.max_train_objects is not None:
        train_ids = train_ids[:args.max_train_objects]
    print(f"Split: {len(train_ids)} fine-tune, {len(eval_ids)} eval objects")

    cond_kw = cond_utils.dataset_cond_kwargs(args)
    train_dataset = RealTactileTransferDataset(args.transfer_dir, train_ids, split='train',
                                               residual=args.residual, **cond_kw)
    eval_dataset = RealTactileTransferDataset(args.transfer_dir, eval_ids, split='val',
                                              residual=args.residual, **cond_kw)
    print(f"Fine-tune frame samples: {len(train_dataset)}, eval objects: {len(eval_ids)}")

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- Model: build, load pretrained weights, then freeze per finetune mode ---
    # Same cond dims as sim pretraining, so the checkpoint loads with no surgery.
    cond_chans, film_chans = cond_utils.cond_dims(args)
    model = build_model(args.model_size, cond_chans, film_chans).to(device)
    load_pretrained(model, args.pretrained, device)
    trainable_params = set_trainable(model, args.finetune_mode)
    print(f"Model: {args.model_size}  |  Device: {device}  |  "
          f"cond_chans={cond_chans} film_chans={film_chans}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Flat LR for fine-tuning: eval PSNR plateaus after ~1 epoch, so a constant
    # rate is used instead of cosine annealing. LambdaLR with a unit multiplier
    # keeps the scheduler.step()/state-dict interface a no-op.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    trainer = Trainer(model, train_loader, eval_dataset, eval_ids,
                      optimizer, scheduler, lpips_model, device, args)

    # --- Resume (fine-tuning run only; pretrained weights already loaded above) ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_psnr = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        print(f"Resumed fine-tuning from epoch {start_epoch-1}, best PSNR={best_psnr:.2f}")

    # --- Wandb ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"finetune_{args.model_size}_{args.finetune_mode}_lr{args.lr}",
        config=vars(args),
        mode='offline' if args.wandb_offline else 'online',
    )

    # --- Baseline eval before any fine-tuning (transfers as-is) ---
    print("\n=== Epoch 0 (pretrained baseline) ===")
    base_metrics = trainer.evaluate(eval_ids, eval_dataset, -1, 'val')
    trainer.log_metrics(base_metrics, -1, 'val')
    trainer.print_metrics(base_metrics, 'val')
    best_psnr = max(best_psnr, base_metrics['PSNR'])

    # --- Fine-tuning loop ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss = trainer.train_epoch(epoch)
        print(f"Train loss: {train_loss:.5f}")

        video_save_dir = None
        if args.video_save:
            video_save_dir = os.path.join(args.save_dir, 'videos', f'epoch_{epoch+1:04d}')

        val_metrics = trainer.evaluate(eval_ids, eval_dataset, epoch, 'val',
                                       video_save_dir=video_save_dir)
        trainer.log_metrics(val_metrics, epoch, 'val')
        trainer.print_metrics(val_metrics, 'val')

        trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pth'), epoch, best_psnr)

        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best.pth'), epoch, best_psnr)
            print(f"  ** New best PSNR: {best_psnr:.2f} — checkpoint saved **")

    wandb.finish()
    print(f"\nFine-tuning complete. Best eval PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
