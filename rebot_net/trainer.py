import os

import cv2
import numpy as np
import torch
import wandb
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def _charbonnier_loss(pred, gt, eps=1e-6):
    return (pred - gt).pow(2).add(eps).sqrt().mean()


def _write_video(path, frames_rgb_float, fps=5.0):
    """Write list of (H,W,3) float32 [0,1] frames to an MP4 file."""
    h, w = frames_rgb_float[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames_rgb_float:
        f_uint8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(f_uint8, cv2.COLOR_RGB2BGR))
    out.release()


def _residual_to_vis(residual_np):
    """Map residual frame from [-1,1] to [0,1] for visualization."""
    return np.clip(residual_np * 0.5 + 0.5, 0, 1)


def _read_video_frames(path):
    """Read an MP4 into a list of float32 [0,1] RGB (H,W,3) arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return frames


def _make_grid_video(path, tl, tr, bl, br, labels=("Reference", "Query", "Transferred", "Refined"), fps=5.0):
    """Write a 2×2 grid MP4 from four float32 [0,1] RGB frame lists.

    Layout:
        | tl (top-left)  | tr (top-right)  |
        | bl (bottom-left)| br (bottom-right)|
    """
    n = min(len(tl), len(tr), len(bl), len(br))
    if n == 0:
        return

    h, w = tl[0].shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    pad = 4

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w * 2, h * 2))

    for i in range(n):
        cells = []
        for frame, label in zip([tl[i], tr[i], bl[i], br[i]], labels):
            cell = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            cell = cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(cell, (pad, pad), (pad + tw + pad, pad + th + pad), (0, 0, 0), -1)
            cv2.putText(cell, label, (pad * 2, pad + th), font, font_scale, (255, 255, 255), thickness)
            cells.append(cell)
        grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
        out.write(grid)

    out.release()


class Trainer:
    def __init__(self, model, train_loader, val_dataset, val_object_ids,
                 optimizer, scheduler, lpips_model, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.val_object_ids = val_object_ids
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lpips_model = lpips_model
        self.device = device
        self.args = args
        self.global_step = 0
        self.residual = getattr(args, 'residual', False)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_steps = len(self.train_loader)
        log_interval = self.args.log_interval

        for i, batch in enumerate(self.train_loader):
            lq = batch['lq'].to(self.device)   # (B, 2, 3(+cond), H, W)
            gt = batch['gt'].to(self.device)   # (B, 3, H, W)
            film = batch.get('film')
            if film is not None:
                film = film.to(self.device)

            pred = self.model(lq, film=film)   # (B, 3, H, W)

            with torch.no_grad():
                # RGB channels only (lq may carry extra conditioning channels);
                # in residual mode lq[:,1,:3] is the transferred residual.
                loss_mask = (gt - lq[:, 1, :3]).abs().amax(dim=1, keepdim=True) > 1e-2
            if not loss_mask.any():
                continue
            loss = _charbonnier_loss(pred[loss_mask.expand_as(pred)],
                                     gt[loss_mask.expand_as(gt)])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if (i + 1) % log_interval == 0:
                print(f"  [train] epoch {epoch+1}  step {i+1}/{n_steps} "
                      f"(global {self.global_step})  loss: {loss.item():.5f}")
                wandb.log({'train/loss': loss.item(), 'train/step': self.global_step})

        self.scheduler.step()
        avg_loss = total_loss / n_steps
        wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch})
        return avg_loss

    @torch.no_grad()
    def evaluate(self, object_ids, dataset, epoch, split, video_save_dir=None):
        self.model.eval()
        if video_save_dir:
            os.makedirs(video_save_dir, exist_ok=True)

        all_mse, all_psnr, all_ssim, all_lpips = [], [], [], []

        # Count total videos upfront for progress display
        total_videos = sum(
            1 for obj_id in object_ids
            for pair_idx in range(dataset.NUM_PAIRS)
            if dataset.lq_video_exists(obj_id, pair_idx)
        )
        video_count = 0

        for obj_id in object_ids:
            for pair_idx in range(dataset.NUM_PAIRS):
                if not dataset.lq_video_exists(obj_id, pair_idx):
                    continue

                video_count += 1
                print(f"  [{split}] obj {obj_id} pair {pair_idx}  "
                      f"[{video_count}/{total_videos}]", end='', flush=True)

                pred_frames, gt_frames, transferred_frames = [], [], []
                pred_residual_frames, gt_residual_frames = [], []

                for lq_pair, gt_frame, blank, film in dataset.iter_video_pairs(obj_id, pair_idx):
                    lq_in = lq_pair.unsqueeze(0).to(self.device)  # (1,2,3(+cond),H,W)
                    film_in = film.unsqueeze(0).to(self.device) if film is not None else None
                    pred = self.model(lq_in, film=film_in).squeeze(0)   # (3,H,W)
                    lq_rgb = lq_pair[1, :3]                              # transferred RGB

                    if self.residual:
                        # pred and gt_frame are residuals; reconstruct absolute
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
                    print()
                    continue

                # Per-video metrics (always on absolute values)
                v_mse, v_psnr, v_ssim, v_lpips = [], [], [], []
                for pred_np, gt_np in zip(pred_frames, gt_frames):
                    mse = compute_mse(gt_np, pred_np)
                    v_mse.append(mse)
                    v_psnr.append(compute_psnr(gt_np, pred_np, data_range=1.0)
                                  if mse > 0 else 100.0)
                    v_ssim.append(compute_ssim(gt_np, pred_np, data_range=1.0,
                                               channel_axis=-1))
                    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
                    pr_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
                    v_lpips.append(self.lpips_model(gt_t, pr_t).item())

                vm, vp, vs, vl = (float(np.mean(v_mse)), float(np.mean(v_psnr)),
                                  float(np.mean(v_ssim)), float(np.mean(v_lpips)))
                all_mse.append(vm)
                all_psnr.append(vp)
                all_ssim.append(vs)
                all_lpips.append(vl)

                print(f"  MSE: {vm:.5f}  PSNR: {vp:.2f}  SSIM: {vs:.4f}  LPIPS: {vl:.4f}")

                if video_save_dir:
                    out_path = os.path.join(video_save_dir,
                                            f"{obj_id}_{pair_idx}_enhanced.mp4")
                    _write_video(out_path, pred_frames)

                    if self.residual and pred_residual_frames:
                        _write_video(
                            os.path.join(video_save_dir,
                                         f"{obj_id}_{pair_idx}_pred_residual.mp4"),
                            pred_residual_frames)
                        _write_video(
                            os.path.join(video_save_dir,
                                         f"{obj_id}_{pair_idx}_gt_residual.mp4"),
                            gt_residual_frames)

                    save_gt = getattr(self.args, 'save_gt', False)
                    if save_gt:
                        _write_video(
                            os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_transferred.mp4"),
                            transferred_frames)
                        video_type = getattr(dataset, 'video_type', 'shadow')
                        ref_path = os.path.join(dataset.transfer_dir, str(obj_id),
                                                f"{pair_idx}_ref_{video_type}.mp4")
                        ref_frames = _read_video_frames(ref_path)
                        _make_grid_video(
                            os.path.join(video_save_dir, f"{obj_id}_{pair_idx}_grid.mp4"),
                            tl=ref_frames, tr=gt_frames,
                            bl=transferred_frames, br=pred_frames)

        metrics = {
            'MSE': float(np.mean(all_mse)) if all_mse else 0.0,
            'PSNR': float(np.mean(all_psnr)) if all_psnr else 0.0,
            'SSIM': float(np.mean(all_ssim)) if all_ssim else 0.0,
            'LPIPS': float(np.mean(all_lpips)) if all_lpips else 0.0,
        }
        return metrics

    def log_metrics(self, metrics, epoch, split):
        wandb.log({f'{split}/{k}': v for k, v in metrics.items()} | {'epoch': epoch})

    def print_metrics(self, metrics, split='val'):
        print(f"\n[{split}] MSE\tPSNR\tSSIM\tLPIPS")
        print(f"      {metrics['MSE']:.5f}\t{metrics['PSNR']:.2f}\t"
              f"{metrics['SSIM']:.4f}\t{metrics['LPIPS']:.4f}")

    def save_checkpoint(self, path, epoch, best_psnr):
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
        return ckpt.get('epoch', 0), ckpt.get('best_psnr', 0.0)
