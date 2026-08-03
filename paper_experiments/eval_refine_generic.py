"""Conditioned ReBotNet refinement eval over an arbitrary object list / layout.

rebot_net/eval.py is hard-wired to the ground-truth-retrieval benchmark: it
takes one flat transfer_dir and always evaluates all_ids[950:] with
NUM_PAIRS = 8. The full-pipeline benchmark (job 2) and the ablation subset
(job 3) need neither of those, so this driver reuses rebot_net's dataset,
model builder and metric code but lets the caller choose:

  * which objects to evaluate (--object_ids / --object_ids_file),
  * where each object's transferred videos live (--layout flat|nested),
  * how many touch indices to probe (--num_pairs),
  * a per-object conditioning directory root.

Metrics (MSE / PSNR / SSIM / LPIPS) are computed exactly as in rebot_net/eval.py
so numbers are comparable across jobs.
"""
import argparse
import json
import os
import pickle
import sys

import lpips
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, os.path.join(ROOT, "rebot_net"))

from dataset import TactileTransferDataset          # noqa: E402
from train import MODEL_CONFIGS, build_model        # noqa: E402
from trainer import _write_video, _read_video_frames, _make_grid_video  # noqa: E402
import cond_utils                                   # noqa: E402


class FlexibleTransferDataset(TactileTransferDataset):
    """TactileTransferDataset with a configurable object layout and pair count.

    layout 'flat'   -> {transfer_dir}/{obj}/            (rebot_net's own layout)
    layout 'nested' -> {transfer_dir}/{obj}/transfer/   (transfer_pipeline.py's)

    cond_root is the directory holding {obj}/{idx}_scale{S}_{modality}.jpg.
    """

    def __init__(self, *args, layout="flat", num_pairs=8, cond_root=None, **kwargs):
        self.layout = layout
        self.cond_root = cond_root
        # NUM_PAIRS is read during __init__, so set it before delegating.
        self.NUM_PAIRS = num_pairs
        super().__init__(*args, **kwargs)

    def _obj_dir(self, obj_id):
        base = os.path.join(self.transfer_dir, str(obj_id))
        return os.path.join(base, "transfer") if self.layout == "nested" else base

    def _cond_base(self, obj_id):
        root = self.cond_root or self.cond_dir
        return os.path.join(root, str(obj_id))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--transfer_dir', required=True)
    p.add_argument('--layout', default='flat', choices=['flat', 'nested'],
                   help="'nested' for transfer_pipeline.py's {obj}/transfer/ output")
    p.add_argument('--object_ids', nargs='+', type=int, default=None)
    p.add_argument('--object_ids_file', default=None,
                   help="Text file with one object id per line (overrides --object_ids)")
    p.add_argument('--num_pairs', type=int, default=8,
                   help="Probe touch indices 0..num_pairs-1 per object")
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model_size', default='rebot_S', choices=list(MODEL_CONFIGS))
    p.add_argument('--save_dir', required=True)
    p.add_argument('--video_type', default='tactile_normal')
    p.add_argument('--bottleneck_hw', type=int, default=24)
    p.add_argument('--video_save', action='store_true')
    p.add_argument('--save_gt', action='store_true')
    p.add_argument('--max_videos', type=int, default=0,
                   help="With --video_save, only write videos for the first N objects "
                        "(0 = all). Keeps qualitative runs cheap.")
    cond_utils.add_cond_args(p)
    return p.parse_args()


def resolve_ids(args):
    if args.object_ids_file:
        with open(args.object_ids_file) as f:
            return [int(line) for line in f if line.strip()]
    if args.object_ids:
        return args.object_ids
    return sorted(int(d) for d in os.listdir(args.transfer_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(args.transfer_dir, d)))


def main():
    args = parse_args()
    cond_utils.check_cond_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obj_ids = resolve_ids(args)
    print(f"Objects: {len(obj_ids)}  |  layout: {args.layout}  |  num_pairs: {args.num_pairs}")

    dataset = FlexibleTransferDataset(
        args.transfer_dir, obj_ids, split='test',
        layout=args.layout, num_pairs=args.num_pairs, cond_root=args.cond_dir,
        video_type=args.video_type,
        **cond_utils.dataset_cond_kwargs(args))

    cond_chans, film_chans = cond_utils.cond_dims(args)
    model = build_model(args.model_size, cond_chans, film_chans,
                        bottleneck_hw=args.bottleneck_hw,
                        time_cond=cond_utils.time_cond_mode(args)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')} "
          f"(best val PSNR {ckpt.get('best_psnr', 0):.2f})")

    lpips_model = lpips.LPIPS(net='alex').to(device)
    for prm in lpips_model.parameters():
        prm.requires_grad_(False)

    video_dir = os.path.join(args.save_dir, 'videos') if args.video_save else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    per_object, per_touch = {}, {}
    agg = {k: [] for k in ('MSE', 'PSNR', 'SSIM', 'LPIPS')}

    model.eval()
    with torch.no_grad():
        for n_obj, obj_id in enumerate(obj_ids):
            obj_vals = {k: [] for k in agg}
            for pair_idx in range(dataset.NUM_PAIRS):
                if not dataset.lq_video_exists(obj_id, pair_idx):
                    continue

                pred_frames, gt_frames, lq_frames = [], [], []
                for lq_pair, gt_frame, _blank, film, t_norm in \
                        dataset.iter_video_pairs(obj_id, pair_idx):
                    lq_in = lq_pair.unsqueeze(0).to(device)
                    film_in = film.unsqueeze(0).to(device) if film is not None else None
                    t_in = torch.tensor([t_norm], device=device)
                    pred = model(lq_in, film=film_in, t=t_in).squeeze(0)
                    pred_frames.append(pred.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                    gt_frames.append(gt_frame.permute(1, 2, 0).numpy())
                    lq_frames.append(lq_pair[1, :3].permute(1, 2, 0).numpy())

                if not pred_frames:
                    continue

                vals = {k: [] for k in agg}
                for pred_np, gt_np in zip(pred_frames, gt_frames):
                    mse = compute_mse(gt_np, pred_np)
                    vals['MSE'].append(mse)
                    vals['PSNR'].append(compute_psnr(gt_np, pred_np, data_range=1.0)
                                        if mse > 0 else 100.0)
                    vals['SSIM'].append(compute_ssim(gt_np, pred_np, data_range=1.0,
                                                     channel_axis=-1))
                    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1)[None].to(device) * 2 - 1
                    pr_t = torch.from_numpy(pred_np).permute(2, 0, 1)[None].to(device) * 2 - 1
                    vals['LPIPS'].append(lpips_model(gt_t, pr_t).item())

                per_touch[f"{obj_id}_{pair_idx}"] = {k: float(np.mean(v)) for k, v in vals.items()}
                for k in agg:
                    obj_vals[k].extend(vals[k])

                if video_dir and (args.max_videos == 0 or n_obj < args.max_videos):
                    _write_video(os.path.join(video_dir, f"{obj_id}_{pair_idx}_enhanced.mp4"),
                                 pred_frames)
                    if args.save_gt:
                        _write_video(os.path.join(video_dir, f"{obj_id}_{pair_idx}_transferred.mp4"),
                                     lq_frames)
                        _write_video(os.path.join(video_dir, f"{obj_id}_{pair_idx}_gt.mp4"),
                                     gt_frames)
                        ref_path = os.path.join(dataset._obj_dir(obj_id),
                                                f"{pair_idx}_ref_{args.video_type}.mp4")
                        ref_frames = _read_video_frames(ref_path) if os.path.exists(ref_path) else []
                        if ref_frames:
                            _make_grid_video(
                                os.path.join(video_dir, f"{obj_id}_{pair_idx}_grid.mp4"),
                                tl=ref_frames, tr=gt_frames, bl=lq_frames, br=pred_frames)

            if not obj_vals['MSE']:
                continue
            m = {k: float(np.mean(v)) for k, v in obj_vals.items()}
            per_object[obj_id] = m
            for k in agg:
                agg[k].append(m[k])
            print(f"  Object {obj_id:5d} — MSE {m['MSE']:.5f}  PSNR {m['PSNR']:.2f}  "
                  f"SSIM {m['SSIM']:.4f}  LPIPS {m['LPIPS']:.4f}", flush=True)

    avg = {k: float(np.mean(v)) for k, v in agg.items()}
    print("\n" + "=" * 60)
    print(f"Average over {len(per_object)} objects ({len(per_touch)} touches)")
    print("MSE\t\tPSNR\tSSIM\tLPIPS")
    print(f"{avg['MSE']:.5f}\t{avg['PSNR']:.2f}\t{avg['SSIM']:.4f}\t{avg['LPIPS']:.4f}")

    out = {'per_object': per_object, 'per_touch': per_touch, 'average': avg,
           'n_objects': len(per_object), 'n_touches': len(per_touch),
           'checkpoint': args.checkpoint, 'transfer_dir': args.transfer_dir}
    with open(os.path.join(args.save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(out, f)
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.save_dir}/metrics.{{pkl,json}}")


if __name__ == '__main__':
    main()
