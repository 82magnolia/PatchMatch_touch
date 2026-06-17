# ReBotNet — Tactile Video Enhancement

Post-processing network that enhances PatchMatch-transferred tactile videos.
Given a transferred video (`{N}_transferred_em.mp4`) as input, the network outputs a video that is closer to the ground-truth query tactile video (`{N}_query_shadow.mp4`).

The model is [ReBotNet](https://arxiv.org/abs/2303.09650), a ConvNeXt + bottleneck-transformer architecture adapted here for the full 640×480 resolution via 2D adaptive average pooling in the bottleneck.

---

## Data Format

The scripts expect the following layout under `--transfer_dir` (e.g. `log/transfer/`):

```
log/transfer/
    {obj_id}/               # integer object ID, 1-1000
        0_transferred_em.mp4    # PatchMatch result (network input)
        0_query_shadow.mp4      # ground-truth query video
        1_transferred_em.mp4
        1_query_shadow.mp4
        ...
        7_transferred_em.mp4
        7_query_shadow.mp4
```

Object IDs are sorted numerically and split as:
- **Train**: objects 1–930 (930 objects)
- **Validation**: objects 931–950 (20 objects)
- **Test**: objects 951–1000 (50 objects)

---

## Dependencies

Install the project environment first (see top-level `CLAUDE.md`), then:

```bash
pip install wandb lpips scikit-image
```

---

## Training

```bash
python rebot_net/train.py \
    --transfer_dir log/transfer \
    --save_dir     log/rebot_checkpoints \
    --model_size   rebot_S \
    --epochs       100 \
    --batch_size   4 \
    --lr           2e-4 \
    --num_workers  4 \
    --wandb_project tactile_enhance \
    --wandb_run_name my_run
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--transfer_dir` | *(required)* | Root of the `log/transfer` directory |
| `--save_dir` | `log/rebot_checkpoints` | Where to write `best.pth` and `latest.pth` |
| `--model_size` | `rebot_S` | Model variant: `rebot_XS`, `rebot_S`, `rebot_M`, `rebot_L` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `4` | Frames per batch |
| `--lr` | `2e-4` | AdamW learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--num_workers` | `4` | DataLoader worker processes |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--video_save` | off | Save enhanced validation videos each epoch to `save_dir/videos/epoch_NNNN/` |
| `--wandb_project` | `tactile_enhance` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--wandb_offline` | off | Run W&B in offline mode |

Checkpoints saved:
- `best.pth` — best validation PSNR
- `latest.pth` — most recent epoch

---

## Evaluation

```bash
python rebot_net/eval.py \
    --transfer_dir log/transfer \
    --checkpoint   log/rebot_checkpoints/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_eval \
    --video_save \
    --save_gt
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--transfer_dir` | *(required)* | Root of the `log/transfer` directory |
| `--checkpoint` | *(required)* | Path to a `.pth` checkpoint file |
| `--model_size` | `rebot_S` | Must match the checkpoint's model size |
| `--save_dir` | `log/rebot_eval` | Directory for `metrics.pkl` and optional videos |
| `--video_save` | off | Save all enhanced test videos to `save_dir/videos/` |
| `--save_gt` | off | Also copy ground-truth query and reference videos alongside enhanced output (requires `--video_save`) |

Outputs:
- Per-object metrics printed to stdout
- `metrics.pkl` — dict with keys `per_object` and `average`

---

## Inference

Runs enhancement on transferred videos without requiring ground truth. Use `CUDA_VISIBLE_DEVICES` to select a GPU.

```bash
# Single video
CUDA_VISIBLE_DEVICES=0 python rebot_net/infer.py \
    --input_video log/transfer/52/0_transferred_em.mp4 \
    --checkpoint  log/rebot_checkpoints/best.pth \
    --model_size  rebot_S \
    --save_dir    log/rebot_infer \
    --save_gt

# All objects in transfer_dir
CUDA_VISIBLE_DEVICES=0 python rebot_net/infer.py \
    --transfer_dir log/transfer \
    --checkpoint   log/rebot_checkpoints/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_infer \
    --save_gt

# Specific objects only
CUDA_VISIBLE_DEVICES=0 python rebot_net/infer.py \
    --transfer_dir log/transfer \
    --object_ids   52 597 381 \
    --checkpoint   log/rebot_checkpoints/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_infer \
    --save_gt
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input_video` | — | Path to a single transferred video (mutually exclusive with `--transfer_dir`) |
| `--transfer_dir` | — | Root of `log/transfer/` to process in bulk (mutually exclusive with `--input_video`) |
| `--object_ids` | all | Subset of object IDs to process (only with `--transfer_dir`) |
| `--checkpoint` | *(required)* | Path to a `.pth` checkpoint file |
| `--model_size` | `rebot_S` | Must match the checkpoint's model size |
| `--save_dir` | `log/rebot_infer` | Directory to write enhanced videos |
| `--save_gt` | off | Also copy ground-truth query and reference videos alongside enhanced output |

Outputs are named `{original_stem}_enhanced.mp4`. When using `--transfer_dir`, videos are written under `save_dir/{obj_id}/`.

---

## Residual Prediction Mode

Instead of predicting absolute tactile frames, the model can operate in **contact residual space**:

```
residual[t] = video[t] - blank
```

where `blank` is frame 0 of the reference video (`{pair_idx}_ref_shadow.mp4`) — the no-contact sensor reading. The model predicts refined residuals; adding `blank` back gives the final absolute tactile video. This representation lets the network focus purely on the contact signal rather than the background texture.

**No architecture changes are needed.** The model's built-in residual connection (`output = network(input) + input`) naturally operates in residual space when the input is a residual frame.

### Training

```bash
python rebot_net/train.py \
    --transfer_dir log/transfer \
    --save_dir     log/rebot_checkpoints_residual \
    --model_size   rebot_S \
    --epochs       100 \
    --batch_size   4 \
    --lr           2e-4 \
    --residual \
    --wandb_project tactile_enhance \
    --wandb_run_name residual_S
```

### Evaluation

```bash
python rebot_net/eval.py \
    --transfer_dir log/transfer \
    --checkpoint   log/rebot_checkpoints_residual/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_eval_residual \
    --video_save --save_gt \
    --residual
```

Metrics (MSE/PSNR/SSIM/LPIPS) are always computed on **absolute** reconstructions (`pred_residual + blank`), matching the non-residual evaluation for fair comparison.

### Inference

```bash
# Single video
CUDA_VISIBLE_DEVICES=0 python rebot_net/infer.py \
    --input_video log/transfer/52/0_transferred_em.mp4 \
    --checkpoint  log/rebot_checkpoints_residual/best.pth \
    --model_size  rebot_S \
    --save_dir    log/rebot_infer_residual \
    --residual

# All objects
CUDA_VISIBLE_DEVICES=0 python rebot_net/infer.py \
    --transfer_dir log/transfer \
    --checkpoint   log/rebot_checkpoints_residual/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_infer_residual \
    --residual
```

The blank frame is loaded from `{pair_idx}_ref_shadow.mp4` in the same directory as the input video. If the reference video is not found, the script falls back to absolute mode for that video.

### Output files in residual mode

| File | Description |
|------|-------------|
| `{stem}_enhanced.mp4` | Absolute tactile video prediction (`pred_residual + blank`) |
| `{stem}_pred_residual.mp4` | Predicted contact residual, visualized as `(r × 0.5 + 0.5)` → [0, 1] |
| `{stem}_gt_residual.mp4` | Ground-truth contact residual visualization (requires `--save_gt`) |

---

## Normal-Image Baseline

A baseline variant that skips PatchMatch entirely: the model receives a static surface normal image at the query touch location (tiled into a video) as input and predicts the same ground-truth tactile video. This establishes a lower bound to quantify the benefit of PatchMatch transfer.

Normal images are read from `{query_normal_dir}/{obj_id}/{pair_idx}_scale100_normal.jpg` (640×480 RGB). The model architecture is identical to the main pipeline; the residual output is computed on top of the tiled normal image.

### Training

```bash
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
```

### Key flags (normal baseline training)

| Flag | Default | Description |
|------|---------|-------------|
| `--transfer_dir` | *(required)* | Root of `log/transfer` (GT query videos) |
| `--query_normal_dir` | *(required)* | Root of `gen_contact_full_query` (normal images) |
| `--save_dir` | `log/rebot_checkpoints_normal` | Where to write `best.pth` and `latest.pth` |
| `--model_size` | `rebot_S` | Model variant: `rebot_XS`, `rebot_S`, `rebot_M`, `rebot_L` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `4` | Frames per batch |
| `--lr` | `2e-4` | AdamW learning rate |
| `--num_workers` | `4` | DataLoader worker processes |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--video_save` | off | Save enhanced validation videos each epoch |
| `--wandb_project` | `tactile_enhance` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |

### Evaluation

```bash
python rebot_net/eval_normal.py \
    --transfer_dir log/transfer \
    --query_normal_dir Taxim/results/gen_contact_full_query \
    --checkpoint log/rebot_checkpoints_normal/best.pth \
    --model_size rebot_S \
    --save_dir log/rebot_eval_normal \
    --video_save \
    --save_gt
```

### Key flags (normal baseline evaluation)

| Flag | Default | Description |
|------|---------|-------------|
| `--transfer_dir` | *(required)* | Root of `log/transfer` (GT query videos) |
| `--query_normal_dir` | *(required)* | Root of `gen_contact_full_query` (normal images) |
| `--checkpoint` | *(required)* | Path to a `.pth` checkpoint file |
| `--model_size` | `rebot_S` | Must match the checkpoint's model size |
| `--save_dir` | `log/rebot_eval_normal` | Directory for `metrics.pkl` and optional videos |
| `--video_save` | off | Save all enhanced test videos to `save_dir/videos/` |
| `--save_gt` | off | Also save the normal-image video and a 2×2 grid (Normal \| GT / Normal \| Predicted) |

---

## Metrics

All metrics match those used in `main_retrieval_transfer_accel.py`:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean squared error (lower is better) |
| **PSNR** | Peak signal-to-noise ratio in dB (higher is better) |
| **SSIM** | Structural similarity index (higher is better, max 1.0) |
| **LPIPS** | Learned perceptual image patch similarity with AlexNet (lower is better) |
