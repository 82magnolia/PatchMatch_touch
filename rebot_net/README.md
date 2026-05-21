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
python rebot-net/train.py \
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
python rebot-net/eval.py \
    --transfer_dir log/transfer \
    --checkpoint   log/rebot_checkpoints/best.pth \
    --model_size   rebot_S \
    --save_dir     log/rebot_eval \
    --video_save
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--transfer_dir` | *(required)* | Root of the `log/transfer` directory |
| `--checkpoint` | *(required)* | Path to a `.pth` checkpoint file |
| `--model_size` | `rebot_S` | Must match the checkpoint's model size |
| `--save_dir` | `log/rebot_eval` | Directory for `metrics.pkl` and optional videos |
| `--video_save` | off | Save all enhanced test videos to `save_dir/videos/` |

Outputs:
- Per-object metrics printed to stdout
- `metrics.pkl` — dict with keys `per_object` and `average`

---

## Metrics

All metrics match those used in `main_retrieval_transfer_accel.py`:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean squared error (lower is better) |
| **PSNR** | Peak signal-to-noise ratio in dB (higher is better) |
| **SSIM** | Structural similarity index (higher is better, max 1.0) |
| **LPIPS** | Learned perceptual image patch similarity with AlexNet (lower is better) |
