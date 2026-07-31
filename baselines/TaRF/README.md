# TaRF baseline for PatchMatch_touch

This directory adapts **Tactile-Augmented Radiance Fields (TaRF)** into the
deterministic, one-shot interface used by this repository. For every query
location, TaRF conditions its image-to-touch diffusion model on egocentric RGB,
depth, and a sensor background, samples multiple tactile images, ranks them with
the learned RGB/tactile encoders, selects one image, and repeats that image to
the query sequence length.

The implementation follows `baselines/Baselines.pdf` when it differs from the
one-page baseline notes. In particular, the output is one estimated touch frame
copied to N video frames. It never reads query tactile ground truth during
condition resolution, generation, or ranking.

## Command convention

Run all commands below from the `PatchMatch_touch` repository root. Inputs use
the locations specified by `baselines/Baselines.pdf`. Use
`python baselines/TaRF/run_baseline.py --help` for the complete CLI.

```text
log/real_data_gt_retrieval/<object>/
Taxim/results/gen_contact_full_pseudo_mini/<object>/
Taxim/results/gen_contact_full_query_pseudo_mini/<object>/
```

The common arguments and pairing semantics match `transfer_pipeline.py`:
`--ref_dir`, `--query_dir`, `--save_dir`, `--scale`, `--video_type`,
`--retrieval_mode`, and optional `--tsv`. Simulation ground-truth retrieval is
identity (`query i -> reference i`); real ground-truth retrieval is odd-to-even
(`query i -> reference i-1`). TaRF additionally needs its RGB/depth conditions,
sensor background, pretrained assets, and—only for real fixed-view
construction—the saved `--sensor_offset_file`.

## Installation

```bash
cd /path/to/PatchMatch_touch
conda env create -f baselines/TaRF/environment.yml
conda activate TaRF
```

The checked-in upstream diffusion implementation has legacy dependencies. The
environment pins NumPy and MKL for compatibility and installs the PyTorch
CUDA 12.1 runtime. The host still needs a compatible NVIDIA driver.

## Required pretrained assets

Full TaRF inference requires three downloaded checkpoint files:

1. `img2touch.ckpt` — image-to-touch latent diffusion model.
2. `reranking_rgb_enc.ckpt` — RGB ranking encoder.
3. `reranking_tac_enc.ckpt` — tactile ranking encoder.

An optional standalone `model.ckpt` can provide the KL first-stage
autoencoder. The official `img2touch.ckpt` used here already embeds the
complete `first_stage_model.*` state, so the separate file is not required.

The [TaRF repository used by this baseline](https://github.com/yjun13568/TaRF) links its
[pretrained model archive](https://www.dropbox.com/scl/fi/5n9vx5991ev8av5l6ca2e/pretrained_models.tar.gz?rlkey=gdbkyot3at0hrr76np0hu220n&st=7krfblmx&dl=0).
After downloading, install it with:

```bash
cd /path/to/PatchMatch_touch/baselines/TaRF/img2touch
tar -xzf /path/to/pretrained_models.tar.gz
```

The resulting layout is:

```text
baselines/TaRF/img2touch/
├── pretrained_models/
│   ├── img2touch.ckpt
│   ├── reranking_rgb_enc.ckpt
│   └── reranking_tac_enc.ckpt
```

The default background is the same pseudo-mini no-contact image used by Taxim
calibration:

```text
baselines/TaRF/img2touch/touch_bg/gelsight_pseudo_background.jpg
```

It is selected automatically when `--background` is omitted. Passing an
explicit `--background` still overrides it.

## Fixed-standoff RGB/depth conditions

By default (`--condition_geometry auto`), the runner creates the two camera
views required by the original TaRF input format:

- `40_50`: target standoff 0.45 m, 50° field of view;
- `0_40`: target standoff 0.05 m, 40.86° field of view.

These are the original TaRF input conventions. The condition filenames remain
exactly `40_50` and `0_40`, and `view_metadata.json` records their geometry and
measured pre-inpaint coverage.

The prepared dataset scales are mapped to TaRF's two views:

| Domain | `40_50` context | `0_40` touch-aligned |
|---|---:|---:|
| simulation | `scale25` | `scale100` |
| real | `scale4` | `scale1` |

This uses the close/detail scale for `0_40`, aligned with the final tactile
footprint, and the 4× wider/coarser scale for `40_50`. The real prepared images
were already centered using the saved ArUco pose and
`log/gelsight_sensor_offset.json`. Their Viridis height previews are decoded
back to the capture pipeline's metric relative-height convention and anchored
at TaRF's corresponding depth before diffusion.

Pass `--condition_geometry files` to consume externally rendered NeRF views
instead. The preferred input is a NeRF export per query:

```text
<conditions_dir>/<query_idx>/
├── rgb/40_50.png
├── rgb/0_40.png
├── depth/40_50.npy
└── depth/0_40.npy
```

File mode requires all four files. A single RGB/height view is not duplicated:
that would produce the expected channel count but would not reproduce TaRF's
two-scale conditioning. A custom JSON manifest may instead be supplied with
`--condition_manifest`; its arrays must contain exactly two entries in
`40_50`, then `0_40`, order:

```json
{
  "0": {
    "rgb": ["conditions/0/rgb_a.png", "conditions/0/rgb_b.png"],
    "depth": ["conditions/0/depth_a.npy", "conditions/0/depth_b.npy"]
  }
}
```

Missing RGB/depth files cause an actionable error. Query `shadow`, `sim`, or
`tactile_normal` videos are never accepted as condition fallbacks.

Both RGB-D pairs are used for every prediction. Each saved 480×480 condition is
center-cropped/resized to 256×256, then concatenated in this exact order with
the 256×256 sensor background:

```text
RGB(40_50, 3) + depth(40_50, 1)
+ RGB(0_40, 3) + depth(0_40, 1)
+ background(3) = 11 conditioning channels
```

The diffusion model produces a 256×256 tactile image. The baseline resizes that
generated image only when packaging it at the query video's output resolution.

## Fine-tuning img2touch on paired simulation data

The simulation preparation pairs both Taxim roots and keeps whole objects in
only one split:

```bash
conda run --no-capture-output -n TaRF \
  python baselines/TaRF/scripts/prepare_sim_training_data.py \
  --roots \
    Taxim/results/gen_contact_full_pseudo_mini \
    Taxim/results/gen_contact_full_query_pseudo_mini \
  --output log/baselines/tarf_training/patchmatch_sim \
  --workers 12
```

For each touch, the target is the `shadow.mp4` frame having the largest contact
area in `mask.mp4`. Conditions reproduce TaRF's two-view, 11-channel input:

```text
40_50 = scale25 RGB + scale25 depth
0_40  = scale100 RGB + scale100 depth
background RGB
```

Taxim's `*_height.npz` is an elevation map, not a camera-depth map. A separate
depth renderer is therefore unnecessary for this dataset. The loader converts
each elevation value `h` to camera depth:

```text
elevation_m = (h - min(h)) * 0.0295e-3
depth_m = standoff_m - elevation_m
condition = 2 * clip(depth_m, 0, 5) / 5 - 1
```

The nominal standoffs are 0.45 m for `40_50` and 0.05 m for `0_40`. Thus a
higher/closer object point has a smaller camera-depth value. Keep the height
files: they encode the geometry used to construct each required depth channel,
but they are not passed to TaRF unchanged.

Training starts from `img2touch/pretrained_models/img2touch.ckpt`; the released
file contains the first-stage autoencoder weights as well. The config uses
batch size 1 because the model peaks near 19.7 GiB per RTX 3090. Before every
run, the wrapper prints `nvidia-smi`, selects the four least-busy GPUs, and
launches DDP:

```bash
bash baselines/TaRF/scripts/train_img2touch_sim.sh
```

Select a different set or count when needed:

```bash
TARF_GPUS=1,2,3,4 bash baselines/TaRF/scripts/train_img2touch_sim.sh
TARF_NUM_GPUS=2 bash baselines/TaRF/scripts/train_img2touch_sim.sh
```

Resume a completed checkpoint without changing the run directory:

```bash
TARF_GPUS=1,2,3,4 \
TARF_RESUME=/path/to/checkpoints/epoch=000000.ckpt \
bash baselines/TaRF/scripts/train_img2touch_sim.sh
```

Validation loss is computed over the validation split, while expensive
diffusion preview generation is limited to one rank-zero batch per validation
epoch. This avoids generating previews repeatedly at the same global step.

The default is 30 epochs. Runs, CSV metrics, generated previews, and checkpoints
are written under `log/baselines/tarf_training/runs/`, outside the baseline
source tree.

## Training a tactile-normal img2touch model

Tactile-normal training uses the same TaRF latent-diffusion architecture and
the same two RGB-D condition views, but changes the prediction target from the
Taxim `shadow.mp4` image to `tactile_normal.mp4`. Prepare the current simulation
data with:

```bash
bash baselines/TaRF/scripts/prepare_sim_tactile_normal_training_data.sh
```

This reads:

```text
Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/
Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/
```

For every touch, the preparation script:

1. maps `scale25` RGB/height to TaRF's `40_50` view;
2. maps `scale100` RGB/height to TaRF's `0_40` view;
3. converts each Taxim elevation map into a camera-depth channel;
4. finds the frame with maximum contact area in `mask.mp4`; and
5. extracts that frame from `tactile_normal.mp4` as the target image.

The output is:

```text
log/baselines/tarf_training/patchmatch_sim_tactile_normal/
├── manifest.json
├── tactile_normal_background.jpg
└── touch/
    └── r<root>_o<object>_t<touch>.jpg
```

The no-contact first frame becomes the normal-map background (approximately
RGB `[125, 126, 250]`). The object-disjoint split is the same as shadow
training: 12,800 train, 1,600 validation, and 1,600 test samples. Object IDs
ending in buckets 1–8 are train, bucket 9 is validation, and bucket 0 is test.

The released `img2touch.ckpt` predicts shadow/RGB tactile appearance, so its
diffusion UNet, EMA, and RGB-D conditioner weights are **not loaded** for this
model. Those components start randomly with the same upstream TaRF structure.
TaRF freezes its first-stage latent autoencoder, however, so a frozen random
autoencoder would make diffusion training invalid. The launcher therefore
extracts only `first_stage_model.*` from the released checkpoint into
`pretrained_models/img2touch_first_stage.ckpt`. It does not transfer the
shadow-trained diffusion mapping.

Before using GPUs, the launcher prints `nvidia-smi` and chooses four idle GPUs:

```bash
bash baselines/TaRF/scripts/train_img2touch_tactile_normal.sh
```

Explicitly select idle devices when sharing the machine:

```bash
TARF_GPUS=0,1,2,3 \
bash baselines/TaRF/scripts/train_img2touch_tactile_normal.sh
```

The training config is
`img2touch/configs/patchmatch_sim_tactile_normal_train.yaml`. It uses batch
size 1 per GPU, 30 epochs, full validation loss, and only one rank-zero
diffusion-preview batch per validation epoch. Runs are saved under:

```text
log/baselines/tarf_training/runs/<timestamp>_patchmatch_sim_tactile_normal/
```

To resume, set `TARF_RESUME` to a completed checkpoint:

```bash
TARF_GPUS=0,1,2,3 \
TARF_RESUME=/absolute/path/to/checkpoints/last.ckpt \
bash baselines/TaRF/scripts/train_img2touch_tactile_normal.sh
```

After training, pass the selected tactile-normal diffusion checkpoint to the
tactile-normal inference wrapper instead of the released shadow checkpoint:

```bash
bash baselines/TaRF/scripts/run_sim_tactile_normal.sh \
  --diffusion_ckpt \
  log/baselines/tarf_training/runs/<run>/checkpoints/<selected>.ckpt \
  --background \
  log/baselines/tarf_training/patchmatch_sim_tactile_normal/tactile_normal_background.jpg
```

## Simulation

GPU validation using the environment file as checked in:

```bash
bash baselines/TaRF/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --query_indices 0 \
  --save_dir log/baselines/tarf/sim_1_gpu \
  --background baselines/TaRF/img2touch/touch_bg/gelsight_pseudo_background.jpg \
  --device cuda --n_samples 1 --skip_eval --debug_images
```

Simulation defaults are scale 100, `shadow`, and identity
`sim_gt_retrieval`. Omit `--n_samples 1` to restore eight candidates. The
checkpoint arguments may be omitted when the files use the documented default
layout.

## Real data

Extract the provided archive first so that one object/session directory
contains the flat touch files. Odd query indices are paired with the preceding
even reference index:

```bash
bash baselines/TaRF/scripts/run_real.sh \
  --ref_dir log/real_data_gt_retrieval/1 \
  --query_dir log/real_data_gt_retrieval/1 \
  --query_indices 1 \
  --save_dir log/baselines/tarf/real_1 \
  --background baselines/TaRF/img2touch/touch_bg/gelsight_pseudo_background.jpg \
  --sensor_offset_file log/gelsight_sensor_offset.json \
  --device cuda --n_samples 1 --skip_eval --debug_images
```

Real defaults are scale 1, `shadow`, and `real_gt_retrieval`. With no
`--conditions_dir`, the default `auto` mode uses the saved ArUco pose, ZED
RGB-D cache, saved marker-to-gel calibration, and both original TaRF views. The
calibrated pseudo-mini no-contact background is used by default. For the full
research configuration, use a
CUDA-enabled environment, remove `--n_samples 1`, and
remove `--skip_eval` when the evaluation dependencies are installed.

## Tactile-normal data

These wrappers select `{idx}_tactile_normal.mp4` instead of `shadow` for the
reference/query tactile stream and evaluation:

```bash
# Simulation
bash baselines/TaRF/scripts/run_sim_tactile_normal.sh \
  --query_indices 0 --device cuda --n_samples 1 --skip_eval

# Real
bash baselines/TaRF/scripts/run_real_tactile_normal.sh \
  --query_indices 1 --device cuda --n_samples 1 --skip_eval
```

The wrapper changes only the tactile modality and dataset roots; the normal
TaRF runner still loads and executes the diffusion model. Its data flow is:

1. resolve the query/reference pair (identity in simulation and odd-to-even
   in real data);
2. load or construct the fixed-view RGB/depth conditions;
3. generate `--n_samples` diffusion candidates and select the best candidate;
4. repeat the selected tactile image over the query render-mask video's frame
   count, resolution, and FPS; and
5. package the reference/query tactile-normal videos and evaluate against the
   query tactile-normal video unless `--skip_eval` is supplied.

Neither the reference nor query tactile-normal pixels condition diffusion.
The query video is post-prediction ground truth, while the reference video is
retained for the common retrieval/output contract. Consequently, changing
from `shadow` to `tactile_normal` does not remove TaRF's RGB/depth
requirements. The output is a static generated touch repeated to match the
query video timing, consistent with the original image-to-touch TaRF model.

They default to object 1, simulation roots
`Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/` and
`Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/`, and real
root `log/real_data_gt_retrieval/`. Select data with
`TACTILE_NORMAL_OBJECT_ID`, `TACTILE_NORMAL_SIM_REF_ROOT`,
`TACTILE_NORMAL_SIM_QUERY_ROOT`, and `TACTILE_NORMAL_REAL_ROOT`. Both wrappers
use the pseudo-mini calibration background by default; it can be overridden
with `TARF_TACTILE_NORMAL_BACKGROUND`. Outputs
default to `log/baselines/tactile_normal/{sim,real}/object_<id>/tarf/`.

TaRF still conditions its diffusion model on the required RGB/depth views;
the tactile-normal video is not substituted for those conditions. The real
wrapper retains the saved ArUco pose, `log/gelsight_sensor_offset.json`, and
the original TaRF `40_50` and `0_40` view conventions. Before running CUDA, inspect
`nvidia-smi` and select an idle GPU, for example
`CUDA_VISIBLE_DEVICES=2 ... --device cuda`.
Use `--dry_run` to check condition discovery, fixed-view geometry, pairing,
and checkpoints without loading diffusion. `--n_samples 1` is a fast
integration run; omit it to use the documented eight-candidate default.

## Checkpoint-free smoke run

The smoke backend validates condition discovery, pairing, candidate ranking,
video repetition, copying, and metadata without loading TaRF checkpoints:

```bash
bash baselines/TaRF/scripts/run_smoke_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --query_indices 0 \
  --save_dir log/baselines/tarf/smoke_1 \
  --background baselines/TaRF/img2touch/touch_bg/gelsight_pseudo_background.jpg
```

Smoke images are procedural condition-only diagnostics and **must not be
reported as TaRF research results**.

The normal `run_real.sh` and `run_sim.sh` paths use the diffusion model.
`--smoke_test` must be supplied explicitly and exists only for unit/integration
diagnostics.

## Outputs

```text
<save_dir>/
├── identity.tsv or odd_to_even.tsv
├── resolved_config.json
├── retrieval/results.pkl
├── generation/
│   ├── metadata.json
│   └── <query_idx>/{candidate_00.png,...,selected.png}
└── transfer/
    ├── <query_idx>_transferred.mp4
    ├── <query_idx>_ref_<video_type>.mp4
    ├── <query_idx>_query_<video_type>.mp4
    └── metrics.pkl
```

The generated still is repeated using frame count, resolution, and FPS read
from `<query_idx>_render_mask.mp4`, so tactile ground-truth pixels are not read
for prediction timing. `metrics.pkl` contains `per_touch` and `average`
MSE/PSNR/SSIM/LPIPS and is compatible with:

```bash
python parse_metrics.py --dir log/baselines/tarf --verbose
```

Use `--skip_eval` when LPIPS evaluation is not installed. Evaluation is the
only stage that decodes the query tactile video.

## Tests

Run tests through the environment's Python module entry point so the local
`patchmatch_tarf` package is resolved correctly:

```bash
cd baselines/TaRF
conda run -n TaRF python -m pytest -q tests
cd ../..
```

## Important options

- `--n_samples 8`, `--ddim_steps 200`, `--guidance_scale 7.5`
- `--seed 42`, `--device cuda`
- `--depth_multiplier 1`, `--depth_clip_max 5`
- `--conditions_dir`, `--condition_manifest`, `--background`
- `--condition_geometry {auto,fixed_views,files}`
- `--sensor_offset_file log/gelsight_sensor_offset.json`
- `--retrieval_mode {dinov3,tsv,sim_gt_retrieval,real_gt_retrieval}`
- `--dry_run` validates pairings, conditions, and full checkpoint paths.

## Limitations

- The original TaRF model was trained with scene-specific NeRF RGBD views and
  sensor backgrounds. PatchMatch static color/height modalities are accepted
  for integration, but are not distribution-equivalent to the original
  conditions.
- Full diffusion inference is computationally heavy and is impractical on CPU.
- Research runs use the default eight diffusion candidates and 200 DDIM steps.
  CPU-only comparison runs may set `TARF_N_SAMPLES=1` while retaining all 200
  steps; such results must be labeled as one-candidate validation results.

To regenerate only the TaRF rows of the eight-object real/sim comparison:

```bash
TARF_DEVICE=cuda \
  test_scripts/run_tarf_comparison_objects.sh 1 2 10 25 50 75 99 100
```

For a CPU validation comparison, use
`TARF_DEVICE=cpu TARF_N_SAMPLES=1`; this changes only candidate count, not the
200-step diffusion schedule. Then rebuild metrics and videos with
`test_scripts/compare_baseline_suite.py`. The evaluator rejects smoke outputs,
always recomputes TaRF metrics, and records each run's sampling and actual
close-view geometry in `tarf_run_settings.json`.
- The pretrained assets are not redistributed here and retain their original
  licenses.
- The source checkout's Nerfstudio viewer remains available, but this adapter
  intentionally replaces its file-polling estimator with a finite batch run.

## Attribution

This work adapts the local checkout originally associated with
[yjun13568/TaRF](https://github.com/yjun13568/TaRF) and the
[official implementation](https://github.com/Dou-Yiming/TaRF).

Paper: [Tactile-Augmented Radiance Fields, CVPR
2024](https://arxiv.org/abs/2405.04534), Yiming Dou, Fengyu Yang, Yi Liu,
Antonio Loquercio, and Andrew Owens.

Use of the upstream code and checkpoints is subject to their original license
and attribution terms.
