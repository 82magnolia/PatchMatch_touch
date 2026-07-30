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

The [official TaRF repository](https://github.com/Dou-Yiming/TaRF) links its
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

A sensor-specific background JPEG is also mandatory. The upstream scene
backgrounds included under `img2touch/touch_bg/` can be used only when they
match the experiment's sensor.

## Fixed-standoff RGB/depth conditions

By default (`--condition_geometry auto`), the runner creates the two camera
views required by the original TaRF input format:

- `40_50`: target standoff 0.45 m, 50° field of view;
- `0_40`: target standoff 0.05 m, 40.86° field of view.

These are the original TaRF input conventions. The condition filenames remain
exactly `40_50` and `0_40`, and `view_metadata.json` records their geometry and
measured pre-inpaint coverage.

For real captures, the target is obtained from the saved ArUco pose using the
calibration in `log/gelsight_sensor_offset.json`. Its positive marker-face to
gel-tip Z distance is applied with the capture convention
`R_marker @ [offset_x, offset_y, -offset_z] + tvec`; the saved theta aligns the
sensor axes. The cached ZED RGB-D surface is then perspective-rasterized from
both virtual cameras. For simulation, the closest available multiscale
RGB/height pair is cropped to each view's physical footprint.

Pass `--condition_geometry files` to consume externally rendered NeRF views
instead. The preferred input is a NeRF export per query:

```text
<conditions_dir>/<query_idx>/
├── rgb/40_50.png
├── rgb/0_40.png
├── depth/40_50.npy
└── depth/0_40.npy
```

The PatchMatch simulation dataset's equivalent static modalities are also
accepted:

```text
<query_dir>/<query_idx>_scale100_color.jpg
<query_dir>/<query_idx>_scale100_height.npz
```

For the extracted real dataset, `<query_idx>_scale1_color.jpg` and the
normalized `<query_idx>_height.jpg` preview are accepted. The latter is mapped
to TaRF's upstream 0–5 depth-conditioning range.

When file mode has only one view, it is explicitly duplicated into the
upstream two-view conditioning layout and recorded in `resolved_config.json`.
A custom JSON manifest may instead be supplied with `--condition_manifest`:

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

## Simulation

GPU validation using the environment file as checked in:

```bash
bash baselines/TaRF/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --query_indices 0 \
  --save_dir log/baselines/tarf/sim_1_gpu \
  --background baselines/TaRF/img2touch/touch_bg/bench_colmap_40_50/bg.jpg \
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
  --background log/real_data_gt_retrieval/1/blank_frame.jpg \
  --sensor_offset_file log/gelsight_sensor_offset.json \
  --device cuda --n_samples 1 --skip_eval --debug_images
```

Real defaults are scale 1, `shadow`, and `real_gt_retrieval`. With no
`--conditions_dir`, the default `auto` mode uses the saved ArUco pose, ZED
RGB-D cache, saved marker-to-gel calibration, and original TaRF `0_40` view. Use a
no-contact background captured by the same GelSight sensor; `blank_frame.jpg`
is the dataset-provided example. For the full research configuration, use a
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
`TACTILE_NORMAL_SIM_QUERY_ROOT`, and `TACTILE_NORMAL_REAL_ROOT`. Simulation
background can be overridden with `TARF_TACTILE_NORMAL_BACKGROUND`. Outputs
default to `log/baselines/tactile_normal/{sim,real}/object_<id>/tarf/`.

TaRF still conditions its diffusion model on the required RGB/depth views;
the tactile-normal video is not substituted for those conditions. The real
wrapper retains the saved ArUco pose, `log/gelsight_sensor_offset.json`, and
the original TaRF `0_40` view convention. Before running CUDA, inspect
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
  --background baselines/TaRF/img2touch/touch_bg/bench_colmap_40_50/bg.jpg
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
