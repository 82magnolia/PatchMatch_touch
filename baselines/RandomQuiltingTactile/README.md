# RandomQuiltingTactile baseline

This directory implements the RandomQuiltingTactile/Tactile DreamFusion
baseline for `PatchMatch_touch`. It follows the common `transfer_pipeline.py`
input and output contract while keeping the imported research repositories
under `ImageQuilting/`, `TactileDreamFusion/`, and `ObjectFolder/`.

The prediction stage uses only the retrieved reference touch. Query tactile
videos are copied for visualization and opened only by the final evaluation
stage. Query render-mask videos supply output frame count, resolution, and FPS.

## Command convention

Run every command below from the `PatchMatch_touch` repository root. Inputs
use the locations specified by `baselines/Baselines.pdf` and shared with
`transfer_pipeline.py`:

```text
log/real_data_gt_retrieval/<object>/
Taxim/results/gen_contact_full_pseudo_mini/<object>/
Taxim/results/gen_contact_full_query_pseudo_mini/<object>/
```

Use `python baselines/RandomQuiltingTactile/run_baseline.py --help` for the
complete CLI. The shell wrappers select the appropriate scale, video type, and
ground-truth retrieval convention for simulation or real data.

## Method

For each retrieved reference, the runner selects its most-contact frame using
`{ref}_render_mask.mp4`. If that mask is unavailable, it uses the largest
difference from `blank_frame.jpg` or from the first reference frame. The
corresponding reference `tactile_normal` frame is preferred when available.

`rqt/quilting.py` performs overlap-SSD block selection, tolerance sampling, and
minimum-cost seam blending with a fixed random seed. The candidate pool is
deterministically capped by `--quilt_max_candidates` (default: 1024) to avoid
materializing every overlapping patch from a full video frame.

Two execution modes are available:

- `fallback` quilts the selected 2D reference touch and repeats it to the query
  render-mask length. This is the documented real-data path when mesh/query
  geometry is unavailable.
- `full` quilts the normal patch, caches Tactile DreamFusion training by object,
  reference, quilting parameters, config, and seed, renders a view-space normal
  at an explicit query point, integrates it to height, and renders the tactile
  result through the imported Taxim/ObjectFolder code.

## Installation

The fallback environment can be created reproducibly with:

```bash
conda env create -f baselines/RandomQuiltingTactile/environment.yml
```

The run scripts automatically use the `RandomQuiltingTactile` environment.
Set `RQT_PYTHON=/path/to/python` to override that behavior.

Evaluation uses the dependencies already required by `PatchMatch_touch`:
PyTorch, LPIPS, and scikit-image. The full mode has separate, older
ObjectFolder and Tactile DreamFusion environments; follow
`ObjectFolder/environment.yml` and `TactileDreamFusion/requirements.txt`.
Tactile DreamFusion also requires its external model checkpoints and CUDA
renderer dependencies.

## Data assumptions

The simulated folders use the standard flat per-object layout:

```text
Taxim/results/gen_contact_full_pseudo_mini/<object>/
Taxim/results/gen_contact_full_query_pseudo_mini/<object>/
```

Files include `{idx}_scale100_normal.jpg`, `{idx}_shadow.mp4`, and
`{idx}_render_mask.mp4`. Identity retrieval pairs query and reference indices.

Extracted real sessions contain `{idx}_scale1_normal.jpg`,
`{idx}_shadow.mp4`, and `{idx}_render_mask.mp4`. Odd query index `n` is paired
with even reference `n-1`. A ZIP file itself is not a valid `--ref_dir`.

## Tactile-normal data

Use the dedicated wrappers to replace the old `shadow` video stream with
`{idx}_tactile_normal.mp4`:

```bash
# Simulation: query 0 is paired with reference 0.
bash baselines/RandomQuiltingTactile/scripts/run_sim_tactile_normal.sh \
  --query_indices 0 --pipeline_mode fallback --skip_eval

# Real: query 1 is paired with reference 0.
bash baselines/RandomQuiltingTactile/scripts/run_real_tactile_normal.sh \
  --query_indices 1 --pipeline_mode fallback --skip_eval
```

Each wrapper delegates to the normal `run_sim.sh` or `run_real.sh` and appends
`--video_type tactile_normal`. The quilting pipeline then:

1. resolves the retrieval pair (identity for simulation, odd query `n` to
   even reference `n-1` for real data);
2. uses the reference render mask to find the strongest-contact frame;
3. reads that frame from the reference `tactile_normal` video and synthesizes
   a full-size quilted tactile-normal image;
4. repeats the generated image using the query render-mask video's frame
   count, resolution, and FPS; and
5. compares the prediction against the query `tactile_normal` video unless
   `--skip_eval` is supplied.

The query tactile-normal video is ground truth for packaging and evaluation;
it is not read to create the prediction. In `fallback` mode the quilted image
is the result. In `full` mode the same quilt is passed through the optional
ObjectFolder/Tactile DreamFusion stages described below.

The wrappers default to object 1 and these dataset roots:

```text
Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/
Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/
log/real_data_gt_retrieval/
```

Select another object with `TACTILE_NORMAL_OBJECT_ID=10`. Override relocated
datasets with `TACTILE_NORMAL_SIM_REF_ROOT`,
`TACTILE_NORMAL_SIM_QUERY_ROOT`, or `TACTILE_NORMAL_REAL_ROOT`. Extra
arguments are forwarded to `run_sim.sh` or `run_real.sh`; because they are
appended last, they may override any wrapper default. Outputs default to
`log/baselines/tactile_normal/{sim,real}/object_<id>/random_quilting/`.
Use `--dry_run` to inspect resolved paths and pairs without generating files,
and `--debug_images` to save the selected reference frame and quilt.

## Simulation

Runnable 2D fallback:

```bash
bash baselines/RandomQuiltingTactile/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --save_dir log/baselines/random_quilting/1 \
  --query_indices 0 \
  --pipeline_mode fallback \
  --debug_images --skip_eval
```

Remove `--skip_eval` to compute MSE, PSNR, SSIM, and LPIPS after installing
PyTorch and LPIPS in the selected environment. Evaluation never affects the
generated result.

A checked single-query example is available at
`examples/object1_query0.tsv`:

```bash
bash baselines/RandomQuiltingTactile/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --save_dir log/baselines/random_quilting/object_1_query0 \
  --retrieval_mode tsv \
  --tsv baselines/RandomQuiltingTactile/examples/object1_query0.tsv \
  --pipeline_mode fallback --debug_images --skip_eval
```

Generate four lightweight examples across objects 1, 10, and 100:

```bash
bash baselines/RandomQuiltingTactile/scripts/run_examples.sh
```

Choose a custom set with `object:query` entries:

```bash
RQT_EXAMPLES="2:1 25:4 250:7" \
  bash baselines/RandomQuiltingTactile/scripts/run_examples.sh
```

The script writes selected-reference images, quilted images, and transferred
videos under `log/baselines/random_quilting/examples/`. For a normal baseline
run, `--query_indices 0 3 7` can similarly restrict execution to a subset.

Full TDF/Taxim execution additionally requires:

```bash
bash baselines/RandomQuiltingTactile/scripts/run_sim.sh \
  --ref_dir <sim-reference-object-dir> \
  --query_dir <sim-query-object-dir> \
  --save_dir <output-dir> \
  --pipeline_mode full \
  --object_id <object-id> \
  --tdf_config <TactileDreamFusion-config.yaml> \
  --tdf_checkpoint <optional-pretrained-stage-checkpoint> \
  --object_mesh <object/model.obj> \
  --query_view_path '<one-point-ply-dir>/{query_idx}.ply' \
  --taxim_calibration <Taxim-calibration-dir> \
  --objectfolder_object_dir <ObjectFolder-object-dir> \
  --objectfile_checkpoint <ObjectFile.pth> \
  --object_sample_ply <existing-dataset-ply-name> \
  --train_if_missing
```

`--tdf_checkpoint` is optional and is forwarded to Tactile DreamFusion's
`load` configuration field. `--tdf_python` and `--objectfolder_python` can point to executables from the
two respective environments. Query point-cloud paths may contain the literal
`{query_idx}` placeholder. Each must contain the geometry for one query only,
because the imported renderer samples one point from the supplied PLY.

To train a cache entry directly:

```bash
bash baselines/RandomQuiltingTactile/scripts/train_tdf.sh \
  --tdf-root baselines/RandomQuiltingTactile/TactileDreamFusion \
  --config <config.yaml> --mesh <model.obj> --texture <quilted.png> \
  --texture-name <name> --cache-key <key>
```

## Real data

```bash
bash baselines/RandomQuiltingTactile/scripts/run_real.sh \
  --ref_dir log/real_data_gt_retrieval/1 \
  --query_dir log/real_data_gt_retrieval/1 \
  --query_indices 1 \
  --save_dir log/baselines/random_quilting/real_object1 \
  --debug_images --skip_eval
```

For a custom TSV, use `--retrieval_mode tsv --tsv pairs.tsv`. DINOv3
retrieval is available with `--retrieval_mode dinov3 --dino_weights <pth>`.

## Outputs and evaluation

```text
<save_dir>/
  identity.tsv or odd_to_even.tsv
  resolved_config.json
  retrieval/results.pkl
  transfer/
    <query>_transferred.mp4
    <query>_ref_<video_type>.mp4
    <query>_query_<video_type>.mp4
    metrics.pkl
```

Predictions match the query render mask's frame count, resolution, and FPS.
`metrics.pkl` uses the `per_touch`/`average` MSE, PSNR, SSIM, and LPIPS schema
consumed by:

```bash
python parse_metrics.py --dir <save_dir>
```

Use `--skip_eval` when evaluation dependencies are unavailable. Use
`--dry_run` to validate discovery, pairing, and resolved configuration without
loading NumPy, OpenCV, checkpoints, or tactile videos.

Tests:

```bash
PYTHONPATH=baselines/RandomQuiltingTactile \
  python -m unittest discover \
  -s baselines/RandomQuiltingTactile/tests -v
```

## Limitations

- The fallback is intentionally geometry-free and produces a static video.
- Full mode needs one-point query PLY geometry, ObjectFolder checkpoints, TDF
  model assets, compatible CUDA environments, and a complete Taxim calibration.
- Existing simulated data without `tactile_normal.mp4` falls back to quilting
  the retrieved reference video frame.
- The imported research code has substantially older dependency constraints
  than the main PatchMatch environment.

### Legacy five-stage scripts

The numbered `scripts/1.obj_normal.sh` through
`scripts/5.normal_to_tactile.sh` are retained as upstream provenance, not as
portable entry points. They assume Conda environments named `ObjectFolder` and
`TDF`, a checkout at `~/Projects/RandomQuiltingTactile`, downloaded
`ObjectFolder1-100` checkpoints/contact PLYs, TDF base meshes/checkpoints, and a
working CUDA renderer. Those assets are not included in this checkout. Use
`run_sim.sh`, `run_real.sh`, and `train_tdf.sh` for the repository-relative
baseline interface.

## References and attribution

- [RandomQuiltingTactile code](https://github.com/yjun13568/RandomQuiltingTactile)
- [Tactile DreamFusion paper](https://arxiv.org/abs/2412.06785)
- [ObjectFolder](https://github.com/rhgao/ObjectFolder)
- [Taxim](https://github.com/CMURoboTouch/Taxim)

The new wrapper code follows the parent project's license. Imported components
retain the licenses and attribution in their respective subdirectories.
