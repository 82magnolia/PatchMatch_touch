# ObjectFolder INR baseline

This baseline adapts ObjectFolder TouchNet to the `PatchMatch_touch` transfer
contract. It predicts a scalar pseudo-height at every tactile pixel from

`x, y, z, theta, cos(phi), sin(phi), displacement, w, h`

using positional encoding and a NeRF-style MLP, then renders each predicted
height map with the ObjectFolder Taxim renderer. The implementation follows
`baselines/Baselines.pdf`; the one-page implementation notes are secondary.

## Command convention

Run the commands below from the `PatchMatch_touch` repository root. The
PatchMatch runner, adapted TouchNet, required ObjectFolder Taxim renderer,
optical calibration, and upstream license are all self-contained in
`baselines/objectfolder_inr`. The full original `baselines/ObjectFolder`
checkout is not a runtime dependency. Inputs follow `baselines/Baselines.pdf`:

```text
log/real_data_gt_retrieval/<object>/
Taxim/results/gen_contact_full_pseudo_mini/<object>/
Taxim/results/gen_contact_full_query_pseudo_mini/<object>/
```

Print the complete runner interface with:

```bash
conda run -n ObjectFolder \
  python baselines/objectfolder_inr/run_baseline.py --help
```

## Installation

The runner scripts use a Conda environment named `ObjectFolder`:

```bash
conda env create -f baselines/objectfolder_inr/environment.yml
conda activate ObjectFolder
```

This installs the PyTorch CUDA 12.1 runtime; the host still needs a compatible
NVIDIA driver.

Set `OBJECTFOLDER_PYTHON=/path/to/python` to make the shell wrappers use a
different environment.

## ArUco pose conditioning

The real dataset was captured with marker ID 6 from `DICT_4X4_50`. Its
reprocessing output already stores the calibrated detections in
`{idx}_pose_contact.npz`; inference does not need to detect the marker again.
For every saved `rvec`/`tvec`, the baseline:

1. loads `log/gelsight_sensor_offset.json` and transforms its calibrated
   marker-to-gel translation into camera coordinates using
   `R_marker @ [offset_x, offset_y, -offset_z] + tvec`;
2. obtains the sensor normal from the marker rotation and converts it to
   inclination `theta` and azimuth `phi`;
3. projects marker translation onto the contact-start sensor normal to obtain
   per-frame indentation depth, aligns it to the saved contact window, and
   re-anchors it at the aligned contact start while clamping later withdrawal
   motion to zero; and
4. linearly fills occasional missing marker detections before resampling to the
   output frame count.

`--sensor_offset_file` selects the capture calibration JSON; numeric offset
arguments are not required.
The pose features used for each prediction are saved under
`<save_dir>/pose_features/`.

Only the query `render_mask.mp4` container is opened before prediction, and
only to obtain frame count, resolution, and FPS. Query tactile RGB is copied
byte-for-byte after prediction and decoded only by evaluation.

## Data and assets

Common inputs are the same as `transfer_pipeline.py`:

- `--ref_dir`, `--query_dir`, `--save_dir`
- `--scale`, `--video_type`
- `--retrieval_mode`, and optional `--tsv`

Real data must be extracted into `log/real_data_gt_retrieval/<object>` before
running because OpenCV and NumPy require ordinary filesystem paths. For one
object, pass that same directory as both `--ref_dir` and `--query_dir`.

Simulation needs the contact-point file used by Taxim generation
(`picked_points_fps.ply`, `.npy`, `.npz`, or `.json`) via `--contact_points`.
The rendered `Taxim/results/gen_contact_full_*_pseudo_mini` folders contain
height/normal images and videos but not the original xyz coordinates. The
`--allow_index_coordinate_fallback` option exists only for synthetic smoke
tests and is not a research configuration.

For `shadow` or `sim`, the runner defaults to the vendored legacy ObjectFolder
calibration:

```text
baselines/objectfolder_inr/vendor/objectfolder/calibs
```

It must contain `polycalib.npz`, `dataPack.npz`, `depth_bg.npy`, and
`real_bg.npy`. Override it with `--taxim_calibration <directory>` only for a
different compatible calibration. `tactile_normal` rendering does not require
those files.

## Training

Training consumes reference numeric height files only. It never trains on
query tactile RGB. Both Taxim `height` arrays and real
`contact_data.height_map_0` are robustly normalized and labeled
`pseudo-height` in the checkpoint and run metadata; this label is deliberately
separate from any RGB experiment.

Simulation:

```bash
bash baselines/objectfolder_inr/scripts/train.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --save_dir log/baselines/objectfolder_inr/object_1_train \
  --scale 100 --retrieval_mode sim_gt_retrieval \
  --contact_points Taxim/results/object_folder_touch/1/picked_points_fps.ply \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/1.pth
```

Real:

```bash
bash baselines/objectfolder_inr/scripts/train.sh \
  --ref_dir log/real_data_gt_retrieval/1 \
  --query_dir log/real_data_gt_retrieval/1 \
  --save_dir log/baselines/objectfolder_inr/real_1_train \
  --scale 1 --retrieval_mode real_gt_retrieval --pose_source aruco \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/real_1.pth
```

Network depth/width, encoding levels, epochs, samples per touch, batch size,
learning rate, seed, device, and checkpoint path are explicit flags. The run
scripts also support `--train_if_missing`.

`--object_mesh` is recorded as asset provenance. `--normalization_stats`
overrides checkpoint `feature_min`/`feature_max` from JSON or NPZ. The runner
accepts its versioned adapted checkpoints and legacy ObjectFolder
`ObjectFile.pth` TouchNet weights via `--object_file`; legacy weights retain
their original `[0,1]`/angle feature convention.

## Inference

One-command simulation integration check using the rendered dataset's explicit
index-coordinate fallback:

```bash
bash baselines/objectfolder_inr/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --save_dir log/baselines/objectfolder_inr/object_1_fallback \
  --query_indices 0 \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/1_fallback.pth \
  --train_if_missing --allow_index_coordinate_fallback \
  --device cuda --debug_images --skip_eval
```

This is runnable without the missing source contact PLY, but is an integration
check rather than a coordinate-conditioned research result.

Research simulation with original xyz contact points:

```bash
bash baselines/objectfolder_inr/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --save_dir log/baselines/objectfolder_inr/object_1 \
  --contact_points Taxim/results/object_folder_touch/1/picked_points_fps.ply \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/1.pth
```

Real:

```bash
bash baselines/objectfolder_inr/scripts/run_real.sh \
  --ref_dir log/real_data_gt_retrieval/1 \
  --query_dir log/real_data_gt_retrieval/1 \
  --save_dir log/baselines/objectfolder_inr/real_1 \
  --query_indices 1 \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/real_1.pth \
  --train_if_missing \
  --sensor_offset_file log/gelsight_sensor_offset.json \
  --device cuda --debug_images --skip_eval
```

## Tactile-normal data

The tactile-normal wrappers use `{idx}_tactile_normal.mp4` in place of the old
`shadow` stream while retaining ObjectFolder-INR's coordinate-conditioned
prediction:

```bash
# Simulation integration check (use measured contact points for research runs).
bash baselines/objectfolder_inr/scripts/run_sim_tactile_normal.sh \
  --query_indices 0 \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/1_tactile_normal.pth \
  --train_if_missing --allow_index_coordinate_fallback \
  --device cuda --skip_eval

# Real data; marker-to-gel calibration is loaded by run_real.sh.
bash baselines/objectfolder_inr/scripts/run_real_tactile_normal.sh \
  --query_indices 1 \
  --checkpoint log/baselines/objectfolder_inr/checkpoints/real_1.pth \
  --device cuda --skip_eval
```

The wrappers delegate to the existing runners with
`--video_type tactile_normal`; they do not change the ObjectFolder-INR model.
For every query, the pipeline:

1. obtains `(x, y, z, theta, phi)` and displacement conditions from the
   simulation contact points or the real ArUco trajectory;
2. predicts a tactile height field with the ObjectFolder-INR checkpoint;
3. renders that height field in `tactile_normal` mode at the query video's
   size, FPS, and frame count; and
4. copies the reference/query tactile-normal videos into `transfer/` and
   evaluates the prediction against the query video unless `--skip_eval` is
   supplied.

The tactile-normal videos are therefore the requested output modality and
evaluation target. Their pixels are not decoded as inputs to the INR height
prediction. Real inference derives the contact coordinate from the ArUco pose
and the complete marker-to-gel calibration in
`log/gelsight_sensor_offset.json`. Simulation research runs instead require
the original contact-point coordinates; `--allow_index_coordinate_fallback`
is only for pipeline validation.

Defaults are object 1, simulation roots
`Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/` and
`Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/`, and real
root `log/real_data_gt_retrieval/`. Set `TACTILE_NORMAL_OBJECT_ID`,
`TACTILE_NORMAL_SIM_REF_ROOT`, `TACTILE_NORMAL_SIM_QUERY_ROOT`, or
`TACTILE_NORMAL_REAL_ROOT` to override them. Outputs default to
`log/baselines/tactile_normal/{sim,real}/object_<id>/objectfolder/`.

Before a CUDA run, inspect `nvidia-smi` and select an idle device, for example
`CUDA_VISIBLE_DEVICES=2 ... --device cuda`. The simulation fallback above is
only an integration check; pass `--contact_points <picked_points_fps.ply>` and
a research checkpoint when evaluating coordinate-conditioned ObjectFolder.
Use `--dry_run` to validate pairing, calibration, paths, and checkpoint
selection without loading the model.

Use `--query_indices 1 3 5`, `--dry_run`, `--debug_images`, or `--skip_eval`
for focused validation.

Three lightweight simulation examples can be trained and rendered with:

```bash
bash baselines/objectfolder_inr/scripts/run_examples.sh
```

The defaults are objects/queries `1:0 10:3 100:7`. Override them with:

```bash
OBJECTFOLDER_EXAMPLES="2:1 25:4" \
  bash baselines/objectfolder_inr/scripts/run_examples.sh
```

These examples use the clearly labeled index-coordinate fallback because the
distributed rendered SimData omits the original contact PLY. They validate the
pipeline and video contract, not coordinate-conditioned research accuracy.

## Outputs and evaluation

The output layout matches the PDF and `transfer_pipeline.py`:

```text
<save_dir>/
  identity.tsv or odd_to_even.tsv
  resolved_config.json
  retrieval/results.pkl
  pose_features/{query_idx}.npz
  debug/{query_idx}_depth0.png
  debug/{query_idx}_peak.png
  transfer/{query_idx}_transferred.mp4
  transfer/{query_idx}_ref_{video_type}.mp4
  transfer/{query_idx}_query_{video_type}.mp4
  transfer/metrics.pkl
```

Predictions preserve query frame count, resolution, and FPS. Unless
`--skip_eval` is used, evaluation writes `per_touch` and `average`
MSE/PSNR/SSIM/LPIPS fields readable by:

```bash
python parse_metrics.py --dir <save_dir>
```

Run the lightweight tests with:

```bash
PYTHONPATH=baselines/objectfolder_inr \
  conda run -n ObjectFolder pytest -q baselines/objectfolder_inr/tests
```

## Limitations

- ArUco-derived coordinates are in the capture camera frame unless an external
  camera-to-object transform is supplied upstream. They are internally
  consistent within one capture object.
- Simulation xyz cannot be reconstructed exactly from rendered images; retain
  the Taxim contact-point file.
- The imported 2022 ObjectFolder Taxim calibration renders internally at
  120x160 and is resized to the query resolution.
- A useful checkpoint needs diverse reference coordinates and displacements;
  the smoke-test settings only verify plumbing.

## References and attribution

- [Adapted ObjectFolder code](https://github.com/yjun13568/ObjectFolder)
- [ObjectFolder 2.0 project](https://ai.stanford.edu/~rhgao/objectfolder2.0/)
- [ObjectFolder 2.0 paper](https://arxiv.org/abs/2204.02389)

The minimal vendored ObjectFolder renderer and calibration remain under their
original CC BY 4.0 license in
`vendor/objectfolder/LICENSE`. New integration code should retain that license
and cite ObjectFolder 2.0 in research use.
