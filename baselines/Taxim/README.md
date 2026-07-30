# Taxim baseline for PatchMatch_touch

This directory contains the direct, non-learning Taxim baseline required by
`baselines/Baselines.pdf`. For every frame it constructs a height map at the
current pressing depth, applies Taxim's gel-deformation approximation, and runs
Taxim's calibrated optical renderer. The PDF is authoritative if it differs
from the one-page baseline notes.

The prediction path never reads the query tactile video. That video is opened
only after rendering, for output packaging and optional evaluation.

## Command convention

Run all commands below from the `PatchMatch_touch` repository root. Inputs use
the locations specified by `baselines/Baselines.pdf`. The wrappers use the
Conda environment named `Taxim`; set `TAXIM_PYTHON=/path/to/python` to
override it.

```text
log/real_data_gt_retrieval/<object>/
Taxim/results/gen_contact_full_pseudo_mini/<object>/
Taxim/results/gen_contact_full_query_pseudo_mini/<object>/
```

Print every available option with:

```bash
conda run -n Taxim python baselines/Taxim/run_baseline.py --help
```

## Environment

The checked-in calibration was generated for Taxim's sensor model. Create the
dedicated environment from the repository root:

```bash
conda env create -f baselines/Taxim/environment.yml
conda activate Taxim
```

For rendering without the optional learned metrics, the lighter environment
used during development is:

```bash
conda create -n Taxim -c defaults python=3.10 numpy=1.26 opencv scipy \
  scikit-image matplotlib pillow pyyaml tqdm pytest pip setuptools=75.8
conda run -n Taxim pip install trimesh open3d pyrender rtree
```

`torch`, `torchvision`, and `lpips` are needed only when evaluation is enabled.
Use `--skip_eval` for a CPU-safe renderer smoke test.

## Required inputs and assets

Common arguments match `transfer_pipeline.py`: `--ref_dir`, `--query_dir`,
`--save_dir`, `--scale`, `--video_type`, `--retrieval_mode`, and optional
`--tsv`. Modalities are `shadow`, `sim`, and `tactile_normal`.

Simulation requires:

- `{idx}_scale{scale}_height.npz`, the known query geometry;
- `{idx}_render_mask.mp4`, used only for output frame count, resolution, and FPS;
- the chosen reference/query modality videos for packaging and evaluation.

The generated pseudo-mini data uses the known `back_forth_press` schedule:
0→10→0 mm in 50 frames. It does not estimate depth from tactile RGB.

Real data must be extracted under
`log/real_data_gt_retrieval/<object>`. Each query requires:

- `{idx}_pose_contact.npz`, containing the per-frame ArUCo `tvecs`;
- `{idx}_contact_data.npz`, containing `tvec_0`, `sensor_z_0`, `height_map_0`,
  `valid_depth_remap`, and `mask_crop`;
- `{idx}_render_mask.mp4` for timing only.

Real pressing depth follows the repository's capture coordinate convention:

```text
depth_mm = clip(
    (depth_sign * dot(sensor_z_0, tvec_i - tvec_0) * 1000)
    * depth_scale + depth_offset_mm
)
```

By default, real rendering now uses `--real_geometry_mode full_pose`. It
reuses `real_data_transfer._gelsight_processing.ortho_project_raw` and the
capture calibration loaded from `log/gelsight_sensor_offset.json` to rerasterize
`object_cache_{view}.npz` at every ArUCo `rvec/tvec`. Contact is consequently
computed between the current gel plane and current sensor-frame object
surface, rather than by shifting one static height map. The former
translation-only implementation remains available as
`--real_geometry_mode legacy_scalar` for controlled comparisons.

Only bounded missing-pose runs of at most `--interp_max_gap` frames are
interpolated. Defaults are a one-frame Gaussian smoothing sigma, `[0, 10]` mm
clipping, and the repository's `-5 mm` surface threshold offset. Capture-specific
calibration is selected with `--sensor_offset_file`; sign, offset, scale,
gap, smoothing, clipping, optical calibration, gel map, background, and mesh
provenance are all explicit CLI options.

Optical assets default to `calibs/gelsight_pseudo_mini/`:
`dataPack.npz`, `polycalib.npz`, `shadowTable.npz`, and `gelmap5.npy`.

## Run

Simulation, object 1:

```bash
bash baselines/Taxim/scripts/run_sim.sh \
  --ref_dir Taxim/results/gen_contact_full_pseudo_mini/1 \
  --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/1 \
  --query_indices 0 \
  --save_dir log/baselines/taxim/sim_object1 \
  --skip_eval --debug_images
```

Real data:

```bash
TAXIM_REAL_OBJECT_ID=1 \
  bash baselines/Taxim/scripts/run_real.sh \
  --query_indices 1 \
  --save_dir log/baselines/taxim/real_object1 \
  --sensor_offset_file log/gelsight_sensor_offset.json \
  --skip_eval --debug_images
```

`TAXIM_REAL_DATA_ROOT` defaults to `log/real_data_gt_retrieval`. Set it when
the extracted dataset is elsewhere, or pass explicit `--ref_dir` and
`--query_dir` after the wrapper defaults.

Use `--dry_run` to validate pairing, paths, geometry, timing, and ArUCo depth
without loading Taxim or writing outputs. Simulation uses identity pairing.
Real retrieval pairs odd query `i` with even reference `i-1`. A TSV may instead
provide `query<TAB>ref`, and `dinov3` uses the repository retrieval runner.

There is no training stage: Taxim is a calibrated analytical/example-based
renderer.

## Tactile-normal data

Use these wrappers to select `{idx}_tactile_normal.mp4` instead of the old
`shadow` stream:

```bash
# Simulation
bash baselines/Taxim/scripts/run_sim_tactile_normal.sh \
  --query_indices 0 --skip_eval --debug_images

# Real
bash baselines/Taxim/scripts/run_real_tactile_normal.sh \
  --query_indices 1 --skip_eval --debug_images
```

The wrappers delegate to the standard Taxim runners with
`--video_type tactile_normal`. Taxim remains an analytical renderer and does
not train a model. For each query it:

1. resolves an identity simulation pair or an odd-query/even-reference real
   pair;
2. loads the simulation height map, or reconstructs the real contact surface
   from the ArUco/ZED geometry;
3. derives the contact-depth sequence and video timing without reading query
   tactile pixels;
4. renders every frame with Taxim's `tactile_normal` output mode; and
5. copies the reference/query tactile-normal videos for the common output
   contract and evaluates against the query video unless `--skip_eval` is
   supplied.

Thus, the query `tactile_normal` video is ground truth rather than a
generation input. Simulation uses the query height NPZ plus render-mask
timing. Real `full_pose` mode uses the saved object surface and calibrated
GelSight trajectory, so contact and deformation can change frame by frame.

Defaults are object 1, simulation roots
`Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/` and
`Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/`, and real
root `log/real_data_gt_retrieval/`. Override them with
`TACTILE_NORMAL_OBJECT_ID`, `TACTILE_NORMAL_SIM_REF_ROOT`,
`TACTILE_NORMAL_SIM_QUERY_ROOT`, or `TACTILE_NORMAL_REAL_ROOT`. Outputs
default to `log/baselines/tactile_normal/{sim,real}/object_<id>/taxim/`.

For real data the delegated `run_real.sh` continues to load
`log/gelsight_sensor_offset.json`; the full saved marker-to-gel pose is used
to align the GelSight contact geometry. Explicit trailing arguments can
override wrapper defaults.
Use `--dry_run` to verify geometry sources, calibration, contact timing, and
pairs without initializing the renderer. `--debug_images` saves rendered,
height, and contact-mask frames at the start, peak contact, and end.

## Outputs and evaluation

The runner writes:

```text
<save_dir>/
  resolved_config.json
  retrieval/results.pkl
  identity.tsv or odd_to_even.tsv
  depth/{query_idx}.npz
  transfer/{query_idx}_transferred.mp4
  transfer/{query_idx}_ref_{video_type}.mp4
  transfer/{query_idx}_query_{video_type}.mp4
  transfer/metrics.pkl
  debug/                         # with --debug_images
```

Predictions preserve query frame count, resolution, and FPS.
`metrics.pkl` is produced by PatchMatch_touch's evaluator and contains
per-touch and average MSE, PSNR, SSIM, and LPIPS. It is compatible with
`parse_metrics.py`.

Run unit tests with:

```bash
cd baselines/Taxim
conda run -n Taxim pytest -q tests
```

## Limitations

- Optical appearance depends on the supplied GelSight calibration and
  background; new hardware should be calibrated with Taxim's calibration tools.
- Real geometry is the saved depth-camera surface from preprocessing, while
  per-frame displacement comes from ArUCo. Long pose gaps intentionally fail
  instead of silently inventing motion.
- The stored static height-map route is used for PatchMatch datasets. `--mesh`
  records the source mesh for provenance; upstream Taxim mesh preprocessing is
  still available for datasets without derived height maps.
- Taxim is CPU-capable but calibrated shadow rendering is substantially slower
  than `sim` or `tactile_normal`.

## Attribution and license

This package reuses the
[PatchMatch_touch Taxim integration](https://github.com/82magnolia/PatchMatch_touch/tree/main/Taxim)
and the existing Taxim functions rather than copying their implementation.
See the [Taxim paper](https://arxiv.org/abs/2109.04027) and
[project page](https://labs.ri.cmu.edu/robotouch/taxim-simulation/).
Taxim is distributed under the included [MIT license](LICENSE).

```bibtex
@article{si2021taxim,
  title={Taxim: An Example-based Simulation Model for GelSight Tactile Sensors},
  author={Si, Zilin and Yuan, Wenzhen},
  journal={arXiv preprint arXiv:2109.04027},
  year={2021}
}
```
