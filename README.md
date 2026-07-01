# PatchMatch

## PatchMatch algorithm for python. 
Currently supports CPU as well as GPU (using pycuda). 
http://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php


See Scratch.ipynb for demo and usage

## Installation
Run the following commands
```
conda create -n pm_touch python=3.10
pip install -r requirements.txt
conda install nvidia::cuda-toolkit
conda install conda-forge::pycuda
```

## Running sample
Use the following command. First mount `Objectfolder_touch` inside `data/`.
```
python demo.py --img_a data/ObjectFolder_touch/36/4_scale_50_normal.jpg --img_b data/ObjectFolder_touch/36/7_scale_50_normal.jpg --img_a_prime data/ObjectFolder_touch/36/4_scale_50_shadow.jpg --img_b_prime data/ObjectFolder_touch/36/7_scale_50_shadow.jpg
```

## Tactile Transfer Pipeline

`transfer_pipeline.py` runs the full three-stage pipeline (retrieval → PatchMatch transfer → optional ReBotNet refinement) for a flat directory of N reference touches and M query touches.

Four `--retrieval_mode` options are available:

| Mode | Description |
|---|---|
| `sim_gt_retrieval` | Auto-generates an identity TSV (query idx = ref idx). For Taxim synthetic data where reference and query are matched pairs. |
| `real_gt_retrieval` | Auto-generates an odd→even TSV (odd idx = query, even idx = ref, e.g. 1→0, 3→2). For real GelSight captures stored in a single directory. |
| `dinov2` | DINOv2 feature retrieval across all reference touches. Supports multi-modality and multi-scale feature concatenation. |
| `tsv` | Explicit TSV file via `--tsv`. |

**i) Taxim synthetic data — ground-truth identity retrieval (`sim_gt_retrieval`)**

Auto-generates an identity TSV so each query is matched to the same-index reference.

```bash
python transfer_pipeline.py \
    --ref_dir Taxim/results/gen_contact_full/52 \
    --query_dir Taxim/results/gen_contact_full_query/52 \
    --scale 100 \
    --retrieval_mode sim_gt_retrieval \
    --use_keyframe --use_accel --use_downsample_em \
    --save_dir log/pipeline/52
```

**ii) Real GelSight data — ground-truth paired retrieval (`real_gt_retrieval`)**

Even-indexed captures are treated as reference, odd-indexed as query. Auto-generates an odd→even TSV (1→0, 3→2, …). Both `--ref_dir` and `--query_dir` point to the same session directory.

```bash
python transfer_pipeline.py \
    --ref_dir log/gelsight_captures/session_01 \
    --query_dir log/gelsight_captures/session_01 \
    --scale 1 \
    --retrieval_mode real_gt_retrieval \
    --use_keyframe --use_accel --use_downsample_em \
    --save_dir log/pipeline/session_01_gt
```

**iii) Multi-modality DINOv2 retrieval — Taxim with ReBotNet**

Concatenates normal + curvature DINOv2 features for retrieval, then runs PatchMatch transfer and neural refinement.

```bash
python transfer_pipeline.py \
    --ref_dir Taxim/results/gen_contact_full/52 \
    --query_dir Taxim/results/gen_contact_full_query/52 \
    --scale 100 \
    --retrieval_mode dinov2 \
    --retrieval_modality normal curvature \
    --transfer_modality raw_normal \
    --use_keyframe --use_accel --use_downsample_em \
    --checkpoint log/rebot_checkpoints/best.pth \
    --save_dir log/pipeline/52_dinov2
```

**iv) Real GelSight data — multi-scale DINOv2 + residual ReBotNet**

Retrieval uses features from three render scales (0.5×, 1×, 2×) concatenated; ReBotNet runs in residual mode.

```bash
python transfer_pipeline.py \
    --ref_dir log/gelsight_captures/session_01 \
    --query_dir log/gelsight_captures/session_01 \
    --scale 0.5 1 2 \
    --retrieval_mode dinov2 \
    --retrieval_modality normal \
    --transfer_modality raw_normal \
    --use_keyframe --use_accel --use_downsample_em \
    --checkpoint log/rebot_checkpoints/best.pth --residual \
    --save_dir log/pipeline/session_01
```

**v) High-resolution NNF seeding (`--init_scale`)**

By default, the very first PatchMatch call for each query/ref pair (the anchor frame in `--use_keyframe` mode, or frame 0 otherwise) starts from a random NNF. `--init_scale` instead computes a seed NNF between a higher-resolution static-image variant of the query/ref pair (ignoring the touch video) and uses it to warm-start that first call, which can converge to a better correspondence than a random start.

This requires `--scale` (the base resolution used for transfer) and `--init_scale_convention` to also be set, since the two scale values are related differently depending on the data source:

| Convention | Use for | Meaning |
|---|---|---|
| `render_scale` | Real GelSight captures (`capture_gelsight_single_shot.py` / `process_single_shot.py`) | Physical FOV is proportional to the scale value, so a *larger* `--init_scale` than `--scale` zooms out to a wider physical footprint at the same pixel resolution. |
| `obj_scale_factor` | Taxim synthetic data (`gen_contact_video.py`) | The sensor's physical FOV is fixed regardless of scale, and a *larger* `obj_scale_factor` renders finer object detail (i.e. is "more zoomed in"), so a *larger* `--init_scale` than `--scale` is the useful direction here too, just for the opposite underlying reason. |

In both cases `--init_scale` must correspond to a physical footprint that is at least as large as `--scale`'s (the pipeline errors out otherwise — a smaller footprint can't cover the base image's full field of view). The `{idx}[_scale{N}]_{modality}` static files for `--init_scale` must already exist in `--ref_dir`/`--query_dir` (e.g. generated by requesting multiple `--obj_scale_factor` values in `gen_contact_video.py`, or multiple `--render_scale` values during GelSight capture).

Example — real GelSight data, seeding from an 8× render-scale static image while transferring at scale 1:

```bash
python transfer_pipeline.py \
    --ref_dir log/RealData/box_norm_fix \
    --query_dir log/RealData/box_norm_fix \
    --scale 1 \
    --retrieval_mode real_gt_retrieval \
    --use_keyframe --use_accel --use_downsample_em --use_mask \
    --init_scale 8 --init_scale_convention render_scale \
    --checkpoint log/rebot_checkpoints_S_240x320_residual/best.pth --residual \
    --save_nnf_figures \
    --save_dir log/pipeline_box_residual/
```

With `--save_nnf_figures`, an extra `{query_idx}_init_nnf.png` diagnostic figure is saved next to the usual `{query_idx}_nnf.png`, showing the init-scale static modalities and the resulting seed NNF.

Outputs are written to `--save_dir/{retrieval,transfer,enhanced}/`. Pass `--skip_refine` to stop after PatchMatch, or `--skip_retrieval` / `--skip_transfer` to resume from a later stage.