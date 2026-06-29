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

Outputs are written to `--save_dir/{retrieval,transfer,enhanced}/`. Pass `--skip_refine` to stop after PatchMatch, or `--skip_retrieval` / `--skip_transfer` to resume from a later stage.