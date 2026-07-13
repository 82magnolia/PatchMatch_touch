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
| `dinov3` | DINOv3 feature retrieval across all reference touches. Supports multi-modality and multi-scale feature concatenation. Requires `--dino_weights` (gated `.pth` checkpoint). |
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

**iii) Multi-modality DINOv3 retrieval — Taxim with ReBotNet**

Concatenates normal + curvature DINOv3 features for retrieval, then runs PatchMatch transfer and neural refinement.

```bash
python transfer_pipeline.py \
    --ref_dir Taxim/results/gen_contact_full/52 \
    --query_dir Taxim/results/gen_contact_full_query/52 \
    --scale 100 \
    --retrieval_mode dinov3 \
    --retrieval_modality normal curvature \
    --dino_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --transfer_modality raw_normal \
    --use_keyframe --use_accel --use_downsample_em \
    --checkpoint log/rebot_checkpoints/best.pth \
    --save_dir log/pipeline/52_dinov3
```

**iv) Real GelSight data — multi-scale DINOv3 + residual ReBotNet**

Retrieval uses features from three render scales (0.5×, 1×, 2×) concatenated; ReBotNet runs in residual mode.

```bash
python transfer_pipeline.py \
    --ref_dir log/gelsight_captures/session_01 \
    --query_dir log/gelsight_captures/session_01 \
    --scale 0.5 1 2 \
    --retrieval_mode dinov3 \
    --retrieval_modality normal \
    --dino_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
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

**vi) DINOv3-based NNF seeding (`--init_dinov3_match_scale`)**

An alternative to `--init_scale` for seeding that same first PatchMatch call: instead of computing the seed NNF with PatchMatch, it's computed via DINOv3 patch-feature matching — sparse patch matches, filtered by RANSAC homography inliers, then interpolated into a dense correspondence field with a thin-plate-spline RBF warp (`dinov3/dense_match.py`, based on the matching logic in the `dinov3/app.py` Gradio demo). `--init_scale` and `--init_dinov3_match_scale` are mutually exclusive — pick one seeding strategy per run.

`--init_dinov3_match_scale` uses the same `--scale`/convention relationship as `--init_scale` (see the table above — pass `--init_dinov3_match_scale_convention`), plus two more required flags:

- `--dinov3_weights`: path to a gated DINOv3 `.pth` checkpoint. Weights aren't bundled with the repo — request access at [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and download the file yourself (see `dinov3/README.md`).
- `--dinov3_model`: which variant the weights are for (`dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16` [default], or `dinov3_vitl16`).

DINOv3 expects RGB input, so the static modalities used for matching (`--modality`, combined) must produce exactly 3 channels — e.g. `--modality normal` or `--modality raw_normal`, not a multi-modality concatenation. The pipeline raises a clear error if this doesn't hold.

Example — same setup as above, but seeding via DINOv3 matching instead of PatchMatch:

```bash
python transfer_pipeline.py \
    --ref_dir log/RealData/box_norm_fix \
    --query_dir log/RealData/box_norm_fix \
    --scale 1 \
    --retrieval_mode real_gt_retrieval \
    --use_keyframe --use_accel --use_downsample_em --use_mask \
    --save_dir log/pipeline_box_residual_dino/ \
    --checkpoint log/rebot_checkpoints_S_240x320_residual/best.pth --save_nnf_figures --residual --scale 1 --init_dinov3_match_scale 1 --init_dinov3_match_scale_convention render_scale --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

With `--save_nnf_figures`, this saves `{query_idx}_dinov3_init_nnf.png` instead of `{query_idx}_init_nnf.png`.

**vii) DINOv3-only transfer, no PatchMatch (`main_retrieval_transfer_feat_match.py` / `--transfer_backend dinov3_feat_match`)**

`main_retrieval_transfer_feat_match.py` is a standalone alternative to `main_retrieval_transfer_accel.py` that has no PatchMatch/CUDA dependency at all. Rather than using DINOv3 to *seed* a PatchMatch EM loop, it computes **one** DINOv3 correspondence field per query/ref pair (same matching logic as `--init_dinov3_match_scale`, via `dinov3/dense_match.py`) and applies that single NNF to warp every frame of the touch video directly — no iterative refinement, keyframe propagation, acceleration, or downsampling. It keeps NNF diagnostic figures and `--eval` metrics from the main script, but has no masking of any kind (no render mask, no reference contact mask, no static mask) — every warped frame is written out as-is.

Because it never touches PatchMatch, running `--help` (or the script itself) doesn't require a CUDA context — useful on machines without a GPU set up for `pycuda`.

Its CLI mirrors the shared pieces of `main_retrieval_transfer_accel.py` (`--query_dir`, `--ref_dir`, `--retrieval_pkl`, `--modality`, `--video_type`, `--scale`, `--eval`, `--no_nnf_figures`) plus a DINOv3-specific surface:

- `--dinov3_weights` (**required** — no PatchMatch fallback exists here) and `--dinov3_model`, same as `--init_dinov3_match_scale`.
- `--dinov3_match_scale` / `--dinov3_match_scale_convention`: optional higher-resolution matching scale, same semantics as `--init_dinov3_match_scale`/`_convention` above — omit to match directly on the `--scale` images.
- `--dinov3_num_points` (default `100`), `--dinov3_stratify_threshold` (default `20.0`): DINOv3-specific sparse-matching hyperparameters.
- `--reproj_threshold` (default `3.0`) and `--transform_type` (default `rbf_homography`): the RANSAC-inlier-selection + geometric-fit hyperparameters shared by every `--matcher` backend (see section viii below) — `affine`/`homography` fit a single global RANSAC-robust transform over all matches (rigid across the whole frame); `rbf_affine`/`rbf_homography` use that same RANSAC fit only to select inliers, then interpolate a non-rigid thin-plate-spline warp through them (handles local deformation, at the cost of being less constrained where matches are sparse).

Run it directly:

```bash
python main_retrieval_transfer_feat_match.py \
    --query_dir log/RealData/box_norm_fix \
    --ref_dir log/RealData/box_norm_fix \
    --retrieval_pkl log/pipeline_box_residual_dino/retrieval/results.pkl \
    --modality normal \
    --video_type shadow \
    --scale 1 \
    --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --save_dir log/transfer_feat_match
```

Or select it from the pipeline via `--transfer_backend dinov3_feat_match`:

```bash
python transfer_pipeline.py \
    --ref_dir log/gelsight_captures/session_01 \
    --query_dir log/gelsight_captures/session_01 \
    --scale 1 --retrieval_mode real_gt_retrieval \
    --transfer_backend dinov3_feat_match \
    --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --save_dir log/pipeline/session_01_dinov3
```

Output filenames differ slightly from the `patchmatch` backend (no `_em` suffix, since there's no EM loop to disambiguate): `{query_idx}_transferred.mp4` instead of `{query_idx}_transferred_em.mp4`. `transfer_pipeline.py` accounts for this automatically in Stages 3–5 (ReBotNet refine, grid viz, eval) based on `--transfer_backend`.

Outputs are written to `--save_dir/{retrieval,transfer,enhanced}/`. Pass `--skip_refine` to stop after PatchMatch, or `--skip_retrieval` / `--skip_transfer` to resume from a later stage.

**viii) Alternative local feature matchers via image-matching-webui (`--matcher`)**

`main_retrieval_transfer_feat_match.py` can swap DINOv3 out for one of five local feature matchers vendored in `image-matching-webui/` (IMCUI), selected with `--matcher`:

| `--matcher` value | Method |
|---|---|
| `dinov3` (default) | DINOv3 patch-feature matching — unchanged from section vii above |
| `disk_lightglue` | DISK keypoints + LightGlue matcher |
| `superpoint_superglue` | SuperPoint keypoints + SuperGlue matcher |
| `loftr` | LoFTR — end-to-end dense matcher, no separate keypoint detector |
| `superpoint_lightglue` | SuperPoint keypoints + LightGlue matcher |
| `sift_lightglue` | SIFT keypoints + LightGlue matcher |

All five run through `imcui_match.py`, which calls IMCUI's `hloc.extract_features`/`hloc.match_features`/`hloc.match_dense` directly (the same functions IMCUI's own `ImageMatchingAPI` calls internally) rather than importing `imcui.api`/`imcui.ui.utils` — those pull in `gradio`/`spaces`/`datasets`/`poselib` at module level just to be importable, none of which this script needs for a single headless matching call. The raw matched keypoints from whichever method is selected are then fed into the *same* RANSAC-inlier-selection + affine/homography/RBF geometric-fit stage `dinov3/dense_match.py` uses for DINOv3 (`_fit_dense_field`), so every backend in this script shares one fitting implementation and produces directly comparable NNFs.

Setup (one-time, only needed for `--matcher` values other than `dinov3`):

```bash
pip install kornia omegaconf h5py pycolmap
pip install git+https://github.com/cvg/LightGlue.git
git clone https://github.com/Vincentqyw/SuperGluePretrainedNetwork.git \
    image-matching-webui/imcui/third_party/SuperGluePretrainedNetwork
```

Notes on the setup steps:
- `kornia` backs the `disk_lightglue` and `loftr` matchers (and is also imported by the `sift_lightglue` extractor); `omegaconf` and `h5py` and `pycolmap` are imported unconditionally by IMCUI's `hloc` modules as soon as any matcher is loaded, regardless of which one — all five are needed even if you only ever use one matcher.
- `LightGlue` has no official PyPI package, so `disk_lightglue`, `superpoint_lightglue`, and `sift_lightglue` need it installed from GitHub directly.
- `SuperGluePretrainedNetwork` provides the **SuperPoint extractor** in addition to SuperGlue itself, so it's required for `superpoint_lightglue` too, not just `superpoint_superglue`. Use **`Vincentqyw/SuperGluePretrainedNetwork`** specifically (a patched fork — this is the exact fork IMCUI's own `.gitmodules` pins), not the vanilla `magicleap/SuperGluePretrainedNetwork`: IMCUI's `SuperPoint._forward` calls `self.net(data, self.conf)`, which only the patched fork's `SuperPoint.forward(self, data, cfg={})` signature accepts — the original upstream `forward(self, data)` raises a `TypeError` at every call.
- No pretrained weights need to be downloaded manually. All five auto-download on first use: LightGlue/SuperGlue/SuperPoint weights come from the `Realcat/imcui_checkpoints` HuggingFace Hub repo (cached under `~/.cache/huggingface/hub/`); DISK and LoFTR weights are fetched by kornia's own hub / torch hub mechanism (cached under `~/.cache/torch/hub/checkpoints/`).

`--dinov3_match_scale`/`--dinov3_match_scale_convention` (higher-resolution matching, see section vii) apply to every `--matcher` choice despite the flag name (kept for backward compatibility — it predates this multi-matcher support). `--reproj_threshold`/`--transform_type` (also section vii) likewise configure the geometric fit uniformly for whichever `--matcher` is selected — there's no separate `--imcui_*` version of these.

```bash
python main_retrieval_transfer_feat_match.py \
    --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/52 \
    --ref_dir   Taxim/results/gen_contact_full_pseudo_mini/52 \
    --retrieval_pkl log/touch_retrieval/52/results.pkl \
    --modality normal \
    --video_type shadow \
    --scale 100 \
    --matcher superpoint_lightglue \
    --save_dir log/transfer_feat_match_spg_lg
```