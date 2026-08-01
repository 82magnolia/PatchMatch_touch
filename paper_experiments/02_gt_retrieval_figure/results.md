# Job 2 — Ground-truth retrieval qualitative figure (and our-method metrics)

## Setup

- Coarse alignment: SuperPoint + SuperGlue on curvature renders at 4x the sensor footprint,
  median translation offset (`main_retrieval_transfer_feat_match.py`).
- Refinement: ReBotNet-S, query normal map concatenated + sinusoidal temporal FiLM,
  `log/rebot_checkpoints_S_geomcat_film/best.pth` (epoch 18).
- Evaluation set: objects 951–1000, 400 touch locations, 50 frames each.
- Hardware: one RTX 2080 Ti.

## Table — ours, ground-truth retrieval benchmark

Averaged over all 400 touches and all frames.

| Method | PSNR (dB) | SSIM | LPIPS | MSE |
|---|---|---|---|---|
| Ours (coarse transfer only) | 22.74 | 0.8304 | 0.2208 | 0.0175 |
| Ours (refined) | 31.20 | 0.9214 | 0.1100 | 0.0056 |

Refinement gain: **+8.46 dB** PSNR.

Standard deviation across touches (for error bars if wanted):
PSNR coarse 8.09, refined 7.30.

## Figure

`log/paper_job02_gt_retrieval_figure/figure_gt_retrieval_3x6.png` — 3 rows (touch locations)
x 6 columns:

1. reference touch, middle frame
2. reference surface-normal render (4x field of view, 1x sensor footprint boxed in red)
3. query surface-normal render (4x field of view, 1x sensor footprint boxed in red)
4. coarse transfer, middle frame
5. refined transfer (ours), middle frame
6. ground-truth query touch, middle frame

Per-cell PNGs for **all** 400 evaluation touches are in
`log/paper_job02_gt_retrieval_figure/assets/`, named
`<object>_<touch>_01_ref_touch.png` … `_06_gt_query.png`, so a different row selection can be
made without re-running anything.

## Caveats

- The middle frame of the 50-frame press is used as the representative frame; because the
  press schedule is press-in-then-withdraw, this is also the deepest-contact frame.
- These are our method's numbers only. Baseline comparisons run on the other machine.
