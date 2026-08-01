# Job 3 — Ablation study

20-object subset of the full-pipeline benchmark.

## Coarse alignment (modality and scale)

Scale naming: object-scale factor 100 = 1x sensor footprint, 50 = 2x, 25 = 4x.

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| Modality: surface normal (4x)  [default] | 20 | 27.71 | 0.8244 | 0.2797 | 0.01876 |
| Modality: RGB colour (4x) | 20 | 26.01 | 0.8211 | 0.2873 | 0.01918 |
| Modality: curvature (4x) | 20 | 27.03 | 0.8172 | 0.2895 | 0.01957 |
| Modality: height map (4x) | 20 | 26.61 | 0.8153 | 0.2853 | 0.01946 |
| Scale: 1x sensor | 20 | 28.49 | 0.8236 | 0.2919 | 0.01999 |
| Scale: 2x sensor | 20 | 28.36 | 0.8216 | 0.2908 | 0.01982 |
| Scale: 4x sensor  [default, same run as mod_normal] | 20 | 27.71 | 0.8244 | 0.2797 | 0.01876 |

## Refinement network

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| w/o neural-network refinement | 20 | 27.71 | 0.8244 | 0.2797 | 0.01876 |
| Ours (full model) | 20 | 31.40 | 0.9074 | 0.1522 | 0.00721 |
| w/o temporal FiLM | 20 | 29.90 | 0.9022 | 0.1735 | 0.00802 |
| w/o normal concatenation | 20 | 25.38 | 0.8540 | 0.2722 | 0.01499 |

LaTeX table body: `paper_experiments/job3_ablation/table_body.tex`.
