# Job 1 — Ground-truth retrieval benchmark

Ground-truth reference touch supplied to every method; 50 held-out objects
(951–1000), 8 touch locations each. TaRF excluded (model still training).

## Main comparison

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| Tactile Normal Quilting | 50 | 14.58 | 0.7227 | 0.4833 | 0.04991 |
| ObjectFolder INR | 50 | 18.26 | 0.8166 | 0.3821 | 0.02401 |
| Ours (coarse transfer) | - | pending | pending | pending | pending |
| Ours (refined) | - | pending | pending | pending | pending |

## Refinement-network ablations (same benchmark)

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| w/o temporal FiLM | 50 | 30.04 | 0.9187 | 0.1163 | 0.00563 |
| w/o normal concatenation | 50 | 25.31 | 0.8756 | 0.1943 | 0.01057 |

**Caveat on "w/o normal concatenation".** The checkpoint the plan specifies for
this arm (`..._cond-film-normal`) was trained with a different recipe from the
other two arms, not only a different conditioning scheme: 37 epochs of a
100-epoch schedule (vs a completed 20-epoch schedule), and without
`zero_init_final` or `lambda_delta`. Best val PSNR 25.93 vs 32.13 for the full
model. The gap therefore mixes conditioning with training recipe and overstates
the effect of concatenation alone. "w/o temporal FiLM" is a clean ablation
(only `--time_cond` differs).

Assets: `log/paper_job1_figure_assets/` (per-panel PNGs + contact sheet),
`log/paper_job1_refine_ours/videos/` (predicted videos).
LaTeX table body: `paper_experiments/job1_gt_retrieval/table_body.tex`.
