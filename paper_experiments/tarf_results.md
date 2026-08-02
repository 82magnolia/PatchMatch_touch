# TaRF baseline vs our method

Three trained TaRF checkpoints, both benchmarks, all run after the float32
conditioning-encoder fix (see below).

## Checkpoints

| Row | File | Training run | Notes |
|---|---|---|---|
| TaRF (epoch 5, finetuned) | `log/tarf_pretrained.ckpt` | `2026-07-31T10-45-38 ..._upstream_finetune_ref_even_query_odd` | Early snapshot (epoch 5 of 30) of the run that starts from the released TaRF weights. Best validation loss so far at that point: 0.1732. |
| TaRF (epoch 29, from scratch) | `log/tarf_pretrained_v2.ckpt` | `2026-07-31T10-14-41 patchmatch_sim_tactile_normal_ref_even_query_odd` | Completed 30-epoch run that deliberately does not import the released diffusion or conditioning weights. Best validation loss 0.1767 (epoch 10); 0.1795 at the saved epoch. |
| TaRF (epoch 29, finetuned) | `log/tarf_pretrained_v3.ckpt` | `2026-07-31T10-45-38 ..._upstream_finetune_ref_even_query_odd` | The same run as the first row, trained to completion. Best validation loss 0.1373 (epoch 26); 0.1411 at the saved epoch — the lowest of the three. |

## Benchmark 1 — ground-truth retrieval (50 objects, 951–1000)

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| Tactile Normal Quilting | 50 | 14.58 | 0.7227 | 0.4833 | 0.04991 |
| ObjectFolder INR | 50 | 18.26 | 0.8166 | 0.3821 | 0.02401 |
| TaRF (epoch 5, finetuned) | 50 | 10.34 | 0.6552 | 0.5391 | 0.09591 |
| TaRF (epoch 29, from scratch) | 50 | 10.48 | 0.4813 | 0.7159 | 0.09926 |
| TaRF (epoch 29, finetuned) | 50 | 11.50 | 0.5634 | 0.6797 | 0.07136 |
| Ours (coarse transfer, normals) | 50 | 22.90 | 0.8363 | 0.2102 | 0.01664 |
| Ours (refined, normals) | 50 | 31.21 | 0.9224 | 0.1085 | 0.00545 |

## Benchmark 2 — full pipeline (100 objects)

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| Tactile Normal Quilting | 100 | 15.94 | 0.7145 | 0.4874 | 0.04390 |
| ObjectFolder INR | 100 | 21.11 | 0.8331 | 0.3769 | 0.01839 |
| TaRF (epoch 5, finetuned) | 100 | 10.94 | 0.6844 | 0.5287 | 0.08498 |
| TaRF (epoch 29, from scratch) | 100 | 6.89 | 0.1144 | 0.8748 | 0.22106 |
| TaRF (epoch 29, finetuned) | 100 | 10.98 | 0.3738 | 0.7380 | 0.08205 |
| Ours (coarse transfer, normals) | 100 | 21.60 | 0.7858 | 0.3304 | 0.02078 |
| Ours (refined, normals) | 100 | 31.58 | 0.9182 | 0.1212 | 0.00485 |

## Do the predictions respond to the query?

Mean absolute difference between the middle frames of *different objects'*
predictions at one touch index, against the ground truth on the same objects.
A method that ignores its conditioning collapses towards 0%.

| Method | Objects | Spread | GT spread | % of ground truth |
|---|---|---|---|---|
| TaRF (epoch 5, finetuned) | 50 | 0.0898 | 0.1824 | 49% |
| TaRF (epoch 29, from scratch) | 50 | 0.2350 | 0.1824 | 129% |
| TaRF (epoch 29, finetuned) | 50 | 0.0735 | 0.1824 | 40% |
| Tactile Normal Quilting | 50 | 0.1707 | 0.1824 | 94% |
| ObjectFolder INR | 50 | 0.0071 | 0.1824 | 4% |
| Ours (coarse transfer, normals) | 50 | 0.1715 | 0.1824 | 94% |
| Ours (refined, normals) | 8 | 0.2007 | 0.2083 | 96% |

Ours 94–96%, quilting 94%: they vary with the query as much as the truth does.
The finetuned TaRF checkpoints reach 40–49% (half the real variation missing);
the from-scratch one hits 129% (noise, not signal); ObjectFolder INR is 4%, i.e.
almost the same image for every object.

Reproduce: `python paper_experiments/job1_gt_retrieval/prediction_diversity.py`.

## The float32 conditioning fix

TaRF inference ran fully in float16. The conditioning encoder overflowed float16
range (>65504) for the epoch-29 finetuned checkpoint, producing inf → NaN → an
all-black uint8 image with no error raised. Same object: 3.37 dB PSNR broken,
11.29 dB fixed. `baselines/TaRF/patchmatch_tarf/generator.py` now runs
`get_learned_conditioning` outside the autocast block; the UNet and decoder still
use float16. No measurable runtime cost.

Raw runs: `log/paper_job{1,2}_baselines/{tarf,tarf_v2,tarf_v3}/`.
Pre-fix runs kept for reference: `log/paper_job{1,2}_baselines/tarf_fp16cond_old/`.
HTML report: `log/paper_tarf_report.html`.
