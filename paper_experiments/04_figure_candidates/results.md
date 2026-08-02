# Figure candidates (Local jobs only)

Full report with images: `log/paper_figure_candidates.html`
Assets: `log/paper_fig_candidates/`

Scope: `fig_gt_retrieval` and `fig_recon` only. `fig_full_pipeline` and `fig_ablation` are Dirac
jobs; `fig_teaser` and `fig_method` are illustrations, not experiment outputs. No TaRF.

## fig_gt_retrieval candidates (each is a complete 3 x 6 figure)

| Candidate | Touches (object_touch) | Verdict |
|---|---|---|
| 1 | 980_6, 964_3, 974_5 | recommended |
| 2 | 978_4, 969_3, 992_3 | strong alternative |
| 3 | 970_7, 951_7, 995_5 | usable |
| 4 | 981_2, 993_2, 994_3 | strong alternative, most detail |
| 5 | 953_4, 952_3, 955_6 | weakest |

Recommended: **candidate 1** (980_6, 964_3, 974_5). Object 964 is the strongest single panel —
the coarse transfer invents a smooth diagonal ridge and the refinement recovers the corner plus
the row of bumps, so the reader sees the network doing real work.

## fig_recon candidates

| Candidate | Touch | Verdict |
|---|---|---|
| 1 | 994_3 | recommended |
| 2 | 981_5 | weakest |
| 3 | 978_4 | strong alternative |
| 4 | 993_2 | best detail |
| 5 | 975_2 | usable, with a caveat |

Recommended: **994_3**. If the paper needs to emphasise fine-detail preservation instead, use
**993_2**; if it needs to emphasise that the prediction follows query geometry rather than copying
the reference, use **978_4**.

## Selection method

Touches must clear hard cut-offs on contact strength, structure (edge density in the contact),
refinement gain, pose difference, and normal-render coverage; `fig_recon` additionally requires
temporal change across the press. Survivors are ranked by a weighted score and spread over distinct
objects. Cut-offs and weights are in `paper_experiments/04_figure_candidates/build_candidates.py`
(`GATES`, `WEIGHTS`); per-touch scores for all 400 touches are in `candidates.json`.

Why not rank by PSNR: the highest-PSNR touches are the ones where nothing visible happens, so they
make empty-looking panels.
