# Job 2 — Full-pipeline benchmark

Retrieval is part of the system under test. 100 objects, 396 held-out query
touches, 1121 reference touches; 4 queries per object (3 when an object has fewer
than 9 touches), seeded random draw. TaRF excluded.

| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|---|
| Tactile Normal Quilting | 100 | 15.94 | 0.7145 | 0.4874 | 0.04390 |
| ObjectFolder INR | 100 | 21.11 | 0.8331 | 0.3769 | 0.01839 |
| Ours (coarse transfer, normals) | 100 | 21.60 | 0.7858 | 0.3304 | 0.02078 |
| Ours (refined, normals) | 100 | 31.58 | 0.9182 | 0.1212 | 0.00485 |
| Ours (coarse transfer, curvature) | 100 | 21.60 | 0.7850 | 0.3354 | 0.02157 |
| Ours (refined, curvature) | 100 | 31.51 | 0.9179 | 0.1222 | 0.00499 |

Split manifest: `paper_experiments/job2_full_pipeline/splits.json`.
LaTeX table body: `paper_experiments/job2_full_pipeline/table_body.tex`.
