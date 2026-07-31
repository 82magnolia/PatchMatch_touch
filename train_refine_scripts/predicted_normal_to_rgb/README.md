# predicted_normal_to_rgb

**The script that turns a predicted tactile-normal map into an RGB (GelSight) tactile image.**

`predicted_normal_to_rgb.py` — predicted normal map → Poisson heightmap → Taxim
optical simulation → GelSight RGB tactile image.

## Core functions (import these to convert your own normal maps)
- `normal_to_taxim(normal_map_hw3_in_0_1)` → RGB tactile `uint8` (sensor res): resizes
  the normal map to sensor resolution, integrates it to a heightmap with the Poisson
  solver, and renders it with Taxim.
- `taxim_rgb(heightmap)` → RGB tactile `uint8`: the Taxim optical model alone
  (calibrated gradient→RGB LUT `Taxim/calibs/polycalib.npz` + gel background
  `dataPack.npz`). Use this if you already have a heightmap.

## Run (report + videos)
```
conda activate pm_touch
CUDA_VISIBLE_DEVICES=0 python predicted_normal_to_rgb.py
```
Produces `log/geomcat_film_taxim_rgb_videos/*.mp4` (H.264) and
`log/tactile_normal_geomcat_film_taxim_rgb_report.html`.

Edit `TOUCHES` / `CKPT` at the top to point at other touches or checkpoints.

## Related (3D reconstruction, same normal→height step)
`../time_cond_sweep/height3d_geomcat_film.py` (3D stills) and
`../time_cond_sweep/height3d_video_geomcat_film.py` (3D videos).
