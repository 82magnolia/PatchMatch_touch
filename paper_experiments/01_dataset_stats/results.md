# Job 1 — Dataset generation statistics

Numbers below are measured directly from the rendered data on disk
(`paper_experiments/01_dataset_stats/collect_stats.py`). Ready to be quoted in
Section "Experiments → Benchmark".

## Headline numbers

| Quantity | Value |
|---|---|
| ObjectFolder meshes used | 1000 |
| Touch locations, all datasets | 17,522 |
| Tactile video frames, all datasets | 876,100 |
| Accompanying rendered maps | 262,830 |
| Total disk footprint | 16.82 GB |

## Simulation settings (identical for all datasets)

| Setting | Value |
|---|---|
| Simulator | Taxim |
| Sensor calibration | `gelsight_pseudo_mini` |
| Sensor resolution | 240 x 320 |
| Press motion | `back_forth_press` (press in, withdraw) |
| Depth schedule | 0 -> 10 depth units over 50 steps |
| Video | 50 frames @ 5 fps |
| Modality recorded | `tactile_normal` (gel surface normal) |

## Per-dataset

| Dataset | Folder | Objects | Touches / object | Total touches | Video frames | Rotation jitter | Disk |
|---|---|---|---|---|---|---|---|
| Reference (GT-retrieval) | `gen_contact_full_tactile_normal_pseudo_mini` | 1000 | 8 (fixed) | 8,000 | 400,000 | none | 7.11 GB |
| Query (GT-retrieval) | `gen_contact_full_query_tactile_normal_pseudo_mini` | 1000 | 8 (fixed) | 8,000 | 400,000 | +/- 15 deg | 7.47 GB |
| Full pipeline | `gen_contact_raw_eval_tactile_normal_pseudo_mini` | 100 | 15.2 (min 6, max 31) | 1,522 | 76,100 | +/- 30 deg | 2.25 GB |

## Renderings per touch

Each touch also carries surface renderings from the sensor viewpoint at three fields of
view — 1x (`scale100`, exactly the sensor footprint), 2x (`scale50`), 4x (`scale25`) —
each 240 x 320 pixels. Modalities: RGB colour, surface normal, height, curvature, shape
index. That is 5 modalities x 3 scales = 15 renderings per touch location.

`normal` and `height` are also stored as float arrays: normal (240, 320, 3),
height (240, 320).

## Split

Objects 1–950 train, objects 951–1000 (50 objects, 400 touch locations) evaluation.
Hard-coded in `rebot_net/eval.py` (`all_ids[950:]`).

## Sentence-ready summary

> We build our benchmark on all 1000 ObjectFolder objects. For the
> ground-truth-retrieval benchmark we simulate 8 reference and 8 query touches per object
> (16,000 touch locations, 800,000 tactile frames); for the full-pipeline benchmark we
> simulate on average 15.2 touches on each of 100
> objects (1,522 touch locations). Every touch is a 50-frame,
> 240 x 320 tactile-normal video at 5 fps produced by Taxim with a scaled GelSight Mini
> calibration, pressing from 0 to 10 depth units and withdrawing, and is accompanied by
> RGB, normal, height, curvature and shape-index renderings of the surface at 1x, 2x and 4x
> the sensor footprint.
