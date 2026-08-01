# Job 3 — 3D surface reconstruction and visuo-tactile sensor simulation

## What the figure shows

Four rows, columns = frames of one touch:

1. reference tactile normal video (the example transferred from)
2. predicted tactile normal video (our refinement network)
3. shaded 3D relief from the predicted heightmap
4. simulated RGB visuo-tactile frames (Taxim optical model)

Primary example: object 951, touch 5
(`log/paper_job03_recon_visuotactile/figure_951_5.png`).
Alternates generated: 977_1, 981_3, 969_5, 967_1, 965_3.

## Method details for the paper text

- Normals -> heights: invert Taxim's normal encoding to slopes
  (`gx = -nx/nz`, `gy = -ny/nz`), integrate with a discrete-cosine-transform Poisson
  solver with Neumann boundary conditions (`poisson_dct_neumann`), then subtract a
  least-squares plane fit (integration is only defined up to a plane).
- Sign disambiguation: the heightmap is flipped, if needed, so the contact region is
  raised relative to the background.
- Visualization: the heightmap is drawn as a shaded 3D surface with plain matte
  (Lambertian) shading computed from its own slopes, light direction (-0.5, -0.6, 0.7),
  ambient 0.28 + diffuse 0.72, viewed at elevation 55 / azimuth -62. This is the same
  style as `train_refine_scripts/time_cond_sweep/height3d_geomcat_film.py`; matplotlib's
  built-in hillshade was avoided because it produces contour-like rings on these
  video-compressed normals.
- Heights -> RGB: Taxim optical simulation at the sensor's native 480 x 640 resolution,
  using the calibrated gradient-to-RGB polynomial lookup table (`Taxim/calibs/polycalib.npz`)
  plus the smoothed gel background frame (`dataPack.npz`).
- Every step after the network is deterministic and needs no ground truth, so the whole
  chain runs at deployment time from one reference touch.

## Sentence-ready summary

> Because our method predicts tactile surface normals, its output can be integrated into a
> heightmap with a Poisson solver, yielding a 3D reconstruction of the object surface at a
> location that was never physically touched. Feeding the same heightmap through Taxim's
> calibrated optical model further produces the RGB image a camera-based tactile sensor
> would have measured, i.e. a virtual visuo-tactile measurement. Both steps are deterministic
> post-processing of the predicted normals and require no additional supervision.

## Caveat

The no-contact frames at the start and end of a press integrate into meaningless ripples
(there is no signal to integrate); figure columns are therefore sampled from the
in-contact portion of the press.
