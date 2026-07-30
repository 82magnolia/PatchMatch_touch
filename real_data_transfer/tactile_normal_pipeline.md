# How the real tactile normal videos are made

This note describes how `{idx}_tactile_normal.mp4` is produced for real GelSight
recordings, and why each step matches what the GelSight vendor code
(`gsrobotics`, the manufacturer's own software development kit, or SDK) does.

Code involved:

| File | Role |
|------|------|
| `_tactile_normal_net.py` | The network and all of the math (a self-contained port of `gsrobotics/utilities/reconstruction.py`) |
| `process_single_shot.py` | Builds the video while processing a capture session |
| `regen_normal_videos_from_shadow.py` | Rebuilds only the normal videos for sessions already on disk |
| `_gelsight_processing.py` | Contact mask, color coding, video writing |
| `gsnormal_models/nnmini.pt` | The vendor's trained network weights (copied in from a GelSight SDK install; not committed) |

---

## 1. What the video contains

For every pixel, the direction the gel surface is facing at that point — the
**surface normal**, three numbers `(nx, ny, nz)`:

- `nx` — how much the surface tilts left/right
- `ny` — how much it tilts up/down
- `nz` — how much it faces the camera (1.0 = perfectly flat, facing straight out)

They are painted into color with `color = (n + 1) / 2 * 255` (see
`normals_to_colormap` in `_gelsight_processing.py`). A flat, untouched patch is
`(0, 0, 1)`, which paints as the pale lavender `(128, 128, 255)`. This is the
identical encoding Taxim's simulated tactile normals use, which is what makes the
real and simulated videos directly comparable.

---

## 2. The pipeline, step by step

The input is the touch's `*_shadow.mp4` — the trimmed and resampled GelSight
camera recording of one press. Frame 0 of that video is, by construction, a
**no-contact frame**: nothing is touching the gel yet.

### Step 1 — Per-pixel network prediction

`net_nxny()` runs `RGB2NormNet` (the vendor's small 4-layer network) on the
frame. Its input per pixel is `(r, g, b, y/H, x/W)`: the pixel's color in 0–1
plus its position normalized by image height and width. Its output is two
numbers, `nx` and `ny`. **The network never predicts `nz`** — that is recovered
later from the unit-length constraint.

This is exactly `Reconstruction3D.get_depthmap`'s feature construction in
`gsrobotics/utilities/reconstruction.py` (`image_contact_normalized`,
`px_positions_normalized`, `np.column_stack`), same weights, same ordering.

### Step 2 — Convert to surface slopes

`nxny_to_gradients()` recovers `nz = sqrt(1 - nx² - ny²)` and forms the surface
slopes

```
gx = -nx / nz        gy = -ny / nz
```

which is verbatim `gradient_x = -normal_x / normal_z` in the vendor code.

One deliberate difference: the network's two outputs are unconstrained, so
`nx² + ny² > 1` happens (about 0.27% of pixels, up to 3% on individual frames)
and the square root is undefined there. Even just inside that rim, `nz → 0` sends
the slope to infinity — unguarded, real frames produced slope magnitudes around
1300, i.e. 89.96° vertical walls the silicone gel cannot physically form. We clamp
the tangential radius to `sin(85°)` before taking the root (`MAX_SLOPE_DEG`),
which fixes both cases at once and keeps `nz ≥ cos(85°)`. Real signal ends far
below the cap — the 99.9th percentile of slope magnitude is 3.6, i.e. 75° — so it
clips only 0.008% of valid pixels. The vendor instead substitutes the frame's mean
`nz` wherever the root goes undefined (`normal_z[np.isnan(normal_z)] = np.nanmean(normal_z)`);
radial clamping is preferred here because it is local and frame-independent,
whereas a frame-wide mean makes each pixel's value depend on the rest of the
frame.

### Step 3 — Subtract the flat-gel baseline (the zeroing step)

`baseline_gradients()` runs steps 1–2 on frame 0 (the no-contact frame) and keeps
the resulting slope field. Every frame of the touch then has that field
subtracted, per pixel:

```
gx ← gx - gx_baseline        gy ← gy - gy_baseline
```

This is the single most important step, and section 3 below explains why it is
required and why it is done on slopes rather than on `(nx, ny)`.

### Step 4 — Gate on the contact mask

`compute_contact_mask()` compares each frame against frame 0 (color difference →
Gaussian blur → threshold at `NORMAL_CONTACT_THRESHOLD = 0.025` → morphological
open then close) to get the region where something is actually pressing. Outside
that region the slopes are set to 0, which is by definition "no deformation".
Because the mask is computed from the same resampled frames the shadow video is
written from, the two videos line up frame for frame.

The vendor code gates instead on `markers_threshold` — it excludes the dark
printed marker dots from the network input. That gating is still available in
`frame_to_normals` (`marker_range=(0, 70)`) as the fallback when no contact mask
is supplied; the contact mask simply supersedes it in production, since we want
the untouched gel to read as flat rather than merely marker-free.

### Step 5 — Optional Poisson blend

`NORMAL_POISSON_BLEND` is currently **off**. When on, `poisson_blend_normals()`
solves over the entire contact mask with the background held fixed at flat,
preserving every interior gradient exactly while absorbing any residual
foreground/background offset. With the baseline subtraction in place the
foreground already meets the flat background smoothly, so the blend corrects
little and costs about 0.17 s per frame. It is kept for the case where a seam
does appear.

### Step 6 — Back to normals, and to color

`gradients_to_normals()` maps slopes back to unit normals:

```
n = [-gx, -gy, 1] / sqrt(gx² + gy² + 1)
```

Unit length holds by construction (verified: maximum deviation 1.2e-7). This is
the same formula as Taxim's `height_map_to_normals`, so both pipelines end up
expressing the same quantity the same way. `normals_to_colormap()` then writes
the color video at `VIDEO_FPS`.

---

## 3. Why this agrees with the gsrobotics conventions

### The zeroing step is the vendor's own required procedure

`nnmini.pt` is calibrated per sensor, and on this GelSight an undeformed gel does
**not** map to `(0, 0, 1)`. Measured directly on a no-contact frame, the raw
network output is about `(nx, ny) = (+0.17, -0.53)`, i.e. `nz ≈ 0.76`, a phantom
41° tilt — plus a fixed spatial pattern (per-pixel standard deviation ~0.13 and
~0.25) that is the sensor's LED lighting gradient being misread as geometry. For
scale: the difference between the no-contact frame and a frame with an object
pressing is only about 0.01–0.05, while the offset itself is 0.53. Without
zeroing, over 90% of what the video shows is the sensor's own bias rather than
the touch. On screen this appeared as a whole-frame magenta cast (`ny` pushed
negative → green channel down).

`Reconstruction3D.get_depthmap` does this unconditionally, at startup:

```python
# averages the first 50 frames while nothing is touching the sensor
if self.depth_map_zero_counter < 50:
    self.depth_map_zero += depth_map
    if self.depth_map_zero_counter == 0:
        log_message("Zeroing depth. Please do not touch the sensor...")
    ...
# and on every frame afterwards:
depth_map -= self.depth_map_zero
```

So zeroing against the untouched gel is not a correction we invented — it is the
vendor's documented startup requirement ("Please do not touch the sensor"). The
same convention appears throughout GelSight work: Taxim's optical simulation also
operates on a difference against a stored blank frame `f0`.

The only structural difference is *which* no-contact frame is used. The vendor
averages 50 live frames at sensor startup; we use frame 0 of each touch's own
recording. Ours is per touch rather than per session, which is if anything
stricter — it tracks any slow drift in the gel or the lighting between presses.

### Why the subtraction happens on slopes, not on `(nx, ny)`

Look at *what* the vendor zeroes: `depth_map`, obtained as
`poisson_dct_neumann(gx, gy)` from the slopes. Poisson integration is a **linear**
operator on those slopes, so subtracting a constant depth field is exactly
subtracting the corresponding constant slope field. It is *not* the same as
subtracting the normal components `(nx, ny)`.

Normally the distinction would be negligible, since for gentle slopes `nz ≈ 1`
and `gx ≈ -nx`. But this baseline is not gentle — it is a 41° tilt with
`nz = 0.76` — so the two choices differ by 3–5° per pixel on average and by up to
50° at the extremes. Slope space is also the better-posed choice on its own
terms: slopes are unbounded and add properly, whereas normal components are
confined to the unit disk and differencing them can leave it, which is not a
well-defined geometric operation. And it is the representation the two pipelines
genuinely share, since Taxim builds its normals as `[-dz/dx, -dz/dy, 1] / norm`.

We skip the Poisson depth integration itself: the video needs per-pixel
directions, not a height map, and integrating only to subtract a constant and
differentiate again would add nothing. Everything up to that point is identical.

### Result

Measured on the contact region, 18 touches per source, against the Taxim
simulated target:

| Contact region | Real — before zeroing | Real — after | Taxim (target) |
|---|---|---|---|
| Average up/down tilt `ny` | −0.44 | −0.0001 | +0.04 |
| Average facing-camera `nz` | 0.78 – 0.85 | 0.90 – 0.99 | 0.69 |
| No-contact frame reads as flat | no (41° tilt) | yes (exactly) | yes |

Over the full regenerated dataset (`log/real_data_gt_retrieval`, 1,061 touches,
120 sampled for statistics), the average `ny` moved from −0.236 to −0.029 against
the Taxim value of −0.010.

---

## 4. Known remaining differences

These are honest gaps, not defects in the port:

- **Real slopes are genuinely gentler.** After the fix the spread of `ny` is about
  0.20 on real data against 0.47 on Taxim, a little under half. Part is real
  physics — the silicone gel smooths sharp geometry, so the sensor cannot see the
  crisp edges Taxim renders straight from a mesh — and part may be that the
  simulated presses go deeper than the real ones. No gain or rescaling was applied
  to hide it; a per-dataset gain on `(nx, ny)` is the obvious knob if the two
  domains should be matched in contrast as well as in center.
- **The contact mask is loose.** `NORMAL_CONTACT_THRESHOLD = 0.025` makes the mask
  cover 13–76% of the frame, more than the visible contact footprint. This
  mattered a lot before the zeroing fix, because everything inside the mask picked
  up the magenta offset; it matters much less now, since outside the real contact
  the corrected slopes are near zero anyway and those pixels come out lavender
  either way.

---

## 5. Reproducing

```bash
conda activate pm_real

# During capture processing (on by default):
python real_data_transfer/process_single_shot.py --session_dir <session> --tactile_normal_video

# Or rebuild only the normal videos for sessions already on disk, in place:
python real_data_transfer/regen_normal_videos_from_shadow.py log/real_data_gt_retrieval
```

The regeneration path reads each touch's `*_shadow.mp4` and rewrites only
`*_tactile_normal.mp4`. No re-segmentation happens and no other artifact changes,
so everything stays frame-for-frame aligned with the shadow and render-mask videos.

Related reports: `log/tactile_normal_color_mismatch/report.html` (diagnosis and
the vendor-code check) and `log/step1_normal_regen/report.html` (the full
regeneration run).
