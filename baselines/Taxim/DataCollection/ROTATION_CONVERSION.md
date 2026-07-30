# Rotating a Taxim calibration folder

`rotate_calib_folder.py` builds a rotated copy of a Taxim calibration folder
(`dataPack.npz`, `polycalib.npz`, `shadowTable.npz`, `gelmap5.npy` — all four
must be present in the input folder) for a sensor that is physically mounted
at a 90-degree rotation relative to another. This came up because the
`gelsight_r1.5` sensor turned out to be the same physical sensor as the
`gelsight` mini, just mounted rotated 90 degrees clockwise — so its raw
calibration data needs de-rotating before it can be used at the mini's native
480x640 orientation without passing `-override_hw` to `simOptical.py`.

A calibration folder isn't just pixel arrays, so a naive `np.rot90` on
everything is wrong. Four distinct things live in these files, and each needs
its own treatment under rotation:

1. **`f0` / `imgs`** — plain pixel arrays. A straight `np.rot90` is correct.
2. **`polycalib.npz`'s `grad_r`/`grad_g`/`grad_b`** — for a given (gradient
   magnitude bin, gradient direction bin), these store the 6 coefficients
   `[x^2, y^2, xy, x, y, 1]` of a bivariate polynomial fit **over pixel
   coordinates** (see `simOptical.py`'s `simulating()`, where `A` is built
   from a pixel meshgrid and `est_r = sum(A * params_r)`). Rotating the image
   means this polynomial must be re-expressed in the rotated pixel coordinate
   system — you cannot just `np.rot90` the coefficient array itself, because
   the coefficients are numbers that parameterize a function of position, not
   samples of an image.
3. **The direction-bin axis of that same table** — separately from (2), the
   *gradient direction* itself rotates with the image (a surface gradient
   vector rotates like any other vector). This shifts which direction bin a
   given physical gradient direction falls into, independent of the
   pixel-coordinate substitution in (2).
4. **`shadowTable.npz`'s direction-bin axis** — `simulating()` uses the same
   `grad_dir` value computed in (3) to index both `polycalib`'s direction
   bins (for shading) and `shadowTable`'s direction bins (for shadow
   casting). If polycalib's direction axis is shifted but shadowTable's
   isn't, the two fall out of sync: a given `grad_dir` now picks a shading
   response from one table and a shadow-cast direction from the other that
   no longer correspond to the same physical direction. This was caught
   empirically — a first version of this conversion left `shadowTable.npz`
   unrotated (reasoning it was "just a borrowed/shared file, not real
   per-sensor data"), which produced a visually wrong, scattered/ring-like
   shadow artifact even though the rest of the image looked fine. Whether the
   *content* is sensor-specific or borrowed is irrelevant — what matters is
   that its direction convention must move together with polycalib's.

`gelmap5.npy` (the gel dome height map) is not direction-binned calibration
content — it's a plain 2D array of heights — so it only needs the same
straight `np.rot90` as `f0`/`imgs`, with no coefficient or direction-bin
transform.

## The math

Let the old image have shape `(H0, W0)` (rows, cols). For `np.rot90(arr, k=1)`
(90° CCW) and `np.rot90(arr, k=-1)` (90° CW), the pixel-coordinate map
(`x`=col, `y`=row) between old and new images is:

- **k=1 (CCW):** `x_old = (W0-1) - y_new`, `y_old = x_new`
- **k=-1 (CW):** `x_old = y_new`, `y_old = (H0-1) - x_new`

Substituting these into the old polynomial
`P_old(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f` and collecting terms
gives an exact new coefficient vector `(a',b',c',d',e',f')` — exact because
the substitution is affine and the polynomial is degree ≤ 2, so no
interpolation error is introduced. Implemented in `rotate_poly_coeffs()`.

For the direction axis: writing the height-map finite-difference gradient as
`(d_row, d_col) = (∂z/∂row, ∂z/∂col)`, the same coordinate substitution gives
(via the chain rule):

- **k=1:** `grad_dir_new = grad_dir_old - pi/2 (mod 2*pi)`
- **k=-1:** `grad_dir_new = grad_dir_old + pi/2 (mod 2*pi)`

exactly, at every pixel — gradient **magnitude** is unaffected (rotation
preserves it). Both of these were verified numerically against
`generate_normals()`'s actual finite-difference formula on synthetic height
maps (zero error, not an approximation) before being used here.

Since `pi/2` in radians rarely lands on an exact integer number of direction
bins (`bins - 1` isn't always divisible by 4), `polycalib`'s direction axis is
shifted with **circular linear interpolation** (`circular_shift_interp()`)
rather than an integer `np.roll`, so the shift is still exact up to ordinary
interpolation between adjacent calibration bins. `shadowTable`'s direction
axis, by contrast, is shifted with an integer `np.roll` (rounding to the
nearest bin) because each `shadowTable[c, n, h]` entry is itself a
variable-length (ragged) 1D array — you can't linearly blend two different-
length shadow-falloff profiles element-wise, so exact index permutation is
the safe option there.

All of the above (both formulas, for both `k=1` and `k=-1`) were checked with
brute-force numeric tests: sampling random pixels/coefficients, applying the
substitution, and comparing against direct evaluation — see the conversation
history for the test snippets. Everything matched to floating-point
precision. The `shadowTable` fix was additionally verified with a synthetic
cone height map (guaranteed shadow-casting edges in every direction):
rotating `shadowTable`'s direction axis turned a scattered, ring-like shadow
artifact into a single coherent directional shadow.

## What does NOT get rotated

`touch_center` / `touch_radius` / `names` inside `dataPack.npz` are left
unrotated — `simOptical.py` never reads them (only `f0` is consumed; grep the
codebase to confirm), and their pixel-coordinate convention was never pinned
down, so transforming them risked silently introducing wrong data for no
benefit.

## How to run it

```bash
cd Taxim/DataCollection
python3 rotate_calib_folder.py \
    --in_calib_folder ../calibs/gelsight_r1.5 \
    --out_calib_folder ../calibs/gelsight_pseudo_mini \
    --k -1   # -1 = 90 deg clockwise, 1 = 90 deg CCW
```

`in_calib_folder` must contain all four files (`dataPack.npz`, `polycalib.npz`,
`shadowTable.npz`, `gelmap5.npy`) — all four are read, rotated, and written to
`out_calib_folder` in one pass.

## How to determine `k`'s sign for a new sensor pair

Rotation direction is not something to guess from a text description — verify
it visually. Render both `np.rot90(f0, k=1)` and `np.rot90(f0, k=-1)` of the
source sensor's blank frame as jpgs and compare the LED illumination pattern
(corner highlights, color gradient direction) against a real blank capture
from the target sensor. This is how `k=-1` was confirmed for
`gelsight_r1.5 -> gelsight_pseudo_mini`.

## Verifying the result

After conversion, run `simOptical.py` with `-data_folder
../calibs/gelsight_pseudo_mini/` and no `-override_hw` flag (since the output
is already in the target sensor's native `H0 x W0` -> `W0 x H0` shape). A
correct conversion should run without the "inconsistent size" or index
out-of-bounds warnings, and produce a coherent (not garbled, not a scattered
shadow ring) simulated tactile image.
