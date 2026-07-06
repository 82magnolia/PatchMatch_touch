"""
Build a rotated copy of a Taxim calibration folder (dataPack.npz + polycalib.npz),
correcting for a physical sensor mounted at a 90-degree rotation relative to another.

Handles three distinct representations that all need consistent treatment under rotation:
  - f0 / imgs: raw pixel arrays -> plain np.rot90
  - polycalib grad_r/g/b: per-(mag_bin, dir_bin) bivariate polynomials over PIXEL coordinates
    -> coefficients must be re-derived via exact coordinate substitution (not just np.rot90'd)
  - direction bin axis of grad_r/g/b: gradient direction itself rotates with the image
    -> the direction axis must be circularly shifted/interpolated, independent of the
       pixel-coordinate substitution above

All transforms below were derived and verified numerically (see conversation) against
np.rot90's actual index convention for k=-1 (90 deg clockwise) and k=1 (90 deg CCW).
"""
import argparse
from os import path as osp

import numpy as np


def rotate_poly_coeffs(coeffs, H0, W0, k):
    """
    coeffs: (..., 6) array, last dim = [x^2, y^2, xy, x, y, 1] bivariate poly coefficients
            fit over pixel coords (x=col, y=row) of an image with old shape (H0, W0).
    k: +1 (CCW, np.rot90 k=1) or -1 (CW, np.rot90 k=-1)
    Returns new coeffs valid over the rotated image of shape (W0, H0).
    """
    a, b, c, d, e, f = [coeffs[..., i] for i in range(6)]
    if k == 1:
        # old: x_old = (W0-1) - y_new, y_old = x_new
        K = W0 - 1
        ap, bp, cp = b, a, -c
        dp = c * K + e
        ep = -2 * a * K - d
        fp = a * K**2 + d * K + f
    elif k == -1:
        # old: x_old = y_new, y_old = (H0-1) - x_new
        J = H0 - 1
        ap, bp, cp = b, a, -c
        dp = -2 * b * J - e
        ep = c * J + d
        fp = b * J**2 + e * J + f
    else:
        raise ValueError("k must be +1 or -1")
    return np.stack([ap, bp, cp, dp, ep, fp], axis=-1)


def circular_shift_interp(arr, shift, axis):
    """new[j] = linear-interpolated value of arr at position (j - shift), circularly."""
    n = arr.shape[axis]
    pos = (np.arange(n) - shift) % n
    idx_lo = np.floor(pos).astype(int) % n
    idx_hi = (idx_lo + 1) % n
    frac = pos - np.floor(pos)
    lo = np.take(arr, idx_lo, axis=axis)
    hi = np.take(arr, idx_hi, axis=axis)
    frac_shape = [1] * arr.ndim
    frac_shape[axis] = n
    frac = frac.reshape(frac_shape)
    return lo * (1 - frac) + hi * frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_calib_folder", required=True, type=str,
                         help="Source calib folder containing dataPack.npz and polycalib.npz")
    parser.add_argument("--out_calib_folder", required=True, type=str,
                         help="Destination folder to write the rotated dataPack.npz/polycalib.npz")
    parser.add_argument("--k", required=True, type=int, choices=[1, -1],
                         help="+1 = rotate 90 deg CCW (np.rot90 k=1), -1 = rotate 90 deg CW (np.rot90 k=-1)")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_calib_folder, exist_ok=True)

    # --- dataPack.npz ---
    dp = dict(np.load(osp.join(args.in_calib_folder, "dataPack.npz"), allow_pickle=True))
    H0, W0 = dp["f0"].shape[:2]

    dp["f0"] = np.rot90(dp["f0"], k=args.k, axes=(0, 1))
    if "imgs" in dp:
        dp["imgs"] = np.rot90(dp["imgs"], k=args.k, axes=(1, 2))
    if "img_size" in dp:
        dp["img_size"] = np.array([dp["f0"].shape[0], dp["f0"].shape[1], dp["f0"].shape[2]])
    # touch_center/touch_radius/names are not consumed by simOptical.py and their pixel-coordinate
    # convention wasn't verified here, so they are carried through unrotated.

    np.savez(osp.join(args.out_calib_folder, "dataPack.npz"), **dp)
    print(f"Wrote dataPack.npz: f0 {H0}x{W0} -> {dp['f0'].shape[:2]}")

    # --- polycalib.npz ---
    pc = dict(np.load(osp.join(args.in_calib_folder, "polycalib.npz"), allow_pickle=True))
    bins = int(pc["bins"])
    binm = bins - 1
    y_binr = 2 * np.pi / binm
    # direction shifts by -pi/2 for k=1, +pi/2 for k=-1 (verified numerically)
    dir_shift_rad = -np.pi / 2 if args.k == 1 else np.pi / 2
    dir_shift_bins = dir_shift_rad / y_binr

    for key in ["grad_r", "grad_g", "grad_b"]:
        rotated = rotate_poly_coeffs(pc[key], H0, W0, args.k)
        pc[key] = circular_shift_interp(rotated, dir_shift_bins, axis=1)

    np.savez(osp.join(args.out_calib_folder, "polycalib.npz"), **pc)
    print(f"Wrote polycalib.npz: bins={bins}, direction axis shifted by {dir_shift_bins:.3f} bins")


if __name__ == "__main__":
    main()
