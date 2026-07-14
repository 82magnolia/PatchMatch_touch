"""Recompute curvature JPGs from already-rendered height.npz files for every
Taxim/results/*_pseudo_mini directory, now that height2laplacian's
boundary-artifact fix (mask-aware, separately-normalized composite of
interior + boundary-ring curvature) has landed.

Unlike the real-data side (which needs a full process_single_shot.py rerun
since curvature isn't separable from the rest of that pipeline), synthetic
curvature is a pure function of a saved height map -- no mesh re-render
needed. Background in Taxim's height maps is an exact, hard 0.0 (see
generateHeightMap's np.zeros init + scatter-write of only in-bounds mesh
vertices), so `H != 0` is an exact reconstruction of the object footprint
mask without needing the original mesh/contact-point inputs.

This intentionally does NOT import Taxim/OpticalSimulation/simOptical.py --
that pulls in open3d/pyrender/trimesh (heavy mesh-rendering deps unrelated to
this pure post-processing step). height2laplacian is inlined below, kept in
sync by hand with simOptical.py's version (same convention already used by
real_data_transfer/_gelsight_processing.py's copy).

Usage:
    python train_refine_scripts/recompute_pseudo_mini_curvature.py
    python train_refine_scripts/recompute_pseudo_mini_curvature.py --dry_run
    python train_refine_scripts/recompute_pseudo_mini_curvature.py \\
        --dirs Taxim/results/gen_contact_full_pseudo_mini
"""
import argparse
import glob
import os
from os import path as osp

import cv2
import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_DIRS = sorted(
    d for d in glob.glob(osp.join(PROJECT_ROOT, "Taxim", "results", "*_pseudo_mini"))
    if osp.isdir(d)
)


# ---- height2laplacian, hand-synced with Taxim/OpticalSimulation/simOptical.py ----

def _erode_footprint_mask(mask, mask_erode_px):
    mask_u8 = (np.asarray(mask) != 0).astype(np.uint8)
    k = 2 * mask_erode_px + 1
    return cv2.erode(mask_u8, np.ones((k, k), np.uint8))


def raw_laplacian(H):
    gy, gx = np.gradient(H)
    L = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    L = cv2.GaussianBlur(L, (5, 5), 0)
    return L


def _normalize_field(L, valid, clip_percentile):
    L_valid = L[valid]
    if L_valid.size == 0:
        return np.full(L.shape, 128, dtype=np.uint8)
    if clip_percentile > 0:
        Lmin = float(np.percentile(L_valid, clip_percentile))
        Lmax = float(np.percentile(L_valid, 100 - clip_percentile))
    else:
        Lmin, Lmax = float(L_valid.min()), float(L_valid.max())
    out = np.where(
        L < 0,
        0.5 * (L - Lmin) / (-Lmin + 1e-8),
        0.5 + 0.5 * L / (Lmax + 1e-8),
    )
    out = np.clip(out, 0.0, 1.0)
    return (255 * out).astype(np.uint8)


def height2laplacian(H, mask=None, mask_erode_px=4, clip_percentile=1.0):
    L = raw_laplacian(H)
    if mask is None:
        return _normalize_field(L, np.ones_like(L, dtype=bool), clip_percentile)
    interior = _erode_footprint_mask(mask, mask_erode_px) != 0
    interior_encoded = _normalize_field(L, interior, clip_percentile)
    boundary_encoded = _normalize_field(L, ~interior, clip_percentile)
    return np.where(interior, interior_encoded, boundary_encoded)


# ---- driver ----

def find_height_npz_files(base_dir):
    return sorted(glob.glob(osp.join(base_dir, "*", "*_height.npz")))


def curvature_path_for(height_npz_path):
    # "{prefix}_height.npz" -> "{prefix}_curvature.jpg"
    assert height_npz_path.endswith("_height.npz")
    return height_npz_path[: -len("_height.npz")] + "_curvature.jpg"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                   help="Taxim/results/*_pseudo_mini directories to process "
                        "(default: auto-discovered).")
    p.add_argument("--dry_run", action="store_true",
                   help="Count files and print the plan without writing anything.")
    args = p.parse_args()

    if not args.dirs:
        print("No *_pseudo_mini directories found under Taxim/results -- nothing to do.")
        return

    all_files = []
    for d in args.dirs:
        files = find_height_npz_files(d)
        print(f"{d}: {len(files)} height.npz files")
        all_files.extend(files)

    total = len(all_files)
    print(f"\nTotal: {total} curvature files to recompute")
    if args.dry_run:
        print("Dry run -- no files written.")
        return

    done = 0
    skipped = 0
    for i, height_path in enumerate(all_files):
        curv_path = curvature_path_for(height_path)
        try:
            H = np.load(height_path)["height"].astype(np.float64)
        except Exception as e:
            print(f"  [{i+1}/{total}] SKIP (failed to load {height_path}): {e}")
            skipped += 1
            continue

        mask = (H != 0)
        curv = height2laplacian(H, mask=mask)
        cv2.imwrite(curv_path, curv)
        done += 1

        if done % 2000 == 0 or done == total:
            print(f"  [{done}/{total}] ({skipped} skipped)")

    print(f"\nDone. {done}/{total} curvature files recomputed ({skipped} skipped).")


if __name__ == "__main__":
    main()
