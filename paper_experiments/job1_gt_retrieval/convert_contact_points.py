"""Convert the binary Open3D contact-point PLYs to .npy.

baselines/objectfolder_inr/objectfolder_inr/pose.py::load_sim_contact_points only
parses ASCII PLY, but happily loads .npy. The sim contact points are written by
Open3D as binary_little_endian, so we re-export them once, next to the original.

Writes {ply_dir}/{stem}.npy for both the reference (picked_points_fps.ply) and
query (picked_points_query.ply) point sets of every requested object.
"""
import os
import sys

import numpy as np
import open3d as o3d

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
REF_PLY = os.path.join(ROOT, "Taxim/results/object_folder_touch/{obj}/picked_points_fps.ply")
QUERY_PLY = os.path.join(ROOT, "Taxim/results/object_folder_touch_query/{obj}/picked_points_query.ply")


def convert(path):
    out = os.path.splitext(path)[0] + ".npy"
    if os.path.exists(out):
        return out, "cached"
    if not os.path.exists(path):
        return None, "missing"
    pts = np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, f"bad shape {pts.shape}"
    np.save(out, pts)
    return out, f"{pts.shape[0]} points"


def main(obj_ids):
    n_ok = n_bad = 0
    for obj in obj_ids:
        for tmpl in (REF_PLY, QUERY_PLY):
            path = tmpl.format(obj=obj)
            out, status = convert(path)
            if out is None:
                print(f"  [FAIL] {path}: {status}")
                n_bad += 1
            else:
                n_ok += 1
    print(f"converted/verified {n_ok} files, {n_bad} failures")
    return 1 if n_bad else 0


if __name__ == "__main__":
    ids = sys.argv[1:] or [str(i) for i in range(951, 1001)]
    sys.exit(main(ids))
