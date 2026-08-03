"""Move a benchmark touch to a nearby spot on the object and re-simulate it.

Works on either side of a pair: --which query moves a query touch (the teaser
case, where the reference is held fixed), --which ref moves a reference touch
(the reconstruction-figure case, where the query is held fixed). The two sets
were generated with slightly different settings -- the query touches got a small
random sensor rotation, the reference touches did not -- and this script keeps
whichever convention it is reproducing.

The teaser reads as more of a challenge when the query location is clearly not
the reference location. Rather than hunting for a better pair among the touches
that happen to exist, this places new contact points a chosen distance away from
an existing one, along the object's own surface, and runs Taxim to produce the
tactile video and the normal renders for each -- exactly the same simulation
settings the benchmark itself was generated with
(train_refine_scripts/gen_contact_query_tactile_normal_pseudo_mini/_run.sh).

The result is a directory of new query touches for one object, which
run_full_pipeline_local.py can then treat as the query set while the object's
original reference touches stay the reference set.

One trap, learned the hard way: gen_contact_video.py derives the object's scale
from the *spread of the points in the contact PLY* (the longest-axis extent is
normalised to --obj_scale_factor millimetres), not from the mesh. Feeding it a
tight cluster of shifted points therefore blows the object up and every render
comes out at the wrong field of view. So the PLY written here always keeps the
object's original contact points as anchors, and the shifted points are appended
after them; the anchors fix the normalisation, and the script checks that the
extent still matches the original before simulating.

    python shift_query_touch.py --obj 951 --touch 7 --dists 0.02 0.04 --n_dirs 4
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
import trimesh

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OBJ_DIR = f"{ROOT}/Taxim/data/ObjectFolder"
QUERY_PTS = f"{ROOT}/Taxim/results/object_folder_touch_query"
REF_PTS = f"{ROOT}/Taxim/results/object_folder_touch"
OUT = f"{ROOT}/log/paper_job04_paper_figures/shifted"
GEN = f"{ROOT}/Taxim/OpticalSimulation/gen_contact_video.py"
CALIB = f"{ROOT}/Taxim/calibs/gelsight_pseudo_mini"


def surface_frame(mesh, point):
    """Outward normal and two tangents of the mesh at the surface point nearest `point`."""
    closest, _, face = trimesh.proximity.closest_point(mesh, [point])
    n = mesh.face_normals[face[0]]
    n = n / np.linalg.norm(n)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return closest[0], n, t1, t2


def shifted_points(mesh, origin, dists, n_dirs):
    """Points `dists` away from `origin` in `n_dirs` tangent directions, pulled
    back onto the surface so they are real contact points and not floating."""
    base, n, t1, t2 = surface_frame(mesh, origin)
    scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    pts, labels = [], []
    for d in dists:
        for k in range(n_dirs):
            a = 2 * np.pi * k / n_dirs
            step = (np.cos(a) * t1 + np.sin(a) * t2) * d * scale
            moved, _, _ = trimesh.proximity.closest_point(mesh, [base + step])
            pts.append(moved[0])
            labels.append(dict(dist_frac=d, direction_deg=round(np.degrees(a)),
                               moved_mm_of_bbox=d * scale))
    return np.array(pts), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, default=951)
    ap.add_argument("--which", default="query", choices=["query", "ref"],
                    help="which side of the benchmark pair to move")
    ap.add_argument("--touch", type=int, default=7, help="touch index to move")
    ap.add_argument("--dists", type=float, nargs="+", default=[0.02, 0.04],
                    help="how far to move, as a fraction of the object's bounding-box diagonal")
    ap.add_argument("--n_dirs", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.obj}_{args.which}{args.touch}"
    save_dir = f"{OUT}/{tag}"
    os.makedirs(save_dir, exist_ok=True)

    mesh = trimesh.load(f"{OBJ_DIR}/{args.obj}/model.obj", force="mesh")
    pts_file = (f"{QUERY_PTS}/{args.obj}/picked_points_query.ply" if args.which == "query"
                else f"{REF_PTS}/{args.obj}/picked_points_fps.ply")
    pcd = o3d.io.read_point_cloud(pts_file)
    origin = np.asarray(pcd.points)[args.touch]
    pts, labels = shifted_points(mesh, origin, args.dists, args.n_dirs)
    print(f"object {args.obj}, {args.which} touch {args.touch} at {origin.round(3)} "
          f"(from {os.path.basename(pts_file)})")

    # Anchors first, shifted points after: the anchors reproduce the original
    # point spread, which is what sets the render scale (see the note above).
    anchors = np.asarray(pcd.points)
    all_pts = np.vstack([anchors, pts])
    first_new = len(anchors)

    def long_axis_extent(p):
        a = int((p.max(axis=0) - p.min(axis=0)).argmax())
        return float(p[:, a].max() - p[:, a].min()), a

    ext_ref, axis_ref = long_axis_extent(anchors)
    ext_new, axis_new = long_axis_extent(all_pts)
    if axis_new != axis_ref or abs(ext_new - ext_ref) > 1e-6 * max(ext_ref, 1.0):
        sys.exit(f"the shifted points changed the point-cloud extent "
                 f"({ext_ref:.5f} -> {ext_new:.5f} on axis {axis_ref} -> {axis_new}); "
                 f"the renders would come out at a different scale. Use smaller "
                 f"distances or a different direction.")
    print(f"point spread preserved: longest-axis extent {ext_ref:.5f} on axis {axis_ref}; "
          f"new touches are indices {first_new}..{len(all_pts) - 1}")

    ply = f"{save_dir}/contact_points.ply"
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    o3d.io.write_point_cloud(ply, out)
    np.save(f"{save_dir}/labels.npy",
            [dict(index=first_new + i, **m) for i, m in enumerate(labels)],
            allow_pickle=True)

    # Same simulation settings as the benchmark's own query touches. No EGL here:
    # on this machine Taxim's renderer needs GLX.
    cmd = [sys.executable, GEN,
           "--obj_path", f"{OBJ_DIR}/{args.obj}/model.obj",
           "--contact_ply", ply,
           "--mode", "back_forth_press",
           "--depth_range_info", "0.", "10.", "50",
           # the benchmark's query touches were simulated with a small random
           # sensor rotation, its reference touches without one
           *(["--rand_contact_theta", "--rand_contact_theta_mag", "0.26179938779"]
             if args.which == "query" else []),
           "--modalities", "tactile_normal",
           "--save_dir", f"{save_dir}/{args.obj}",
           "--obj_scale_factor", "100.", "50.", "25.",
           "--override_hw", "240", "320",
           "--data_folder", CALIB]
    for i, m in enumerate(labels):
        print(f"  touch {first_new + i}: {m['dist_frac'] * 100:.1f}% of the bounding box "
              f"({m['moved_mm_of_bbox'] / ext_ref * 100:.1f} mm at the benchmark's "
              f"100 mm normalisation), direction {m['direction_deg']} deg")
    print("\nsimulating", len(all_pts), "touches "
          f"({len(pts)} new, {len(anchors)} anchors) ...")
    r = subprocess.run(cmd, cwd=f"{ROOT}/Taxim/OpticalSimulation")
    if r.returncode != 0:
        sys.exit(f"gen_contact_video.py failed ({r.returncode})")
    print("wrote", f"{save_dir}/{args.obj}")


if __name__ == "__main__":
    main()
