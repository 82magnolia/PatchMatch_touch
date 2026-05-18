import argparse
import os

import numpy as np
import open3d as o3d
import trimesh


def main():
    parser = argparse.ArgumentParser(
        description="Sample contact points near a pre-picked reference set via FPS-dense sampling.")
    parser.add_argument("--obj_path", required=True, help="Path to .obj file")
    parser.add_argument("--prepicked_ply", required=True,
                        help="Path to .ply with pre-picked reference contact points")
    parser.add_argument("--output_ply", required=True,
                        help="Output .ply file path")
    parser.add_argument("--r_low", default=0.01, type=float,
                        help="Minimum distance from each reference point (inclusive, default: 0.01)")
    parser.add_argument("--r_high", default=0.05, type=float,
                        help="Maximum distance from each reference point (inclusive, default: 0.2)")
    parser.add_argument("--num_dense_points", default=100_000, type=int,
                        help="Number of points in the dense surface sample (default: 100000)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load pre-picked reference points
    ref_pcd = o3d.io.read_point_cloud(args.prepicked_ply)
    ref_points = np.asarray(ref_pcd.points)
    print(f"Loaded {len(ref_points)} reference point(s) from {args.prepicked_ply}")

    # Dense surface sample
    tr_mesh = trimesh.load(args.obj_path, force="mesh", process=False)
    dense_pts = trimesh.sample.sample_surface(tr_mesh, args.num_dense_points)[0]
    print(f"Sampled {len(dense_pts)} dense surface points from {args.obj_path}")

    # For each reference point, pick one random dense point in [r_low, r_high]
    picked = []
    for i, ref_pt in enumerate(ref_points):
        dists = np.linalg.norm(dense_pts - ref_pt, axis=1)
        in_range = np.where((dists >= args.r_low) & (dists <= args.r_high))[0]

        if len(in_range) == 0:
            print(f"  [Note] Reference point {i}: no dense points in "
                  f"[{args.r_low}, {args.r_high}] — falling back to closest point.")
            chosen = dense_pts[np.argmin(dists)]
        else:
            chosen = dense_pts[np.random.choice(in_range)]

        picked.append(chosen)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.array(picked))
    o3d.io.write_point_cloud(args.output_ply, out_pcd)
    print(f"Saved {len(picked)} point(s) → {args.output_ply}")


if __name__ == "__main__":
    main()
