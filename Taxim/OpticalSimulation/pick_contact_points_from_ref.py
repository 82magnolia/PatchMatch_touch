import argparse
import os

import numpy as np
import open3d as o3d
import trimesh


def sample_shell_offset(r_low, r_high):
    """Sample a 3D offset uniformly distributed within the spherical shell
    [r_low, r_high] (uniform by volume, not by radius)."""
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)
    r = (r_low ** 3 + np.random.uniform() * (r_high ** 3 - r_low ** 3)) ** (1.0 / 3.0)
    return direction * r


def main():
    parser = argparse.ArgumentParser(
        description="Sample contact points near a pre-picked reference set by "
                    "perturbing within a spherical shell and projecting onto the mesh surface.")
    parser.add_argument("--obj_path", required=True, help="Path to .obj file")
    parser.add_argument("--prepicked_ply", required=True,
                        help="Path to .ply with pre-picked reference contact points")
    parser.add_argument("--output_ply", required=True,
                        help="Output .ply file path")
    parser.add_argument("--r_low", default=0.002, type=float,
                        help="Minimum shell radius from each reference point (inclusive, default: 0.005)")
    parser.add_argument("--r_high", default=0.005, type=float,
                        help="Maximum shell radius from each reference point (inclusive, default: 0.02)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load pre-picked reference points
    ref_pcd = o3d.io.read_point_cloud(args.prepicked_ply)
    ref_points = np.asarray(ref_pcd.points)
    print(f"Loaded {len(ref_points)} reference point(s) from {args.prepicked_ply}")

    tr_mesh = trimesh.load(args.obj_path, force="mesh", process=False)
    prox = trimesh.proximity.ProximityQuery(tr_mesh)

    # For each reference point: sample an offset within the [r_low, r_high]
    # spherical shell, then project the resulting point onto the mesh surface.
    candidates = np.array([ref_pt + sample_shell_offset(args.r_low, args.r_high)
                           for ref_pt in ref_points])
    picked, _, _ = prox.on_surface(candidates)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.array(picked))
    o3d.io.write_point_cloud(args.output_ply, out_pcd)
    print(f"Saved {len(picked)} point(s) → {args.output_ply}")


if __name__ == "__main__":
    main()
