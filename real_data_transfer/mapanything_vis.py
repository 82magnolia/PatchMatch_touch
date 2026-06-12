"""
Visualize MapAnything reconstruction outputs with Open3D.

Shows:
  - Colored point cloud  (pointcloud.ply)
  - Input camera poses   (poses.json from capture_dir) — blue frustums
  - Output camera poses  (poses_ma.json from recon_dir) — red frustums

Usage:
  # recon_dir defaults to <capture_dir>/mapanything_out
  python real_data_transfer/mapanything_vis.py --capture_dir log/captures

  # or point directly at the reconstruction directory
  python real_data_transfer/mapanything_vis.py \
      --capture_dir log/captures \
      --recon_dir   log/captures/mapanything_out
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Camera frustum helpers
# ---------------------------------------------------------------------------

def make_frustum(T_world_from_cam: np.ndarray,
                 K: np.ndarray,
                 width: int, height: int,
                 depth: float,
                 color: list) -> o3d.geometry.LineSet:
    """Return an Open3D LineSet representing a camera frustum in world frame.

    T_world_from_cam: (4, 4) cam→world
    K: (3, 3) pinhole intrinsics
    depth: how far the frustum extends along -Z (display scale, metres)
    color: [R, G, B] in [0, 1]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Four image corners in camera frame at the given depth
    corners_cam = np.array([
        [(0   - cx) / fx * depth, (0      - cy) / fy * depth, depth],
        [(width - cx) / fx * depth, (0      - cy) / fy * depth, depth],
        [(width - cx) / fx * depth, (height - cy) / fy * depth, depth],
        [(0   - cx) / fx * depth, (height - cy) / fy * depth, depth],
    ])  # (4, 3)

    # Camera origin in camera frame
    origin_cam = np.zeros((1, 3))
    pts_cam = np.vstack([origin_cam, corners_cam])  # (5, 3)

    # Transform to world frame
    pts_h = np.hstack([pts_cam, np.ones((5, 1))])
    pts_world = (T_world_from_cam @ pts_h.T).T[:, :3]

    # Edges: 4 rays from origin + rectangle at the far plane
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def make_axis(T_world_from_cam: np.ndarray,
              size: float = 0.02) -> o3d.geometry.LineSet:
    """Small RGB axis cross at the camera centre."""
    origin = T_world_from_cam[:3, 3]
    x_end = origin + T_world_from_cam[:3, 0] * size
    y_end = origin + T_world_from_cam[:3, 1] * size
    z_end = origin + T_world_from_cam[:3, 2] * size

    pts = np.array([origin, x_end, y_end, z_end])
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_input_poses(capture_dir: str) -> tuple[list[np.ndarray], np.ndarray, int, int]:
    """Load input poses from poses.json + intrinsics.json.

    Returns (T_list, K, width, height).
    """
    intr_path = os.path.join(capture_dir, "intrinsics.json")
    poses_path = os.path.join(capture_dir, "poses.json")

    if not os.path.exists(intr_path):
        sys.exit(f"intrinsics.json not found in {capture_dir}")
    if not os.path.exists(poses_path):
        sys.exit(f"poses.json not found in {capture_dir}")

    with open(intr_path) as f:
        intr = json.load(f)
    K = np.array([
        [intr["fx"],       0.0, intr["cx"]],
        [      0.0, intr["fy"], intr["cy"]],
        [      0.0,       0.0,       1.0],
    ], dtype=np.float64)
    width, height = int(intr["width"]), int(intr["height"])

    with open(poses_path) as f:
        records = json.load(f)

    T_list = []
    for rec in sorted(records, key=lambda r: r["pick_idx"]):
        T = rec.get("T_world_from_cam")
        if T is not None:
            T_list.append(np.array(T, dtype=np.float64))

    return T_list, K, width, height


def load_output_poses(recon_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load refined poses from poses_ma.json.

    Returns (T_list, K_list) — K may differ per frame if MapAnything refined intrinsics.
    """
    path = os.path.join(recon_dir, "poses_ma.json")
    if not os.path.exists(path):
        print(f"  poses_ma.json not found in {recon_dir} — skipping output poses")
        return [], []

    with open(path) as f:
        records = json.load(f)

    T_list, K_list = [], []
    for rec in sorted(records, key=lambda r: r["pick_idx"]):
        T = rec.get("T_world_from_cam")
        K = rec.get("intrinsics")
        if T is not None:
            T_list.append(np.array(T, dtype=np.float64))
            K_list.append(np.array(K, dtype=np.float64) if K is not None else None)

    return T_list, K_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize MapAnything reconstruction: point cloud + camera poses"
    )
    p.add_argument("--capture_dir", required=True,
                   help="Capture directory (contains intrinsics.json, poses.json)")
    p.add_argument("--recon_dir", default=None,
                   help="MapAnything output directory (default: <capture_dir>/mapanything_out)")
    p.add_argument("--frustum_depth", type=float, default=0.05,
                   help="Frustum display depth in metres (default: 0.05)")
    p.add_argument("--frustum_axis_size", type=float, default=0.02,
                   help="Camera axis cross size in metres (default: 0.02)")
    p.add_argument("--no_pointcloud", action="store_true",
                   help="Skip loading the point cloud (faster startup)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.recon_dir is None:
        args.recon_dir = os.path.join(args.capture_dir, "mapanything_out")

    geometries = []

    # ── world origin axes ────────────────────────────────────────────────────
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frustum_axis_size * 2)
    geometries.append(axes)

    # ── point cloud ──────────────────────────────────────────────────────────
    pcd_path = os.path.join(args.recon_dir, "pointcloud.ply")
    if not args.no_pointcloud:
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            print(f"Loaded point cloud: {len(pcd.points):,} pts from {pcd_path}")
            geometries.append(pcd)
        else:
            print(f"  pointcloud.ply not found in {args.recon_dir} — skipping")

    # ── input poses (blue) ───────────────────────────────────────────────────
    INPUT_COLOR = [0.2, 0.4, 1.0]   # blue
    OUTPUT_COLOR = [1.0, 0.3, 0.2]  # red

    in_poses, K_in, W, H = load_input_poses(args.capture_dir)
    print(f"Input poses:  {len(in_poses)} (blue)")
    for T in in_poses:
        geometries.append(
            make_frustum(T, K_in, W, H, args.frustum_depth, INPUT_COLOR)
        )
        geometries.append(make_axis(T, args.frustum_axis_size))

    # ── output poses (red) ───────────────────────────────────────────────────
    out_poses, K_out_list = load_output_poses(args.recon_dir)
    print(f"Output poses: {len(out_poses)} (red)")
    for i, T in enumerate(out_poses):
        K_out = K_out_list[i] if K_out_list[i] is not None else K_in
        geometries.append(
            make_frustum(T, K_out, W, H, args.frustum_depth, OUTPUT_COLOR)
        )
        geometries.append(make_axis(T, args.frustum_axis_size))

    # ── legend printed to terminal ───────────────────────────────────────────
    print()
    print("  Blue  frustums — input poses  (from poses.json)")
    print("  Red   frustums — output poses (from poses_ma.json)")
    print()
    print("Controls: left-drag=rotate  scroll=zoom  right-drag=pan  q=quit")
    print()

    o3d.visualization.draw_geometries(
        geometries,
        window_name="MapAnything Reconstruction",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
