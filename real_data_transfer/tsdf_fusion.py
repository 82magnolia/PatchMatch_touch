"""
TSDF fusion from turntable captures.

Reads the output of capture_turntable.py and fuses masked depth maps into a
colored triangle mesh using Open3D's ScalableTSDFVolume.

Usage:
  python real_data_transfer/tsdf_fusion.py \
      --capture_dir log/captures \
      --output     log/captures/mesh.ply \
      --voxel_size 0.002 \
      --max_depth  0.8

The script needs intrinsics.json and poses.json written by capture_turntable.py
(available after the first capture in the session).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d


# ── helpers ───────────────────────────────────────────────────────────────────

def load_intrinsics(capture_dir: str, args) -> o3d.camera.PinholeCameraIntrinsic:
    """Load from intrinsics.json or fall back to CLI overrides."""
    intr_path = os.path.join(capture_dir, "intrinsics.json")
    if os.path.exists(intr_path):
        with open(intr_path) as f:
            d = json.load(f)
        fx, fy = d["fx"], d["fy"]
        cx, cy = d["cx"], d["cy"]
        w, h = d["width"], d["height"]
        print(f"Intrinsics loaded from {intr_path}  "
              f"(fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} {w}×{h})")
    elif all(v is not None for v in [args.fx, args.fy, args.cx, args.cy, args.width, args.height]):
        fx, fy = args.fx, args.fy
        cx, cy = args.cx, args.cy
        w, h = args.width, args.height
        print("Intrinsics from command-line flags.")
    else:
        sys.exit(
            "ERROR: intrinsics.json not found in capture_dir and no --fx/fy/cx/cy/width/height given.\n"
            "Either run capture_turntable.py first (it writes intrinsics.json on first capture),\n"
            "or pass the intrinsics manually."
        )
    return o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)


def load_captures(capture_dir: str, use_masked: bool) -> list[dict]:
    """Return list of {idx, T_world_from_cam, depth_path, color_path}."""
    poses_path = os.path.join(capture_dir, "poses.json")
    if not os.path.exists(poses_path):
        sys.exit(f"ERROR: poses.json not found in {capture_dir}")

    with open(poses_path) as f:
        records = json.load(f)

    captures = []
    for rec in records:
        T = rec.get("T_world_from_cam")
        if T is None:
            print(f"  WARNING: capture {rec['pick_idx']:03d} has no pose — skipping.")
            continue

        idx = rec["pick_idx"]
        suffix = "_masked" if use_masked else ""
        depth_path = os.path.join(capture_dir, f"{idx:03d}_depth{suffix}.npy")
        color_path = os.path.join(
            capture_dir,
            f"{idx:03d}_rgb{'_masked' if use_masked else ''}.png"
        )

        if not os.path.exists(depth_path) or not os.path.exists(color_path):
            print(f"  WARNING: capture {idx:03d} missing files — skipping.")
            continue

        captures.append({
            "idx": idx,
            "T_world_from_cam": np.array(T, dtype=np.float64),
            "depth_path": depth_path,
            "color_path": color_path,
        })

    print(f"Found {len(captures)} captures with valid poses.")
    return captures


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TSDF fusion of turntable captures")
    p.add_argument("--capture_dir", default="log/captures",
                   help="Directory written by capture_turntable.py")
    p.add_argument("--output", default=None,
                   help="Output mesh path (default: <capture_dir>/mesh.ply)")
    p.add_argument("--voxel_size", type=float, default=0.002,
                   help="TSDF voxel size in metres (default: 0.002 = 2 mm)")
    p.add_argument("--sdf_trunc", type=float, default=None,
                   help="SDF truncation distance in metres (default: 4 × voxel_size)")
    p.add_argument("--max_depth", type=float, default=0.8,
                   help="Maximum depth to integrate in metres (default: 0.8)")
    p.add_argument("--depth_scale", type=float, default=1000.0,
                   help="Divide raw depth values by this to get metres (default: 1000 for uint16 mm)")
    p.add_argument("--no_mask", action="store_true",
                   help="Use full (unmasked) depth/color instead of SAM-masked files")
    # Intrinsics overrides (used when intrinsics.json is absent)
    p.add_argument("--fx", type=float)
    p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float)
    p.add_argument("--cy", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or os.path.join(args.capture_dir, "mesh.ply")
    sdf_trunc = args.sdf_trunc or 4.0 * args.voxel_size

    intr = load_intrinsics(args.capture_dir, args)
    captures = load_captures(args.capture_dir, use_masked=not args.no_mask)

    if not captures:
        sys.exit("No valid captures found — nothing to fuse.")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print(f"\nIntegrating {len(captures)} frames "
          f"(voxel={args.voxel_size*1000:.1f} mm, trunc={sdf_trunc*1000:.1f} mm) …")

    for cap in captures:
        idx = cap["idx"]

        depth_mm = np.load(cap["depth_path"])  # uint16, mm
        color_bgr = cv2.imread(cap["color_path"])
        if color_bgr is None:
            print(f"  WARNING: could not read {cap['color_path']} — skipping.")
            continue
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Open3D RGBD: depth_scale divides the raw value to give metres.
        color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_mm.astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=args.depth_scale,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )

        # Open3D integrate expects T_cam_from_world (world → camera).
        extrinsic = np.linalg.inv(cap["T_world_from_cam"])

        volume.integrate(rgbd, intr, extrinsic)
        print(f"  Integrated capture {idx:03d}")

    print("\nExtracting mesh …")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Remove unreferenced vertices that can appear at the volume boundary.
    mesh.remove_unreferenced_vertices()

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    o3d.io.write_triangle_mesh(output, mesh)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"Mesh saved → {output}  ({n_v:,} vertices, {n_t:,} triangles)")

    # Quick preview
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="TSDF Mesh",
        width=960, height=540,
    )


if __name__ == "__main__":
    main()
