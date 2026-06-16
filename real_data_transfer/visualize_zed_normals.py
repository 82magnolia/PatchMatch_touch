"""
Real-time surface-normal visualizer for StereoLabs ZED 2i.

Two windows:
  - cv2:   RGB (left) | normal map (right)
            Normal (nx, ny, nz) → color ((nx+1)/2, (ny+1)/2, (nz+1)/2)
            so pointing-toward-camera surfaces appear cyan/purple,
            left/right surfaces appear red/cyan, up/down appear green/magenta.
  - open3d: interactive point cloud colored by surface normal

Press 'q' in the cv2 window (or close the open3d window) to exit.
"""

import sys
import argparse
import numpy as np
import cv2
import open3d as o3d

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found. Install with: pip install pyzed")

DEPTH_MODES = {
    "performance": sl.DEPTH_MODE.PERFORMANCE,
    "quality":     sl.DEPTH_MODE.QUALITY,
    "ultra":       sl.DEPTH_MODE.ULTRA,
    "neural":      sl.DEPTH_MODE.NEURAL,
    "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
}


def build_camera(depth_mode: sl.DEPTH_MODE, min_depth: float) -> sl.Camera:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = min_depth
    init.depth_maximum_distance = 3.0
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Failed to open ZED camera: {status}")

    info = cam.get_camera_information()
    print(f"Found device: {info.camera_model}  (serial {info.serial_number})")
    return cam


def normals_to_colormap(normals_np: np.ndarray) -> np.ndarray:
    """
    normals_np: (H, W, 4) float32 — ZED NORMALS measure (XYZ + padding).
    Returns (H, W, 3) uint8 BGR image where each normal direction maps to a color.
    Invalid pixels (NaN) are rendered black.
    """
    nxyz = normals_np[:, :, :3]                          # (H, W, 3)
    valid = np.isfinite(nxyz).all(axis=2)                # (H, W) bool

    # Map [-1, 1] → [0, 255]; invalid pixels stay black
    rgb = np.zeros((*nxyz.shape[:2], 3), dtype=np.uint8)
    rgb[valid] = ((nxyz[valid] + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

    # OpenCV expects BGR
    return rgb[:, :, ::-1]


def pointcloud_colored_by_normals(
    xyz_np: np.ndarray, normals_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    xyz_np:     (H, W, 4) float32 — ZED XYZRGBA (we only use XYZ)
    normals_np: (H, W, 4) float32 — ZED NORMALS
    Returns (xyz, rgb) with invalid points removed and ~1/4 subsampling.
    """
    xyz = xyz_np[:, :, :3].reshape(-1, 3)
    nxyz = normals_np[:, :, :3].reshape(-1, 3)

    valid = (np.isfinite(xyz).all(axis=1) & np.isfinite(nxyz).all(axis=1))
    xyz = xyz[valid][::4]
    rgb = ((nxyz[valid][::4] + 1.0) * 0.5).clip(0.0, 1.0)  # float [0,1] for Open3D
    return xyz, rgb


def main():
    parser = argparse.ArgumentParser(description="ZED 2i real-time surface normal viewer")
    parser.add_argument(
        "--depth_mode",
        choices=list(DEPTH_MODES.keys()),
        default="neural_plus",
        help="ZED depth estimation mode (default: neural_plus)",
    )
    parser.add_argument(
        "--min_depth", type=float, default=0.1,
        help="Minimum depth in metres (default: 0.1)",
    )
    parser.add_argument(
        "--confidence_threshold", type=int, default=95,
        help="ZED depth confidence threshold 0-100 (default: 95; lower accepts noisier pixels)",
    )
    args = parser.parse_args()

    cam = build_camera(DEPTH_MODES[args.depth_mode], args.min_depth)
    print(f"Depth mode: {args.depth_mode}  min_depth: {args.min_depth} m  "
          f"confidence_threshold: {args.confidence_threshold}")

    rt = sl.RuntimeParameters()
    rt.enable_fill_mode = False
    rt.confidence_threshold = args.confidence_threshold

    image_sl  = sl.Mat()
    normals_sl = sl.Mat()
    pc_sl     = sl.Mat()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud (colored by normal)", width=960, height=540)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    display_w, display_h = 640, 360
    print("Streaming — press 'q' in the RGB-D window to quit.")

    try:
        while True:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(image_sl, sl.VIEW.LEFT)
            cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)
            cam.retrieve_measure(pc_sl, sl.MEASURE.XYZRGBA)

            color_bgr  = image_sl.get_data()[:, :, :3]
            normals_np = normals_sl.get_data()             # (H, W, 4) float32
            pc_np      = pc_sl.get_data()                  # (H, W, 4) float32

            # --- cv2 grid: RGB | normal map ---
            color_disp  = cv2.resize(color_bgr, (display_w, display_h))
            normal_disp = cv2.resize(normals_to_colormap(normals_np), (display_w, display_h))
            cv2.imshow("RGB (left)  |  Normals (right)", np.hstack([color_disp, normal_disp]))

            # --- open3d point cloud colored by normal direction ---
            xyz, rgb = pointcloud_colored_by_normals(pc_np, normals_np)
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            if first_frame:
                vis.add_geometry(pcd)
                vc = vis.get_view_control()
                vc.set_zoom(0.5)
                first_frame = False
            else:
                vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cam.close()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
