"""
Real-time RGB-D visualizer for StereoLabs ZED 2i.

Two windows:
  - cv2:   RGB (left) | colorized depth (right)
  - open3d: interactive colored point cloud (mouse to rotate/zoom/pan)

Press 'q' in the cv2 window (or close the open3d window) to exit.
"""

import sys
import numpy as np
import cv2
import open3d as o3d

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found. Install with: pip install pyzed")


DEPTH_MIN_M = 0.3   # 0.3 m
DEPTH_MAX_M = 3.0   # 3.0 m


def build_camera() -> sl.Camera:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720   # 1280×720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = DEPTH_MIN_M
    init.depth_maximum_distance = DEPTH_MAX_M
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Failed to open ZED camera: {status}")

    info = cam.get_camera_information()
    model = info.camera_model
    serial = info.serial_number
    print(f"Found device: {model}  (serial {serial})")
    return cam


def depth_to_colormap(depth_np: np.ndarray) -> np.ndarray:
    """depth_np is float32 in metres; invalid pixels are nan/inf/0."""
    depth = np.where(np.isfinite(depth_np) & (depth_np > 0), depth_np, 0.0)
    depth_clipped = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
    depth_norm = ((depth_clipped - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M) * 255).astype(np.uint8)
    # Zero out pixels that had no valid depth
    depth_norm[depth == 0] = 0
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def pointcloud_from_zed(xyzrgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    xyzrgba: (H, W, 4) float32 — ZED XYZRGBA measure.
    Returns (xyz, rgb) arrays with invalid points removed and ~1/4 subsampling.
    """
    flat = xyzrgba.reshape(-1, 4)

    xyz = flat[:, :3]
    # RGBA packed into a single float32; unpack via uint32 view
    rgba_raw = flat[:, 3].view(np.uint32)
    r = ((rgba_raw) & 0xFF).astype(np.float32) / 255.0
    g = ((rgba_raw >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((rgba_raw >> 16) & 0xFF).astype(np.float32) / 255.0
    rgb = np.stack([r, g, b], axis=1)

    valid = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
    xyz, rgb = xyz[valid][::4], rgb[valid][::4]
    return xyz, rgb


def main():
    cam = build_camera()

    rt = sl.RuntimeParameters()
    rt.enable_fill_mode = False  # keep holes; avoids edge bleeding

    image_sl = sl.Mat()
    depth_sl = sl.Mat()
    pc_sl = sl.Mat()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", width=960, height=540)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    display_w, display_h = 640, 360
    print("Streaming — press 'q' in the RGB-D window to quit.")

    try:
        while True:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            # --- retrieve frames ---
            cam.retrieve_image(image_sl, sl.VIEW.LEFT)
            cam.retrieve_measure(depth_sl, sl.MEASURE.DEPTH)
            cam.retrieve_measure(pc_sl, sl.MEASURE.XYZRGBA)

            color_bgr = image_sl.get_data()[:, :, :3]   # drop alpha channel from BGRA
            depth_np = depth_sl.get_data()               # float32, metres
            pc_np = pc_sl.get_data()                     # (H, W, 4) XYZRGBA

            # --- cv2 grid ---
            color_disp = cv2.resize(color_bgr, (display_w, display_h))
            depth_colored = cv2.resize(depth_to_colormap(depth_np), (display_w, display_h))
            cv2.imshow("RGB (left)  |  Depth (right)", np.hstack([color_disp, depth_colored]))

            # --- open3d point cloud ---
            xyz, rgb = pointcloud_from_zed(pc_np)
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
