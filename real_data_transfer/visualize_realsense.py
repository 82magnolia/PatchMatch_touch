"""
Real-time RGB-D visualizer for Intel RealSense D435i.

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
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not found. Install with: pip install pyrealsense2")


def detect_device() -> str:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit("No RealSense device found. Check USB connection.")
    dev = devices[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"Found device: {name}  (serial {serial})")
    return serial


def build_pipeline(serial: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(cfg)
    return pipeline


DEPTH_MIN_MM = 100   # 0.1 m
DEPTH_MAX_MM = 3000  # 3.0 m


def depth_to_colormap(depth_frame) -> np.ndarray:
    depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_img = np.clip(depth_img, DEPTH_MIN_MM, DEPTH_MAX_MM)
    depth_norm = ((depth_img - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def frames_to_pointcloud(pc: rs.pointcloud, color_frame, depth_frame) -> tuple[np.ndarray, np.ndarray]:
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    # Subsample depth resolution for the point cloud to reduce upload cost
    color_img = np.asanyarray(color_frame.get_data())  # BGR, (H, W, 3)
    h, w = color_img.shape[:2]

    u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
    v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)
    colors_bgr = color_img[v, u]  # (N, 3)
    colors_rgb = colors_bgr[:, ::-1].astype(np.float32) / 255.0

    # Drop invalid points (z == 0) and subsample to ~1/4 of points
    valid = vtx[:, 2] > 0
    vtx, colors_rgb = vtx[valid][::4], colors_rgb[valid][::4]
    return vtx, colors_rgb


def main():
    serial = detect_device()
    pipeline = build_pipeline(serial)
    align = rs.align(rs.stream.color)

    pc = rs.pointcloud()

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.1)  # lower = stronger smoothing
    temporal.set_option(rs.option.filter_smooth_delta, 40)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", width=960, height=540)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    # Display size for the cv2 grid (halved from capture resolution)
    display_w, display_h = 640, 360

    print("Streaming — press 'q' in the RGB-D window to quit.")

    try:
        while True:
            frameset = pipeline.wait_for_frames()
            aligned = align.process(frameset)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            depth_frame = temporal.process(depth_frame)

            # --- cv2 grid ---
            color_img = cv2.resize(np.asanyarray(color_frame.get_data()), (display_w, display_h))
            depth_colored = cv2.resize(depth_to_colormap(depth_frame), (display_w, display_h))
            cv2.imshow("RGB (left)  |  Depth (right)", np.hstack([color_img, depth_colored]))

            # --- open3d point cloud ---
            xyz, rgb = frames_to_pointcloud(pc, color_frame, depth_frame)
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
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
