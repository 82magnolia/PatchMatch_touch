"""
ARuCO board calibration script.

Captures multiple views of the turntable marker board to establish the 3D positions
of all markers in the board's coordinate frame (origin = lowest marker ID).  The
result is saved as board_config.json and can be loaded by capture_turntable.py via
--board_config for more accurate joint board pose estimation.

Usage:
  python real_data_transfer/calibrate_board.py --log_dir log/captures --marker_size 0.035

Controls:
  c — capture current frame (accumulate ARuCO detections)
  b — compute board layout from accumulated frames and save
  q — quit

Calibration tips:
  - Ensure all markers are visible in most frames.
  - Capture from at least 3 distinct viewpoints (tilt/shift the camera slightly,
    or rotate the turntable to different angles).
  - 20+ frames gives a stable result; watch the co-visible count in the HUD.
  - After pressing 'b', white squares in the 3D view show the computed board layout.
    Per-frame coloured squares should cluster tightly around the white ones.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not found. Install with: pip install pyrealsense2")

# ── constants ─────────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 640, 360
CAPTURE_W, CAPTURE_H = 1280, 720
ARUCO_DICT = cv2.aruco.DICT_4X4_50

_COLORS = [
    [1.00, 0.25, 0.25], [0.25, 1.00, 0.25], [0.25, 0.45, 1.00],
    [1.00, 1.00, 0.20], [0.20, 1.00, 1.00], [1.00, 0.20, 1.00],
    [1.00, 0.60, 0.00], [0.60, 0.00, 1.00],
]


# ── realsense helpers ─────────────────────────────────────────────────────────

def detect_device() -> str:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit("No RealSense device found.")
    dev = devices[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"Found device: {name}  (serial {serial})")
    return serial


def build_pipeline(serial: str):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, CAPTURE_W, CAPTURE_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, CAPTURE_W, CAPTURE_H, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    return pipeline, profile


# ── aruco helpers ─────────────────────────────────────────────────────────────

def build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(frame_bgr, detector, camera_matrix, dist_coeffs, marker_size):
    """Returns ({marker_id: T_marker_in_cam}, corners, ids)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}, None, None
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size, camera_matrix, dist_coeffs
    )
    poses = {}
    for i in range(len(ids)):
        mid = int(ids[i].flat[0])
        R, _ = cv2.Rodrigues(rvecs[i].flatten())
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvecs[i].flatten()
        poses[mid] = T
    return poses, corners, ids


# ── board layout computation ──────────────────────────────────────────────────

def average_transforms(T_list: list) -> np.ndarray:
    Rs = [T[:3, :3] for T in T_list]
    ts = [T[:3, 3] for T in T_list]
    R_mean = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        R_mean = U @ np.diag([1, 1, -1]) @ Vt
    T = np.eye(4)
    T[:3, :3] = R_mean
    T[:3, 3] = np.mean(ts, axis=0)
    return T


def compute_board_layout(frames: list, marker_size: float):
    """Compute each marker's corner positions in the board (origin-marker) frame.

    Returns (layout, origin_id) where layout = {marker_id: corners (4,3)}.
    """
    all_ids = set()
    for f in frames:
        all_ids.update(f.keys())
    if not all_ids:
        return None, None

    origin_id = min(all_ids)
    h = marker_size / 2.0
    local_corners = np.array([
        [-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0],
    ], dtype=np.float64)

    layout = {origin_id: local_corners.copy()}

    for mid in sorted(all_ids):
        if mid == origin_id:
            continue

        samples = []
        for f in frames:
            if origin_id in f and mid in f:
                T_rel = np.linalg.inv(f[origin_id]) @ f[mid]
                samples.append(T_rel)

        if not samples:
            print(f"  WARNING: marker {mid} never co-visible with origin "
                  f"{origin_id} — skipped.")
            continue

        T_avg = average_transforms(samples) if len(samples) > 1 else samples[0]
        corners_h = np.hstack([local_corners, np.ones((4, 1))])
        corners_in_board = (T_avg @ corners_h.T).T[:, :3]
        layout[mid] = corners_in_board
        print(f"  Marker {mid}: {len(samples)} co-visible frames, "
              f"mean z offset = {corners_in_board[:, 2].mean() * 1000:.2f} mm")

    # Fit a plane to all corners and project onto it.
    # This removes the depth-noise z-offset that persists after averaging SE(3).
    all_corners = np.vstack(list(layout.values()))
    centroid = all_corners.mean(axis=0)
    _, _, Vt = np.linalg.svd(all_corners - centroid)
    normal = Vt[-1]                          # plane normal (least-variance direction)
    for mid in layout:
        c = layout[mid]
        layout[mid] = c - np.outer(np.dot(c - centroid, normal), normal)

    all_corners_proj = np.vstack(list(layout.values()))
    z_spread = float(all_corners_proj[:, 2].max() - all_corners_proj[:, 2].min())
    print(f"\nPlanarity check (after projection): Z-spread = {z_spread * 1000:.2f} mm "
          f"({'good' if z_spread < 0.005 else 'poor — try more varied viewpoints'})")

    return layout, origin_id


def save_board_config(layout: dict, origin_id: int, marker_size: float,
                      output_path: str):
    data = {
        "aruco_dict": "DICT_4X4_50",
        "marker_size": marker_size,
        "origin_marker_id": origin_id,
        "markers": {str(mid): corners.tolist() for mid, corners in layout.items()},
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Board config saved → {output_path}")


# ── 3-D calibration view ──────────────────────────────────────────────────────

class CalibrationView:
    """Open3D window showing accumulated camera frustums and marker squares.

    World frame = origin-marker's coordinate frame.
    Camera k's position in world = inv(T_origin_in_cam_k).
    Marker m at frame k in world = inv(T_origin_in_cam_k) @ T_m_in_cam_k.

    When the board is rigid, repeated observations of each marker should overlap
    tightly.  After pressing 'b', the computed (averaged) board layout is shown
    as white squares for comparison.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 img_w: int, img_h: int, marker_size: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.img_w, self.img_h = img_w, img_h
        self.marker_size = marker_size
        self._alive = True
        self._origin_id: int | None = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Board Calibration 3D", width=800, height=600)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.marker_size)
        self.vis.add_geometry(axes)

    # ── geometry builders ──────────────────────────────────────────────────────

    def _frustum(self, T_cam_in_world: np.ndarray, color: list,
                 depth: float = 0.08) -> o3d.geometry.LineSet:
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        w, h = self.img_w, self.img_h
        local = np.array([
            [0, 0, 0],
            [(0 - cx) / fx * depth, (0 - cy) / fy * depth, depth],
            [(w - cx) / fx * depth, (0 - cy) / fy * depth, depth],
            [(w - cx) / fx * depth, (h - cy) / fy * depth, depth],
            [(0 - cx) / fx * depth, (h - cy) / fy * depth, depth],
        ])
        pts_h = np.hstack([local, np.ones((5, 1))])
        world_pts = (T_cam_in_world @ pts_h.T).T[:, :3]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(world_pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    def _marker_square(self, corners_world: np.ndarray,
                       color: list) -> o3d.geometry.LineSet:
        """Square outline + normal stub from (4,3) world-frame corners."""
        center = corners_world.mean(axis=0)
        x = corners_world[1] - corners_world[0]
        y = corners_world[0] - corners_world[3]
        normal = np.cross(x, y)
        nlen = np.linalg.norm(normal)
        if nlen > 1e-9:
            normal = normal / nlen * self.marker_size * 0.5
        pts = np.vstack([corners_world, center[None], (center + normal)[None]])
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    # ── helpers ────────────────────────────────────────────────────────────────

    def _local_corners(self) -> np.ndarray:
        h = self.marker_size / 2.0
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                        dtype=np.float64)

    def _T_cam_in_world(self, poses: dict) -> np.ndarray | None:
        if self._origin_id not in poses:
            return None
        return np.linalg.inv(poses[self._origin_id])

    def _corners_in_world(self, T_marker_in_cam: np.ndarray) -> np.ndarray:
        lc = self._local_corners()
        lc_h = np.hstack([lc, np.ones((4, 1))])
        return (T_marker_in_cam @ lc_h.T).T[:, :3]

    # ── public interface ───────────────────────────────────────────────────────

    def add_frame(self, poses: dict, frame_idx: int):
        """Add camera frustum + per-marker squares for a newly captured frame."""
        if not poses:
            return

        # Establish world origin on first call
        if self._origin_id is None:
            self._origin_id = min(poses.keys())

        T_cam_in_world = self._T_cam_in_world(poses)
        if T_cam_in_world is None:
            print(f"  (3D view) origin marker {self._origin_id} not in frame — skipped.")
            return

        color = _COLORS[frame_idx % len(_COLORS)]
        self.vis.add_geometry(self._frustum(T_cam_in_world, color))

        lc_h = np.hstack([self._local_corners(), np.ones((4, 1))])
        for mid, T_marker_in_cam in poses.items():
            T_marker_in_world = T_cam_in_world @ T_marker_in_cam
            corners_world = (T_marker_in_world @ lc_h.T).T[:, :3]
            marker_color = _COLORS[mid % len(_COLORS)]
            self.vis.add_geometry(self._marker_square(corners_world, marker_color))

        self.vis.reset_view_point(True)

    def add_board_layout(self, layout: dict):
        """Overlay the computed board layout as white squares (board-frame coords)."""
        for corners in layout.values():
            self.vis.add_geometry(
                self._marker_square(np.array(corners, dtype=np.float64),
                                    [0.0, 0.0, 0.0])
            )
        self.vis.reset_view_point(True)
        print("Board layout added to 3D view (white squares).")

    def update(self) -> bool:
        if not self._alive:
            return False
        if not self.vis.poll_events():
            self._alive = False
            return False
        self.vis.update_renderer()
        return True

    def close(self):
        if self._alive:
            self.vis.destroy_window()
            self._alive = False


# ── display helpers ───────────────────────────────────────────────────────────

def draw_status(img: np.ndarray, n_frames: int,
                all_ids: set, origin_id: int | None) -> np.ndarray:
    out = img.copy()
    ids_str = str(sorted(all_ids)) if all_ids else "none"
    cv2.putText(out,
                f"frames: {n_frames}  markers: {ids_str}"
                f"  origin: {origin_id if origin_id is not None else '?'}",
                (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(out, "c=capture  b=compute+save  q=quit",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate ARuCO board layout")
    p.add_argument("--log_dir", default="log/captures",
                   help="Directory to save board_config.json")
    p.add_argument("--marker_size", type=float, default=0.035,
                   help="ARuCO marker side length in metres")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    output_path = os.path.join(args.log_dir, "board_config.json")

    serial = detect_device()
    pipeline, profile = build_pipeline(serial)
    align = rs.align(rs.stream.color)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
    temporal.set_option(rs.option.filter_smooth_delta, 40)

    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    camera_matrix = np.array(
        [[intr.fx, 0, intr.ppx],
         [0, intr.fy, intr.ppy],
         [0, 0, 1]], dtype=np.float64,
    )
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    detector = build_aruco_detector()

    view = CalibrationView(
        intr.fx, intr.fy, intr.ppx, intr.ppy, CAPTURE_W, CAPTURE_H,
        marker_size=args.marker_size,
    )

    frames: list[dict] = []
    all_seen_ids: set = set()

    print("Streaming — press 'c' to accumulate frames, 'b' to compute, 'q' to quit.")

    try:
        while True:
            frameset = pipeline.wait_for_frames()
            aligned = align.process(frameset)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            depth_frame = temporal.process(depth_frame)

            color_bgr = np.asanyarray(color_frame.get_data())

            poses, corners, ids = detect_markers(
                color_bgr, detector, camera_matrix, dist_coeffs, args.marker_size
            )

            # cv2 display: downscaled color + ARuCO overlay
            display = cv2.resize(color_bgr, (DISPLAY_W, DISPLAY_H))
            if ids is not None:
                scale = DISPLAY_W / CAPTURE_W
                corners_disp = [c * scale for c in corners]
                cv2.aruco.drawDetectedMarkers(display, corners_disp, ids)
                rvecs_tmp, tvecs_tmp, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker_size, camera_matrix, dist_coeffs
                )
                cm_disp = camera_matrix.copy()
                cm_disp[0, 2] *= scale
                cm_disp[1, 2] *= scale
                cm_disp[0, 0] *= scale
                cm_disp[1, 1] *= scale
                for i in range(len(ids)):
                    cv2.drawFrameAxes(display, cm_disp, dist_coeffs,
                                      rvecs_tmp[i], tvecs_tmp[i],
                                      args.marker_size * 0.5)

            origin_id = view._origin_id if view._origin_id is not None else (
                min(all_seen_ids) if all_seen_ids else None
            )
            display = draw_status(display, len(frames), all_seen_ids, origin_id)
            cv2.imshow("Calibrate Board", display)

            view.update()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if not poses:
                    print("No markers visible — skipped.")
                else:
                    view.add_frame(poses, len(frames))
                    frames.append(poses)
                    all_seen_ids.update(poses.keys())
                    print(f"Captured frame {len(frames)} — "
                          f"markers {sorted(poses.keys())}")
            elif key == ord('b'):
                if len(frames) < 5:
                    print("Need at least 5 frames. Keep pressing 'c'.")
                    continue
                print(f"\nComputing board layout from {len(frames)} frames...")
                layout, origin_id = compute_board_layout(frames, args.marker_size)
                if layout is None:
                    print("No markers found in accumulated frames.")
                    continue
                save_board_config(layout, origin_id, args.marker_size, output_path)
                view.add_board_layout(layout)

    finally:
        pipeline.stop()
        view.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
