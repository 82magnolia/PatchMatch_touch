"""
Turntable capture script: RGB-D streaming + ARuCO pose tracking + SAM segmentation.

Pose estimation: every detected ARuCO marker is tracked independently.  For each
capture a per-marker pose dictionary is stored.  The relative transform T_relative
between a capture and pick 0 is computed by averaging the per-marker relative
transforms over all markers that are co-visible in both frames (SVD-re-orthogonalised
mean R, mean t).  This makes the estimate more robust than tracking a single marker
and degrades gracefully when some markers leave the frame.

Controls (main window):
  c  — freeze frame and enter capture mode
  q  — quit

Capture mode (frozen frame window):
  drag (left panel)   — draw bounding box for SAM prompt
  r                   — redraw box
  Enter               — run SAM inference
  s                   — save current capture
  Esc                 — cancel and return to live stream
"""

import argparse
import json
import os
import sys
import glob

import cv2
import numpy as np
import open3d as o3d
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not found. Install with: pip install pyrealsense2")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sys.exit("segment-anything not found. Install with: pip install segment-anything")

# ── constants ────────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 640, 360
CAPTURE_W, CAPTURE_H = 1280, 720

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 3000

ARUCO_DICT = cv2.aruco.DICT_4X4_50


# ── realsense helpers ─────────────────────────────────────────────────────────

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


def build_pipeline(serial: str):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, CAPTURE_W, CAPTURE_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, CAPTURE_W, CAPTURE_H, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    return pipeline, profile


def depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    d = depth_img.astype(np.float32)
    d = np.clip(d, DEPTH_MIN_MM, DEPTH_MAX_MM)
    d = ((d - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


# ── aruco helpers ─────────────────────────────────────────────────────────────

def build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco_pose(frame_bgr, detector, camera_matrix, dist_coeffs, marker_size):
    """Returns (corners, ids, rvecs, tvecs) or (None, None, None, None)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None, None
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size, camera_matrix, dist_coeffs
    )
    return corners, ids, rvecs, tvecs


def draw_aruco_overlay(frame_bgr, corners, ids, rvecs, tvecs,
                       camera_matrix, dist_coeffs, marker_size):
    out = frame_bgr.copy()
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    for i in range(len(ids)):
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs,
                          rvecs[i], tvecs[i], marker_size * 0.5)
    return out


# ── pose math ─────────────────────────────────────────────────────────────────

def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.flatten())
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def average_transforms(T_list: list) -> np.ndarray:
    """Average SE(3) matrices: SVD-re-orthogonalised mean R, mean t."""
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


def build_marker_poses(corners, ids, rvecs, tvecs) -> dict:
    """Return {marker_id: T_marker_in_cam (4×4)} for all detected markers."""
    if ids is None:
        return {}
    return {
        int(ids[i].flat[0]): rvec_tvec_to_T(rvecs[i], tvecs[i])
        for i in range(len(ids))
    }


def compute_relative_transform(ref_poses: dict, cur_poses: dict):
    """Average per-marker relative transforms over co-visible markers.

    Returns (T_relative, co_visible_ids) or (None, []) if no overlap.
    T_relative = mean over k of inv(ref_poses[k]) @ cur_poses[k].
    """
    co_visible = [mid for mid in cur_poses if mid in ref_poses]
    if not co_visible:
        return None, []
    T_rels = [np.linalg.inv(ref_poses[mid]) @ cur_poses[mid] for mid in co_visible]
    return average_transforms(T_rels), co_visible


# ── accumulated 3-D view ──────────────────────────────────────────────────────

class AccumulatedView:
    """Open3D window: masked point clouds + camera frustums for all captures.

    Coordinate frame: pick-0 camera frame.
    T_world_from_cam = inv(T_relative), where T_relative = mean over co-visible
    markers of inv(T_ref_k) @ T_current_k.  At pick 0 T_relative = I so the
    pick-0 camera sits at the world origin.
    """

    _COLORS = [
        [1.00, 0.25, 0.25], [0.25, 1.00, 0.25], [0.25, 0.45, 1.00],
        [1.00, 1.00, 0.20], [0.20, 1.00, 1.00], [1.00, 0.20, 1.00],
        [1.00, 0.60, 0.00], [0.60, 0.00, 1.00],
    ]

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 img_w: int, img_h: int):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.img_w, self.img_h = img_w, img_h
        self._alive = True

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Accumulated Captures", width=960, height=540)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        self.vis.add_geometry(axes)

    def _frustum(self, T_cam_in_world: np.ndarray, color: list,
                 depth: float = 0.08) -> o3d.geometry.LineSet:
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        w, h = self.img_w, self.img_h
        local_pts = np.array([
            [0, 0, 0],
            [(0 - cx) / fx * depth, (0 - cy) / fy * depth, depth],
            [(w - cx) / fx * depth, (0 - cy) / fy * depth, depth],
            [(w - cx) / fx * depth, (h - cy) / fy * depth, depth],
            [(0 - cx) / fx * depth, (h - cy) / fy * depth, depth],
        ])
        pts_h = np.hstack([local_pts, np.ones((5, 1))])
        world_pts = (T_cam_in_world @ pts_h.T).T[:, :3]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(world_pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    def add_capture(self, color_bgr: np.ndarray, depth_img: np.ndarray,
                    mask: np.ndarray, T_world_from_cam, capture_idx: int):
        if T_world_from_cam is None:
            T_world_from_cam = np.eye(4)

        vs, us = np.where(mask > 0)
        z = depth_img[vs, us].astype(np.float32) / 1000.0
        valid = z > 0
        us, vs, z = us[valid], vs[valid], z[valid]
        xyz_cam = np.stack(
            [(us - self.cx) / self.fx * z,
             (vs - self.cy) / self.fy * z,
             z], axis=1,
        )
        rgb = color_bgr[vs, us][:, ::-1].astype(np.float64) / 255.0

        xyz_cam, rgb = xyz_cam[::4], rgb[::4]
        xyz_world = (T_world_from_cam @ np.hstack(
            [xyz_cam, np.ones((len(xyz_cam), 1))]).T).T[:, :3]

        pcd = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz_world))
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        color = self._COLORS[capture_idx % len(self._COLORS)]
        frustum = self._frustum(T_world_from_cam, color)

        self.vis.add_geometry(pcd)
        self.vis.add_geometry(frustum)
        self.vis.reset_view_point(True)

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

def make_grid(color_bgr: np.ndarray, depth_img: np.ndarray) -> np.ndarray:
    color_small = cv2.resize(color_bgr, (DISPLAY_W, DISPLAY_H))
    depth_small = cv2.resize(depth_to_colormap(depth_img), (DISPLAY_W, DISPLAY_H))
    return np.hstack([color_small, depth_small])


def overlay_banner(img: np.ndarray, text: str, alpha: float = 0.55) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.putText(out, text, (20, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    out = color_bgr.copy()
    overlay = out.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


# ── capture state ─────────────────────────────────────────────────────────────

class CaptureState:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.records: list[dict] = []
        # Per-marker poses at pick 0, used as reference for all subsequent T_relative
        self.T_ref_per_marker: dict[int, np.ndarray] = {}
        # Camera pose in the object frame at pick 0: inv(avg T_marker_in_cam at pick 0).
        # Combined with inv(T_relative) this gives the correct orbiting camera position.
        self.T_cam_in_obj0: "np.ndarray | None" = None

    def next_idx(self) -> int:
        return len(glob.glob(os.path.join(self.save_dir, "*_rgb.png")))

    def save(self, idx: int, color_bgr, depth_img, mask,
             corners, ids, rvecs, tvecs) -> "np.ndarray | None":
        """Save capture files and update poses.json.

        Returns T_world_from_cam (4×4) = inv(T_relative), or None when no
        co-visible markers are available.
        """
        cur_poses = build_marker_poses(corners, ids, rvecs, tvecs)

        # At pick 0 establish the reference
        if not self.T_ref_per_marker:
            if not cur_poses:
                print("  WARNING: no ARuCO markers detected at pick 0 — "
                      "relative poses will be unavailable.")
            self.T_ref_per_marker = dict(cur_poses)
            if cur_poses:
                # Camera position in object frame: inv of averaged marker poses.
                # Used so virtual cameras orbit at the correct distance from object.
                self.T_cam_in_obj0 = np.linalg.inv(
                    average_transforms(list(cur_poses.values()))
                )

        T_relative, co_visible = compute_relative_transform(
            self.T_ref_per_marker, cur_poses
        )

        if not co_visible and idx > 0:
            print(f"  WARNING: no co-visible markers with pick 0 — "
                  f"T_relative recorded as null.")

        # inv(T_relative) gives the virtual camera rotation; premultiplying by
        # T_cam_in_obj0 places it at the correct distance/angle from the object.
        if T_relative is not None and self.T_cam_in_obj0 is not None:
            T_world_from_cam = np.linalg.inv(T_relative) @ self.T_cam_in_obj0
        elif self.T_cam_in_obj0 is not None:
            T_world_from_cam = self.T_cam_in_obj0
        else:
            T_world_from_cam = None

        # ── write files ────────────────────────────────────────────────────────
        prefix = os.path.join(self.save_dir, f"{idx:03d}")

        cv2.imwrite(f"{prefix}_rgb.png", color_bgr)
        np.save(f"{prefix}_depth.npy", depth_img)
        cv2.imwrite(f"{prefix}_depth_vis.png", depth_to_colormap(depth_img))
        cv2.imwrite(f"{prefix}_mask.png", mask)

        masked_color = color_bgr.copy()
        masked_color[mask == 0] = 0
        cv2.imwrite(f"{prefix}_rgb_masked.png", masked_color)

        masked_depth = depth_img.copy()
        masked_depth[mask == 0] = 0
        np.save(f"{prefix}_depth_masked.npy", masked_depth)

        self.records.append({
            "pick_idx": idx,
            "marker_poses": {
                str(mid): T.tolist() for mid, T in cur_poses.items()
            },
            "co_visible_marker_ids": co_visible if co_visible else None,
            "T_relative": T_relative.tolist() if T_relative is not None else None,
        })
        with open(os.path.join(self.save_dir, "poses.json"), "w") as f:
            json.dump(self.records, f, indent=2)

        n_co = len(co_visible)
        print(f"Saved capture {idx:03d} — "
              f"{len(cur_poses)} markers detected, {n_co} co-visible with pick 0.")
        return T_world_from_cam


# ── capture flow ──────────────────────────────────────────────────────────────

def run_capture_flow(
    color_bgr: np.ndarray,
    depth_img: np.ndarray,
    aruco_overlay: np.ndarray,
    corners, ids, rvecs, tvecs,
    predictor: "SamPredictor",
    state: CaptureState,
    accumulated_view: AccumulatedView,
):
    WIN = "Capture (drag left panel for box, Enter to run SAM, r to redraw, Esc to cancel)"

    box_pts = [None]
    drag_start = [None]

    def to_full(xd, yd):
        return (int(xd * CAPTURE_W / DISPLAY_W),
                int(yd * CAPTURE_H / DISPLAY_H))

    def on_mouse(event, x, y, flags, param):
        x = min(max(x, 0), DISPLAY_W - 1)
        y = min(max(y, 0), DISPLAY_H - 1)
        if event == cv2.EVENT_LBUTTONDOWN and x < DISPLAY_W:
            drag_start[0] = (x, y)
            box_pts[0] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag_start[0] is not None:
            x1f, y1f = to_full(*drag_start[0])
            x2f, y2f = to_full(x, y)
            box_pts[0] = [min(x1f, x2f), min(y1f, y2f),
                          max(x1f, x2f), max(y1f, y2f)]
        elif event == cv2.EVENT_LBUTTONUP and drag_start[0] is not None:
            x1f, y1f = to_full(*drag_start[0])
            x2f, y2f = to_full(x, y)
            box_pts[0] = [min(x1f, x2f), min(y1f, y2f),
                          max(x1f, x2f), max(y1f, y2f)]
            drag_start[0] = None

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    mask = None

    def box_display(b):
        return (int(b[0] * DISPLAY_W / CAPTURE_W),
                int(b[1] * DISPLAY_H / CAPTURE_H),
                int(b[2] * DISPLAY_W / CAPTURE_W),
                int(b[3] * DISPLAY_H / CAPTURE_H))

    while True:
        color_small = cv2.resize(aruco_overlay, (DISPLAY_W, DISPLAY_H))
        depth_small = cv2.resize(depth_to_colormap(depth_img), (DISPLAY_W, DISPLAY_H))
        display = np.hstack([color_small, depth_small])

        if box_pts[0] is not None and mask is None:
            dx1, dy1, dx2, dy2 = box_display(box_pts[0])
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            if drag_start[0] is None:
                cv2.putText(display, "Press Enter to run SAM, r to redraw",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        if mask is not None:
            mask_small = cv2.resize(mask, (DISPLAY_W, DISPLAY_H),
                                    interpolation=cv2.INTER_NEAREST)
            color_small = overlay_mask(color_small, mask_small)
            display = np.hstack([color_small, depth_small])
            cv2.putText(display, "Press s to save, r to redraw, Esc to cancel",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            break
        elif key == ord('r'):
            box_pts[0] = None
            drag_start[0] = None
            mask = None
        elif key == 13 and box_pts[0] is not None and drag_start[0] is None and mask is None:
            banner = overlay_banner(display, "Running SAM... please wait")
            cv2.imshow(WIN, banner)
            cv2.waitKey(1)

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                predictor.set_image(color_rgb)
                masks, _, _ = predictor.predict(
                    box=np.array(box_pts[0], dtype=np.float32),
                    multimask_output=False,
                )
            mask = masks[0].astype(np.uint8) * 255

        elif key == ord('s') and mask is not None:
            idx = state.next_idx()
            T_world_from_cam = state.save(
                idx, color_bgr, depth_img, mask,
                corners, ids, rvecs, tvecs,
            )
            accumulated_view.add_capture(
                color_bgr, depth_img, mask, T_world_from_cam, idx
            )
            break

    cv2.destroyWindow(WIN)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Turntable RGB-D capture with SAM + ARuCO")
    p.add_argument("--log_dir", default="log/captures",
                   help="Directory to save capture outputs")
    p.add_argument("--marker_size", type=float, default=0.035,
                   help="ARuCO marker side length in metres")
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth",
                   help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam_model_type", default="vit_b",
                   choices=["vit_h", "vit_l", "vit_b"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Loading SAM ({args.sam_model_type}, {args.sam_checkpoint})...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM ready.")

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
    state = CaptureState(args.log_dir)
    accumulated_view = AccumulatedView(
        intr.fx, intr.fy, intr.ppx, intr.ppy, CAPTURE_W, CAPTURE_H
    )

    print("Streaming — press 'c' to capture, 'q' to quit.")

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
            depth_img = np.asanyarray(depth_frame.get_data())

            corners, ids, rvecs, tvecs = detect_aruco_pose(
                color_bgr, detector, camera_matrix, dist_coeffs, args.marker_size
            )

            if ids is not None:
                display_frame = draw_aruco_overlay(
                    color_bgr, corners, ids, rvecs, tvecs,
                    camera_matrix, dist_coeffs, args.marker_size,
                )
            else:
                display_frame = color_bgr

            grid = make_grid(display_frame, depth_img)
            n_markers = 0 if ids is None else len(ids)
            ref_str = (f"ref={sorted(state.T_ref_per_marker.keys())}"
                       if state.T_ref_per_marker else "ref=?")
            cv2.putText(
                grid,
                f"ARuCO: {n_markers}  {ref_str}"
                f"  |  captures: {state.next_idx()}"
                "  |  c=capture  q=quit",
                (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )
            cv2.imshow("RGB (left)  |  Depth (right)", grid)

            accumulated_view.update()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_capture_flow(
                    color_bgr, depth_img, display_frame,
                    corners, ids, rvecs, tvecs,
                    predictor, state, accumulated_view,
                )

    finally:
        pipeline.stop()
        accumulated_view.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
