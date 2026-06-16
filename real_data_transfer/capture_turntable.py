"""
Turntable capture script: RGB-D streaming + ARuCO pose tracking + SAM segmentation.

Pose estimation — two modes:

  Default: every detected ARuCO marker is tracked independently.  T_relative is
  averaged over all co-visible markers (SVD-re-orthogonalised mean R, mean t).

  Board mode (--board_config <path>): loads the board layout from calibrate_board.py
  and uses cv2.aruco.estimatePoseBoard to fit a single jointly-constrained pose to
  all detected markers simultaneously.  This enforces the known coplanarity of the
  markers and gives significantly more accurate and stable poses.

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

sys.path.insert(0, os.path.dirname(__file__))
from camera_utils import build_camera

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
BURST_FRAMES = 5   # frames averaged per capture to reduce random pose noise


def depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    d = depth_img.astype(np.float32)
    d = np.clip(d, DEPTH_MIN_MM, DEPTH_MAX_MM)
    d = ((d - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


# ── aruco helpers ─────────────────────────────────────────────────────────────

def build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def detect_aruco_pose(frame_bgr, detector, camera_matrix, dist_coeffs, marker_size):
    """Returns (corners, ids, rvecs, tvecs) or (None, None, None, None)."""
    gray = _clahe.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
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


# ── board helpers ─────────────────────────────────────────────────────────────

def load_board(board_config_path: str, dictionary):
    """Load board_config.json, build cv2.aruco.Board, return (board, corners_in_board).

    corners_in_board: {marker_id: (4,3) float32 array of corner positions in board frame}
    """
    with open(board_config_path) as f:
        cfg = json.load(f)

    obj_points = []
    ids = []
    corners_in_board = {}

    for mid_str, corners in cfg["markers"].items():
        mid = int(mid_str)
        pts = np.array(corners, dtype=np.float32)
        obj_points.append(pts)
        ids.append(mid)
        corners_in_board[mid] = pts

    board = cv2.aruco.Board(obj_points, dictionary, np.array(ids))
    marker_size = float(cfg.get("marker_size", 0.035))
    print(f"Board loaded: {len(ids)} markers {sorted(ids)}, "
          f"origin={cfg['origin_marker_id']}, marker_size={marker_size}")
    return board, corners_in_board, marker_size


def detect_board_pose(frame_bgr, detector, board, camera_matrix, dist_coeffs):
    """Estimate a single joint board pose using all detected markers.

    Returns (corners, ids, rvec, tvec); rvec/tvec are None if no board pose found.
    """
    gray = _clahe.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None, None
    n, rvec, tvec = cv2.aruco.estimatePoseBoard(
        corners, ids, board, camera_matrix, dist_coeffs, None, None
    )
    if n == 0:
        return corners, ids, None, None
    return corners, ids, rvec, tvec


def corners_to_T(corners: np.ndarray) -> np.ndarray:
    """Recover a 4×4 pose from (4,3) marker corners (ARuCO corner order)."""
    center = corners.mean(axis=0)
    x = ((corners[1] - corners[0]) + (corners[2] - corners[3])) / 2
    x /= np.linalg.norm(x)
    y = ((corners[0] - corners[3]) + (corners[1] - corners[2])) / 2
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    T = np.eye(4)
    T[:3, 0] = x.astype(np.float64)
    T[:3, 1] = y.astype(np.float64)
    T[:3, 2] = z.astype(np.float64)
    T[:3, 3] = center.astype(np.float64)
    return T


def build_marker_poses_from_board(T_board_in_cam: np.ndarray,
                                  corners_in_board: dict) -> dict:
    """Compute per-marker T_marker_in_cam from a single board pose + board layout."""
    poses = {}
    for mid, corners in corners_in_board.items():
        T_marker_in_board = corners_to_T(corners.astype(np.float64))
        poses[mid] = T_board_in_cam @ T_marker_in_board
    return poses


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
    """
    co_visible = [mid for mid in cur_poses if mid in ref_poses]
    if not co_visible:
        return None, []
    T_rels = [np.linalg.inv(ref_poses[mid]) @ cur_poses[mid] for mid in co_visible]
    return average_transforms(T_rels), co_visible


# ── accumulated 3-D view ──────────────────────────────────────────────────────

class AccumulatedView:
    """Open3D window: masked point clouds + camera frustums + marker squares."""

    _COLORS = [
        [1.00, 0.25, 0.25], [0.25, 1.00, 0.25], [0.25, 0.45, 1.00],
        [1.00, 1.00, 0.20], [0.20, 1.00, 1.00], [1.00, 0.20, 1.00],
        [1.00, 0.60, 0.00], [0.60, 0.00, 1.00],
    ]

    def __init__(self, img_w: int, img_h: int, marker_size: float = 0.035):
        self.img_w, self.img_h = img_w, img_h
        self.marker_size = marker_size
        self._alive = True

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Accumulated Captures", width=960, height=540)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        self.vis.add_geometry(axes)

    def _frustum(self, T_cam_in_world: np.ndarray, color: list,
                 intr: dict, depth: float = 0.08) -> o3d.geometry.LineSet:
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        w, h = intr["width"], intr["height"]
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

    def _marker_lineset(self, T_marker_in_world: np.ndarray,
                        color: list) -> o3d.geometry.LineSet:
        """Square outline + normal stub for one ARuCO marker."""
        h = self.marker_size / 2.0
        local = np.array([
            [-h,  h, 0], [ h,  h, 0], [ h, -h, 0], [-h, -h, 0],
            [ 0,  0, 0], [ 0,  0, h * 0.8],
        ], dtype=np.float64)
        pts_h = np.hstack([local, np.ones((len(local), 1))])
        world = (T_marker_in_world @ pts_h.T).T[:, :3]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(world),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    def add_capture(self, color_bgr: np.ndarray, depth_mm: np.ndarray,
                    mask: np.ndarray, intr: dict,
                    T_world_from_cam, cur_poses: dict, capture_idx: int):
        if T_world_from_cam is None:
            T_world_from_cam = np.eye(4)

        # Backproject depth pixels to 3D camera-frame points.
        ys, xs = np.where((depth_mm > 0) & (mask > 0))
        z = depth_mm[ys, xs].astype(np.float64) / 1000.0
        x = (xs - intr["cx"]) * z / intr["fx"]
        y = (ys - intr["cy"]) * z / intr["fy"]
        xyz_cam = np.stack([x, y, z], axis=1)[::4]
        rgb = color_bgr[ys, xs][:, ::-1].astype(np.float64)[::4] / 255.0

        xyz_world = (T_world_from_cam @ np.hstack(
            [xyz_cam, np.ones((len(xyz_cam), 1))]).T).T[:, :3]

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_world))
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        color = self._COLORS[capture_idx % len(self._COLORS)]
        self.vis.add_geometry(pcd)
        self.vis.add_geometry(self._frustum(T_world_from_cam, color, intr))

        for mid, T_marker_in_cam in cur_poses.items():
            T_marker_in_world = T_world_from_cam @ T_marker_in_cam
            marker_color = self._COLORS[mid % len(self._COLORS)]
            self.vis.add_geometry(self._marker_lineset(T_marker_in_world, marker_color))

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
        self.T_ref_per_marker: dict[int, np.ndarray] = {}
        self.T_cam_in_obj0: "np.ndarray | None" = None
        self.T_board_ref: "np.ndarray | None" = None

    def next_idx(self) -> int:
        return len(glob.glob(os.path.join(self.save_dir, "*_rgb.png")))

    def save(self, idx: int, color_bgr, depth_img, mask,
             cur_poses: dict,
             T_board_in_cam: "np.ndarray | None" = None) -> "np.ndarray | None":
        """Save capture files and update poses.json.

        cur_poses: {marker_id: T_marker_in_cam} — pre-computed by the caller.
        T_board_in_cam: if provided (board mode), T_relative is derived directly
        from the board pose instead of averaging per-marker relative transforms.

        Returns T_world_from_cam (4×4) or None when pose is unavailable.
        """
        if T_board_in_cam is not None:
            # Board mode: use the joint board pose directly.
            if self.T_board_ref is None:
                self.T_board_ref = T_board_in_cam
                self.T_cam_in_obj0 = np.linalg.inv(T_board_in_cam)
                T_relative = None
                co_visible = []
            else:
                T_relative = np.linalg.inv(self.T_board_ref) @ T_board_in_cam
                co_visible = list(cur_poses.keys())
        else:
            # Independent mode: average per-marker relative transforms.
            if not self.T_ref_per_marker:
                if not cur_poses:
                    print("  WARNING: no ARuCO markers detected at pick 0 — "
                          "relative poses will be unavailable.")
                self.T_ref_per_marker = dict(cur_poses)
                if cur_poses:
                    self.T_cam_in_obj0 = np.linalg.inv(
                        average_transforms(list(cur_poses.values()))
                    )

            T_relative, co_visible = compute_relative_transform(
                self.T_ref_per_marker, cur_poses
            )

            if not co_visible and idx > 0:
                print(f"  WARNING: no co-visible markers with pick 0 — "
                      f"T_relative recorded as null.")

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
            "T_world_from_cam": T_world_from_cam.tolist() if T_world_from_cam is not None else None,
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
    cur_poses: dict,
    predictor: "SamPredictor",
    state: CaptureState,
    accumulated_view: AccumulatedView,
    intr: dict,
    T_board_in_cam: "np.ndarray | None" = None,
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
                idx, color_bgr, depth_img, mask, cur_poses, T_board_in_cam
            )
            accumulated_view.add_capture(
                color_bgr, depth_img, mask, intr, T_world_from_cam, cur_poses, idx
            )
            break

    cv2.destroyWindow(WIN)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Turntable RGB-D capture with SAM + ARuCO")
    p.add_argument("--log_dir", default="log/captures",
                   help="Directory to save capture outputs")
    p.add_argument("--marker_size", type=float, default=0.035,
                   help="ARuCO marker side length in metres (overridden by board_config)")
    p.add_argument("--board_config",
                   help="Path to board_config.json from calibrate_board.py "
                        "(enables joint board pose estimation)")
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth",
                   help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam_model_type", default="vit_b",
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--camera", choices=["realsense", "zed"], default="zed",
                   help="Camera to use (default: zed)")
    p.add_argument("--depth_mode",
                   choices=["performance", "quality", "ultra", "neural", "neural_plus"],
                   default="neural_plus",
                   help="ZED depth mode (default: neural_plus; ignored for realsense)")
    p.add_argument("--confidence_threshold", type=int, default=95,
                   help="ZED depth confidence threshold 0-100 (default: 95; "
                        "lower accepts noisier pixels; ignored for realsense)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Loading SAM ({args.sam_model_type}, {args.sam_checkpoint})...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM ready.")

    cam = build_camera(args)
    intr = cam.intrinsics
    camera_matrix = intr["camera_matrix"]
    dist_coeffs = intr["dist_coeffs"]

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = build_aruco_detector()

    # Board mode setup
    board = None
    corners_in_board = {}
    if args.board_config:
        board, corners_in_board, board_marker_size = load_board(
            args.board_config, dictionary
        )
        args.marker_size = board_marker_size
        print("Board mode enabled — using estimatePoseBoard.")
    else:
        print("Default mode — using per-marker independent poses.")

    # Save camera intrinsics once so tsdf_fusion.py can read them offline.
    intr_path = os.path.join(args.log_dir, "intrinsics.json")
    if not os.path.exists(intr_path):
        with open(intr_path, "w") as _f:
            json.dump({
                "fx": intr["fx"], "fy": intr["fy"],
                "cx": intr["cx"], "cy": intr["cy"],
                "width": intr["width"], "height": intr["height"],
            }, _f, indent=2)
        print(f"Intrinsics saved → {intr_path}")

    state = CaptureState(args.log_dir)
    accumulated_view = AccumulatedView(
        intr["width"], intr["height"], marker_size=args.marker_size,
    )

    print("Streaming — press 'c' to capture, 'q' to quit.")

    board_rvec = None   # initialise so burst loop can safely check before first detection
    T_board = None

    try:
        while True:
            if not cam.grab():
                continue

            color_bgr = cam.color_bgr
            depth_img = cam.depth_mm

            if board is not None:
                # Board mode: single joint pose
                corners, ids, board_rvec, board_tvec = detect_board_pose(
                    color_bgr, detector, board, camera_matrix, dist_coeffs
                )
                if board_rvec is not None:
                    T_board = rvec_tvec_to_T(board_rvec, board_tvec)
                    cur_poses = build_marker_poses_from_board(T_board, corners_in_board)
                    display_frame = color_bgr.copy()
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs,
                                      board_rvec, board_tvec, args.marker_size)
                else:
                    cur_poses = {}
                    display_frame = color_bgr
            else:
                # Default mode: per-marker independent poses
                corners, ids, rvecs, tvecs = detect_aruco_pose(
                    color_bgr, detector, camera_matrix, dist_coeffs, args.marker_size
                )
                cur_poses = build_marker_poses(corners, ids, rvecs, tvecs)
                if ids is not None:
                    display_frame = draw_aruco_overlay(
                        color_bgr, corners, ids, rvecs, tvecs,
                        camera_matrix, dist_coeffs, args.marker_size,
                    )
                else:
                    display_frame = color_bgr

            grid = make_grid(display_frame, depth_img)
            n_markers = len(cur_poses)
            mode_str = "board" if board is not None else "indep"
            ref_str = (f"ref={sorted(state.T_ref_per_marker.keys())}"
                       if state.T_ref_per_marker else "ref=?")
            cv2.putText(
                grid,
                f"[{mode_str}] ARuCO: {n_markers}  {ref_str}"
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
                # Burst: average BURST_FRAMES consecutive pose detections to
                # reduce single-frame noise before entering the capture flow.
                if board is not None:
                    # Board mode: average board poses directly (not per-marker).
                    board_T_list = [T_board] if board_rvec is not None else []
                    for _ in range(BURST_FRAMES - 1):
                        if not cam.grab():
                            continue
                        _, _, br, bt = detect_board_pose(
                            cam.color_bgr, detector, board, camera_matrix, dist_coeffs
                        )
                        if br is not None:
                            board_T_list.append(rvec_tvec_to_T(br, bt))
                    if board_T_list:
                        averaged_T_board = average_transforms(board_T_list)
                    else:
                        averaged_T_board = T_board if board_rvec is not None else None
                    averaged_poses = (
                        build_marker_poses_from_board(averaged_T_board, corners_in_board)
                        if averaged_T_board is not None else {}
                    )
                    run_capture_flow(
                        color_bgr, depth_img, display_frame,
                        averaged_poses, predictor, state, accumulated_view, intr,
                        T_board_in_cam=averaged_T_board,
                    )
                else:
                    burst: dict[int, list] = {mid: [T] for mid, T in cur_poses.items()}
                    for _ in range(BURST_FRAMES - 1):
                        if not cam.grab():
                            continue
                        c2, i2, r2, t2 = detect_aruco_pose(
                            cam.color_bgr, detector, camera_matrix, dist_coeffs,
                            args.marker_size
                        )
                        for mid, T in build_marker_poses(c2, i2, r2, t2).items():
                            burst.setdefault(mid, []).append(T)
                    averaged_poses = {
                        mid: average_transforms(Ts) if len(Ts) > 1 else Ts[0]
                        for mid, Ts in burst.items()
                    }
                    run_capture_flow(
                        color_bgr, depth_img, display_frame,
                        averaged_poses, predictor, state, accumulated_view, intr,
                    )

    finally:
        cam.close()
        accumulated_view.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
