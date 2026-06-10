"""
Turntable capture script: RGB-D streaming + ARuCO pose tracking + SAM segmentation.

Controls (main window):
  c  — freeze frame and enter capture mode
  q  — quit

Capture mode (frozen frame window):
  click (left panel)  — select SAM point prompt
  r                   — re-select point
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
DISPLAY_W, DISPLAY_H = 640, 360   # per-panel display size
CAPTURE_W, CAPTURE_H = 1280, 720  # capture resolution

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


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.flatten())
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


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


def draw_aruco_overlay(frame_bgr, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs, marker_size):
    out = frame_bgr.copy()
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    for i in range(len(ids)):
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size * 0.5)
    return out


def best_marker_T(rvecs, tvecs) -> np.ndarray:
    """Average transform across all detected markers (simple mean in R/t space)."""
    Rs = [cv2.Rodrigues(r.flatten())[0] for r in rvecs]
    ts = [t.flatten() for t in tvecs]
    R_mean = np.mean(Rs, axis=0)
    # Re-orthogonalise via SVD
    U, _, Vt = np.linalg.svd(R_mean)
    R_mean = U @ Vt
    t_mean = np.mean(ts, axis=0)
    T = np.eye(4)
    T[:3, :3] = R_mean
    T[:3, 3] = t_mean
    return T


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
    cv2.putText(out, text, (20, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    out = color_bgr.copy()
    overlay = out.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


# ── capture state ─────────────────────────────────────────────────────────────

class CaptureState:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.records: list[dict] = []
        self.T_ref: np.ndarray | None = None

        poses_path = os.path.join(save_dir, "poses.json")
        if os.path.exists(poses_path):
            with open(poses_path) as f:
                self.records = json.load(f)
            # Restore reference transform from pick 0
            for r in self.records:
                if r["pick_idx"] == 0 and r["T_marker_in_cam"] is not None:
                    self.T_ref = np.array(r["T_marker_in_cam"])
                    break
            print(f"Resumed: {len(self.records)} existing captures found.")

    def next_idx(self) -> int:
        existing = glob.glob(os.path.join(self.save_dir, "*_rgb.png"))
        return len(existing)

    def save(self, idx: int, color_bgr, depth_img, mask, T_marker_in_cam):
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

        # Pose bookkeeping
        T_list = T_marker_in_cam.tolist() if T_marker_in_cam is not None else None
        if self.T_ref is None and T_marker_in_cam is not None:
            self.T_ref = T_marker_in_cam

        T_rel = None
        if self.T_ref is not None and T_marker_in_cam is not None:
            T_rel = (np.linalg.inv(self.T_ref) @ T_marker_in_cam).tolist()

        self.records.append({
            "pick_idx": idx,
            "T_marker_in_cam": T_list,
            "T_relative": T_rel,
        })

        with open(os.path.join(self.save_dir, "poses.json"), "w") as f:
            json.dump(self.records, f, indent=2)

        print(f"Saved capture {idx:03d} → {self.save_dir}/")


# ── capture flow ──────────────────────────────────────────────────────────────

def run_capture_flow(
    color_bgr: np.ndarray,
    depth_img: np.ndarray,
    aruco_overlay: np.ndarray,
    T_marker_in_cam,
    predictor: "SamPredictor",
    state: CaptureState,
):
    WIN = "Capture (click left panel, Enter to confirm, r to retry, Esc to cancel)"

    clicked = [None]  # [x_full, y_full] or None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x < DISPLAY_W:
            x_full = int(x * CAPTURE_W / DISPLAY_W)
            y_full = int(y * CAPTURE_H / DISPLAY_H)
            clicked[0] = (x_full, y_full)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    mask = None

    while True:
        color_small = cv2.resize(aruco_overlay, (DISPLAY_W, DISPLAY_H))
        depth_small = cv2.resize(depth_to_colormap(depth_img), (DISPLAY_W, DISPLAY_H))
        display = np.hstack([color_small, depth_small])

        if clicked[0] is not None and mask is None:
            px = int(clicked[0][0] * DISPLAY_W / CAPTURE_W)
            py = int(clicked[0][1] * DISPLAY_H / CAPTURE_H)
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(display, "Press Enter to run SAM, r to re-select",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        if mask is not None:
            mask_small = cv2.resize(mask, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
            color_small = overlay_mask(color_small, mask_small)
            display = np.hstack([color_small, depth_small])
            cv2.putText(display, "Press s to save, r to re-select, Esc to cancel",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc — cancel
            break

        elif key == ord('r'):
            clicked[0] = None
            mask = None

        elif key == 13 and clicked[0] is not None and mask is None:  # Enter — run SAM
            banner = overlay_banner(display, "Running SAM... please wait")
            cv2.imshow(WIN, banner)
            cv2.waitKey(1)

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            x, y = clicked[0]
            with torch.inference_mode():
                predictor.set_image(color_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
            mask = masks[0].astype(np.uint8) * 255

        elif key == ord('s') and mask is not None:
            idx = state.next_idx()
            state.save(idx, color_bgr, depth_img, mask, T_marker_in_cam)
            break

    cv2.destroyWindow(WIN)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Turntable RGB-D capture with SAM + ARuCO")
    p.add_argument("--log_dir", default="log/captures", help="Directory to save capture outputs")
    p.add_argument("--marker_size", type=float, default=0.05, help="ARuCO marker side length in metres")
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth", help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam_model_type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
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

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0, intr.ppx],
         [0, intr.fy, intr.ppy],
         [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    detector = build_aruco_detector()
    state = CaptureState(args.log_dir)

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

            # ARuCO detection
            corners, ids, rvecs, tvecs = detect_aruco_pose(
                color_bgr, detector, camera_matrix, dist_coeffs, args.marker_size
            )

            if ids is not None:
                display_frame = draw_aruco_overlay(
                    color_bgr, corners, ids, rvecs, tvecs,
                    camera_matrix, dist_coeffs, args.marker_size
                )
                T_marker_in_cam = best_marker_T(rvecs, tvecs)
            else:
                display_frame = color_bgr
                T_marker_in_cam = None

            grid = make_grid(display_frame, depth_img)
            n_markers = 0 if ids is None else len(ids)
            cv2.putText(grid, f"ARuCO markers: {n_markers}  |  captures: {state.next_idx()}  |  c=capture  q=quit",
                        (10, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("RGB (left)  |  Depth (right)", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_capture_flow(
                    color_bgr, depth_img, display_frame,
                    T_marker_in_cam, predictor, state,
                )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
