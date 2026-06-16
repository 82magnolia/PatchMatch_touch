"""
GelSight Mini capture script for the PatchMatch tactile transfer pipeline.

Saves {idx}_normal.jpg/.npz, {idx}_color.jpg, {idx}_shadow.mp4 per touch location —
the same file layout expected by main_retrieval_transfer_accel.py when --scale is omitted.

Stage 1 (Object Selection):
  Live ZED stream. Press 'c' to freeze, SAM-segment object, confirm with 'y'.
  Captures GelSight blank (no-contact) frame on confirmation.

Stage 2 (Touch Recording):
  Tracks GelSight via ARuCO marker ID=6 (DICT_4X4_50) on the holder back.
  Press 'r' to start recording, 's' to stop/save, 'a' to abort, 'q' to quit.
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_zed_normal_sim import (
    build_camera, normals_to_colormap, inpaint_normals, overlay_banner,
    DEPTH_MODES, GELSIGHT_FOV_W_MM, GELSIGHT_FOV_H_MM,
)

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found.  Install with: pip install pyzed")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sys.exit("segment-anything not found.  Install with: pip install segment-anything")

# ── Constants ──────────────────────────────────────────────────────────────────
GELSIGHT_MARKER_ID = 6
HOLDER_HEIGHT_M    = 0.030    # gsmini_holder.stl: Z range 0–30 mm
GEL_THICKNESS_M    = 0.00425  # GelSight Mini specs: 4.25 mm gel
ARUCO_TO_CONTACT_M = HOLDER_HEIGHT_M  # gel contact flush with holder bottom

ZED_DISPLAY_W, ZED_DISPLAY_H = 640, 360
ZED_W, ZED_H                 = 1280, 720
GELSIGHT_W, GELSIGHT_H       = 320, 240


# ── ARuCO ─────────────────────────────────────────────────────────────────────

def build_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_gelsight_marker(color_bgr, detector, camera_matrix, dist_coeffs, marker_size):
    """Return (rvec, tvec) for marker ID=6, or (None, None) if not visible."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None
    for i, mid in enumerate(ids.flatten()):
        if mid == GELSIGHT_MARKER_ID:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], marker_size, camera_matrix, dist_coeffs)
            return rvecs[0].flatten(), tvecs[0].flatten()
    return None, None


def compute_contact_pixel(rvec, tvec, intr):
    """Map ARuCO pose to the gel contact point in ZED pixel coordinates."""
    R, _ = cv2.Rodrigues(rvec)
    p = R @ np.array([0.0, 0.0, -ARUCO_TO_CONTACT_M]) + tvec
    if p[2] <= 0:
        return None
    px = int(intr["fx"] * p[0] / p[2] + intr["cx"])
    py = int(intr["fy"] * p[1] / p[2] + intr["cy"])
    return (min(max(px, 0), ZED_W - 1), min(max(py, 0), ZED_H - 1))


# ── GelSight frame processing (mirrors gsrobotics GelSightMini.update()) ──────

def _crop_and_resize_gs(image, target_w, target_h, border_fraction):
    border_fraction = min(max(0.0, border_fraction), 0.49)
    bx = int(image.shape[0] * border_fraction)
    by = int(image.shape[1] * border_fraction)
    cropped = image[bx: image.shape[0] - bx, by: image.shape[1] - by]
    return cv2.resize(cropped, (target_w, target_h))


def read_gelsight_frame(cap, border_fraction=0.15):
    """BGR -> RGB -> crop border -> resize to GELSIGHT_W x GELSIGHT_H -> BGR."""
    ret, frame = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = _crop_and_resize_gs(rgb, GELSIGHT_W, GELSIGHT_H, border_fraction)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Orthographic projection ───────────────────────────────────────────────────

def ortho_project_raw(normals_np, color_bgr, depth_m, mask, px, py, intr, method, rvec=None):
    """
    Orthographic projection at (px, py) for the GelSight Mini FoV.

    Returns (normal_bgr, raw_normals_hw3, color_bgr_crop) all at GELSIGHT_W x GELSIGHT_H,
    or None if no valid depth at the pick location.

    normal_bgr:       (H_out, W_out, 3) uint8 BGR colormap
    raw_normals_hw3:  (H_out, W_out, 3) float32 in [-1, 1], normals in sensor frame
    color_bgr_crop:   (H_out, W_out, 3) uint8 BGR
    """
    H, W = normals_np.shape[:2]
    r = 4
    d_patch = depth_m[max(0, py - r): py + r + 1, max(0, px - r): px + r + 1]
    valid_d = d_patch[np.isfinite(d_patch) & (d_patch > 0)]
    if len(valid_d) == 0:
        return None
    Z_m = float(np.median(valid_d))

    fx, fy = intr["fx"], intr["fy"]
    w_px = max(1, round(GELSIGHT_FOV_W_MM * 1e-3 * fx / Z_m))
    h_px = max(1, round(GELSIGHT_FOV_H_MM * 1e-3 * fy / Z_m))
    half_w, half_h = w_px // 2, h_px // 2

    y1_raw, y2_raw = py - half_h, py - half_h + h_px
    x1_raw, x2_raw = px - half_w, px - half_w + w_px
    y1, y2 = max(0, y1_raw), min(H, y2_raw)
    x1, x2 = max(0, x1_raw), min(W, x2_raw)
    pad_y, pad_x = y1 - y1_raw, x1 - x1_raw
    ah, aw = y2 - y1, x2 - x1

    normals_crop = np.full((h_px, w_px, 4), np.nan, dtype=np.float32)
    mask_crop = np.zeros((h_px, w_px), dtype=np.uint8)
    color_crop = np.zeros((h_px, w_px, 3), dtype=np.uint8)

    normals_crop[pad_y:pad_y + ah, pad_x:pad_x + aw] = normals_np[y1:y2, x1:x2]
    mask_crop[pad_y:pad_y + ah, pad_x:pad_x + aw] = mask[y1:y2, x1:x2]
    color_crop[pad_y:pad_y + ah, pad_x:pad_x + aw] = color_bgr[y1:y2, x1:x2]

    normals_crop[mask_crop == 0] = np.nan
    normals_filled = inpaint_normals(normals_crop, method)

    # Re-orient normals from ZED camera frame into the GelSight sensor (marker) frame
    if rvec is not None:
        R, _ = cv2.Rodrigues(rvec)
        nxyz = normals_filled[:, :, :3]
        valid_n = np.isfinite(nxyz).all(axis=-1)
        if valid_n.any():
            nxyz_rot = np.full_like(nxyz, np.nan)
            nxyz_rot[valid_n] = (R.T @ nxyz[valid_n].T).T
            normals_filled = normals_filled.copy()
            normals_filled[:, :, :3] = nxyz_rot

    color_crop[mask_crop == 0] = 0

    out_w, out_h = GELSIGHT_W, GELSIGHT_H

    normal_bgr = normals_to_colormap(normals_filled)
    normal_bgr[mask_crop == 0] = 0
    normal_bgr_out = cv2.resize(normal_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    raw_norm = normals_filled[:, :, :3].copy()
    raw_norm_out = cv2.resize(raw_norm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.resize(mask_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    raw_norm_out[mask_out == 0] = 0.0

    color_out = cv2.resize(color_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return normal_bgr_out, raw_norm_out, color_out


# ── Video helpers ─────────────────────────────────────────────────────────────

def write_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def trim_and_resample(frames, blank_bgr, threshold, num_frames):
    """Trim stale frames before/after contact, resample to num_frames.

    Returns a list of num_frames uint8 BGR frames, or None if no contact detected.
    """
    blank = blank_bgr.astype(np.float32) / 255.0
    diffs = np.array([
        np.linalg.norm(f.astype(np.float32) / 255.0 - blank, axis=-1).mean()
        for f in frames
    ])
    above = np.where(diffs > threshold)[0]
    if len(above) == 0:
        return None
    contact = frames[above[0]: above[-1] + 1]
    if len(contact) == 0:
        return None
    idx = np.linspace(0, len(contact) - 1, num_frames).round().astype(int)
    return [contact[i] for i in idx]


# ── Stage 1: Object selection ──────────────────────────────────────────────────

def run_object_selection(color_bgr, normals_np, predictor):
    """Interactive SAM segmentation flow. Returns mask (H,W) uint8 or None."""
    WIN = "Object Selection: drag=box  Enter=SAM  y=confirm  r=redo  Esc=cancel"

    phase = ["box"]
    box_pts = [None]
    drag_start = [None]
    mask = [None]

    def to_full(xd, yd):
        return (int(xd * ZED_W / ZED_DISPLAY_W), int(yd * ZED_H / ZED_DISPLAY_H))

    def on_mouse(event, x, y, flags, param):
        x = min(max(x, 0), 2 * ZED_DISPLAY_W - 1)
        y = min(max(y, 0), ZED_DISPLAY_H - 1)
        if phase[0] != "box" or x >= ZED_DISPLAY_W:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start[0] = (x, y)
            box_pts[0] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag_start[0] is not None:
            x1f, y1f = to_full(*drag_start[0])
            x2f, y2f = to_full(x, y)
            box_pts[0] = [min(x1f, x2f), min(y1f, y2f), max(x1f, x2f), max(y1f, y2f)]
        elif event == cv2.EVENT_LBUTTONUP and drag_start[0] is not None:
            x1f, y1f = to_full(*drag_start[0])
            x2f, y2f = to_full(x, y)
            box_pts[0] = [min(x1f, x2f), min(y1f, y2f), max(x1f, x2f), max(y1f, y2f)]
            drag_start[0] = None

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    result = [None]

    try:
        while True:
            if mask[0] is None:
                c_panel = cv2.resize(color_bgr, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                n_panel = cv2.resize(normals_to_colormap(normals_np),
                                     (ZED_DISPLAY_W, ZED_DISPLAY_H))
            else:
                c_m = color_bgr.copy()
                c_m[mask[0] == 0] = 0
                n_m = normals_np.copy()
                n_m[mask[0] == 0] = np.nan
                c_panel = cv2.resize(c_m, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                n_panel = cv2.resize(normals_to_colormap(n_m), (ZED_DISPLAY_W, ZED_DISPLAY_H))

            disp = np.hstack([c_panel, n_panel])

            if phase[0] == "box":
                if box_pts[0] is not None:
                    b = box_pts[0]
                    dx1 = int(b[0] * ZED_DISPLAY_W / ZED_W)
                    dy1 = int(b[1] * ZED_DISPLAY_H / ZED_H)
                    dx2 = int(b[2] * ZED_DISPLAY_W / ZED_W)
                    dy2 = int(b[3] * ZED_DISPLAY_H / ZED_H)
                    cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    if drag_start[0] is None:
                        cv2.putText(disp, "Enter=SAM  r=redraw  Esc=cancel",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(disp, "Drag bounding box around object  |  Esc=cancel",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            elif phase[0] == "confirm":
                cv2.putText(disp, "y=confirm and proceed  r=redo  Esc=cancel",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                break
            elif key == ord('r'):
                mask[0] = None
                box_pts[0] = None
                drag_start[0] = None
                phase[0] = "box"
            elif key == ord('y') and phase[0] == "confirm":
                result[0] = mask[0]
                break
            elif (key == 13 and phase[0] == "box"
                  and box_pts[0] is not None and drag_start[0] is None):
                banner = overlay_banner(disp, "Running SAM... please wait")
                cv2.imshow(WIN, banner)
                cv2.waitKey(1)
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    predictor.set_image(color_rgb)
                    masks_out, _, _ = predictor.predict(
                        box=np.array(box_pts[0], dtype=np.float32),
                        multimask_output=False)
                mask[0] = masks_out[0].astype(np.uint8) * 255
                phase[0] = "confirm"
    finally:
        cv2.destroyWindow(WIN)

    return result[0]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture GelSight tactile data for the PatchMatch transfer pipeline."
    )
    p.add_argument("--depth_mode", choices=list(DEPTH_MODES.keys()), default="neural_plus",
                   help="ZED depth estimation mode (default: neural_plus)")
    p.add_argument("--zed_confidence", type=int, default=95,
                   help="ZED depth confidence 0-100 (default: 95)")
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth",
                   help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam_model_type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--gelsight_device", type=str, default="0",
                   help="cv2.VideoCapture device index or video path (default: 0)")
    p.add_argument("--marker_size", type=float, default=0.037,
                   help="ARuCO marker physical size in metres (default: 0.037)")
    p.add_argument("--num_frames", type=int, default=50,
                   help="Target frame count after resampling (default: 50, "
                        "matches gen_contact_query.sh --depth_range_info 0 10 50)")
    p.add_argument("--contact_threshold", type=float, default=0.05,
                   help="Mean L2 diff vs blank frame for contact detection (default: 0.05)")
    p.add_argument("--border_fraction", type=float, default=0.15,
                   help="Border fraction to crop from each GelSight frame edge (default: 0.15, "
                        "matches gsrobotics GelSightMini default)")
    p.add_argument("--inpaint_method", default="telea", choices=["telea", "ns", "nearest"],
                   help="Normal-map hole inpainting method (default: telea)")
    p.add_argument("--save_dir", default="log/gelsight_captures",
                   help="Output directory (default: log/gelsight_captures)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        gs_device = int(args.gelsight_device)
    except ValueError:
        gs_device = args.gelsight_device

    print(f"Loading SAM ({args.sam_model_type}) from {args.sam_checkpoint} ...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM ready.")

    cam, rt, intr = build_camera(args.depth_mode, args.zed_confidence)
    print(f"ZED opened.  fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
          f"cx={intr['cx']:.1f}  cy={intr['cy']:.1f}")

    gs_cap = cv2.VideoCapture(gs_device)
    gs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, GELSIGHT_W)
    gs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GELSIGHT_H)
    if not gs_cap.isOpened():
        cam.close()
        sys.exit(f"Cannot open GelSight camera: {gs_device}")
    gs_fps = gs_cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"GelSight opened at {gs_fps:.1f} fps  ({GELSIGHT_W}x{GELSIGHT_H})")
    print(f"Holder height: {HOLDER_HEIGHT_M*1000:.0f} mm  "
          f"gel thickness: {GEL_THICKNESS_M*1000:.2f} mm  "
          f"marker size: {args.marker_size*1000:.0f} mm")

    aruco_detector = build_aruco_detector()
    camera_matrix = np.array([[intr["fx"], 0, intr["cx"]],
                               [0, intr["fy"], intr["cy"]],
                               [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_sl   = sl.Mat()
    normals_sl = sl.Mat()
    depth_sl   = sl.Mat()
    xyz_sl     = sl.Mat()

    # ── Stage 1: Object Selection ─────────────────────────────────────────────
    print("\n--- Stage 1: Object Selection ---")
    print("Press 'c' to freeze and segment object, 'q' to quit.")

    LIVE_WIN  = "ZED: RGB | Normals  (c=capture  q=quit)"
    CACHE_WIN = "Object Cache: RGB | Normals (masked)"

    color_bgr_cached = normals_cached = depth_cached = mask_cached = blank_frame = None

    while color_bgr_cached is None:
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue

        cam.retrieve_image(image_sl, sl.VIEW.LEFT)
        cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)
        cam.retrieve_measure(depth_sl, sl.MEASURE.DEPTH)

        color_bgr  = image_sl.get_data()[:, :, :3].copy()
        normals_np = normals_sl.get_data().copy()
        depth_raw  = depth_sl.get_data().copy()
        depth_m    = depth_raw.squeeze() if depth_raw.ndim == 3 else depth_raw

        c_d = cv2.resize(color_bgr, (ZED_DISPLAY_W, ZED_DISPLAY_H))
        n_d = cv2.resize(normals_to_colormap(normals_np), (ZED_DISPLAY_W, ZED_DISPLAY_H))
        grid = np.hstack([c_d, n_d])
        cv2.putText(grid, "c=capture  q=quit  |  RGB (left)  Normals (right)",
                    (10, ZED_DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)
        cv2.imshow(LIVE_WIN, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cam.close()
            gs_cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('c'):
            mask = run_object_selection(color_bgr, normals_np, predictor)
            if mask is None:
                print("  Object selection cancelled — try again.")
                continue

            gs_blank = read_gelsight_frame(gs_cap, args.border_fraction)
            if gs_blank is None:
                print("  WARNING: could not read GelSight blank frame — using black.")
                gs_blank = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
            blank_frame = gs_blank
            cv2.imwrite(os.path.join(args.save_dir, "blank_frame.jpg"), blank_frame)

            cam.retrieve_measure(xyz_sl, sl.MEASURE.XYZRGBA)
            xyz_np = xyz_sl.get_data()[:, :, :3].copy()
            np.savez_compressed(
                os.path.join(args.save_dir, "object_cache.npz"),
                color=color_bgr, normals=normals_np,
                depth=depth_m, xyz=xyz_np, mask=mask)

            color_bgr_cached, normals_cached, depth_cached, mask_cached = (
                color_bgr, normals_np, depth_m, mask)

            c_m = color_bgr.copy(); c_m[mask == 0] = 0
            n_m = normals_np.copy(); n_m[mask == 0] = np.nan
            cache_disp = np.hstack([
                cv2.resize(c_m, (ZED_DISPLAY_W, ZED_DISPLAY_H)),
                cv2.resize(normals_to_colormap(n_m), (ZED_DISPLAY_W, ZED_DISPLAY_H)),
            ])
            cv2.imshow(CACHE_WIN, cache_disp)
            print("  Cached: blank_frame.jpg + object_cache.npz")

    cv2.destroyWindow(LIVE_WIN)

    # ── Stage 2: Touch Recording Loop ─────────────────────────────────────────
    print("\n--- Stage 2: Touch Recording ---")
    print(f"  ARuCO marker ID={GELSIGHT_MARKER_ID} (DICT_4X4_50), "
          f"size={args.marker_size*1000:.0f} mm")
    print("  Keys: r=start recording  s=stop+save  a=abort  q=quit")

    ZED_WIN   = "ZED: ARuCO Tracking  (r=record  s=stop  a=abort  q=quit)"
    GS_WIN    = "GelSight Live"
    ORTHO_WIN = "Orthographic Normal Preview"

    touch_idx  = 0
    recording  = False
    buffer     = []
    last_rvec  = None
    last_tvec  = None
    last_px_py = None
    ortho_prev = None

    try:
        while True:
            # ZED grab
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                color_live = image_sl.get_data()[:, :, :3].copy()
            else:
                color_live = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

            # GelSight grab
            gs_frame = read_gelsight_frame(gs_cap, args.border_fraction)
            if gs_frame is None:
                gs_frame = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
            if recording:
                buffer.append(gs_frame.copy())

            # ARuCO detection
            rvec, tvec = detect_gelsight_marker(
                color_live, aruco_detector, camera_matrix, dist_coeffs, args.marker_size)
            vis = color_live.copy()

            if rvec is not None:
                last_rvec, last_tvec = rvec, tvec
                cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                                   rvec, tvec, args.marker_size * 0.5)
                px_py = compute_contact_pixel(rvec, tvec, intr)
                if px_py is not None:
                    last_px_py = px_py
                    cv2.circle(vis, px_py, 8, (0, 0, 255), -1)
                    result = ortho_project_raw(
                        normals_cached, color_bgr_cached, depth_cached, mask_cached,
                        px_py[0], px_py[1], intr, args.inpaint_method, rvec=rvec)
                    if result is not None:
                        ortho_prev = result[0]
            else:
                cv2.putText(vis, "Marker ID=6 not found",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ZED display
            zed_disp = cv2.resize(vis, (ZED_DISPLAY_W, ZED_DISPLAY_H))
            status_text = "RECORDING" if recording else f"Touch #{touch_idx} ready"
            status_color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(zed_disp, status_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(zed_disp, "r=record  s=stop  a=abort  q=quit",
                        (10, ZED_DISPLAY_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow(ZED_WIN, zed_disp)

            # GelSight display
            gs_disp = gs_frame.copy()
            if recording:
                cv2.putText(gs_disp, f"RECORDING  {len(buffer)} frames",
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow(GS_WIN, gs_disp)

            # Ortho preview
            if ortho_prev is not None:
                cv2.imshow(ORTHO_WIN, ortho_prev)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r') and not recording:
                recording = True
                buffer = []
                print(f"  [Touch #{touch_idx}] Recording started ...")

            elif key == ord('a') and recording:
                recording = False
                buffer = []
                print(f"  [Touch #{touch_idx}] Recording aborted.")

            elif key == ord('s') and recording:
                recording = False
                n_buf = len(buffer)
                print(f"  [Touch #{touch_idx}] Stopped. {n_buf} raw frames.")

                if last_px_py is None:
                    print("  ERROR: no valid ARuCO detection — cannot save. Try again.")
                    buffer = []
                    continue

                resampled = trim_and_resample(
                    buffer, blank_frame, args.contact_threshold, args.num_frames)
                if resampled is None:
                    print(f"  WARNING: no contact detected "
                          f"(threshold={args.contact_threshold}). "
                          "Adjust --contact_threshold or retry.")
                    buffer = []
                    continue

                res = ortho_project_raw(
                    normals_cached, color_bgr_cached, depth_cached, mask_cached,
                    last_px_py[0], last_px_py[1], intr,
                    args.inpaint_method, rvec=last_rvec)
                if res is None:
                    print("  ERROR: no valid depth at contact point. Try again.")
                    buffer = []
                    continue
                normal_bgr_out, raw_norm_out, color_out = res

                prefix = os.path.join(args.save_dir, str(touch_idx))
                cv2.imwrite(f"{prefix}_normal.jpg", normal_bgr_out)
                np.savez_compressed(f"{prefix}_normal.npz", normal=raw_norm_out)
                cv2.imwrite(f"{prefix}_color.jpg", color_out)
                write_video(f"{prefix}_shadow.mp4", resampled, gs_fps)
                with open(f"{prefix}_meta.json", "w") as f:
                    json.dump({
                        "touch_idx": touch_idx,
                        "px": int(last_px_py[0]),
                        "py": int(last_px_py[1]),
                        "rvec": last_rvec.tolist() if last_rvec is not None else None,
                        "tvec": last_tvec.tolist() if last_tvec is not None else None,
                        "n_raw_frames": n_buf,
                        "n_resampled_frames": len(resampled),
                    }, f, indent=2)

                print(f"  Saved touch #{touch_idx}: "
                      f"{touch_idx}_normal.jpg/.npz  {touch_idx}_color.jpg  "
                      f"{touch_idx}_shadow.mp4  ({len(resampled)} frames)")
                touch_idx += 1
                buffer = []

    finally:
        cam.close()
        gs_cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {touch_idx} touch location(s) saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
