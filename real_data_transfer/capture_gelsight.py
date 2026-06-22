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
import threading
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
ARUCO_TO_CONTACT_M = 0.050    # measured: 50 mm from marker face to gel tip

ZED_DISPLAY_W, ZED_DISPLAY_H = 640, 360
ZED_W, ZED_H                 = 1280, 720
GELSIGHT_W, GELSIGHT_H       = 320, 240
HEIGHT_CUTOFF_M              = 0.050   # display saturation: depths beyond 50 mm clamp to max color
HEIGHT_MASK_THRES_M          = 0.000   # contact mask threshold: height_map < this value is contact
RENDER_MASK_THRES_M          = 0.000   # per-frame video mask threshold (shifts with pressing depth)
VIDEO_FPS                    = 5.0      # output video fps, matches optical simulation

def _make_Rz(degrees):
    """Rotation matrix around the Z axis by `degrees` (counter-clockwise)."""
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_rvec_z(rvec, R_z):
    """Apply an extra Z-axis rotation (in marker frame) to a Rodrigues vector."""
    R, _ = cv2.Rodrigues(rvec)
    rvec_new, _ = cv2.Rodrigues(R @ R_z)
    return rvec_new.flatten()


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


class GelSightCapture:
    """Background thread that drains the VideoCapture buffer continuously,
    keeping only the latest processed frame to eliminate display lag."""

    def __init__(self, cap, border_fraction=0.15):
        self._cap = cap
        self._border_fraction = border_fraction
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            frame = read_gelsight_frame(self._cap, self._border_fraction)
            if frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


# ── Orthographic projection ───────────────────────────────────────────────────

def ortho_project_raw(normals_np, color_bgr, mask, depth_m, intr, method,
                      rvec=None, tvec=None, render_scale=1.0):
    """
    True orthographic projection of the GelSight Mini FoV using sensor pose.

    Builds a regular (GELSIGHT_H × GELSIGHT_W) grid of physical offsets covering
    the sensor FoV (18.6 × 14.3 mm) multiplied by render_scale, places it in the
    sensor plane using rvec/tvec, and projects each point into ZED pixel
    coordinates to sample normals and color.
    Requires rvec and tvec; returns None if either is missing or all samples are invalid.

    rvec should already encode the z-axis alignment correction via _rotate_rvec_z().
    R_sensor (with R_z baked in) is used for spatial grid placement only.
    Normal re-orientation uses R_orig @ R_flip (R_z undone) to avoid swapping nx↔ny.

    Returns (normal_bgr, raw_normals_hw3, color_bgr_crop) at GELSIGHT_W × GELSIGHT_H.
    normal_bgr:      (H, W, 3) uint8 BGR colormap
    raw_normals_hw3: (H, W, 3) float32 in [-1, 1], expressed in sensor frame
    color_bgr_crop:  (H, W, 3) uint8 BGR
    """
    if rvec is None or tvec is None:
        return None

    H, W = normals_np.shape[:2]
    out_h, out_w = GELSIGHT_H, GELSIGHT_W

    # Sensor frame: marker Z points toward camera; gel contact faces opposite side.
    # rvec encodes R_orig @ R_z(90°) (via _rotate_rvec_z). R_z(90°) corrects the
    # in-plane spatial layout but must NOT be included in the normal re-orientation:
    # R_z(90°) swaps nx↔ny (since it maps x→-y, y→x), causing cyan↔violet.
    # R_sensor (with R_z) is used for spatial sampling only.
    # R_sensor_normals (without R_z) = R_orig @ R_flip is used for normal re-orientation.
    R, _ = cv2.Rodrigues(rvec)
    R_flip   = np.diag([1.0, -1.0, -1.0])  # marker frame → sensor frame (180° around X)
    R_sensor = R @ R_flip                   # for spatial sampling (includes R_z)
    R_z_align = _make_Rz(90)               # the hardcoded alignment rotation
    R_sensor_normals = R @ R_z_align.T @ R_flip  # = R_orig @ R_flip, for normal rotation

    # Contact point in camera frame
    p_contact = R @ np.array([0.0, 0.0, -ARUCO_TO_CONTACT_M]) + tvec  # (3,)

    # Sensor-plane basis vectors in camera frame
    x_axis = R_sensor[:, 0]  # sensor X in camera frame
    y_axis = R_sensor[:, 1]  # sensor Y in camera frame

    # Grid of physical offsets (metres) covering the scaled GelSight FoV.
    fov_w_mm = GELSIGHT_FOV_W_MM * render_scale
    fov_h_mm = GELSIGHT_FOV_H_MM * render_scale
    u = np.linspace(-fov_w_mm / 2, fov_w_mm / 2, out_w) * 1e-3
    v = np.linspace(-fov_h_mm / 2, fov_h_mm / 2, out_h) * 1e-3
    uu, vv = np.meshgrid(u, v)  # (out_h, out_w)

    # 3D points in camera frame for each output pixel
    P_cam = (p_contact[:, None, None]
             + x_axis[:, None, None] * uu[None]
             + y_axis[:, None, None] * vv[None])  # (3, out_h, out_w)

    Pz = P_cam[2]
    valid_proj = Pz > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        spx_f = np.where(valid_proj,
                         intr["fx"] * P_cam[0] / Pz + intr["cx"], -1).astype(np.float32)
        spy_f = np.where(valid_proj,
                         intr["fy"] * P_cam[1] / Pz + intr["cy"], -1).astype(np.float32)

    # Bilinear sampling via cv2.remap — eliminates nearest-neighbour jaggies when
    # the GelSight FoV covers far fewer ZED pixels than the 320×240 output grid.
    remap_kw = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Height map: sample ZED depth at each output pixel's ZED sample location,
    # backproject to 3D, then measure displacement along physical sensor Z.
    depth_safe = np.where(np.isfinite(depth_m) & (depth_m > 0),
                          depth_m, 0.0).astype(np.float32)
    depth_sampled = cv2.remap(depth_safe, spx_f, spy_f, cv2.INTER_LINEAR, **remap_kw)
    valid_depth_remap = depth_sampled > 0
    P_obj = np.stack([
        (spx_f - intr["cx"]) / intr["fx"] * depth_sampled,
        (spy_f - intr["cy"]) / intr["fy"] * depth_sampled,
        depth_sampled,
    ], axis=-1)  # (out_h, out_w, 3) in camera frame
    sensor_z_hmap = R_sensor_normals[:, 2]  # physical sensor Z (excludes R_z correction)
    height_map = np.einsum("hwc,c->hw", P_obj - p_contact, sensor_z_hmap)

    color_crop = cv2.remap(color_bgr, spx_f, spy_f,
                           cv2.INTER_LINEAR, **remap_kw)

    # Normals: zero-fill NaN before bilinear remap, restore invalidity after
    nxyz = normals_np[:, :, :3]
    n_valid = np.isfinite(nxyz).all(axis=2)
    nxyz_safe = np.where(n_valid[:, :, None], nxyz, 0.0).astype(np.float32)
    norm_remap = cv2.remap(nxyz_safe, spx_f, spy_f, cv2.INTER_LINEAR, **remap_kw)
    valid_remap = cv2.remap(n_valid.astype(np.uint8), spx_f, spy_f,
                            cv2.INTER_NEAREST, **remap_kw)
    normals_crop = np.full((out_h, out_w, 4), np.nan, dtype=np.float32)
    normals_crop[:, :, :3] = norm_remap
    normals_crop[valid_remap == 0] = np.nan

    mask_crop = cv2.remap(mask, spx_f, spy_f, cv2.INTER_NEAREST, **remap_kw)

    # Invalidate pixels that projected behind the camera
    normals_crop[~valid_proj] = np.nan
    mask_crop[~valid_proj] = 0
    color_crop[~valid_proj] = 0

    normals_crop[mask_crop == 0] = np.nan
    if not np.isfinite(normals_crop[:, :, 0]).any():
        return None

    normals_filled = inpaint_normals(normals_crop, method)

    # Re-orient normals from ZED camera frame into the GelSight sensor frame.
    # Use R_sensor_normals (= R_orig @ R_flip, without R_z) so spatial alignment
    # rotation does not swap nx↔ny in the normal colormap.
    # NOTE: the re-oriented normals here are in the physical sensor frame, so their
    # colormap will differ from the cached object normals shown in the dashboard
    # (which are in ZED camera frame). This is expected — the cached panel applies a
    # global rotation (the ZED→sensor transformation) before the point cloud is
    # rendered, making the color encoding appear different even for the same surface.
    nxyz = normals_filled[:, :, :3]
    valid_n = np.isfinite(nxyz).all(axis=-1)
    if valid_n.any():
        nxyz_rot = np.full_like(nxyz, np.nan)
        nxyz_rot[valid_n] = (R_sensor_normals.T @ nxyz[valid_n].T).T
        normals_filled = normals_filled.copy()
        normals_filled[:, :, :3] = nxyz_rot

    color_crop[mask_crop == 0] = 0

    normal_bgr = normals_to_colormap(normals_filled)
    normal_bgr[mask_crop == 0] = 0

    raw_norm = normals_filled[:, :, :3].copy()
    raw_norm[mask_crop == 0] = 0.0

    # Render contact mask: object surface above the virtual gel plane (height_map < 0
    # means the object is closer to the camera than the gel tip, i.e. in contact).
    contact_mask = (height_map < HEIGHT_MASK_THRES_M) & valid_depth_remap & (mask_crop > 0)
    h_u8 = (np.clip(-height_map / HEIGHT_CUTOFF_M, 0, 1) * 255).astype(np.uint8)
    height_vis = cv2.applyColorMap(h_u8, cv2.COLORMAP_VIRIDIS)

    return (normal_bgr, raw_norm, color_crop, height_vis, contact_mask,
            height_map, sensor_z_hmap, valid_depth_remap, mask_crop)


# ── Video helpers ─────────────────────────────────────────────────────────────

def write_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def make_render_mask_video(height_map_0, valid_depth_remap, mask_crop,
                           sensor_z_0, tvec_0, pose_contact_slice, num_frames):
    """Generate per-frame render mask images by shifting height_map threshold by pressing depth.

    pressing_depth_i = dot(sensor_z_0, tvec_i - tvec_0)
    contact at frame i: height_map_0 < pressing_depth_i + RENDER_MASK_THRES_M

    Returns a list of num_frames BGR images (white=contact, black=no contact).
    Falls back to a static mask (depth=0) if fewer than 2 valid poses exist.
    """
    from scipy.interpolate import RBFInterpolator

    n_contact = len(pose_contact_slice)
    raw_x, raw_y = [], []
    for i, (rv, tv) in enumerate(pose_contact_slice):
        if tv is not None:
            d = float(np.dot(sensor_z_0, tv - tvec_0))
            raw_x.append(i / max(n_contact - 1, 1))
            raw_y.append(d)

    # Evaluate smoothed depths at the num_frames resampled positions in [0, 1].
    eval_x = np.linspace(0, 1, num_frames)
    if len(raw_x) >= 2:
        rbf = RBFInterpolator(
            np.array(raw_x)[:, None], np.array(raw_y),
            kernel="thin_plate_spline", smoothing=0.1)
        depths = rbf(eval_x[:, None])
    else:
        print("  WARNING: fewer than 2 valid poses in contact range — "
              "contact mask video uses static threshold.")
        depths = np.zeros(num_frames)

    frames_out = []
    for d in depths:
        threshold = float(d) + RENDER_MASK_THRES_M
        cm = (height_map_0 < threshold) & valid_depth_remap & (mask_crop > 0)
        vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
        vis[cm] = 255
        frames_out.append(vis)
    return frames_out


def trim_and_resample(frames, blank_bgr, threshold, num_frames):
    """Trim stale frames before/after contact, resample to num_frames.

    Returns (resampled_frames, peak_raw_idx, contact_start_idx, contact_end_idx)
    where indices are into the original `frames` list.
    Returns (None, None, None, None) if no contact detected.
    """
    blank = blank_bgr.astype(np.float32) / 255.0
    diffs = np.array([
        np.linalg.norm(f.astype(np.float32) / 255.0 - blank, axis=-1).mean()
        for f in frames
    ])
    above = np.where(diffs > threshold)[0]
    if len(above) == 0:
        return None, None, None, None
    peak_raw_idx = int(np.argmax(diffs))
    cs_idx, ce_idx = int(above[0]), int(above[-1])
    contact = frames[cs_idx: ce_idx + 1]
    if len(contact) == 0:
        return None, None, None, None
    idx = np.linspace(0, len(contact) - 1, num_frames).round().astype(int)
    return [contact[i] for i in idx], peak_raw_idx, cs_idx, ce_idx


# ── Display helpers ───────────────────────────────────────────────────────────

def _label_panel(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def _blank_panel(width, height, title):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return _label_panel(img, title)


def _panel_or_blank(img, width, height, title):
    if img is None:
        return _blank_panel(width, height, title)
    return _label_panel(img, title)


def _pad_to(img, width, height):
    h, w = img.shape[:2]
    out = np.zeros((height, width, 3), dtype=img.dtype)
    out[:h, :w] = img
    return out


def _hstack_padded(panels):
    height = max(p.shape[0] for p in panels)
    return np.hstack([_pad_to(p, p.shape[1], height) for p in panels])


def _vstack_padded(rows):
    width = max(r.shape[1] for r in rows)
    return np.vstack([_pad_to(r, width, r.shape[0]) for r in rows])


def _format_scale(scale):
    return f"{scale:g}"


def build_touch_dashboard(zed_disp, cache_disp, gs_disp, normal_prev=None,
                          color_prev=None, height_prev=None,
                          contact_mask_prev=None, multiscale_prevs=None,
                          debug_disp=None):
    zed_vis = cv2.resize(zed_disp, (ZED_W // 2, ZED_H // 2)) if zed_disp is not None else None
    cache_vis = (cv2.resize(cache_disp, (ZED_W, ZED_H // 2))
                 if cache_disp is not None else None)
    top = _hstack_padded([
        _panel_or_blank(zed_vis, ZED_W // 2, ZED_H // 2, "ZED tracking"),
        _panel_or_blank(cache_vis, ZED_W, ZED_H // 2, "Object cache"),
    ])
    bottom = _hstack_padded([
        _panel_or_blank(gs_disp, GELSIGHT_W, GELSIGHT_H, "GelSight live"),
        _panel_or_blank(normal_prev, GELSIGHT_W, GELSIGHT_H, "Normals"),
        _panel_or_blank(color_prev, GELSIGHT_W, GELSIGHT_H, "RGB"),
        _panel_or_blank(height_prev, GELSIGHT_W, GELSIGHT_H, "Height map"),
        _panel_or_blank(contact_mask_prev, GELSIGHT_W, GELSIGHT_H,
                        "Contact mask"),
    ])
    rows = [top, bottom]
    if debug_disp is not None:
        rows.append(_label_panel(debug_disp, "Sensor alignment debug"))
    if multiscale_prevs:
        scale_panels = []
        for scale, normal_img, color_img in multiscale_prevs:
            scale_label = _format_scale(scale)
            scale_panels.extend([
                _panel_or_blank(normal_img, GELSIGHT_W, GELSIGHT_H,
                                f"Normals x{scale_label}"),
                _panel_or_blank(color_img, GELSIGHT_W, GELSIGHT_H,
                                f"RGB x{scale_label}"),
            ])
        rows.append(_hstack_padded(scale_panels))
    return _vstack_padded(rows)


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
    p.add_argument("--render_scale", type=float, nargs="+", default=[1.0],
                   help="One or more FoV multipliers for normal/RGB orthographic "
                        "renders, each still output at 320x240 (default: 1)")
    p.add_argument("--save_dir", default="log/gelsight_captures",
                   help="Output directory (default: log/gelsight_captures)")
    p.add_argument("--debug_sensor_align", action="store_true",
                   help="Show a debug window with normals/RGB blended over the live "
                        "GelSight frame (alpha 0.5/0.5) to verify sensor alignment.")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if any(scale <= 0 for scale in args.render_scale):
        sys.exit("--render_scale values must be positive")
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
    color_bgr_cached = normals_cached = depth_cached = mask_cached = blank_frame = None
    cache_disp_base = None

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
            gs_cap.release()  # GelSightCapture not yet started; release directly
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
                c_m,
                normals_to_colormap(n_m),
            ])
            cache_disp_base = cache_disp.copy()
            print("  Cached: blank_frame.jpg + object_cache.npz")

    cv2.destroyWindow(LIVE_WIN)

    # ── Stage 2: Touch Recording Loop ─────────────────────────────────────────
    print("\n--- Stage 2: Touch Recording ---")
    print(f"  ARuCO marker ID={GELSIGHT_MARKER_ID} (DICT_4X4_50), "
          f"size={args.marker_size*1000:.0f} mm")
    print("  Keys: r=start recording  s=stop+save  a=abort  q=quit")

    # Background thread drains the GelSight buffer so the main loop always
    # gets the latest frame instead of a buffered (lagging) one.
    gs_capture = GelSightCapture(gs_cap, args.border_fraction)

    DASHBOARD_WIN = "GelSight Capture Dashboard  (r=record  s=stop  a=abort  q=quit)"

    # 90° CCW z-axis rotation corrects the in-plane misalignment between the
    # ARuCO marker frame and the physical GelSight sensor orientation.
    R_z = _make_Rz(90)
    print("Z-axis alignment rotation: 90° CCW")

    touch_idx   = 0
    recording   = False
    buffer      = []
    pose_buffer = []  # (rvec, tvec) per recorded frame, parallel to buffer
    last_rvec   = None
    last_tvec   = None
    last_px_py  = None
    normal_prev = None
    color_prev  = None
    height_prev = None
    contact_mask_prev = None
    multiscale_prevs = []

    try:
        while True:
            # ZED grab
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                color_live = image_sl.get_data()[:, :, :3].copy()
            else:
                color_live = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

            # GelSight grab — always latest frame (background thread drains buffer)
            gs_frame = gs_capture.read()
            if gs_frame is None:
                gs_frame = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
            if recording:
                buffer.append(gs_frame.copy())
                pose_buffer.append((
                    last_rvec.copy() if last_rvec is not None else None,
                    last_tvec.copy() if last_tvec is not None else None,
                ))

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
                    aligned_rvec = _rotate_rvec_z(rvec, R_z)
                    result = ortho_project_raw(
                        normals_cached, color_bgr_cached, mask_cached,
                        depth_cached, intr, args.inpaint_method,
                        rvec=aligned_rvec, tvec=tvec)
                    if result is not None:
                        rcm_vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                        rcm_vis[result[4]] = (255, 255, 255)
                        normal_prev = result[0]
                        color_prev = result[2]
                        height_prev = result[3]
                        contact_mask_prev = rcm_vis
                    scale_prevs = []
                    for scale in args.render_scale:
                        if np.isclose(scale, 1.0) and result is not None:
                            scale_prevs.append((scale, result[0], result[2]))
                            continue
                        scale_res = ortho_project_raw(
                            normals_cached, color_bgr_cached, mask_cached,
                            depth_cached, intr, args.inpaint_method,
                            rvec=aligned_rvec, tvec=tvec,
                            render_scale=scale)
                        if scale_res is None:
                            scale_prevs.append((scale, None, None))
                        else:
                            scale_prevs.append((scale, scale_res[0], scale_res[2]))
                    multiscale_prevs = scale_prevs
            else:
                cv2.putText(vis, "Marker ID=6 not found",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ZED display
            zed_disp = vis.copy()
            status_text = "RECORDING" if recording else f"Touch #{touch_idx} ready"
            status_color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(zed_disp, status_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(zed_disp, "r=record  s=stop  a=abort  q=quit",
                        (10, ZED_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            # Cache display: segmented RGB | normals with contact dot
            cache_disp = None
            if cache_disp_base is not None:
                cache_disp = cache_disp_base.copy()
                if last_px_py is not None:
                    dx, dy = last_px_py
                    cv2.circle(cache_disp, (dx, dy), 8, (0, 0, 255), -1)
                    cv2.circle(cache_disp, (dx + ZED_W, dy), 8, (0, 0, 255), -1)

            # GelSight display
            gs_disp = gs_frame.copy()
            if recording:
                cv2.putText(gs_disp, f"RECORDING  {len(buffer)} frames",
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Sensor alignment debug: blend ortho normal/RGB with live GelSight.
            debug_disp = None
            if (args.debug_sensor_align and normal_prev is not None
                    and color_prev is not None):
                gs_vis = gs_frame if gs_frame is not None else np.zeros(
                    (GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                blend_normal = cv2.addWeighted(normal_prev, 0.5, gs_vis, 0.5, 0)
                blend_color  = cv2.addWeighted(color_prev,  0.5, gs_vis, 0.5, 0)
                debug_disp = np.hstack([blend_normal, blend_color])

            dashboard = build_touch_dashboard(
                zed_disp, cache_disp, gs_disp, normal_prev, color_prev,
                height_prev, contact_mask_prev, multiscale_prevs, debug_disp)
            cv2.imshow(DASHBOARD_WIN, dashboard)

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
                pose_buffer = []
                print(f"  [Touch #{touch_idx}] Recording aborted.")

            elif key == ord('s') and recording:
                recording = False
                n_buf = len(buffer)
                print(f"  [Touch #{touch_idx}] Stopped. {n_buf} raw frames.")

                if last_px_py is None:
                    print("  ERROR: no valid ARuCO detection — cannot save. Try again.")
                    buffer = []
                    continue

                resampled, peak_idx, cs_idx, ce_idx = trim_and_resample(
                    buffer, blank_frame, args.contact_threshold, args.num_frames)
                if resampled is None:
                    print(f"  WARNING: no contact detected "
                          f"(threshold={args.contact_threshold}). "
                          "Adjust --contact_threshold or retry.")
                    buffer = []
                    pose_buffer = []
                    continue

                # Use the pose closest in time to the peak-contact GelSight frame.
                peak_rvec, peak_tvec = pose_buffer[peak_idx]
                if peak_rvec is None:
                    # Fall back to last known pose if marker wasn't visible at peak.
                    peak_rvec, peak_tvec = last_rvec, last_tvec
                    print("  WARNING: no marker pose at peak-contact frame — "
                          "using last known pose.")

                # pose_buffer[i] is the ARuCO pose detected on the ZED frame that was
                # current when GelSight frame i was captured.  Using the peak-contact
                # index gives the sensor pose at the moment of deepest press, which best
                # represents the contact geometry saved in the static normal/color files.
                peak_aligned_rvec = _rotate_rvec_z(peak_rvec, R_z)
                res = ortho_project_raw(
                    normals_cached, color_bgr_cached, mask_cached,
                    depth_cached, intr, args.inpaint_method,
                    rvec=peak_aligned_rvec, tvec=peak_tvec)
                if res is None:
                    print("  ERROR: no valid depth at contact point. Try again.")
                    buffer = []
                    pose_buffer = []
                    continue
                normal_bgr_out, raw_norm_out, color_out, height_vis_out, rcm_out = res[:5]

                scaled_static = {}
                for scale in args.render_scale:
                    if np.isclose(scale, 1.0):
                        scaled_static[scale] = (normal_bgr_out, raw_norm_out, color_out)
                        continue
                    scale_res = ortho_project_raw(
                        normals_cached, color_bgr_cached, mask_cached,
                        depth_cached, intr, args.inpaint_method,
                        rvec=peak_aligned_rvec, tvec=peak_tvec,
                        render_scale=scale)
                    if scale_res is None:
                        print(f"  WARNING: scale {scale:g} ortho failed — skipping.")
                    else:
                        scaled_static[scale] = (scale_res[0], scale_res[1], scale_res[2])

                # Per-frame contact mask video: use the contact-start pose for height_map_0
                # so the threshold shifts by pressing depth for each resampled frame.
                cs_rvec_raw, cs_tvec_raw = pose_buffer[cs_idx]
                if cs_rvec_raw is None:
                    cs_rvec_raw, cs_tvec_raw = peak_rvec, peak_tvec
                    print("  WARNING: no marker pose at contact start — "
                          "using peak pose for contact mask video.")
                cs_res = ortho_project_raw(
                    normals_cached, color_bgr_cached, mask_cached,
                    depth_cached, intr, args.inpaint_method,
                    rvec=_rotate_rvec_z(cs_rvec_raw, R_z), tvec=cs_tvec_raw)
                if cs_res is not None:
                    hmap_0, sz_0, vdr_0, mc_0 = cs_res[5], cs_res[6], cs_res[7], cs_res[8]
                    pose_contact = pose_buffer[cs_idx: ce_idx + 1]
                    rm_frames = make_render_mask_video(
                        hmap_0, vdr_0, mc_0, sz_0, cs_tvec_raw,
                        pose_contact, args.num_frames)
                else:
                    print("  WARNING: contact-start ortho failed — "
                          "render mask video will be static.")
                    rm_vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                    rm_vis[rcm_out] = 255
                    rm_frames = [rm_vis] * args.num_frames

                prefix = os.path.join(args.save_dir, str(touch_idx))
                cv2.imwrite(f"{prefix}_normal.jpg", normal_bgr_out)
                np.savez_compressed(f"{prefix}_normal.npz", normal=raw_norm_out)
                cv2.imwrite(f"{prefix}_color.jpg", color_out)
                for scale, (normal_bgr_s, raw_norm_s, color_s) in scaled_static.items():
                    scale_tag = _format_scale(scale)
                    cv2.imwrite(f"{prefix}_scale{scale_tag}_normal.jpg", normal_bgr_s)
                    np.savez_compressed(
                        f"{prefix}_scale{scale_tag}_normal.npz", normal=raw_norm_s)
                    cv2.imwrite(f"{prefix}_scale{scale_tag}_color.jpg", color_s)
                cv2.imwrite(f"{prefix}_height.jpg", height_vis_out)
                cv2.imwrite(f"{prefix}_contact_mask.jpg",
                            (rcm_out.astype(np.uint8) * 255))
                write_video(f"{prefix}_shadow.mp4", resampled, VIDEO_FPS)
                write_video(f"{prefix}_render_mask.mp4", rm_frames, VIDEO_FPS)
                side_by_side = [np.hstack([s, r]) for s, r in zip(resampled, rm_frames)]
                write_video(f"{prefix}_shadow_render_mask.mp4", side_by_side, VIDEO_FPS)
                with open(f"{prefix}_meta.json", "w") as f:
                    json.dump({
                        "touch_idx": touch_idx,
                        "px": int(last_px_py[0]),
                        "py": int(last_px_py[1]),
                        "rvec": peak_rvec.tolist() if peak_rvec is not None else None,
                        "tvec": peak_tvec.tolist() if peak_tvec is not None else None,
                        "n_raw_frames": n_buf,
                        "n_resampled_frames": len(resampled),
                        "render_scale": args.render_scale,
                    }, f, indent=2)

                print(f"  Saved touch #{touch_idx}: "
                      f"{touch_idx}_normal.jpg/.npz  {touch_idx}_color.jpg  "
                      f"{touch_idx}_shadow.mp4  ({len(resampled)} frames)")
                touch_idx += 1
                buffer = []
                pose_buffer = []

    finally:
        cam.close()
        gs_capture.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {touch_idx} touch location(s) saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
