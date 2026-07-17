"""
GUI for calibrating the X/Y/Z translation offset between the ARuCO marker
frame and the GelSight sensor's true physical contact point.

The marker and sensor planes are assumed parallel, but their in-plane angle
may differ. Marker->contact X/Y/Z translation and tactile-image theta are
therefore calibrated here. Production currently uses a hand-measured Z-only
constant (ARUCO_TO_CONTACT_M in _gelsight_processing.py) with no lateral
(X/Y) or angular term, which can cause pose-dependent geometric error.

Stage 1 (Scene Capture):
  Live ZED stream of a static scene. Press 'c' to run FoundationStereo once
  and cache normals/color/depth. No SAM segmentation -- the whole frame is
  used, since the contact-mask/render-mask machinery isn't needed here.

Stage 2 (Offset Calibration):
  Move the GelSight sensor (marker attached) around in front of the now-
  static cached scene. Four trackbars (X/Y/Z in mm, Theta in degrees) control
  the current offset guess. Theta rotates the GelSight image
  around its center to compensate for in-plane marker/sensor misalignment.
  Every frame, the GUI re-renders orthographic normal/RGB
  crops at scales 1/2/4/8 from the live marker pose using the current
  offset, plus:
    - a red box on scale>1 crops delineating where the scale=1 sensor
      footprint falls within that wider crop
    - a red dot on the live ZED view and on the cached full-scene panel
      showing the assumed contact point
  Adjust X/Y/Z until the red box / red dot track the sensor's true physical
  contact location as you move it around the scene. Press 's' to print and
  save the calibrated offset, 'r' to re-capture the scene, 'q' to quit.

Usage:
  python real_data_transfer/calibrate_sensor_offset.py \
      --geometry_mode fast_foundation_stereo \
      --fs_model_dir real_data_transfer/Fast-FoundationStereo/checkpoint.pth
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gelsight_processing import (
    _make_Rz, _rotate_rvec_z, ortho_project_raw, normals_to_colormap,
    GELSIGHT_W, GELSIGHT_H, ARUCO_TO_CONTACT_M,
)
from capture_gelsight import (
    build_aruco_detector, detect_gelsight_marker, GelSightCapture,
    _label_panel, _hstack_padded, _vstack_padded,
    ZED_DISPLAY_W, ZED_DISPLAY_H, ZED_W, ZED_H,
)
from visualize_zed_normal_sim import build_camera, overlay_banner, DEPTH_MODES
from visualize_zed_fs import load_model, run_inference, normals_from_xyz

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found.  Install with: pip install pyzed")

try:
    import cv2.aruco  # noqa: F401
except AttributeError:
    sys.exit("opencv-contrib-python (with cv2.aruco) is required.")


# ── Trackbar resolution ─────────────────────────────────────────────────────
TRACKBAR_STEP_MM = 0.1  # each trackbar tick = 0.1 mm


def _mm_to_ticks(mm, lo_mm):
    return int(round((mm - lo_mm) / TRACKBAR_STEP_MM))


def _ticks_to_mm(ticks, lo_mm):
    return lo_mm + ticks * TRACKBAR_STEP_MM


TRACKBAR_X_NAME = "X offset mm (marker right = +)"
TRACKBAR_Y_NAME = "Y offset mm (marker down  = +)"
TRACKBAR_Z_NAME = "Z offset mm (marker->gel = +)"
TRACKBAR_THETA_NAME = "Theta deg (CCW = +)"
THETA_STEP_DEG = 0.1


def _deg_to_ticks(degrees, lo_deg):
    return int(round((degrees - lo_deg) / THETA_STEP_DEG))


def _ticks_to_deg(ticks, lo_deg):
    return lo_deg + ticks * THETA_STEP_DEG


def _build_legend_image():
    # Many OpenCV GUI backends (GTK/Qt-less builds) never render the trackbar
    # name string next to the slider -- only the slider itself. Rather than
    # depend on that, draw the labels into an image and show it in the same
    # window; imshow content always renders regardless of backend.
    legend = np.zeros((140, 640, 3), dtype=np.uint8)
    lines = [
        "Slider 1 (top)    = X offset  (marker right = +)",
        "Slider 2 (middle) = Y offset  (marker down  = +)",
        "Slider 3 (bottom) = Z offset  (marker -> gel = +)",
        "Slider 4          = Theta      (tactile CCW rotation = +)",
        "Or press x / y / z / t in the dashboard to type an exact value.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(legend, line, (8, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return legend


def _setup_trackbars(win, init_x_mm, init_y_mm, init_z_mm,
                     init_theta_deg, xy_range_mm, z_max_mm, theta_range_deg):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 640, 300)
    x_max_ticks = _mm_to_ticks(xy_range_mm, -xy_range_mm)
    y_max_ticks = _mm_to_ticks(xy_range_mm, -xy_range_mm)
    z_max_ticks = _mm_to_ticks(z_max_mm, 0.0)
    theta_max_ticks = _deg_to_ticks(theta_range_deg, -theta_range_deg)
    cv2.createTrackbar(TRACKBAR_X_NAME, win, _mm_to_ticks(init_x_mm, -xy_range_mm),
                       x_max_ticks, lambda v: None)
    cv2.createTrackbar(TRACKBAR_Y_NAME, win, _mm_to_ticks(init_y_mm, -xy_range_mm),
                       y_max_ticks, lambda v: None)
    cv2.createTrackbar(TRACKBAR_Z_NAME, win, _mm_to_ticks(init_z_mm, 0.0),
                       z_max_ticks, lambda v: None)
    cv2.createTrackbar(
        TRACKBAR_THETA_NAME, win,
        _deg_to_ticks(init_theta_deg, -theta_range_deg),
        theta_max_ticks, lambda v: None)
    print(f"  Trackbar order in '{win}' window (top to bottom): "
          f"1) {TRACKBAR_X_NAME}  2) {TRACKBAR_Y_NAME}  "
          f"3) {TRACKBAR_Z_NAME}  4) {TRACKBAR_THETA_NAME}")

    legend = _build_legend_image()
    cv2.imshow(win, legend)
    cv2.waitKey(1)
    return legend


def _read_trackbars(win, xy_range_mm, theta_range_deg):
    x_mm = _ticks_to_mm(cv2.getTrackbarPos(TRACKBAR_X_NAME, win), -xy_range_mm)
    y_mm = _ticks_to_mm(cv2.getTrackbarPos(TRACKBAR_Y_NAME, win), -xy_range_mm)
    z_mm = _ticks_to_mm(cv2.getTrackbarPos(TRACKBAR_Z_NAME, win), 0.0)
    theta_deg = _ticks_to_deg(
        cv2.getTrackbarPos(TRACKBAR_THETA_NAME, win), -theta_range_deg)
    return x_mm, y_mm, z_mm, theta_deg


def _set_trackbar_value(win, axis, value, xy_range_mm, z_range_mm,
                        theta_range_deg):
    """Clamp an exact-entry value, push it to its trackbar, and return it."""
    if axis in ("x", "y"):
        value = float(np.clip(value, -xy_range_mm, xy_range_mm))
        name = TRACKBAR_X_NAME if axis == "x" else TRACKBAR_Y_NAME
        ticks = _mm_to_ticks(value, -xy_range_mm)
        cv2.setTrackbarPos(name, win, ticks)
        value = _ticks_to_mm(ticks, -xy_range_mm)
    elif axis == "z":
        value = float(np.clip(value, 0.0, z_range_mm))
        ticks = _mm_to_ticks(value, 0.0)
        cv2.setTrackbarPos(TRACKBAR_Z_NAME, win, ticks)
        value = _ticks_to_mm(ticks, 0.0)
    else:
        value = float(np.clip(value, -theta_range_deg, theta_range_deg))
        ticks = _deg_to_ticks(value, -theta_range_deg)
        cv2.setTrackbarPos(
            TRACKBAR_THETA_NAME, win, ticks)
        value = _ticks_to_deg(ticks, -theta_range_deg)
    return value


# ── Contact-point projection (mirrors compute_contact_pixel, offset-aware) ──

def _contact_point_cam(rvec, tvec, offset_m):
    R, _ = cv2.Rodrigues(rvec)
    ox, oy, oz = offset_m
    return R @ np.array([ox, oy, -oz]) + tvec


def _project_point(p_cam, intr, w, h):
    if p_cam[2] <= 0:
        return None
    px = int(intr["fx"] * p_cam[0] / p_cam[2] + intr["cx"])
    py = int(intr["fy"] * p_cam[1] / p_cam[2] + intr["cy"])
    return (min(max(px, 0), w - 1), min(max(py, 0), h - 1))


# ── Red-box overlay: where does the scale=1 footprint fall in a scale=N crop ─

def _titled_panel(img, title):
    """Like _label_panel, but stacks the title bar above the image instead of
    overlaying it onto the top rows -- the scale panels below carry a red
    footprint box centered on the *full* render, so overwriting rows would
    make the box look off-center relative to what's actually visible."""
    h, w = img.shape[:2]
    bar = np.zeros((24, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _draw_footprint_box(img, scale, theta_deg=0.0):
    out = img.copy()
    h, w = out.shape[:2]
    half_w = w / (2.0 * scale)
    half_h = h / (2.0 * scale)
    cx, cy = w / 2.0, h / 2.0
    corners = np.array([[
        [cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h], [cx - half_w, cy + half_h],
    ]], dtype=np.float32)
    rotation = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    corners = cv2.transform(corners, rotation)[0]
    cv2.polylines(out, [np.rint(corners).astype(np.int32)], True,
                  (0, 0, 255), 2, cv2.LINE_AA)
    return out


# ── Tactile/normal overlay: place the live GelSight frame at its true ───────
# physical size within a scale-N crop (shrunk + centered, same box the red
# footprint outline marks), then alpha-blend with the rendered normals.

def _embed_gs_frame(gs_frame, out_w, out_h, scale, theta_deg=0.0):
    if gs_frame is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    inner_w = out_w if np.isclose(scale, 1.0) else max(1, int(round(out_w / scale)))
    inner_h = out_h if np.isclose(scale, 1.0) else max(1, int(round(out_h / scale)))
    resized = cv2.resize(gs_frame, (inner_w, inner_h))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - inner_w) // 2
    y0 = (out_h - inner_h) // 2
    canvas[y0:y0 + inner_h, x0:x0 + inner_w] = resized
    if np.isclose(theta_deg, 0.0):
        return canvas
    rotation = cv2.getRotationMatrix2D(
        ((out_w - 1) / 2.0, (out_h - 1) / 2.0), theta_deg, 1.0)
    return cv2.warpAffine(canvas, rotation, (out_w, out_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _tactile_normal_overlay(normal_img, gs_frame, scale, theta_deg=0.0,
                            gs_alpha=0.65):
    h, w = normal_img.shape[:2]
    gs_embed = _embed_gs_frame(gs_frame, w, h, scale, theta_deg)
    return cv2.addWeighted(normal_img, 1.0 - gs_alpha, gs_embed, gs_alpha, 0)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate the ARuCO marker -> GelSight contact-point "
                    "X/Y/Z translation and tactile theta offsets with a live GUI."
    )
    p.add_argument("--depth_mode", choices=list(DEPTH_MODES.keys()), default="neural_plus",
                   help="ZED depth mode, only used when --geometry_mode zed.")
    p.add_argument("--zed_confidence", type=int, default=95)
    p.add_argument("--gelsight_device", type=str, default="0",
                   help="cv2.VideoCapture device index or path. Only used for "
                        "the live GelSight preview panel -- not required for "
                        "calibration itself (marker tracking uses the ZED feed).")
    p.add_argument("--no_gelsight", action="store_true",
                   help="Skip opening the GelSight camera entirely (no live preview panel).")
    p.add_argument("--marker_size", type=float, default=0.037,
                   help="ARuCO marker physical size in metres (default: 0.037)")
    p.add_argument("--border_fraction", type=float, default=0.15)
    p.add_argument("--inpaint_method", default="telea", choices=["telea", "ns", "nearest"])
    p.add_argument("--render_scales", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0],
                   help="FoV multipliers to render side-by-side (default: 1 2 4 8).")
    p.add_argument("--init_x_mm", type=float, default=None,
                   help="Initial X in mm; overrides the saved offset.")
    p.add_argument("--init_y_mm", type=float, default=None,
                   help="Initial Y in mm; overrides the saved offset.")
    p.add_argument("--init_z_mm", type=float, default=None,
                   help="Initial Z in mm; overrides the saved offset.")
    p.add_argument("--init_theta_deg", type=float, default=None,
                   help="Initial CCW tactile rotation in degrees; overrides "
                        "the saved offset.")
    p.add_argument("--xy_range_mm", type=float, default=30.0,
                   help="X/Y trackbar half-range in mm (default: +-30).")
    p.add_argument("--z_range_mm", type=float, default=100.0,
                   help="Z trackbar max in mm (default: 0-100).")
    p.add_argument("--theta_range_deg", type=float, default=180.0,
                   help="Theta trackbar half-range in degrees (default: +-180).")
    p.add_argument("--geometry_mode",
                   choices=["zed", "foundation_stereo", "fast_foundation_stereo"],
                   default="fast_foundation_stereo")
    p.add_argument("--fs_model_dir", default=None,
                   help="Path to FoundationStereo checkpoint. Required unless --geometry_mode zed.")
    p.add_argument("--fs_valid_iters", type=int, default=None)
    p.add_argument("--fs_max_disp", type=int, default=192)
    p.add_argument("--fs_scale", type=float, default=1.0)
    p.add_argument("--save_path", default="log/gelsight_sensor_offset.json",
                   help="Where 's' writes the calibrated offset (default: "
                        "log/gelsight_sensor_offset.json).")
    p.add_argument("--no_load_saved_offset", action="store_true",
                   help="Ignore an existing --save_path and start from built-in "
                        "defaults unless --init_* values are supplied.")
    return p.parse_args()


def _initialize_offsets(args):
    """Resolve initial controls from CLI overrides, saved JSON, then defaults."""
    saved = {}
    if not args.no_load_saved_offset and os.path.isfile(args.save_path):
        try:
            with open(args.save_path, "r") as f:
                saved = json.load(f)
            print(f"Loaded saved offset from {args.save_path}.")
        except (OSError, ValueError, TypeError) as exc:
            print(f"WARNING: could not load saved offset {args.save_path}: {exc}")

    defaults = {
        "init_x_mm": 0.0,
        "init_y_mm": 0.0,
        "init_z_mm": ARUCO_TO_CONTACT_M * 1000.0,
        "init_theta_deg": 0.0,
    }
    def saved_number(key, scale, fallback):
        try:
            return float(saved[key]) * scale
        except (KeyError, TypeError, ValueError):
            return fallback

    saved_values = {
        "init_x_mm": saved_number("offset_x_m", 1e3, defaults["init_x_mm"]),
        "init_y_mm": saved_number("offset_y_m", 1e3, defaults["init_y_mm"]),
        "init_z_mm": saved_number("offset_z_m", 1e3, defaults["init_z_mm"]),
        "init_theta_deg": saved_number(
            "offset_theta_deg", 1.0, defaults["init_theta_deg"]),
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, float(saved_values.get(name, fallback)))

    print("Initial offset: "
          f"X={args.init_x_mm:.2f}mm  Y={args.init_y_mm:.2f}mm  "
          f"Z={args.init_z_mm:.2f}mm  Theta={args.init_theta_deg:.2f}deg")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if any(s <= 0 for s in args.render_scales):
        sys.exit("--render_scales values must be positive")
    if args.theta_range_deg <= 0:
        sys.exit("--theta_range_deg must be positive")
    if args.geometry_mode != "zed" and not args.fs_model_dir:
        sys.exit("--fs_model_dir is required when --geometry_mode != zed")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    _initialize_offsets(args)

    if args.geometry_mode != "zed":
        if args.fs_valid_iters is None:
            args.fs_valid_iters = 8 if args.geometry_mode == "fast_foundation_stereo" else 32
        print(f"Loading {args.geometry_mode} from {args.fs_model_dir} ...")
        fs_model = load_model(args.geometry_mode, args.fs_model_dir,
                              args.fs_valid_iters, args.fs_max_disp)
        print("FS model ready.")
    else:
        fs_model = None

    cam, rt, intr = build_camera(args.depth_mode, args.zed_confidence)
    print(f"ZED opened.  fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
          f"cx={intr['cx']:.1f}  cy={intr['cy']:.1f}")

    fs_baseline = None
    image_r_sl = None
    if args.geometry_mode != "zed":
        info = cam.get_camera_information()
        fs_baseline = info.camera_configuration.calibration_parameters.get_camera_baseline()
        image_r_sl = sl.Mat()
        print(f"FS stereo baseline: {fs_baseline*1000:.1f} mm")

    gs_capture = None
    if not args.no_gelsight:
        try:
            gs_device = int(args.gelsight_device)
        except ValueError:
            gs_device = args.gelsight_device
        gs_cap = cv2.VideoCapture(gs_device)
        gs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, GELSIGHT_W)
        gs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GELSIGHT_H)
        if gs_cap.isOpened():
            gs_capture = GelSightCapture(gs_cap, args.border_fraction)
            print("GelSight live preview enabled.")
        else:
            print("WARNING: could not open GelSight camera -- preview disabled "
                  "(not required for calibration).")

    aruco_detector = build_aruco_detector()
    camera_matrix = np.array([[intr["fx"], 0, intr["cx"]],
                               [0, intr["fy"], intr["cy"]],
                               [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_sl   = sl.Mat()
    normals_sl = sl.Mat()
    depth_sl   = sl.Mat()

    R_z = _make_Rz(90)  # in-plane alignment, same fixed rotation as production

    TRACKBAR_WIN = "Offset Controls"
    CAPTURE_WIN = "ZED live  (c=capture scene  q=quit)"
    DASHBOARD_WIN = "Sensor Offset Calibration  (r=recapture  s=save  q=quit)"

    try:
        while True:
            # ── Stage 1: Scene Capture ──────────────────────────────────────
            print("\n--- Stage 1: Scene Capture ---")
            print("Point the ZED at a static, textured scene. Press 'c' to "
                  "capture, 'q' to quit.")
            color_cached = normals_cached = depth_cached = None
            right_bgr = None

            while color_cached is None:
                if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                    continue
                cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)
                cam.retrieve_measure(depth_sl, sl.MEASURE.DEPTH)
                if args.geometry_mode != "zed":
                    cam.retrieve_image(image_r_sl, sl.VIEW.RIGHT)
                    right_bgr = image_r_sl.get_data()[:, :, :3].copy()

                color_bgr  = image_sl.get_data()[:, :, :3].copy()
                normals_np = normals_sl.get_data().copy()
                depth_raw  = depth_sl.get_data().copy()
                depth_m    = depth_raw.squeeze() if depth_raw.ndim == 3 else depth_raw

                disp = cv2.resize(color_bgr, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                cv2.putText(disp, "c=capture  q=quit", (10, ZED_DISPLAY_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow(CAPTURE_WIN, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                elif key == ord('c'):
                    if args.geometry_mode != "zed" and right_bgr is not None:
                        banner = overlay_banner(disp, f"Running {args.geometry_mode}...")
                        cv2.imshow(CAPTURE_WIN, banner)
                        cv2.waitKey(1)
                        print(f"  Running {args.geometry_mode} for depth/normals...")
                        left_rgb  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
                        _, fs_dl, fs_xyz = run_inference(
                            fs_model, args.geometry_mode, left_rgb, right_rgb,
                            intr, fs_baseline, args.fs_scale, args.fs_valid_iters,
                            z_near=0.1, z_far=5.0)
                        if fs_dl.shape != (ZED_H, ZED_W):
                            depth_cached = cv2.resize(fs_dl, (ZED_W, ZED_H),
                                                      interpolation=cv2.INTER_LINEAR)
                            fn_lr = normals_from_xyz(fs_xyz)
                            normals_cached = cv2.resize(fn_lr, (ZED_W, ZED_H),
                                                        interpolation=cv2.INTER_LINEAR)
                            nrm = np.linalg.norm(normals_cached, axis=2, keepdims=True)
                            normals_cached = np.where(
                                nrm > 1e-8, normals_cached / nrm, 0.0).astype(np.float32)
                        else:
                            depth_cached = fs_dl
                            normals_cached = normals_from_xyz(fs_xyz)
                        normals_cached[~np.isfinite(depth_cached)] = np.nan
                    else:
                        normals_cached = normals_np
                        depth_cached = depth_m
                    color_cached = color_bgr
                    print("  Scene cached.")

            cv2.destroyWindow(CAPTURE_WIN)
            mask_full = np.full(normals_cached.shape[:2], 255, dtype=np.uint8)
            cached_disp_base = cv2.resize(
                np.hstack([color_cached, normals_to_colormap(normals_cached)]),
                (ZED_DISPLAY_W * 2, ZED_DISPLAY_H))

            # ── Stage 2: Offset Calibration ─────────────────────────────────
            print("\n--- Stage 2: Offset Calibration ---")
            print("  Move the sensor around the scene. Adjust X/Y/Z/Theta "
                  "until the red box, red dot, and tactile image align.")
            print("  Keys: r=recapture scene  s=save offset  q=quit")

            _setup_trackbars(TRACKBAR_WIN, args.init_x_mm, args.init_y_mm,
                             args.init_z_mm, args.init_theta_deg,
                             args.xy_range_mm, args.z_range_mm,
                             args.theta_range_deg)
            print("  Press x/y/z/t in the dashboard window to type an exact "
                  "offset (mm for X/Y/Z, degrees for Theta); "
                  "Enter=commit  Esc=cancel.")

            typing_axis = None  # None, or x/y/z/t while in numeric-entry mode
            typing_buf = ""

            recapture = False
            while not recapture:
                if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                    color_live = image_sl.get_data()[:, :, :3].copy()
                else:
                    color_live = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

                x_mm, y_mm, z_mm, theta_deg = _read_trackbars(
                    TRACKBAR_WIN, args.xy_range_mm, args.theta_range_deg)
                offset_m = (x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3)

                gs_disp = None
                if gs_capture is not None:
                    gs_frame = gs_capture.read()
                    if gs_frame is not None:
                        gs_disp = gs_frame

                rvec, tvec = detect_gelsight_marker(
                    color_live, aruco_detector, camera_matrix, dist_coeffs, args.marker_size)

                vis = color_live.copy()
                cached_disp = cached_disp_base.copy()
                scale_panels = []

                if rvec is not None:
                    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                                      rvec, tvec, args.marker_size * 0.5)
                    p_contact = _contact_point_cam(rvec, tvec, offset_m)
                    px_py = _project_point(p_contact, intr, ZED_W, ZED_H)
                    if px_py is not None:
                        cv2.circle(vis, px_py, 10, (0, 0, 255), -1)
                        # Same point, drawn on the cached (display-scaled) scene panel.
                        dx = int(px_py[0] * ZED_DISPLAY_W / ZED_W)
                        dy = int(px_py[1] * ZED_DISPLAY_H / ZED_H)
                        cv2.circle(cached_disp, (dx, dy), 8, (0, 0, 255), -1)
                        cv2.circle(cached_disp, (dx + ZED_DISPLAY_W, dy), 8, (0, 0, 255), -1)

                    aligned_rvec = _rotate_rvec_z(rvec, R_z)
                    for scale in args.render_scales:
                        res = ortho_project_raw(
                            normals_cached, color_cached, mask_full, depth_cached,
                            intr, args.inpaint_method, rvec=aligned_rvec, tvec=tvec,
                            render_scale=scale, apply_mask=False, offset=offset_m)
                        if res is None:
                            normal_raw = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                        else:
                            normal_raw = res[0]
                        overlay_raw = _tactile_normal_overlay(
                            normal_raw, gs_disp, scale, theta_deg)
                        if scale > 1.0:
                            normal_img = _draw_footprint_box(normal_raw, scale, theta_deg)
                            overlay_img = _draw_footprint_box(overlay_raw, scale, theta_deg)
                        else:
                            normal_img, overlay_img = normal_raw, overlay_raw
                        tag = f"{scale:g}"
                        scale_panels.append(_titled_panel(normal_img, f"x{tag} normal"))
                        scale_panels.append(_titled_panel(overlay_img, f"x{tag} tactile overlay"))
                else:
                    cv2.putText(vis, "Marker ID=6 not found",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    for scale in args.render_scales:
                        tag = f"{scale:g}"
                        blank = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                        scale_panels.append(_titled_panel(blank, f"x{tag} normal"))
                        scale_panels.append(_titled_panel(blank, f"x{tag} tactile overlay"))

                zed_disp = cv2.resize(vis, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                cv2.putText(zed_disp,
                            f"offset X={x_mm:+.1f}mm Y={y_mm:+.1f}mm "
                            f"Z={z_mm:.1f}mm Theta={theta_deg:+.1f}deg",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
                cv2.putText(zed_disp, "r=recapture s=save q=quit x/y/z/t=type value",
                            (10, ZED_DISPLAY_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                top_row = _hstack_padded([
                    _label_panel(zed_disp, "ZED live (red dot = assumed contact point)"),
                    _label_panel(cached_disp, "Cached scene: color | normals"),
                ])
                mid_panels = []
                if gs_disp is not None:
                    mid_panels.append(_titled_panel(gs_disp, "GelSight live"))
                mid_panels.extend(scale_panels)
                mid_row = _hstack_padded(mid_panels)
                dashboard = _vstack_padded([top_row, mid_row])

                # Legend/typing banner -- drawn into the image itself (not the
                # trackbar window) so it renders identically on every OpenCV
                # GUI backend, regardless of whether trackbar name labels do.
                banner_h = 26
                banner = np.zeros((banner_h, dashboard.shape[1], 3), dtype=np.uint8)
                if typing_axis is not None:
                    unit = "deg" if typing_axis == "t" else "mm"
                    text = (f"Type {typing_axis.upper()} offset ({unit}): {typing_buf}_"
                           f"   [Enter=commit  Esc=cancel]")
                    color = (0, 255, 255)
                else:
                    text = ("Trackbars = X, Y, Z (mm), Theta (deg)   |   "
                           "press x/y/z/t then type a number + Enter")
                    color = (200, 200, 200)
                cv2.putText(banner, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                           0.55, color, 1, cv2.LINE_AA)
                dashboard = np.vstack([banner, dashboard])
                cv2.imshow(DASHBOARD_WIN, dashboard)

                key = cv2.waitKey(1) & 0xFF

                if typing_axis is not None:
                    if key in (13, 10):  # Enter
                        if typing_buf not in ("", "-", "."):
                            try:
                                val = float(typing_buf)
                                clamped = _set_trackbar_value(
                                    TRACKBAR_WIN, typing_axis, val,
                                    args.xy_range_mm, args.z_range_mm,
                                    args.theta_range_deg)
                                unit = "deg" if typing_axis == "t" else "mm"
                                print(f"  {typing_axis.upper()} offset set to "
                                      f"{clamped:.2f}{unit}")
                            except ValueError:
                                print(f"  Could not parse '{typing_buf}' as a number.")
                        typing_axis = None
                        typing_buf = ""
                    elif key == 27:  # Esc
                        typing_axis = None
                        typing_buf = ""
                    elif key in (8, 127):  # Backspace
                        typing_buf = typing_buf[:-1]
                    elif key == ord('-') and typing_buf == "":
                        typing_buf += "-"
                    elif key == ord('.') and '.' not in typing_buf:
                        typing_buf += "."
                    elif ord('0') <= key <= ord('9'):
                        typing_buf += chr(key)
                elif key == ord('q'):
                    raise KeyboardInterrupt
                elif key == ord('r'):
                    recapture = True
                elif key == ord('s'):
                    result = {
                        "offset_x_m": x_mm * 1e-3,
                        "offset_y_m": y_mm * 1e-3,
                        "offset_z_m": z_mm * 1e-3,
                        "offset_theta_deg": theta_deg,
                    }
                    with open(args.save_path, "w") as f:
                        json.dump(result, f, indent=2)
                    # Preserve the newly saved calibration if the user recaptures
                    # the scene during this same process.
                    args.init_x_mm = x_mm
                    args.init_y_mm = y_mm
                    args.init_z_mm = z_mm
                    args.init_theta_deg = theta_deg
                    print(f"  Saved offset to {args.save_path}: "
                          f"X={x_mm:.2f}mm  Y={y_mm:.2f}mm  Z={z_mm:.2f}mm  "
                          f"Theta={theta_deg:.2f}deg")
                elif key in (ord('x'), ord('y'), ord('z'), ord('t')):
                    typing_axis = chr(key)
                    typing_buf = ""

            cv2.destroyWindow(DASHBOARD_WIN)
            cv2.destroyWindow(TRACKBAR_WIN)

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if gs_capture is not None:
            gs_capture.release()
        cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()
