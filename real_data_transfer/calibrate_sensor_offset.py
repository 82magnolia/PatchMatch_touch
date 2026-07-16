"""
GUI for calibrating the X/Y/Z translation offset between the ARuCO marker
frame and the GelSight sensor's true physical contact point.

Rotation between the marker and the sensor is assumed exact (known from the
holder's CAD model, see gsmini_holder.stl) -- only the marker->contact
translation is calibrated here. In production this is a single hand-measured
Z-only constant (ARUCO_TO_CONTACT_M in _gelsight_processing.py) with no
lateral (X/Y) term, which is a likely source of pose-dependent geometric
error in real captures.

Stage 1 (Scene Capture):
  Live ZED stream of a static scene. Press 'c' to run FoundationStereo once
  and cache normals/color/depth. No SAM segmentation -- the whole frame is
  used, since the contact-mask/render-mask machinery isn't needed here.

Stage 2 (Offset Calibration):
  Move the GelSight sensor (marker attached) around in front of the now-
  static cached scene. Three trackbars (X/Y/Z, mm) control the current
  offset guess. Every frame, the GUI re-renders orthographic normal/RGB
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


def _setup_trackbars(win, init_x_mm, init_y_mm, init_z_mm,
                     xy_range_mm, z_max_mm):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 140)
    x_max_ticks = _mm_to_ticks(xy_range_mm, -xy_range_mm)
    y_max_ticks = _mm_to_ticks(xy_range_mm, -xy_range_mm)
    z_max_ticks = _mm_to_ticks(z_max_mm, 0.0)
    cv2.createTrackbar("X (mm)", win, _mm_to_ticks(init_x_mm, -xy_range_mm),
                       x_max_ticks, lambda v: None)
    cv2.createTrackbar("Y (mm)", win, _mm_to_ticks(init_y_mm, -xy_range_mm),
                       y_max_ticks, lambda v: None)
    cv2.createTrackbar("Z (mm)", win, _mm_to_ticks(init_z_mm, 0.0),
                       z_max_ticks, lambda v: None)


def _read_trackbars(win, xy_range_mm):
    x_mm = _ticks_to_mm(cv2.getTrackbarPos("X (mm)", win), -xy_range_mm)
    y_mm = _ticks_to_mm(cv2.getTrackbarPos("Y (mm)", win), -xy_range_mm)
    z_mm = _ticks_to_mm(cv2.getTrackbarPos("Z (mm)", win), 0.0)
    return x_mm, y_mm, z_mm


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

def _draw_footprint_box(img, scale):
    out = img.copy()
    h, w = out.shape[:2]
    half_w = w / (2.0 * scale)
    half_h = h / (2.0 * scale)
    cx, cy = w / 2.0, h / 2.0
    p1 = (int(round(cx - half_w)), int(round(cy - half_h)))
    p2 = (int(round(cx + half_w)), int(round(cy + half_h)))
    cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate the ARuCO marker -> GelSight contact-point "
                    "X/Y/Z translation offset with a live GUI."
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
    p.add_argument("--init_x_mm", type=float, default=0.0)
    p.add_argument("--init_y_mm", type=float, default=0.0)
    p.add_argument("--init_z_mm", type=float, default=ARUCO_TO_CONTACT_M * 1000.0)
    p.add_argument("--xy_range_mm", type=float, default=30.0,
                   help="X/Y trackbar half-range in mm (default: +-30).")
    p.add_argument("--z_range_mm", type=float, default=100.0,
                   help="Z trackbar max in mm (default: 0-100).")
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
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if any(s <= 0 for s in args.render_scales):
        sys.exit("--render_scales values must be positive")
    if args.geometry_mode != "zed" and not args.fs_model_dir:
        sys.exit("--fs_model_dir is required when --geometry_mode != zed")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

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
            print("  Move the sensor around the scene. Adjust X/Y/Z trackbars "
                  "until the red box / red dot track the true contact point.")
            print("  Keys: r=recapture scene  s=save offset  q=quit")

            _setup_trackbars(TRACKBAR_WIN, args.init_x_mm, args.init_y_mm,
                             args.init_z_mm, args.xy_range_mm, args.z_range_mm)

            recapture = False
            while not recapture:
                if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                    color_live = image_sl.get_data()[:, :, :3].copy()
                else:
                    color_live = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

                x_mm, y_mm, z_mm = _read_trackbars(TRACKBAR_WIN, args.xy_range_mm)
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
                            normal_img = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                            color_img = normal_img.copy()
                        else:
                            normal_img, color_img = res[0], res[2]
                            if scale > 1.0:
                                normal_img = _draw_footprint_box(normal_img, scale)
                                color_img = _draw_footprint_box(color_img, scale)
                        tag = f"{scale:g}"
                        scale_panels.append(_label_panel(normal_img, f"\xd7{tag} normal"))
                        scale_panels.append(_label_panel(color_img, f"\xd7{tag} color"))
                else:
                    cv2.putText(vis, "Marker ID=6 not found",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    for scale in args.render_scales:
                        tag = f"{scale:g}"
                        blank = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
                        scale_panels.append(_label_panel(blank, f"\xd7{tag} normal"))
                        scale_panels.append(_label_panel(blank, f"\xd7{tag} color"))

                zed_disp = cv2.resize(vis, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                cv2.putText(zed_disp, f"offset  X={x_mm:+.1f}mm  Y={y_mm:+.1f}mm  Z={z_mm:.1f}mm",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
                cv2.putText(zed_disp, "r=recapture  s=save  q=quit",
                            (10, ZED_DISPLAY_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                top_row = _hstack_padded([
                    _label_panel(zed_disp, "ZED live (red dot = assumed contact point)"),
                    _label_panel(cached_disp, "Cached scene: color | normals"),
                ])
                mid_panels = []
                if gs_disp is not None:
                    mid_panels.append(_label_panel(gs_disp, "GelSight live"))
                mid_panels.extend(scale_panels)
                mid_row = _hstack_padded(mid_panels)
                dashboard = _vstack_padded([top_row, mid_row])
                cv2.imshow(DASHBOARD_WIN, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                elif key == ord('r'):
                    recapture = True
                elif key == ord('s'):
                    result = {
                        "offset_x_m": x_mm * 1e-3,
                        "offset_y_m": y_mm * 1e-3,
                        "offset_z_m": z_mm * 1e-3,
                    }
                    with open(args.save_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"  Saved offset to {args.save_path}: "
                          f"X={x_mm:.2f}mm  Y={y_mm:.2f}mm  Z={z_mm:.2f}mm")

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
