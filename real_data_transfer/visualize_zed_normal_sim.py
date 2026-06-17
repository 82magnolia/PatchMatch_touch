"""
GelSight Mini normal map simulator for ZED 2i.

Streams RGB + surface normals from the ZED camera.  On 'c', freezes the frame,
lets the user SAM-segment an object, pick a contact point, and renders an
orthographic projection of the normal map at the GelSight Mini sensor FoV
(18.6 × 14.3 mm).

Controls (main window):
  c — freeze frame and enter capture mode
  q — quit

Capture mode:
  drag (left panel)  — draw bounding box for SAM prompt
  Enter              — run SAM segmentation
  r                  — redraw box / re-segment (discard mask)
  Esc                — cancel, return to live stream

After segmentation:
  left-click (left panel, on object)  — pick contact point, compute projection
  p                                   — re-pick contact point
  r                                   — re-segment
  Esc                                 — close capture window
"""

import sys
import argparse
import numpy as np
import cv2
import torch

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found.  Install with: pip install pyzed")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sys.exit("segment-anything not found.  Install with: pip install segment-anything")

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ── constants ─────────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 640, 360
CAPTURE_W, CAPTURE_H = 1280, 720

GELSIGHT_FOV_W_MM = 18.6
GELSIGHT_FOV_H_MM = 14.3

DEPTH_MODES = {
    "performance": sl.DEPTH_MODE.PERFORMANCE,
    "quality": sl.DEPTH_MODE.QUALITY,
    "ultra": sl.DEPTH_MODE.ULTRA,
    "neural": sl.DEPTH_MODE.NEURAL,
    "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
}


# ── camera ────────────────────────────────────────────────────────────────────

def build_camera(depth_mode_str: str, confidence: int):
    """Open ZED 2i and return (cam, runtime_params, intrinsics_dict)."""
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = DEPTH_MODES[depth_mode_str]
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.1
    init.depth_maximum_distance = 3.0
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Failed to open ZED camera: {status}")

    info = cam.get_camera_information()
    print(f"Found device: {info.camera_model}  (serial {info.serial_number})")

    rt = sl.RuntimeParameters()
    rt.enable_fill_mode = False
    rt.confidence_threshold = confidence

    cal = info.camera_configuration.calibration_parameters.left_cam
    intr = {
        "fx": float(cal.fx),
        "fy": float(cal.fy),
        "cx": float(cal.cx),
        "cy": float(cal.cy),
        "width": int(info.camera_configuration.resolution.width),
        "height": int(info.camera_configuration.resolution.height),
    }
    return cam, rt, intr


# ── image helpers ─────────────────────────────────────────────────────────────

def normals_to_colormap(normals_np: np.ndarray) -> np.ndarray:
    """(H,W,4) float32 → (H,W,3) uint8 BGR.  Invalid (NaN) → black."""
    nxyz = normals_np[:, :, :3]
    valid = np.isfinite(nxyz).all(axis=2)
    rgb = np.zeros((*nxyz.shape[:2], 3), dtype=np.uint8)
    rgb[valid] = ((nxyz[valid] + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    out = color_bgr.copy()
    overlay = out.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def overlay_banner(img: np.ndarray, text: str, alpha: float = 0.55) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.putText(out, text, (20, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# ── inpainting ────────────────────────────────────────────────────────────────

def inpaint_normals(crop: np.ndarray, method: str = "telea") -> np.ndarray:
    """Fill NaN holes in (H,W,4) float32 normal crop. Returns same shape."""
    out = crop.copy()
    invalid = ~np.isfinite(crop[:, :, 0])   # (H,W) True = hole
    if not invalid.any():
        return out

    if method == "nearest":
        if not _SCIPY_OK:
            print("  WARNING: scipy unavailable, falling back to telea.")
            method = "telea"
        else:
            valid = ~invalid
            if not valid.any():
                return out
            nn_idx = _edt(invalid, return_distances=False, return_indices=True)
            for c in range(3):
                ch = out[:, :, c].copy()
                ch[invalid] = crop[nn_idx[0][invalid], nn_idx[1][invalid], c]
                out[:, :, c] = ch
            return out

    # telea / ns: work in uint8 space per channel
    hole_u8 = invalid.astype(np.uint8)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    for c in range(3):
        ch = crop[:, :, c].copy()
        ch[invalid] = 0.0
        ch_u8 = ((ch + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        ch_inp = cv2.inpaint(ch_u8, hole_u8, 3, flag)
        out[:, :, c] = ch_inp.astype(np.float32) / 255.0 * 2.0 - 1.0
    return out


# ── orthographic projection ───────────────────────────────────────────────────

def ortho_project(
    normals_np: np.ndarray,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    px: int,
    py: int,
    intr: dict,
    method: str,
    dpm: float,
) -> "np.ndarray | None":
    """
    Orthographic projection of the normal map and RGB image centred at (px, py).

    Builds a regular (out_h × out_w) grid of physical offsets covering the GelSight
    FoV (18.6 × 14.3 mm) in the sensor plane. Since no ARuCO pose is available here,
    the sensor plane is assumed parallel to the image plane (x_axis=[1,0,0],
    y_axis=[0,1,0]). Each grid point is projected into ZED pixel coordinates to
    sample normals and color directly — no intermediate resize needed.

    Returns (H_out, W_out*2, 3) uint8 BGR image (normals left | RGB right),
    or None if depth or normals are unavailable.
    """
    H, W = normals_np.shape[:2]
    out_w = max(1, round(GELSIGHT_FOV_W_MM * dpm))
    out_h = max(1, round(GELSIGHT_FOV_H_MM * dpm))

    # Depth at pick: median over 9×9 neighbourhood
    r = 4
    d_patch = depth_m[max(0, py - r): py + r + 1, max(0, px - r): px + r + 1]
    valid_d = d_patch[np.isfinite(d_patch) & (d_patch > 0)]
    if len(valid_d) == 0:
        print("  WARNING: no valid depth near picked point — try another location.")
        return None
    Z_m = float(np.median(valid_d))

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    # Contact point in camera frame
    p_contact = np.array([(px - cx) * Z_m / fx, (py - cy) * Z_m / fy, Z_m])

    # Orthographic grid in the sensor plane (parallel to image plane)
    u = np.linspace(-GELSIGHT_FOV_W_MM / 2, GELSIGHT_FOV_W_MM / 2, out_w) * 1e-3
    v = np.linspace(-GELSIGHT_FOV_H_MM / 2, GELSIGHT_FOV_H_MM / 2, out_h) * 1e-3
    uu, vv = np.meshgrid(u, v)  # (out_h, out_w)

    # 3D points: x/y offset in sensor plane, z constant at contact depth
    Px = p_contact[0] + uu
    Py = p_contact[1] + vv

    # Project into ZED pixel coordinates
    spx = np.clip((fx * Px / Z_m + cx).round().astype(int), 0, W - 1)
    spy = np.clip((fy * Py / Z_m + cy).round().astype(int), 0, H - 1)

    # Sample ZED data at projected positions
    normals_crop = normals_np[spy, spx].copy()  # (out_h, out_w, 4)
    color_crop = color_bgr[spy, spx].copy()     # (out_h, out_w, 3)
    mask_crop = mask[spy, spx].copy()           # (out_h, out_w)

    normals_crop[mask_crop == 0] = np.nan
    if not np.isfinite(normals_crop[:, :, 0]).any():
        print("  WARNING: no valid normals in projection area — try another location.")
        return None

    normals_filled = inpaint_normals(normals_crop, method)

    normal_bgr = normals_to_colormap(normals_filled)
    normal_bgr[mask_crop == 0] = 0
    color_crop[mask_crop == 0] = 0

    ann = (f"GelSight FoV: {GELSIGHT_FOV_W_MM}x{GELSIGHT_FOV_H_MM} mm  "
           f"depth: {Z_m * 100:.1f} cm  [{method}]")
    cv2.putText(normal_bgr, ann, (6, out_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(color_crop, "RGB", (6, out_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return np.hstack([normal_bgr, color_crop])


# ── capture flow ──────────────────────────────────────────────────────────────

def run_capture_flow(
    color_bgr: np.ndarray,
    normals_np: np.ndarray,
    depth_m: np.ndarray,
    predictor: "SamPredictor",
    intr: dict,
    inpaint_method: str,
    ortho_dpm: float,
) -> None:
    """Interactive capture window: segment → pick → orthographic projection."""
    WIN = "Capture: drag=box  Enter=SAM  left-click=pick  r=redo  Esc=cancel"
    ORTHO_WIN = "GelSight Sim: Normals (left) | RGB (right)"

    # ── shared mutable state ─────────────────────────────────────────────────
    phase = ["box"]       # "box" | "pick" | "done"
    box_pts = [None]      # [x1,y1,x2,y2] in full-res
    drag_start = [None]   # display-res start of drag
    mask = [None]            # (H,W) uint8 SAM mask
    pick_xy = [None]         # (px, py) full-res
    ortho_img = [None]       # (H,W,3) uint8 BGR
    need_compute = [False]   # trigger projection computation from main loop

    def to_full(xd, yd):
        return (int(xd * CAPTURE_W / DISPLAY_W), int(yd * CAPTURE_H / DISPLAY_H))

    def to_disp(xf, yf):
        return (int(xf * DISPLAY_W / CAPTURE_W), int(yf * DISPLAY_H / CAPTURE_H))

    def on_mouse(event, x, y, flags, param):
        x = min(max(x, 0), 2 * DISPLAY_W - 1)
        y = min(max(y, 0), DISPLAY_H - 1)

        if phase[0] == "box":
            if x >= DISPLAY_W:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
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

        elif phase[0] == "pick":
            if x >= DISPLAY_W or event != cv2.EVENT_LBUTTONDOWN:
                return
            px, py = to_full(x, y)
            px = min(max(px, 0), CAPTURE_W - 1)
            py = min(max(py, 0), CAPTURE_H - 1)
            if mask[0] is not None and mask[0][py, px] == 0:
                print("  Picked point is outside object mask — try again.")
                return
            pick_xy[0] = (px, py)
            need_compute[0] = True

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    def _reset_to_box():
        mask[0] = None
        box_pts[0] = None
        drag_start[0] = None
        pick_xy[0] = None
        ortho_img[0] = None
        need_compute[0] = False
        phase[0] = "box"
        try:
            cv2.destroyWindow(ORTHO_WIN)
        except Exception:
            pass

    try:
        while True:
            # ── build left / right display panels ────────────────────────────
            if mask[0] is None:
                color_panel = cv2.resize(color_bgr, (DISPLAY_W, DISPLAY_H))
                normal_panel = cv2.resize(normals_to_colormap(normals_np),
                                          (DISPLAY_W, DISPLAY_H))
            else:
                c_masked = color_bgr.copy()
                c_masked[mask[0] == 0] = 0
                n_masked = normals_np.copy()
                n_masked[mask[0] == 0] = np.nan
                color_panel = cv2.resize(c_masked, (DISPLAY_W, DISPLAY_H))
                normal_panel = cv2.resize(normals_to_colormap(n_masked),
                                          (DISPLAY_W, DISPLAY_H))

            display = np.hstack([color_panel, normal_panel])

            # ── phase overlays ────────────────────────────────────────────────
            if phase[0] == "box":
                if box_pts[0] is not None:
                    b = box_pts[0]
                    dx1 = int(b[0] * DISPLAY_W / CAPTURE_W)
                    dy1 = int(b[1] * DISPLAY_H / CAPTURE_H)
                    dx2 = int(b[2] * DISPLAY_W / CAPTURE_W)
                    dy2 = int(b[3] * DISPLAY_H / CAPTURE_H)
                    cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    if drag_start[0] is None:
                        cv2.putText(display,
                                    "Press Enter to run SAM  |  r=redraw  Esc=cancel",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(display,
                                "Drag left panel to draw bounding box  |  Esc=cancel",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)

            elif phase[0] == "pick":
                cv2.putText(display,
                            "Left-click on object to pick contact point"
                            "  |  r=re-segment  Esc=cancel",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 0), 1)

            elif phase[0] == "done" and pick_xy[0] is not None:
                dx, dy = to_disp(*pick_xy[0])
                cv2.circle(display, (dx, dy), 5, (0, 0, 255), -1)
                cv2.circle(display, (dx + DISPLAY_W, dy), 5, (0, 0, 255), -1)
                cv2.putText(display,
                            "p=re-pick  r=re-segment  Esc=close",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)
                if ortho_img[0] is not None:
                    cv2.imshow(ORTHO_WIN, ortho_img[0])

            cv2.imshow(WIN, display)
            key = cv2.waitKey(20) & 0xFF

            # ── orthographic computation (deferred from mouse callback) ───────
            if need_compute[0] and pick_xy[0] is not None:
                need_compute[0] = False
                banner = overlay_banner(display,
                                        "Computing orthographic projection...")
                cv2.imshow(WIN, banner)
                cv2.waitKey(1)

                px, py = pick_xy[0]
                result = ortho_project(
                    normals_np, color_bgr, depth_m, mask[0],
                    px, py, intr, inpaint_method, ortho_dpm,
                )
                if result is not None:
                    ortho_img[0] = result
                    phase[0] = "done"
                else:
                    pick_xy[0] = None
                    print("  No valid depth — pick another point.")

            # ── key handling ─────────────────────────────────────────────────
            if key in (27, ord('q')):           # Esc / q → close capture
                break
            elif key == ord('r'):               # re-segment from scratch
                _reset_to_box()
            elif key == ord('p') and phase[0] == "done":   # re-pick
                pick_xy[0] = None
                ortho_img[0] = None
                need_compute[0] = False
                phase[0] = "pick"
                try:
                    cv2.destroyWindow(ORTHO_WIN)
                except Exception:
                    pass
            elif (key == 13 and  # Enter → run SAM
                  phase[0] == "box" and
                  box_pts[0] is not None and
                  drag_start[0] is None):
                banner = overlay_banner(display, "Running SAM... please wait")
                cv2.imshow(WIN, banner)
                cv2.waitKey(1)

                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    predictor.set_image(color_rgb)
                    masks_out, _, _ = predictor.predict(
                        box=np.array(box_pts[0], dtype=np.float32),
                        multimask_output=False,
                    )
                mask[0] = masks_out[0].astype(np.uint8) * 255
                phase[0] = "pick"

    finally:
        for w in (WIN, ORTHO_WIN):
            try:
                cv2.destroyWindow(w)
            except Exception:
                pass


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate GelSight Mini normal map capture from ZED 2i"
    )
    p.add_argument("--depth_mode", choices=list(DEPTH_MODES.keys()),
                   default="neural_plus",
                   help="ZED depth estimation mode (default: neural_plus)")
    p.add_argument("--confidence", type=int, default=95,
                   help="ZED depth confidence threshold 0-100 (default: 95; "
                        "lower accepts noisier pixels)")
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth",
                   help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam_model_type", default="vit_b",
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--inpaint_method", default="telea",
                   choices=["telea", "ns", "nearest"],
                   help="Normal-map hole inpainting method (default: telea)")
    p.add_argument("--ortho_dpm", type=float, default=20.0,
                   help="Output resolution in dots-per-mm for the orthographic "
                        "projection (default: 20 → 372×286 px)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading SAM ({args.sam_model_type}) from {args.sam_checkpoint} ...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM ready.")

    cam, rt, intr = build_camera(args.depth_mode, args.confidence)
    print(f"Depth mode: {args.depth_mode}  confidence: {args.confidence}  "
          f"inpaint: {args.inpaint_method}  dpm: {args.ortho_dpm}")
    print(f"Intrinsics: fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
          f"cx={intr['cx']:.1f}  cy={intr['cy']:.1f}")

    image_sl = sl.Mat()
    normals_sl = sl.Mat()
    depth_sl = sl.Mat()

    print("Streaming — press 'c' to capture, 'q' to quit.")

    try:
        while True:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(image_sl, sl.VIEW.LEFT)
            cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)
            cam.retrieve_measure(depth_sl, sl.MEASURE.DEPTH)

            color_bgr = image_sl.get_data()[:, :, :3].copy()
            normals_np = normals_sl.get_data().copy()   # (H, W, 4) float32
            depth_raw = depth_sl.get_data().copy()      # (H, W[, 1]) float32

            # Ensure depth is 2-D
            depth_m = depth_raw.squeeze() if depth_raw.ndim == 3 else depth_raw

            # Live display: RGB | normal colormap
            color_disp = cv2.resize(color_bgr, (DISPLAY_W, DISPLAY_H))
            normal_disp = cv2.resize(normals_to_colormap(normals_np),
                                     (DISPLAY_W, DISPLAY_H))
            grid = np.hstack([color_disp, normal_disp])
            cv2.putText(grid,
                        "c=capture  q=quit  |  RGB (left)  Normals (right)",
                        (10, DISPLAY_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("RGB (left)  |  Normals (right)", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_capture_flow(
                    color_bgr, normals_np, depth_m,
                    predictor, intr,
                    args.inpaint_method, args.ortho_dpm,
                )

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
