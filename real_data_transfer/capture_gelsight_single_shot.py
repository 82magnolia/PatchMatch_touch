"""
Single-shot GelSight capture for the PatchMatch tactile transfer pipeline.

Unlike capture_gelsight.py (which requires explicit 'r'/'s' per touch),
this script records the entire GelSight stream + ARuCO poses continuously
from Stage 2 start until 'q' is pressed, then automatically segments the
recording into individual contact events and processes them post-hoc.

Stage 1 (Object Selection):
  Live ZED stream. Press 'c' to freeze, SAM-segment object, confirm with 'y'.
  Captures GelSight blank (no-contact) frame on confirmation.

Stage 2 (Continuous Recording):
  Records GelSight + ARuCO poses. Press 't' to re-capture object at new turntable
  angle, 'q' to stop and process. Live dashboard shows ZED tracking, GelSight
  live feed, and a rolling diff-from-blank waveform.

Post-hoc processing:
  Segments the recording, trims each contact event, runs ortho projection,
  and saves outputs in the same format as capture_gelsight.py.
  Re-process with different parameters using process_single_shot.py.
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gelsight_processing import (
    _make_Rz, _rotate_rvec_z,
    ortho_project_raw, normals_to_colormap,
    write_video, make_render_mask_video, trim_and_resample,
    segment_contacts, _render_diff_plot, _format_scale,
    GELSIGHT_W, GELSIGHT_H, VIDEO_FPS, MASK_OPEN_PX,
    RENDER_MASK_THRES_M,
)
from capture_gelsight import (
    build_aruco_detector, detect_gelsight_marker, compute_contact_pixel,
    GelSightCapture, read_gelsight_frame,
    run_object_selection,
    _label_panel, _hstack_padded, _vstack_padded,
    GELSIGHT_MARKER_ID,
    ZED_DISPLAY_W, ZED_DISPLAY_H, ZED_W, ZED_H,
)
from visualize_zed_normal_sim import build_camera, overlay_banner, DEPTH_MODES
from visualize_zed_fs import load_model, run_inference, normals_from_xyz

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found.  Install with: pip install pyzed")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sys.exit("segment-anything not found.  Install with: pip install segment-anything")

# ── Constants ──────────────────────────────────────────────────────────────────
PLOT_W        = 960   # pixels wide for rolling diff plot
PLOT_H        = 160   # pixels tall
PLOT_WINDOW   = 800   # how many recent frames to show in rolling plot


# ── Diff-from-blank (single frame) ───────────────────────────────────────────

def _frame_diff(frame_bgr, blank_bgr):
    blank = blank_bgr.astype(np.float32) / 255.0
    return float(np.linalg.norm(
        frame_bgr.astype(np.float32) / 255.0 - blank, axis=-1).mean())


# ── Rolling diff plot ─────────────────────────────────────────────────────────

PLOT_Y_MAX = 0.1  # fixed y-axis ceiling for the live rolling diff plot

def _render_rolling_plot(diffs, seg_threshold, width=PLOT_W, height=PLOT_H,
                         window=PLOT_WINDOW, y_max=PLOT_Y_MAX,
                         min_gap_frames=10, peak_ratio=0.2, merge_gap=0,
                         boundary_pad=0):
    """Render a rolling waveform of the last `window` diffs with threshold line and segment boundaries."""
    from scipy.ndimage import gaussian_filter1d
    diffs_show = np.array(diffs[-window:], dtype=np.float32)
    n = len(diffs_show)
    if n < 2:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        ty = int((1.0 - min(seg_threshold, y_max) / y_max) * (height - 1))
        cv2.line(img, (0, ty), (width - 1, ty), (0, 0, 255), 1)
        return img
    # Pad to fixed width on left if shorter; use the first real value to avoid
    # creating an artificial step that segment_contacts would detect as a contact.
    pad_offset = max(0, width - n)
    if n < width:
        pad = np.full(pad_offset, diffs_show[0], dtype=np.float32)
        diffs_plot = np.concatenate([pad, diffs_show])
    else:
        diffs_plot = diffs_show[-width:]
        pad_offset = 0
    smooth_plot = gaussian_filter1d(diffs_plot, sigma=2.0)

    # Draw curves + threshold line (no segment markers yet)
    img = _render_diff_plot(
        diffs_plot, smooth_plot,
        cs_idx=None, ce_idx=None, peak_idx=None,
        threshold=seg_threshold, width=width, height=height, y_max=y_max)

    # Detect and overlay all segment boundaries in the visible window
    if seg_threshold > 0:
        segments = segment_contacts(smooth_plot, seg_threshold,
                                    min_gap_frames=min_gap_frames,
                                    peak_ratio=peak_ratio,
                                    merge_gap=merge_gap,
                                    boundary_pad=boundary_pad)
        n_plot = len(diffs_plot)
        for cs, ce, pk, _ in segments:
            for idx_val, color in [(cs, (0, 255, 0)), (ce, (0, 255, 0)), (pk, (0, 255, 255))]:
                x = int(idx_val / (n_plot - 1) * (width - 1))
                cv2.line(img, (x, 0), (x, height - 1), color, 1)
    return img


# ── Per-segment save ──────────────────────────────────────────────────────────

def _save_segment(touch_idx, gs_buffer, pose_buffer, blank_frame,
                  normals_cached, color_bgr_cached, mask_cached, depth_cached,
                  intr, inpaint_method, render_scale_list,
                  seg_cs, seg_ce, seg_peak, seg_trim_threshold,
                  num_frames, render_mask_type, mask_temperature,
                  save_dir, view_idx):
    """Process one detected segment and save all outputs.

    Returns True on success, False if ortho projection fails.
    """
    contact_morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * MASK_OPEN_PX + 1, 2 * MASK_OPEN_PX + 1))
    R_z = _make_Rz(90)

    # Pose at peak (for static normal/color/height outputs)
    peak_rvec, peak_tvec = pose_buffer[seg_peak]
    if peak_rvec is None:
        # Fall back to last valid pose before peak
        for k in range(seg_peak, -1, -1):
            if pose_buffer[k][0] is not None:
                peak_rvec, peak_tvec = pose_buffer[k]
                break
    if peak_rvec is None:
        print(f"  [Touch #{touch_idx}] No valid ARuCO pose — skipping.")
        return False

    peak_aligned_rvec = _rotate_rvec_z(peak_rvec, R_z)
    res = ortho_project_raw(
        normals_cached, color_bgr_cached, mask_cached,
        depth_cached, intr, inpaint_method,
        rvec=peak_aligned_rvec, tvec=peak_tvec)
    if res is None:
        print(f"  [Touch #{touch_idx}] Ortho projection at peak failed — skipping.")
        return False
    normal_bgr_out, raw_norm_out, color_out, height_vis_out, rcm_out = res[:5]

    # Scaled renders
    scaled_static = {}
    for scale in render_scale_list:
        if np.isclose(scale, 1.0):
            scaled_static[scale] = (normal_bgr_out, raw_norm_out, color_out)
            continue
        sr = ortho_project_raw(
            normals_cached, color_bgr_cached, mask_cached,
            depth_cached, intr, inpaint_method,
            rvec=peak_aligned_rvec, tvec=peak_tvec, render_scale=scale)
        if sr is not None:
            scaled_static[scale] = (sr[0], sr[1], sr[2])

    # Pose at contact start (for render mask video)
    cs_rvec, cs_tvec = pose_buffer[seg_cs]
    if cs_rvec is None:
        cs_rvec, cs_tvec = peak_rvec, peak_tvec

    cs_res = ortho_project_raw(
        normals_cached, color_bgr_cached, mask_cached,
        depth_cached, intr, inpaint_method,
        rvec=_rotate_rvec_z(cs_rvec, R_z), tvec=cs_tvec)

    pose_contact = pose_buffer[seg_cs: seg_ce + 1]
    hmap_0 = sz_0 = vdr_0 = mc_0 = None
    if cs_res is not None:
        hmap_0, sz_0, vdr_0, mc_0 = cs_res[5], cs_res[6], cs_res[7], cs_res[8]
        rm_frames = make_render_mask_video(
            hmap_0, vdr_0, mc_0, sz_0, cs_tvec,
            pose_contact, num_frames,
            render_mask_thres=RENDER_MASK_THRES_M,
            render_mask_type=render_mask_type,
            mask_temperature=mask_temperature)
    else:
        rm_vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
        rm_vis[rcm_out] = 255
        rm_frames = [rm_vis] * num_frames

    # Resample GelSight frames within the segment window
    seg_frames = gs_buffer[seg_cs: seg_ce + 1]
    resampled, peak_rel, cs_rel, ce_rel, diffs_seg, smooth_seg, trim_thres = \
        trim_and_resample(seg_frames, blank_frame, num_frames)
    if resampled is None:
        # fall back: uniform subsample
        idx = np.linspace(0, len(seg_frames) - 1, num_frames).round().astype(int)
        resampled = [seg_frames[i] for i in idx]

    prefix = os.path.join(save_dir, str(touch_idx))
    cv2.imwrite(f"{prefix}_normal.jpg", normal_bgr_out)
    np.savez_compressed(f"{prefix}_normal.npz", normal=raw_norm_out)
    cv2.imwrite(f"{prefix}_color.jpg", color_out)
    for scale, (normal_s, raw_s, color_s) in scaled_static.items():
        tag = _format_scale(scale)
        cv2.imwrite(f"{prefix}_scale{tag}_normal.jpg", normal_s)
        np.savez_compressed(f"{prefix}_scale{tag}_normal.npz", normal=raw_s)
        cv2.imwrite(f"{prefix}_scale{tag}_color.jpg", color_s)
    cv2.imwrite(f"{prefix}_height.jpg", height_vis_out)

    rcm_vis = (rcm_out.astype(np.uint8) * 255)
    rcm_vis = cv2.morphologyEx(rcm_vis, cv2.MORPH_OPEN, contact_morph_kernel)
    cv2.imwrite(f"{prefix}_contact_mask.jpg", rcm_vis)

    write_video(f"{prefix}_shadow.mp4", resampled, VIDEO_FPS)
    write_video(f"{prefix}_render_mask.mp4", rm_frames, VIDEO_FPS)
    sbs = [np.hstack([s, r]) for s, r in zip(resampled, rm_frames)]
    write_video(f"{prefix}_shadow_render_mask.mp4", sbs, VIDEO_FPS)

    if hmap_0 is not None:
        np.savez_compressed(
            f"{prefix}_contact_data.npz",
            height_map_0=hmap_0.astype(np.float32),
            valid_depth_remap=vdr_0,
            mask_crop=mc_0,
            sensor_z_0=sz_0.astype(np.float32),
            tvec_0=np.array(cs_tvec, dtype=np.float64),
            cs_rvec=np.array(cs_rvec, dtype=np.float64),
        )

    n_contact = len(pose_contact)
    pc_rvecs = np.full((n_contact, 3), np.nan, dtype=np.float64)
    pc_tvecs = np.full((n_contact, 3), np.nan, dtype=np.float64)
    for _i, (_rv, _tv) in enumerate(pose_contact):
        if _rv is not None:
            pc_rvecs[_i] = _rv
            pc_tvecs[_i] = _tv
    np.savez_compressed(f"{prefix}_pose_contact.npz",
                        rvecs=pc_rvecs, tvecs=pc_tvecs)

    # diffs for the segment window (for render_masks.py compatibility)
    np.savez_compressed(f"{prefix}_diffs.npz",
                        diffs=diffs_seg.astype(np.float32),
                        smooth_diffs=smooth_seg.astype(np.float32))

    with open(f"{prefix}_meta.json", "w") as f:
        json.dump({
            "touch_idx": touch_idx,
            "px": None,
            "py": None,
            "rvec": peak_rvec.tolist(),
            "tvec": peak_tvec.tolist(),
            "n_raw_frames": len(seg_frames),
            "n_resampled_frames": len(resampled),
            "render_scale": render_scale_list,
            "cs_idx": int(cs_rel),
            "ce_idx": int(ce_rel),
            "peak_idx": int(peak_rel),
            "trim_threshold": float(trim_thres),
        }, f, indent=2)

    with open(f"{prefix}_seg_meta.json", "w") as f:
        json.dump({
            "cs_idx_in_session": int(seg_cs),
            "ce_idx_in_session": int(seg_ce),
            "peak_idx_in_session": int(seg_peak),
            "view_idx": int(view_idx),
            "seg_threshold": float(seg_trim_threshold),
        }, f, indent=2)

    return True


# ── Post-hoc processing ───────────────────────────────────────────────────────

def process_session(gs_buffer, pose_buffer, blank_frame,
                    object_caches, view_boundaries,
                    intr, inpaint_method, render_scale_list,
                    seg_threshold, min_gap_frames,
                    num_frames, render_mask_type, mask_temperature,
                    save_dir, peak_ratio=0.4, smooth_sigma=2.0, merge_gap=0,
                    boundary_pad=0):
    """Segment the full session and save each contact event.

    object_caches: list of (normals, color, mask, depth) dicts, one per view.
    view_boundaries: list of {'view_idx': N, 'gs_frame_start': M}.
    Returns number of successfully saved touches.
    """
    from scipy.ndimage import gaussian_filter1d

    print(f"  Computing diff curve ({len(gs_buffer)} frames)...")
    blank = blank_frame.astype(np.float32) / 255.0
    diffs = np.array([
        float(np.linalg.norm(f.astype(np.float32) / 255.0 - blank, axis=-1).mean())
        for f in gs_buffer
    ])
    smooth_diffs = gaussian_filter1d(diffs, sigma=smooth_sigma)

    segments = segment_contacts(smooth_diffs, seg_threshold,
                                min_gap_frames=min_gap_frames,
                                peak_ratio=peak_ratio,
                                merge_gap=merge_gap,
                                boundary_pad=boundary_pad)
    print(f"  Found {len(segments)} contact segment(s).")

    # Lookup helper: which view covers a given session frame index
    def _view_for(frame_idx):
        view_idx = 0
        for vb in view_boundaries:
            if vb["gs_frame_start"] <= frame_idx:
                view_idx = vb["view_idx"]
        return view_idx

    saved = 0
    for touch_idx, (cs, ce, peak, trim_thr) in enumerate(segments):
        print(f"  [Touch #{touch_idx}] cs={cs}  ce={ce}  peak={peak}  "
              f"frames={ce - cs + 1}", end="  ")
        vid = _view_for(peak)
        cache = object_caches[vid]
        ok = _save_segment(
            touch_idx, gs_buffer, pose_buffer, blank_frame,
            cache["normals"], cache["color"], cache["mask"], cache["depth"],
            intr, inpaint_method, render_scale_list,
            cs, ce, peak, trim_thr,
            num_frames, render_mask_type, mask_temperature,
            save_dir, view_idx=vid)
        if ok:
            saved += 1
            print("saved.")
        else:
            print("FAILED.")
    return saved


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-shot continuous GelSight capture with post-hoc segmentation."
    )
    p.add_argument("--depth_mode", choices=list(DEPTH_MODES.keys()), default="neural_plus")
    p.add_argument("--zed_confidence", type=int, default=95)
    p.add_argument("--sam_checkpoint", default="log/sam_vit_b_01ec64.pth")
    p.add_argument("--sam_model_type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--gelsight_device", type=str, default="0")
    p.add_argument("--marker_size", type=float, default=0.037)
    p.add_argument("--num_frames", type=int, default=50,
                   help="Output frames per contact event (default: 50)")
    p.add_argument("--seg_threshold", type=float, default=None,
                   help="Diff-from-blank threshold for segment detection. "
                        "If omitted, computed automatically from the first "
                        "--calib_frames frames as median + --calib_alpha.")
    p.add_argument("--calib_frames", type=int, default=15,
                   help="Number of no-touch frames at Stage 2 start used to "
                        "auto-compute seg_threshold (default: 30)")
    p.add_argument("--calib_alpha", type=float, default=0.005,
                   help="Margin added to the calibration median to set "
                        "seg_threshold: threshold = median + alpha (default: 0.01)")
    p.add_argument("--min_gap_frames", type=int, default=10,
                   help="Min idle gap in frames between segments (default: 10)")
    p.add_argument("--merge_gap", type=int, default=0,
                   help="Merge consecutive segments whose gap is <= this many frames; "
                        "use >= 1 to collapse multi-peak contacts into one (default: 0)")
    p.add_argument("--boundary_pad", type=int, default=10,
                   help="Extra frames added before cs_idx and after ce_idx of each "
                        "detected segment (default: 0)")
    p.add_argument("--peak_ratio", type=float, default=0.01,
                   help="Contact boundary threshold as a fraction between seg_threshold "
                        "and the peak: threshold = seg_threshold + (peak - seg_threshold) "
                        "* peak_ratio (default: 0.2)")
    p.add_argument("--border_fraction", type=float, default=0.15)
    p.add_argument("--inpaint_method", default="telea",
                   choices=["telea", "ns", "nearest"])
    p.add_argument("--render_scale", type=float, nargs="+", default=[1.0])
    p.add_argument("--render_mask_type", choices=["hard", "soft"], default="hard")
    p.add_argument("--mask_temperature", type=float, default=0.002)
    p.add_argument("--save_dir", default="log/gelsight_captures")
    p.add_argument("--geometry_mode",
                   choices=["zed", "foundation_stereo", "fast_foundation_stereo"],
                   default="zed")
    p.add_argument("--fs_model_dir", default=None)
    p.add_argument("--fs_valid_iters", type=int, default=None)
    p.add_argument("--fs_max_disp", type=int, default=192)
    p.add_argument("--fs_scale", type=float, default=1.0)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if any(s <= 0 for s in args.render_scale):
        sys.exit("--render_scale values must be positive")
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        gs_device = int(args.gelsight_device)
    except ValueError:
        gs_device = args.gelsight_device

    if args.geometry_mode != "zed":
        if not args.fs_model_dir:
            sys.exit("--fs_model_dir is required when --geometry_mode != zed")
        if args.fs_valid_iters is None:
            args.fs_valid_iters = 8 if args.geometry_mode == "fast_foundation_stereo" else 32
        print(f"Loading {args.geometry_mode} from {args.fs_model_dir} ...")
        fs_model = load_model(args.geometry_mode, args.fs_model_dir,
                              args.fs_valid_iters, args.fs_max_disp)
        print("FS model ready.")
    else:
        fs_model = None

    print(f"Loading SAM ({args.sam_model_type}) from {args.sam_checkpoint} ...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM ready.")

    cam, rt, intr = build_camera(args.depth_mode, args.zed_confidence)
    print(f"ZED opened.  fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
          f"cx={intr['cx']:.1f}  cy={intr['cy']:.1f}")

    fs_baseline = None
    image_r_sl = None
    right_bgr = None
    if args.geometry_mode != "zed":
        info = cam.get_camera_information()
        fs_baseline = info.camera_configuration.calibration_parameters.get_camera_baseline()
        image_r_sl = sl.Mat()

    gs_cap = cv2.VideoCapture(gs_device)
    gs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, GELSIGHT_W)
    gs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GELSIGHT_H)
    if not gs_cap.isOpened():
        cam.close()
        sys.exit(f"Cannot open GelSight camera: {gs_device}")
    gs_fps = gs_cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"GelSight {GELSIGHT_W}x{GELSIGHT_H} @ {gs_fps:.1f} fps")

    aruco_detector = build_aruco_detector()
    camera_matrix = np.array([[intr["fx"], 0, intr["cx"]],
                               [0, intr["fy"], intr["cy"]],
                               [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_sl   = sl.Mat()
    normals_sl = sl.Mat()
    depth_sl   = sl.Mat()
    xyz_sl     = sl.Mat()

    # Save intrinsics for offline processing
    with open(os.path.join(args.save_dir, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)

    # ── Stage 1: Object Selection ─────────────────────────────────────────────
    print("\n--- Stage 1: Object Selection ---")
    print("Press 'c' to freeze and segment, 'q' to quit.")

    LIVE_WIN = "ZED: RGB | Normals  (c=capture  q=quit)"
    color_bgr_cached = normals_cached = depth_cached = mask_cached = blank_frame = None
    cache_disp_base = None

    def _run_fs_inference(color_bgr, right_bgr_fs, disp_img, win):
        banner = disp_img.copy()
        cv2.putText(banner, f"Running {args.geometry_mode}... please wait",
                    (10, ZED_DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
        cv2.imshow(win, banner)
        cv2.waitKey(1)
        left_rgb  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr_fs, cv2.COLOR_BGR2RGB)
        _, fs_dl, fs_xyz = run_inference(
            fs_model, args.geometry_mode, left_rgb, right_rgb,
            intr, fs_baseline, args.fs_scale, args.fs_valid_iters,
            z_near=0.1, z_far=5.0)
        if fs_dl.shape != (ZED_H, ZED_W):
            fd = cv2.resize(fs_dl, (ZED_W, ZED_H), interpolation=cv2.INTER_LINEAR)
            fn_lr = normals_from_xyz(fs_xyz)
            fn = cv2.resize(fn_lr, (ZED_W, ZED_H), interpolation=cv2.INTER_LINEAR)
            nrms = np.linalg.norm(fn, axis=2, keepdims=True)
            fn = np.where(nrms > 1e-8, fn / nrms, 0.0).astype(np.float32)
        else:
            fd = fs_dl
            fn = normals_from_xyz(fs_xyz)
        fn[~np.isfinite(fd)] = np.nan
        return fd, fn

    while color_bgr_cached is None:
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

        c_d = cv2.resize(color_bgr, (ZED_DISPLAY_W, ZED_DISPLAY_H))
        n_d = cv2.resize(normals_to_colormap(normals_np), (ZED_DISPLAY_W, ZED_DISPLAY_H))
        grid = np.hstack([c_d, n_d])
        cv2.putText(grid, "c=capture  q=quit", (10, ZED_DISPLAY_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(LIVE_WIN, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cam.close(); gs_cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord('c'):
            fs_depth_full = fs_normals_full = None
            normals_for_sel = normals_np
            if args.geometry_mode != "zed" and right_bgr is not None:
                print(f"  Running {args.geometry_mode}...")
                fs_depth_full, fs_normals_full = _run_fs_inference(
                    color_bgr, right_bgr, grid, LIVE_WIN)
                normals_for_sel = fs_normals_full

            mask = run_object_selection(color_bgr, normals_for_sel, predictor)
            if mask is None:
                print("  Cancelled — try again.")
                continue

            for _ in range(10):
                gs_cap.read()
            gs_blank = read_gelsight_frame(gs_cap, args.border_fraction)
            if gs_blank is None:
                gs_blank = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
            blank_frame = gs_blank
            cv2.imwrite(os.path.join(args.save_dir, "blank_frame.jpg"), blank_frame)

            cam.retrieve_measure(xyz_sl, sl.MEASURE.XYZRGBA)
            xyz_np = xyz_sl.get_data()[:, :, :3].copy()

            if args.geometry_mode != "zed" and fs_depth_full is not None:
                invalid = (mask == 0) | ~np.isfinite(fs_depth_full)
                fs_normals_full[invalid] = np.nan
                fs_depth_full[invalid] = np.nan
                normals_to_cache = fs_normals_full
                depth_to_cache   = fs_depth_full
            else:
                normals_to_cache = normals_np
                depth_to_cache   = depth_m

            # View 0 cache
            np.savez_compressed(
                os.path.join(args.save_dir, "object_cache_0.npz"),
                color=color_bgr, normals=normals_to_cache,
                depth=depth_to_cache, xyz=xyz_np, mask=mask)

            color_bgr_cached  = color_bgr
            normals_cached    = normals_to_cache
            depth_cached      = depth_to_cache
            mask_cached       = mask

            c_m = color_bgr.copy(); c_m[mask == 0] = 0
            n_m = normals_cached.copy(); n_m[mask == 0] = np.nan
            cache_disp_base = np.hstack([c_m, normals_to_colormap(n_m)])
            print("  Cached: blank_frame.jpg + object_cache_0.npz")

    cv2.destroyWindow(LIVE_WIN)

    # ── Stage 2: Continuous Recording ────────────────────────────────────────
    print("\n--- Stage 2: Continuous Recording ---")
    print("  Keys: t=new turntable view  q=stop+process")

    gs_capture = GelSightCapture(gs_cap, args.border_fraction)
    # Wait for the background thread to deliver its first frame before entering
    # the loop — otherwise the very first read() returns None → zero frame →
    # artificially large diff against the blank.
    while gs_capture.read() is None:
        pass
    R_z = _make_Rz(90)

    # Object caches and view tracking
    object_caches = [{
        "normals": normals_cached,
        "color":   color_bgr_cached,
        "mask":    mask_cached,
        "depth":   depth_cached,
    }]
    view_boundaries = [{"view_idx": 0, "gs_frame_start": 0}]
    current_view_idx = 0

    gs_buffer   = []
    pose_buffer = []
    diffs_live  = []
    last_rvec   = last_tvec = None

    seg_threshold = args.seg_threshold  # None → auto-calibrate
    calibrating   = seg_threshold is None

    ortho_mid_panels = []  # updated on every fresh ARuCO detection

    # VideoWriter opened immediately; written frame by frame during Stage 2
    session_mp4_path = os.path.join(args.save_dir, "session_gs.mp4")
    session_writer = cv2.VideoWriter(
        session_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS, (GELSIGHT_W, GELSIGHT_H))

    DASHBOARD_WIN = "Single-Shot Capture  (t=new view  q=stop+process)"

    try:
        while True:
            # ZED grab
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                color_live = image_sl.get_data()[:, :, :3].copy()
            else:
                color_live = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

            # GelSight grab
            gs_frame = gs_capture.read()

            # Buffer + write
            gs_buffer.append(gs_frame.copy())
            session_writer.write(gs_frame)

            d = _frame_diff(gs_frame, blank_frame)
            diffs_live.append(d)

            # Auto-calibrate seg_threshold from the first N no-touch frames
            if calibrating and len(diffs_live) >= args.calib_frames:
                seg_threshold = float(np.median(diffs_live[:args.calib_frames])) \
                    + args.calib_alpha
                calibrating = False
                print(f"  seg_threshold auto-set to {seg_threshold:.4f} "
                      f"(median={seg_threshold - args.calib_alpha:.4f} "
                      f"+ alpha={args.calib_alpha})")

            # ARuCO detection
            rvec, tvec = detect_gelsight_marker(
                color_live, aruco_detector, camera_matrix, dist_coeffs, args.marker_size)
            if rvec is not None:
                last_rvec, last_tvec = rvec, tvec
            pose_buffer.append((
                last_rvec.copy() if last_rvec is not None else None,
                last_tvec.copy() if last_tvec is not None else None,
            ))

            # ZED display + live ortho projection (on every fresh ARuCO detection)
            vis = color_live.copy()
            if rvec is not None:
                cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                                  rvec, tvec, args.marker_size * 0.5)
                px_py = compute_contact_pixel(rvec, tvec, intr)
                if px_py is not None:
                    cv2.circle(vis, px_py, 8, (0, 0, 255), -1)
                aligned_rvec = _rotate_rvec_z(rvec, R_z)
                res = ortho_project_raw(
                    normals_cached, color_bgr_cached, mask_cached,
                    depth_cached, intr, args.inpaint_method,
                    rvec=aligned_rvec, tvec=tvec)
                if res is not None:
                    ortho_mid_panels = [
                        _label_panel(res[0], "normal"),
                        _label_panel(res[2], "color"),
                    ]
                    for scale in args.render_scale:
                        if np.isclose(scale, 1.0):
                            continue
                        sr = ortho_project_raw(
                            normals_cached, color_bgr_cached, mask_cached,
                            depth_cached, intr, args.inpaint_method,
                            rvec=aligned_rvec, tvec=tvec,
                            render_scale=scale)
                        if sr is not None:
                            ortho_mid_panels.append(
                                _label_panel(sr[0], f"\xd7{scale:.2g} normal"))
                            ortho_mid_panels.append(
                                _label_panel(sr[2], f"\xd7{scale:.2g} color"))
            else:
                cv2.putText(vis, "Marker ID=6 not found",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            n_frames = len(gs_buffer)
            marker_ok = "OK" if last_rvec is not None else "---"
            zed_disp = cv2.resize(vis, (ZED_W // 2, ZED_H // 2))
            cv2.putText(zed_disp,
                        f"Recording  {n_frames} frames  |  ARuCO: {marker_ok}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
            cv2.putText(zed_disp, "t=new view  q=stop+process",
                        (10, zed_disp.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # GelSight panel
            gs_disp = _label_panel(gs_frame.copy(), "GelSight live")

            # Rolling diff plot
            thr_display = seg_threshold if seg_threshold is not None else 0.0
            plot_img = _render_rolling_plot(diffs_live, thr_display,
                                            width=PLOT_W, height=PLOT_H,
                                            min_gap_frames=args.min_gap_frames,
                                            peak_ratio=args.peak_ratio,
                                            merge_gap=args.merge_gap,
                                            boundary_pad=args.boundary_pad)
            if calibrating:
                remaining = args.calib_frames - len(diffs_live)
                cv2.putText(plot_img, f"Computing seg_threshold... ({remaining} frames left)",
                            (10, PLOT_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                thr_label = "threshold=calibrating"
            else:
                thr_label = f"threshold={thr_display:.4f}"
            plot_img = _label_panel(plot_img,
                                    f"Diff-from-blank  {thr_label}"
                                    f"  view={current_view_idx}")

            # Cache panel (segmented object)
            cache_disp = None
            if cache_disp_base is not None:
                cache_disp = cache_disp_base.copy()
                if last_rvec is not None:
                    px_py2 = compute_contact_pixel(last_rvec, last_tvec, intr)
                    if px_py2 is not None:
                        cv2.circle(cache_disp, px_py2, 8, (0, 0, 255), -1)
                        cv2.circle(cache_disp, (px_py2[0] + ZED_W, px_py2[1]),
                                   8, (0, 0, 255), -1)

            # Dashboard layout
            top_panels = [_label_panel(zed_disp, "ZED tracking")]
            if cache_disp is not None:
                top_panels.append(_label_panel(
                    cv2.resize(cache_disp, (ZED_W, ZED_H // 2)), "Object cache"))
            top_row = _hstack_padded(top_panels)
            mid_row = _hstack_padded([gs_disp] + ortho_mid_panels)
            dashboard = _vstack_padded([top_row, mid_row, plot_img])
            cv2.imshow(DASHBOARD_WIN, dashboard)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('t'):
                # ── New turntable view ────────────────────────────────────────
                print("  [New view] Rotate turntable, then press 'c'.  Esc=cancel.")
                TRACK_WIN = "New view: c=capture  Esc=cancel"
                cv2.namedWindow(TRACK_WIN)
                right_bgr_t = None
                done_t = False
                try:
                    while not done_t:
                        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                            continue
                        cam.retrieve_image(image_sl, sl.VIEW.LEFT)
                        cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)
                        cam.retrieve_measure(depth_sl, sl.MEASURE.DEPTH)
                        if args.geometry_mode != "zed":
                            cam.retrieve_image(image_r_sl, sl.VIEW.RIGHT)
                            right_bgr_t = image_r_sl.get_data()[:, :, :3].copy()
                        color_t = image_sl.get_data()[:, :, :3].copy()
                        normals_t = normals_sl.get_data().copy()
                        depth_raw_t = depth_sl.get_data().copy()
                        depth_t = depth_raw_t.squeeze() if depth_raw_t.ndim == 3 else depth_raw_t

                        c_d_t = cv2.resize(color_t, (ZED_DISPLAY_W, ZED_DISPLAY_H))
                        n_d_t = cv2.resize(normals_to_colormap(normals_t),
                                           (ZED_DISPLAY_W, ZED_DISPLAY_H))
                        track_grid = np.hstack([c_d_t, n_d_t])
                        cv2.putText(track_grid, "c=capture  Esc=cancel",
                                    (10, ZED_DISPLAY_H - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                        cv2.imshow(TRACK_WIN, track_grid)

                        key_t = cv2.waitKey(1) & 0xFF
                        if key_t == 27:
                            print("  [New view] Cancelled.")
                            break
                        elif key_t == ord('c'):
                            normals_for_t = normals_t
                            fs_d_t = fs_n_t = None
                            if args.geometry_mode != "zed" and right_bgr_t is not None:
                                print(f"  Running {args.geometry_mode}...")
                                fs_d_t, fs_n_t = _run_fs_inference(
                                    color_t, right_bgr_t, track_grid, TRACK_WIN)
                                normals_for_t = fs_n_t

                            new_mask = run_object_selection(color_t, normals_for_t, predictor)
                            if new_mask is None:
                                print("  Cancelled — try again.")
                                continue

                            if args.geometry_mode != "zed" and fs_d_t is not None:
                                inv_t = (new_mask == 0) | ~np.isfinite(fs_d_t)
                                fs_n_t[inv_t] = np.nan
                                fs_d_t[inv_t] = np.nan
                                nc_t = fs_n_t
                                dc_t = fs_d_t
                            else:
                                nc_t = normals_t
                                dc_t = depth_t

                            current_view_idx += 1
                            object_caches.append({
                                "normals": nc_t,
                                "color":   color_t,
                                "mask":    new_mask,
                                "depth":   dc_t,
                            })
                            view_boundaries.append({
                                "view_idx": current_view_idx,
                                "gs_frame_start": len(gs_buffer),
                            })
                            np.savez_compressed(
                                os.path.join(args.save_dir,
                                             f"object_cache_{current_view_idx}.npz"),
                                color=color_t, normals=nc_t,
                                depth=dc_t, mask=new_mask)

                            # Update live cache display
                            c_m = color_t.copy(); c_m[new_mask == 0] = 0
                            n_m = nc_t.copy(); n_m[new_mask == 0] = np.nan
                            cache_disp_base = np.hstack([c_m, normals_to_colormap(n_m)])
                            normals_cached = nc_t
                            color_bgr_cached = color_t
                            mask_cached = new_mask
                            depth_cached = dc_t

                            print(f"  [New view] Cache updated (view {current_view_idx}).")
                            ortho_mid_panels = []
                            done_t = True
                finally:
                    cv2.destroyWindow(TRACK_WIN)

    finally:
        session_writer.release()
        cam.close()
        gs_capture.release()
        cv2.destroyAllWindows()

    n_frames = len(gs_buffer)
    print(f"\nStopped. {n_frames} raw frames captured.")

    # Save raw session data
    print("Saving session data...")
    diffs_arr = np.array(diffs_live, dtype=np.float32)
    rvecs_arr = np.full((n_frames, 3), np.nan, dtype=np.float64)
    tvecs_arr = np.full((n_frames, 3), np.nan, dtype=np.float64)
    for i, (rv, tv) in enumerate(pose_buffer):
        if rv is not None:
            rvecs_arr[i] = rv
            tvecs_arr[i] = tv
    np.savez_compressed(os.path.join(args.save_dir, "session_poses.npz"),
                        rvecs=rvecs_arr, tvecs=tvecs_arr)
    np.savez_compressed(os.path.join(args.save_dir, "session_diffs.npz"),
                        diffs=diffs_arr)
    with open(os.path.join(args.save_dir, "session_views.json"), "w") as f:
        json.dump(view_boundaries, f, indent=2)
    with open(os.path.join(args.save_dir, "session_meta.json"), "w") as f:
        json.dump({
            "seg_threshold": float(seg_threshold) if seg_threshold is not None else None,
            "calib_alpha": args.calib_alpha,
            "calib_frames": args.calib_frames,
            "min_gap_frames": args.min_gap_frames,
            "peak_ratio": args.peak_ratio,
            "merge_gap": args.merge_gap,
            "boundary_pad": args.boundary_pad,
        }, f, indent=2)
    print(f"  session_gs.mp4  session_poses.npz  session_diffs.npz  "
          f"session_views.json  ({current_view_idx + 1} view(s))")

    # Post-hoc processing
    print("\nPost-hoc processing...")
    saved = process_session(
        gs_buffer, pose_buffer, blank_frame,
        object_caches, view_boundaries,
        intr, args.inpaint_method, args.render_scale,
        seg_threshold=seg_threshold if seg_threshold is not None else args.calib_alpha,
        min_gap_frames=args.min_gap_frames,
        num_frames=args.num_frames,
        render_mask_type=args.render_mask_type,
        mask_temperature=args.mask_temperature,
        save_dir=args.save_dir,
        peak_ratio=args.peak_ratio,
        merge_gap=args.merge_gap,
        boundary_pad=args.boundary_pad)

    print(f"\nDone. {saved} touch(es) saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
