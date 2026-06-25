"""
Offline re-processor for single-shot GelSight sessions.

Re-runs segmentation and/or ortho projection on a saved session from
capture_gelsight_single_shot.py with different parameters — no ZED or
GelSight hardware required.

Files read from --session_dir:
  session_gs.mp4          full GelSight video
  session_poses.npz       rvecs (N,3) + tvecs (N,3), NaN where marker absent
  session_diffs.npz       diff-from-blank per frame (avoids video re-decode)
  blank_frame.jpg         reference blank frame
  object_cache_N.npz      one per turntable view (color, normals, depth, mask)
  session_views.json      view boundary list
  intrinsics.json         camera fx/fy/cx/cy

Usage:
  # Re-segment with a looser threshold
  python real_data_transfer/process_single_shot.py \\
      --session_dir log/gelsight_captures/session_01 \\
      --seg_threshold 0.10

  # Re-render with soft masks and a different height threshold
  python real_data_transfer/process_single_shot.py \\
      --session_dir log/gelsight_captures/session_01 \\
      --render_mask_type soft --mask_temperature 0.0005 \\
      --render_mask_thres -0.006

  # Write outputs to a separate directory
  python real_data_transfer/process_single_shot.py \\
      --session_dir log/gelsight_captures/session_01 \\
      --output_dir log/reprocess/session_01
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gelsight_processing import (
    _make_Rz, _rotate_rvec_z,
    ortho_project_raw,
    write_video, read_video_frames,
    make_render_mask_video,
    trim_and_resample, segment_contacts,
    GELSIGHT_W, GELSIGHT_H, VIDEO_FPS, MASK_OPEN_PX,
    RENDER_MASK_THRES_M, _format_scale,
)


# ── Per-segment save (mirrors capture_gelsight_single_shot._save_segment) ────

def _save_segment(touch_idx, gs_frames_seg, pose_buffer_seg, blank_frame,
                  normals_cached, color_bgr_cached, mask_cached, depth_cached,
                  intr, inpaint_method, render_scale_list,
                  seg_cs_abs, seg_ce_abs, seg_peak_abs,
                  num_frames, render_mask_type, mask_temperature,
                  render_mask_thres, output_dir, view_idx):
    """Process one detected segment and write all outputs.

    gs_frames_seg: frames[seg_cs_abs : seg_ce_abs+1]
    pose_buffer_seg: matching slice of (rvec, tvec) tuples (None where missing)
    seg_*_abs: absolute frame indices within the full session
    Returns True on success.
    """
    contact_morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * MASK_OPEN_PX + 1, 2 * MASK_OPEN_PX + 1))
    R_z = _make_Rz(90)

    # Peak pose
    rel_peak = seg_peak_abs - seg_cs_abs
    rel_peak = max(0, min(rel_peak, len(pose_buffer_seg) - 1))
    peak_rvec, peak_tvec = pose_buffer_seg[rel_peak]
    if peak_rvec is None:
        for k in range(rel_peak, -1, -1):
            if pose_buffer_seg[k][0] is not None:
                peak_rvec, peak_tvec = pose_buffer_seg[k]
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
        print(f"  [Touch #{touch_idx}] Ortho projection failed — skipping.")
        return False
    normal_bgr_out, raw_norm_out, color_out, height_vis_out, rcm_out = res[:5]

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

    # Contact-start pose
    cs_rvec, cs_tvec = pose_buffer_seg[0]
    if cs_rvec is None:
        cs_rvec, cs_tvec = peak_rvec, peak_tvec

    cs_res = ortho_project_raw(
        normals_cached, color_bgr_cached, mask_cached,
        depth_cached, intr, inpaint_method,
        rvec=_rotate_rvec_z(cs_rvec, R_z), tvec=cs_tvec)

    hmap_0 = sz_0 = vdr_0 = mc_0 = None
    if cs_res is not None:
        hmap_0, sz_0, vdr_0, mc_0 = cs_res[5], cs_res[6], cs_res[7], cs_res[8]
        rm_frames = make_render_mask_video(
            hmap_0, vdr_0, mc_0, sz_0, cs_tvec,
            pose_buffer_seg, num_frames,
            render_mask_thres=render_mask_thres,
            render_mask_type=render_mask_type,
            mask_temperature=mask_temperature)
    else:
        rm_vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
        rm_vis[rcm_out] = 255
        rm_frames = [rm_vis] * num_frames

    resampled, peak_rel, cs_rel, ce_rel, diffs_seg, smooth_seg, trim_thres = \
        trim_and_resample(gs_frames_seg, blank_frame, num_frames)
    if resampled is None:
        idx = np.linspace(0, len(gs_frames_seg) - 1, num_frames).round().astype(int)
        resampled = [gs_frames_seg[i] for i in idx]

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, str(touch_idx))

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

    n_contact = len(pose_buffer_seg)
    pc_rvecs = np.full((n_contact, 3), np.nan, dtype=np.float64)
    pc_tvecs = np.full((n_contact, 3), np.nan, dtype=np.float64)
    for _i, (_rv, _tv) in enumerate(pose_buffer_seg):
        if _rv is not None:
            pc_rvecs[_i] = _rv
            pc_tvecs[_i] = _tv
    np.savez_compressed(f"{prefix}_pose_contact.npz",
                        rvecs=pc_rvecs, tvecs=pc_tvecs)
    np.savez_compressed(f"{prefix}_diffs.npz",
                        diffs=diffs_seg.astype(np.float32),
                        smooth_diffs=smooth_seg.astype(np.float32))

    with open(f"{prefix}_meta.json", "w") as f:
        json.dump({
            "touch_idx": touch_idx,
            "px": None, "py": None,
            "rvec": peak_rvec.tolist(),
            "tvec": peak_tvec.tolist(),
            "n_raw_frames": len(gs_frames_seg),
            "n_resampled_frames": len(resampled),
            "render_scale": render_scale_list,
            "cs_idx": int(cs_rel),
            "ce_idx": int(ce_rel),
            "peak_idx": int(peak_rel),
            "trim_threshold": float(trim_thres),
        }, f, indent=2)

    with open(f"{prefix}_seg_meta.json", "w") as f:
        json.dump({
            "cs_idx_in_session": int(seg_cs_abs),
            "ce_idx_in_session": int(seg_ce_abs),
            "peak_idx_in_session": int(seg_peak_abs),
            "view_idx": int(view_idx),
        }, f, indent=2)

    return True


# ── Main processing ───────────────────────────────────────────────────────────

def reprocess(session_dir, output_dir, seg_threshold, min_gap_frames,
              peak_ratio, num_frames, render_mask_type, mask_temperature,
              render_mask_thres, inpaint_method, render_scale_list,
              dry_run, smooth_sigma=2.0, merge_gap=0, boundary_pad=0):
    """Load saved session and re-process with the given parameters."""
    from scipy.ndimage import gaussian_filter1d

    # ── Load session files ──────────────────────────────────────────────────
    diffs_path = os.path.join(session_dir, "session_diffs.npz")
    poses_path = os.path.join(session_dir, "session_poses.npz")
    gs_mp4_path = os.path.join(session_dir, "session_gs.mp4")
    blank_path = os.path.join(session_dir, "blank_frame.jpg")
    views_path = os.path.join(session_dir, "session_views.json")
    intr_path = os.path.join(session_dir, "intrinsics.json")

    for p in [diffs_path, poses_path, gs_mp4_path, blank_path, intr_path]:
        if not os.path.exists(p):
            sys.exit(f"Missing required file: {p}")

    diffs_arr = np.load(diffs_path)["diffs"].astype(np.float32)
    poses_npz = np.load(poses_path)
    rvecs_arr = poses_npz["rvecs"]  # (N, 3)
    tvecs_arr = poses_npz["tvecs"]  # (N, 3)
    blank_frame = cv2.imread(blank_path)

    with open(intr_path) as f:
        intr = json.load(f)

    # Reconstruct pose_buffer as list of (rvec|None, tvec|None)
    pose_buffer = []
    for i in range(len(rvecs_arr)):
        rv = rvecs_arr[i]
        tv = tvecs_arr[i]
        if np.isfinite(rv).all() and np.isfinite(tv).all():
            pose_buffer.append((rv, tv))
        else:
            pose_buffer.append((None, None))

    # View boundaries + object caches
    view_boundaries = [{"view_idx": 0, "gs_frame_start": 0}]
    if os.path.exists(views_path):
        with open(views_path) as f:
            view_boundaries = json.load(f)

    object_caches = {}
    for vb in view_boundaries:
        vidx = vb["view_idx"]
        cache_path = os.path.join(session_dir, f"object_cache_{vidx}.npz")
        if not os.path.exists(cache_path):
            sys.exit(f"Missing object cache: {cache_path}")
        cd = np.load(cache_path)
        object_caches[vidx] = {
            "normals": cd["normals"],
            "color": cd["color"],
            "mask": cd["mask"],
            "depth": cd["depth"],
        }

    def _view_for(frame_idx):
        view_idx = 0
        for vb in view_boundaries:
            if vb["gs_frame_start"] <= frame_idx:
                view_idx = vb["view_idx"]
        return view_idx

    # Load session meta to recover the calibrated seg_threshold when the
    # caller did not supply an explicit override.
    meta_path = os.path.join(session_dir, "session_meta.json")
    if seg_threshold is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            session_meta = json.load(f)
        seg_threshold = session_meta.get("seg_threshold")
        if seg_threshold is not None:
            print(f"  Using seg_threshold={seg_threshold:.4f} from session_meta.json")
    if seg_threshold is None:
        sys.exit("seg_threshold could not be determined. "
                 "Pass --seg_threshold explicitly or re-run capture to generate session_meta.json.")

    # ── Segment ────────────────────────────────────────────────────────────
    smooth_diffs = gaussian_filter1d(diffs_arr, sigma=smooth_sigma)
    segments = segment_contacts(smooth_diffs, seg_threshold,
                                min_gap_frames=min_gap_frames,
                                peak_ratio=peak_ratio,
                                merge_gap=merge_gap,
                                boundary_pad=boundary_pad)
    n_seg = len(segments)
    print(f"Found {n_seg} contact segment(s) in {len(diffs_arr)} frames.")

    if dry_run:
        for i, (cs, ce, peak, thr) in enumerate(segments):
            vid = _view_for(peak)
            print(f"  [Touch #{i}] cs={cs}  ce={ce}  peak={peak}  "
                  f"frames={ce-cs+1}  view={vid}")
        print("Dry run complete — no files written.")
        return

    # ── Load video frames (only if we have segments) ─────────────────────
    if n_seg == 0:
        print("No segments — nothing to save.")
        return

    print(f"Loading {gs_mp4_path} ...")
    gs_frames = read_video_frames(gs_mp4_path)
    n_vid = len(gs_frames)
    n_diff = len(diffs_arr)
    if n_vid != n_diff:
        print(f"  WARNING: video has {n_vid} frames but diffs has {n_diff} — "
              f"using min({n_vid}, {n_diff}).")

    if num_frames is None:
        num_frames = 50

    # ── Process segments ──────────────────────────────────────────────────
    saved = 0
    for touch_idx, (cs, ce, peak, trim_thr) in enumerate(segments):
        cs = min(cs, len(gs_frames) - 1)
        ce = min(ce, len(gs_frames) - 1)
        peak = min(peak, len(gs_frames) - 1)
        vid = _view_for(peak)
        cache = object_caches[vid]

        seg_frames = gs_frames[cs: ce + 1]
        seg_poses = pose_buffer[cs: ce + 1]

        print(f"  [Touch #{touch_idx}] cs={cs}  ce={ce}  peak={peak}  "
              f"frames={len(seg_frames)}  view={vid}", end="  ")

        ok = _save_segment(
            touch_idx, seg_frames, seg_poses, blank_frame,
            cache["normals"], cache["color"], cache["mask"], cache["depth"],
            intr, inpaint_method, render_scale_list,
            cs, ce, peak,
            num_frames, render_mask_type, mask_temperature,
            render_mask_thres, output_dir, view_idx=vid)
        if ok:
            saved += 1
            print("saved.")
        else:
            print("FAILED.")

    print(f"\nDone. {saved}/{n_seg} touch(es) saved to: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Re-process a saved single-shot GelSight session.")
    p.add_argument("--session_dir", default="log/gelsight_captures",
                   help="Directory written by capture_gelsight_single_shot.py")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: session_dir)")
    p.add_argument("--seg_threshold", type=float, default=None,
                   help="Diff-from-blank threshold for segment detection. "
                        "Defaults to the value saved in session_meta.json from capture.")
    p.add_argument("--min_gap_frames", type=int, default=10,
                   help="Min idle gap between segments in frames (default: 10)")
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
    p.add_argument("--num_frames", type=int, default=None,
                   help="Output frames per contact event (default: 50)")
    p.add_argument("--render_mask_thres", type=float, default=RENDER_MASK_THRES_M,
                   help=f"Height threshold for render mask (default: {RENDER_MASK_THRES_M})")
    p.add_argument("--render_mask_type", choices=["hard", "soft"], default="hard")
    p.add_argument("--mask_temperature", type=float, default=0.002)
    p.add_argument("--inpaint_method", default="telea",
                   choices=["telea", "ns", "nearest"])
    p.add_argument("--render_scale", type=float, nargs="+", default=[1.0])
    p.add_argument("--dry_run", action="store_true",
                   help="Print detected segments without writing files.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.session_dir
    reprocess(
        session_dir=args.session_dir,
        output_dir=output_dir,
        seg_threshold=args.seg_threshold,
        min_gap_frames=args.min_gap_frames,
        peak_ratio=args.peak_ratio,
        num_frames=args.num_frames,
        render_mask_type=args.render_mask_type,
        mask_temperature=args.mask_temperature,
        render_mask_thres=args.render_mask_thres,
        inpaint_method=args.inpaint_method,
        render_scale_list=args.render_scale,
        dry_run=args.dry_run,
        merge_gap=args.merge_gap,
        boundary_pad=args.boundary_pad)


if __name__ == "__main__":
    main()
