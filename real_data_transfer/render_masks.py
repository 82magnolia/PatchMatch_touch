"""
Post-hoc contact mask renderer for GelSight captures.

Re-generates {idx}_render_mask.mp4 and {idx}_shadow_render_mask.mp4 from the
raw geometry saved by capture_gelsight.py, without needing the ZED or GelSight
connected.  All thresholds and the number of output frames can be adjusted freely.

Saved files used:
  {idx}_contact_data.npz   -- height_map_0, valid_depth_remap, mask_crop,
                               sensor_z_0, tvec_0
  {idx}_pose_contact.npz   -- rvecs / tvecs for each frame in [cs_idx, ce_idx]
  {idx}_diffs.npz          -- diffs, smooth_diffs (for re-trimming)
  {idx}_shadow.mp4         -- resampled GelSight frames (already trimmed)
  {idx}_meta.json          -- cs_idx, ce_idx, peak_idx, trim_threshold

Usage:
  # Re-render touch 3 with a looser contact threshold
  python real_data_transfer/render_masks.py \\
      --data_dir log/gelsight_captures/session_01 \\
      --touch_idx 3 \\
      --render_mask_thres 0.001

  # Write re-rendered videos to a separate directory
  python real_data_transfer/render_masks.py \\
      --data_dir log/gelsight_captures/session_01 \\
      --output_dir log/remask/session_01 \\
      --touch_idx 3 \\
      --render_mask_thres 0.001

  # Re-trim AND re-render (change peak_ratio then re-threshold)
  python real_data_transfer/render_masks.py \\
      --data_dir log/gelsight_captures/session_01 \\
      --touch_idx 3 \\
      --peak_ratio 0.3 \\
      --render_mask_thres -0.0005

  # Batch: re-render all touches, output alongside originals
  python real_data_transfer/render_masks.py \\
      --data_dir log/gelsight_captures/session_01
"""

import sys
import os
import json
import argparse
import glob
import shutil
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GELSIGHT_H = 240
GELSIGHT_W = 320
VIDEO_FPS  = 5.0


# ── Helpers copied / shared with capture_gelsight ────────────────────────────

def write_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def read_video_frames(path):
    """Load all frames from an mp4 as a list of (H, W, 3) uint8 BGR arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def make_render_mask_video(height_map_0, valid_depth_remap, mask_crop,
                           sensor_z_0, tvec_0, pose_contact_rvecs,
                           pose_contact_tvecs, num_frames,
                           render_mask_thres=0.0):
    """Regenerate per-frame render mask images with an adjustable threshold.

    pressing_depth_i = dot(sensor_z_0, tvec_i - tvec_0)
    contact at frame i: height_map_0 < pressing_depth_i + render_mask_thres

    pose_contact_rvecs / tvecs are (N, 3) float64 arrays; NaN rows = missing pose.
    """
    from scipy.interpolate import RBFInterpolator

    n_contact = len(pose_contact_tvecs)
    raw_x, raw_y = [], []
    for i in range(n_contact):
        tv = pose_contact_tvecs[i]
        if np.isfinite(tv).all():
            d = float(np.dot(sensor_z_0, tv - tvec_0))
            raw_x.append(i / max(n_contact - 1, 1))
            raw_y.append(d)

    eval_x = np.linspace(0, 1, num_frames)
    if len(raw_x) >= 2:
        rbf = RBFInterpolator(
            np.array(raw_x)[:, None], np.array(raw_y),
            kernel="thin_plate_spline", smoothing=0.1)
        depths = rbf(eval_x[:, None])
    else:
        print("  WARNING: fewer than 2 valid poses — using static threshold.")
        depths = np.zeros(num_frames)

    frames_out = []
    for d in depths:
        threshold = float(d) + render_mask_thres
        cm = (height_map_0 < threshold) & valid_depth_remap & (mask_crop > 0)
        vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
        vis[cm] = 255
        frames_out.append(vis)
    return frames_out


def retrim_contact_window(diffs, smooth_diffs, peak_ratio=0.4, n_neighbors=3):
    """Re-compute cs_idx, ce_idx, peak_idx from saved diff arrays.

    Returns (peak_idx, cs_idx, ce_idx, threshold).
    """
    peak_idx = int(np.argmax(smooth_diffs))
    peak_val = float(smooth_diffs[peak_idx])
    if peak_val <= 0:
        return peak_idx, 0, len(diffs) - 1, 0.0

    threshold = peak_val * peak_ratio
    cs_idx = 0
    for i in range(peak_idx):
        if smooth_diffs[i] < threshold:
            right = smooth_diffs[i + 1: i + 1 + n_neighbors]
            if len(right) == n_neighbors and np.all(right > threshold):
                cs_idx = i

    ce_idx = len(diffs) - 1
    for j in range(len(diffs) - 1, peak_idx, -1):
        if smooth_diffs[j] < threshold:
            left = smooth_diffs[max(0, j - n_neighbors): j]
            if len(left) == n_neighbors and np.all(left > threshold):
                ce_idx = j

    return peak_idx, cs_idx, ce_idx, threshold


# ── Per-touch rendering ───────────────────────────────────────────────────────

def render_touch(data_dir, touch_idx, render_mask_thres, peak_ratio, num_frames,
                 output_dir=None, dry_run=False):
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    in_prefix  = os.path.join(data_dir,   str(touch_idx))
    out_prefix = os.path.join(output_dir, str(touch_idx))

    contact_data_path  = f"{in_prefix}_contact_data.npz"
    pose_contact_path  = f"{in_prefix}_pose_contact.npz"
    diffs_path         = f"{in_prefix}_diffs.npz"
    meta_path          = f"{in_prefix}_meta.json"
    shadow_path        = f"{in_prefix}_shadow.mp4"

    for p in [contact_data_path, pose_contact_path, shadow_path]:
        if not os.path.exists(p):
            print(f"  [touch {touch_idx}] Missing {p} — skipping.")
            return False

    # Load geometry
    cd = np.load(contact_data_path)
    height_map_0      = cd["height_map_0"]
    valid_depth_remap = cd["valid_depth_remap"].astype(bool)
    mask_crop         = cd["mask_crop"]
    sensor_z_0        = cd["sensor_z_0"]
    tvec_0            = cd["tvec_0"]

    # Load pose sequence
    pc = np.load(pose_contact_path)
    pc_rvecs = pc["rvecs"]
    pc_tvecs = pc["tvecs"]

    # Load meta for original trim indices (used when peak_ratio is unchanged)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    orig_cs = meta.get("cs_idx", 0)
    orig_ce = meta.get("ce_idx", len(pc_tvecs) - 1)

    # Optionally re-trim using saved diff curve
    if peak_ratio is not None and os.path.exists(diffs_path):
        dd = np.load(diffs_path)
        _, cs_idx, ce_idx, threshold = retrim_contact_window(
            dd["diffs"], dd["smooth_diffs"], peak_ratio=peak_ratio)
        # Remap to pose_contact indices (pose_contact starts at orig_cs in the full buffer)
        cs_rel = max(0, cs_idx - orig_cs)
        ce_rel = min(len(pc_tvecs) - 1, ce_idx - orig_cs)
        pose_rvecs = pc_rvecs[cs_rel: ce_rel + 1]
        pose_tvecs = pc_tvecs[cs_rel: ce_rel + 1]
        print(f"  [touch {touch_idx}] Re-trimmed: cs={cs_idx}  ce={ce_idx}  "
              f"threshold={threshold:.5f}")
    else:
        pose_rvecs = pc_rvecs
        pose_tvecs = pc_tvecs

    # Load resampled shadow frames for the side-by-side video
    shadow_frames = read_video_frames(shadow_path)
    if num_frames is None:
        num_frames = meta.get("n_resampled_frames", len(shadow_frames))

    if dry_run:
        print(f"  [touch {touch_idx}] dry-run OK  "
              f"(contact frames={len(pose_tvecs)}  out_frames={num_frames}  "
              f"thres={render_mask_thres:.6f})")
        return True

    rm_frames = make_render_mask_video(
        height_map_0, valid_depth_remap, mask_crop,
        sensor_z_0, tvec_0,
        pose_rvecs, pose_tvecs,
        num_frames=num_frames,
        render_mask_thres=render_mask_thres)

    write_video(f"{out_prefix}_render_mask.mp4", rm_frames, VIDEO_FPS)
    if shadow_frames:
        n = min(len(shadow_frames), len(rm_frames))
        sbs = [np.hstack([shadow_frames[i], rm_frames[i]]) for i in range(n)]
        write_video(f"{out_prefix}_shadow_render_mask.mp4", sbs, VIDEO_FPS)

    # Copy shadow.mp4 to output_dir when it differs from data_dir so the
    # output directory is self-contained (shadow + render_mask + side-by-side).
    out_shadow = f"{out_prefix}_shadow.mp4"
    if os.path.abspath(output_dir) != os.path.abspath(data_dir):
        shutil.copy2(shadow_path, out_shadow)

    print(f"  [touch {touch_idx}] Saved {out_prefix}_render_mask.mp4  "
          f"({len(rm_frames)} frames  thres={render_mask_thres:.6f})")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Re-render contact mask videos from saved capture_gelsight data.")
    p.add_argument("--data_dir", default="log/gelsight_captures",
                   help="Directory written by capture_gelsight.py containing the raw capture data "
                        "(default: log/gelsight_captures)")
    p.add_argument("--output_dir", default=None,
                   help="Directory to write re-rendered videos. Defaults to --data_dir.")
    p.add_argument("--touch_idx", type=int, default=None,
                   help="Touch index to re-render. Omit to process all touches in save_dir.")
    p.add_argument("--render_mask_thres", type=float, default=0.0,
                   help="Height threshold in metres for contact detection "
                        "(negative = tighter, positive = looser; default: 0.0)")
    p.add_argument("--peak_ratio", type=float, default=None,
                   help="Re-trim the contact window using this peak_ratio on the saved "
                        "diff curve before rendering. Omit to use the original trim.")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Number of output frames (default: same as original capture).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be done without writing any files.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.touch_idx is not None:
        indices = [args.touch_idx]
    else:
        # Auto-discover all touches by finding *_contact_data.npz files
        pattern = os.path.join(args.data_dir, "*_contact_data.npz")
        found = sorted(glob.glob(pattern))
        indices = []
        for p in found:
            name = os.path.basename(p).replace("_contact_data.npz", "")
            try:
                indices.append(int(name))
            except ValueError:
                pass
        indices.sort()
        if not indices:
            sys.exit(f"No *_contact_data.npz files found in {args.data_dir}")
        print(f"Found {len(indices)} touch(es): {indices}")

    ok = 0
    for idx in indices:
        if render_touch(args.data_dir, idx,
                        render_mask_thres=args.render_mask_thres,
                        peak_ratio=args.peak_ratio,
                        num_frames=args.num_frames,
                        output_dir=args.output_dir,
                        dry_run=args.dry_run):
            ok += 1

    print(f"\nDone. {ok}/{len(indices)} touch(es) processed.")


if __name__ == "__main__":
    main()
