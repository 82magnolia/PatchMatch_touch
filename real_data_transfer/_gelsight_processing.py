"""
Pure-NumPy / cv2 / scipy processing helpers shared between
capture_gelsight_single_shot.py and process_single_shot.py.
No hardware dependencies (pyzed, segment-anything, torch).
"""

import numpy as np
import cv2

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────
GELSIGHT_W, GELSIGHT_H   = 320, 240
GELSIGHT_FOV_W_MM        = 18.6
GELSIGHT_FOV_H_MM        = 14.3
HEIGHT_CUTOFF_M          = 0.050
HEIGHT_MASK_THRES_M      = 0.000
RENDER_MASK_THRES_M      = -0.005
VIDEO_FPS                = 5.0
MASK_OPEN_PX             = 4
ARUCO_TO_CONTACT_M       = 0.050


# ── Normal helpers ─────────────────────────────────────────────────────────────

def normals_to_colormap(normals_np: np.ndarray) -> np.ndarray:
    """(H,W,4) or (H,W,3) float32 → (H,W,3) uint8 BGR.  Invalid (NaN) → black."""
    nxyz = normals_np[:, :, :3]
    valid = np.isfinite(nxyz).all(axis=2)
    rgb = np.zeros((*nxyz.shape[:2], 3), dtype=np.uint8)
    rgb[valid] = ((nxyz[valid] + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR


def inpaint_normals(crop: np.ndarray, method: str = "telea") -> np.ndarray:
    """Fill NaN holes in (H,W,4) float32 normal crop. Returns same shape."""
    out = crop.copy()
    invalid = ~np.isfinite(crop[:, :, 0])
    if not invalid.any():
        return out

    if method == "nearest":
        if not _SCIPY_OK:
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

    hole_u8 = invalid.astype(np.uint8)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    for c in range(3):
        ch = crop[:, :, c].copy()
        ch[invalid] = 0.0
        ch_u8 = ((ch + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        ch_inp = cv2.inpaint(ch_u8, hole_u8, 3, flag)
        out[:, :, c] = ch_inp.astype(np.float32) / 255.0 * 2.0 - 1.0
    return out


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _make_Rz(degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_rvec_z(rvec, R_z):
    R, _ = cv2.Rodrigues(rvec)
    rvec_new, _ = cv2.Rodrigues(R @ R_z)
    return rvec_new.flatten()


# ── Orthographic projection ───────────────────────────────────────────────────

def ortho_project_raw(normals_np, color_bgr, mask, depth_m, intr, method,
                      rvec=None, tvec=None, render_scale=1.0):
    """
    True orthographic projection of the GelSight Mini FoV using sensor pose.
    Returns (normal_bgr, raw_normals_hw3, color_bgr_crop, height_vis,
             contact_mask, height_map, sensor_z_hmap, valid_depth_remap, mask_crop)
    or None if the projection fails.
    """
    if rvec is None or tvec is None:
        return None

    out_h, out_w = GELSIGHT_H, GELSIGHT_W

    R, _ = cv2.Rodrigues(rvec)
    R_flip          = np.diag([1.0, -1.0, -1.0])
    R_sensor        = R @ R_flip
    R_z_align       = _make_Rz(90)
    R_sensor_normals = R @ R_z_align.T @ R_flip

    p_contact = R @ np.array([0.0, 0.0, -ARUCO_TO_CONTACT_M]) + tvec

    x_axis = R_sensor[:, 0]
    y_axis = R_sensor[:, 1]

    fov_w_mm = GELSIGHT_FOV_W_MM * render_scale
    fov_h_mm = GELSIGHT_FOV_H_MM * render_scale
    u = np.linspace(-fov_w_mm / 2, fov_w_mm / 2, out_w) * 1e-3
    v = np.linspace(-fov_h_mm / 2, fov_h_mm / 2, out_h) * 1e-3
    uu, vv = np.meshgrid(u, v)

    P_cam = (p_contact[:, None, None]
             + x_axis[:, None, None] * uu[None]
             + y_axis[:, None, None] * vv[None])

    Pz = P_cam[2]
    valid_proj = Pz > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        spx_f = np.where(valid_proj,
                         intr["fx"] * P_cam[0] / Pz + intr["cx"], -1).astype(np.float32)
        spy_f = np.where(valid_proj,
                         intr["fy"] * P_cam[1] / Pz + intr["cy"], -1).astype(np.float32)

    remap_kw = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    depth_safe = np.where(np.isfinite(depth_m) & (depth_m > 0),
                          depth_m, 0.0).astype(np.float32)
    depth_sampled = cv2.remap(depth_safe, spx_f, spy_f, cv2.INTER_LINEAR, **remap_kw)
    valid_depth_remap = depth_sampled > 0
    P_obj = np.stack([
        (spx_f - intr["cx"]) / intr["fx"] * depth_sampled,
        (spy_f - intr["cy"]) / intr["fy"] * depth_sampled,
        depth_sampled,
    ], axis=-1)
    sensor_z_hmap = R_sensor_normals[:, 2]
    height_map = np.einsum("hwc,c->hw", P_obj - p_contact, sensor_z_hmap)

    color_crop = cv2.remap(color_bgr, spx_f, spy_f, cv2.INTER_LINEAR, **remap_kw)

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

    normals_crop[~valid_proj] = np.nan
    mask_crop[~valid_proj] = 0
    color_crop[~valid_proj] = 0
    normals_crop[mask_crop == 0] = np.nan

    if not np.isfinite(normals_crop[:, :, 0]).any():
        return None

    normals_filled = inpaint_normals(normals_crop, method)

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
    # Re-normalize: bilinear remap and inpainting can break unit length
    norms = np.linalg.norm(raw_norm, axis=-1, keepdims=True)
    valid_px = (norms[..., 0] > 1e-6) & (mask_crop > 0)
    raw_norm[valid_px] /= norms[valid_px]
    raw_norm[mask_crop == 0] = 0.0

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


# ── Contact mask video ────────────────────────────────────────────────────────

def make_render_mask_video(height_map_0, valid_depth_remap, mask_crop,
                           sensor_z_0, tvec_0, pose_contact_slice, num_frames,
                           render_mask_thres=RENDER_MASK_THRES_M,
                           render_mask_type="hard", mask_temperature=0.002):
    """Generate per-frame render mask BGR images.

    pose_contact_slice: list of (rvec, tvec) tuples with tvec=None for missing poses.
    render_mask_type: "hard" (binary) or "soft" (sigmoid grayscale).
    Returns list of num_frames BGR uint8 images.
    """
    from scipy.interpolate import RBFInterpolator

    n_contact = len(pose_contact_slice)
    raw_x, raw_y = [], []
    for i, (rv, tv) in enumerate(pose_contact_slice):
        if tv is not None:
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
        depths = np.zeros(num_frames)

    invalid = ~valid_depth_remap | (mask_crop == 0)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * MASK_OPEN_PX + 1, 2 * MASK_OPEN_PX + 1))
    frames_out = []
    for d in depths:
        threshold = float(d) + render_mask_thres
        if render_mask_type == "soft":
            penetration = (threshold - height_map_0) / max(mask_temperature, 1e-9)
            soft = 1.0 / (1.0 + np.exp(-penetration))
            soft[invalid] = 0.0
            intensity = (soft * 255).clip(0, 255).astype(np.uint8)
            vis = np.stack([intensity, intensity, intensity], axis=-1)
        else:
            cm = (height_map_0 < threshold) & valid_depth_remap & (mask_crop > 0)
            vis = np.zeros((GELSIGHT_H, GELSIGHT_W, 3), dtype=np.uint8)
            vis[cm] = 255
        vis = cv2.morphologyEx(vis, cv2.MORPH_OPEN, morph_kernel)
        frames_out.append(vis)
    return frames_out


def make_render_mask_video_from_arrays(height_map_0, valid_depth_remap, mask_crop,
                                       sensor_z_0, tvec_0,
                                       pose_rvecs, pose_tvecs, num_frames,
                                       render_mask_thres=RENDER_MASK_THRES_M,
                                       render_mask_type="hard", mask_temperature=0.002):
    """Variant used by process_single_shot: poses as (N,3) arrays with NaN rows."""
    n = len(pose_tvecs)
    pose_slice = []
    for i in range(n):
        tv = pose_tvecs[i]
        rv = pose_rvecs[i]
        if np.isfinite(tv).all():
            pose_slice.append((rv, tv))
        else:
            pose_slice.append((None, None))
    return make_render_mask_video(
        height_map_0, valid_depth_remap, mask_crop,
        sensor_z_0, tvec_0, pose_slice, num_frames,
        render_mask_thres=render_mask_thres,
        render_mask_type=render_mask_type,
        mask_temperature=mask_temperature)


# ── Contact trimming ──────────────────────────────────────────────────────────

def trim_and_resample(frames, blank_bgr, num_frames,
                      peak_ratio=0.4, n_neighbors=3, smooth_sigma=2.0):
    """Trim frames to the contact window and resample to num_frames.

    Returns (resampled_frames, peak_raw_idx, cs_idx, ce_idx,
             diffs, smooth_diffs, threshold).
    resampled_frames is None if no contact detected.
    """
    from scipy.ndimage import gaussian_filter1d

    blank = blank_bgr.astype(np.float32) / 255.0
    diffs = np.array([
        np.linalg.norm(f.astype(np.float32) / 255.0 - blank, axis=-1).mean()
        for f in frames
    ])
    smooth_diffs = gaussian_filter1d(diffs, sigma=smooth_sigma)

    peak_raw_idx = int(np.argmax(smooth_diffs))
    peak_val = float(smooth_diffs[peak_raw_idx])
    if peak_val <= 0:
        return None, peak_raw_idx, 0, len(frames) - 1, diffs, smooth_diffs, 0.0

    threshold = peak_val * peak_ratio
    cs_idx = 0
    for i in range(peak_raw_idx):
        if smooth_diffs[i] < threshold:
            right = smooth_diffs[i + 1: i + 1 + n_neighbors]
            if len(right) == n_neighbors and np.all(right > threshold):
                cs_idx = i

    ce_idx = len(diffs) - 1
    for j in range(len(diffs) - 1, peak_raw_idx, -1):
        if smooth_diffs[j] < threshold:
            left = smooth_diffs[max(0, j - n_neighbors): j]
            if len(left) == n_neighbors and np.all(left > threshold):
                ce_idx = j

    contact = frames[cs_idx: ce_idx + 1]
    if len(contact) == 0:
        return None, peak_raw_idx, cs_idx, ce_idx, diffs, smooth_diffs, threshold

    idx = np.linspace(0, len(contact) - 1, num_frames).round().astype(int)
    return [contact[i] for i in idx], peak_raw_idx, cs_idx, ce_idx, diffs, smooth_diffs, threshold


# ── Session segmentation ──────────────────────────────────────────────────────

def segment_contacts(smooth_diffs, seg_threshold, min_gap_frames=10,
                     peak_ratio=0.4, n_neighbors=3, merge_gap=0,
                     boundary_pad=0):
    """Find contact events in a continuous smooth diff-from-blank trace.

    Returns list of (cs_idx, ce_idx, peak_idx, trim_threshold) for each event.

    merge_gap: merge consecutive segments whose gap (next_cs - prev_ce - 1) is
    <= merge_gap frames.  Use merge_gap >= 1 to collapse multiple peaks within a
    single contact into one segment.
    boundary_pad: extra frames added before cs_idx and after ce_idx on every
    segment after merging; clamped to valid array bounds.
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(smooth_diffs, height=seg_threshold,
                          distance=max(1, min_gap_frames))
    results = []
    for peak_idx in peaks:
        peak_idx = int(peak_idx)
        peak_val = float(smooth_diffs[peak_idx])
        threshold = seg_threshold + (peak_val - seg_threshold) * peak_ratio

        cs_idx = peak_idx - 1
        for i in range(peak_idx):
            if smooth_diffs[i] < threshold:
                right = smooth_diffs[i + 1: i + 1 + n_neighbors]
                if len(right) == n_neighbors and np.all(right > threshold):
                    cs_idx = i

        ce_idx = min(len(smooth_diffs) - 1, peak_idx + 1)
        for j in range(len(smooth_diffs) - 1, peak_idx, -1):
            if smooth_diffs[j] < threshold:
                left = smooth_diffs[max(0, j - n_neighbors): j]
                if len(left) == n_neighbors and np.all(left > threshold):
                    ce_idx = j

        results.append((cs_idx, ce_idx, peak_idx, threshold))

    # Merge segments whose gap is within merge_gap frames
    merged = []
    for cs, ce, pk, thr in results:
        if merged and cs - merged[-1][1] - 1 <= merge_gap:
            prev_cs, prev_ce, prev_pk, prev_thr = merged[-1]
            if smooth_diffs[pk] > smooth_diffs[prev_pk]:
                merged[-1] = (prev_cs, max(prev_ce, ce), pk, thr)
            else:
                merged[-1] = (prev_cs, max(prev_ce, ce), prev_pk, prev_thr)
        else:
            merged.append((cs, ce, pk, thr))

    if boundary_pad:
        n = len(smooth_diffs)
        merged = [(max(0, cs - boundary_pad), min(n - 1, ce + boundary_pad), pk, thr)
                  for cs, ce, pk, thr in merged]
    return merged


# ── Display helpers ───────────────────────────────────────────────────────────

def _render_diff_plot(diffs, smooth_diffs, cs_idx, ce_idx, peak_idx,
                      threshold, width=640, height=200, y_max=None):
    """Render raw diff curve, smoothed curve, and threshold line as a BGR image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(diffs)
    if n < 2:
        return img
    if y_max is None:
        max_val = max(float(smooth_diffs.max()), 1e-6)
    else:
        max_val = float(y_max)

    def to_px(i, v):
        x = int(i / (n - 1) * (width - 1))
        y = int((1.0 - min(v, max_val) / max_val) * (height - 1))
        return (x, max(0, min(height - 1, y)))

    ty = to_px(0, threshold)[1]
    cv2.line(img, (0, ty), (width - 1, ty), (0, 0, 255), 1)

    pts_raw = np.array([to_px(i, diffs[i]) for i in range(n)], dtype=np.int32)
    cv2.polylines(img, [pts_raw.reshape(-1, 1, 2)], False, (80, 80, 80), 1)

    pts_sm = np.array([to_px(i, smooth_diffs[i]) for i in range(n)], dtype=np.int32)
    cv2.polylines(img, [pts_sm.reshape(-1, 1, 2)], False, (255, 255, 255), 2)

    for idx_val, color in [(cs_idx, (0, 255, 0)), (ce_idx, (0, 255, 0)),
                           (peak_idx, (0, 255, 255))]:
        if idx_val is not None:
            x = int(idx_val / (n - 1) * (width - 1))
            cv2.line(img, (x, 0), (x, height - 1), color, 1)
    return img


def _format_scale(scale):
    return f"{scale:g}"
