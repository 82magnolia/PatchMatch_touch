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
# Calibrated marker->contact offset (calibrate_sensor_offset.py, 's' output).
# X/Y are in the marker's local frame; Z is marker face -> gel tip.
ARUCO_TO_CONTACT_X_M     = 0.0
ARUCO_TO_CONTACT_Y_M     = -0.0033
ARUCO_TO_CONTACT_M       = 0.0512
# In-plane marker/sensor misalignment. The calibration GUI rotates the *tactile*
# image by +theta to match the render, so the render must rotate by -theta to
# come out aligned with the raw (unrotated) tactile image.
ARUCO_TO_CONTACT_THETA_DEG = -6.6


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


# ── Tactile-image contact mask ────────────────────────────────────────────────

def compute_contact_mask(ref_frame, base_frame, threshold=0.05,
                         blur_sigma=3.0, morph_radius=5):
    """Binary mask of pixels where contact has occurred, from the tactile image.

    Ported from main_retrieval_transfer_accel.py (also copied in
    test_scripts/aggregate_midframe_metrics.py) so the capture pipeline can
    score its geometry-derived render masks against an image-derived reference
    without importing that module's pycuda/PatchMatchCuda chain.

    Pipeline: L2 diff magnitude -> Gaussian blur -> threshold -> morphological
    open (denoise) -> close (fill holes).

    ref_frame/base_frame: (H, W, 3) float in [0, 1].
    Returns float32 (H, W, 1).
    """
    diff = np.abs(ref_frame - base_frame)
    magnitude = np.linalg.norm(diff, axis=-1).astype(np.float32)
    blurred = cv2.GaussianBlur(magnitude, (0, 0), blur_sigma)
    binary = (blurred > threshold).astype(np.uint8)
    k = morph_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary[..., np.newaxis].astype(np.float32)


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
                      rvec=None, tvec=None, render_scale=1.0, apply_mask=True,
                      offset=None, theta_deg=None):
    """
    True orthographic projection of the GelSight Mini FoV using sensor pose.
    Returns (normal_bgr, raw_normals_hw3, color_bgr_crop, height_vis,
             contact_mask, height_map, sensor_z_hmap, valid_depth_remap, mask_crop)
    or None if the projection fails.

    apply_mask: when False, color/normal outputs are not clipped to the SAM
    mask (still clipped to valid_proj, i.e. in front of the camera) -- but
    contact_mask and the returned mask_crop are unaffected, still using the
    real mask, so contact/render-mask detection stays exactly as before.

    offset: (ox, oy, oz) marker->contact translation in the marker's local
    frame, metres. Defaults to the calibrated
    (ARUCO_TO_CONTACT_X_M, ARUCO_TO_CONTACT_Y_M, ARUCO_TO_CONTACT_M).
    Overridden by calibrate_sensor_offset.py while sweeping x/y/z to find the
    true marker->gel-contact translation.

    theta_deg: in-plane marker/sensor misalignment in degrees, defaulting to
    the calibrated ARUCO_TO_CONTACT_THETA_DEG. The render is rotated by
    -theta_deg about the sensor's own Z so it lands aligned with the raw
    tactile image. calibrate_sensor_offset.py passes 0.0 here because it
    instead rotates the tactile overlay by +theta -- applying both would
    double-count the misalignment.
    """
    if rvec is None or tvec is None:
        return None

    if offset is None:
        offset = (ARUCO_TO_CONTACT_X_M, ARUCO_TO_CONTACT_Y_M, ARUCO_TO_CONTACT_M)
    ox, oy, oz = offset

    if theta_deg is None:
        theta_deg = ARUCO_TO_CONTACT_THETA_DEG

    out_h, out_w = GELSIGHT_H, GELSIGHT_W

    R, _ = cv2.Rodrigues(rvec)
    R_flip          = np.diag([1.0, -1.0, -1.0])
    # Post-multiplying by R_theta rotates the sensor frame about its own Z, so
    # the in-plane correction reaches both the sampling axes (x_axis/y_axis
    # below) and the normal re-orientation -- a grid-only rotation would move
    # the pixels but leave each normal vector's nx/ny unrotated.
    R_theta         = _make_Rz(-theta_deg)
    R_sensor        = R @ R_flip @ R_theta
    R_z_align       = _make_Rz(90)
    R_sensor_normals = R @ R_z_align.T @ R_flip @ R_theta

    p_contact = R @ np.array([ox, oy, -oz]) + tvec

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

    # render_mask_crop gates color/normal outputs only; mask_crop (real SAM mask)
    # is left untouched for contact_mask and the returned mask_crop, so --no_mask
    # never changes contact/render-mask detection.
    render_mask_crop = mask_crop if apply_mask else (valid_proj.astype(np.uint8) * 255)
    normals_crop[render_mask_crop == 0] = np.nan

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

    color_crop[render_mask_crop == 0] = 0
    normal_bgr = normals_to_colormap(normals_filled)
    # normals_np (object_cache) is raw, full-frame ZED normals -- real data
    # exists across the whole scene (background/table included), not just the
    # object footprint. inpaint_normals fills *any* NaN hole though, including
    # spans where the source genuinely had no data (stereo failure); when
    # unmasked, render_mask_crop (valid_proj) no longer bounds that void, so
    # the final output must also require valid_remap (real source data) to
    # avoid showing a hallucinated surface where none was ever measured.
    normal_valid_out = (render_mask_crop > 0) & (valid_remap > 0)
    normal_bgr[~normal_valid_out] = 0
    raw_norm = normals_filled[:, :, :3].copy()
    # Re-normalize: bilinear remap and inpainting can break unit length
    norms = np.linalg.norm(raw_norm, axis=-1, keepdims=True)
    valid_px = (norms[..., 0] > 1e-6) & normal_valid_out
    raw_norm[valid_px] /= norms[valid_px]
    raw_norm[~normal_valid_out] = 0.0

    contact_mask = (height_map < HEIGHT_MASK_THRES_M) & valid_depth_remap & (mask_crop > 0)
    h_u8 = (np.clip(-height_map / HEIGHT_CUTOFF_M, 0, 1) * 255).astype(np.uint8)
    height_vis = cv2.applyColorMap(h_u8, cv2.COLORMAP_VIRIDIS)

    return (normal_bgr, raw_norm, color_crop, height_vis, contact_mask,
            height_map, sensor_z_hmap, valid_depth_remap, mask_crop)


def _normalize_field(L, valid, clip_percentile):
    """Zero-anchored uint8 encoding of L (see height2laplacian), with the
    min/max (or percentile) stats computed only from L[valid] -- so a region
    with a different curvature scale than the rest of the image (e.g. a
    boundary artifact ring) can be normalized on its own terms.
    """
    L_valid = L[valid]
    if L_valid.size == 0:
        return np.full(L.shape, 128, dtype=np.uint8)

    if clip_percentile > 0:
        Lmin = float(np.percentile(L_valid, clip_percentile))
        Lmax = float(np.percentile(L_valid, 100 - clip_percentile))
    else:
        Lmin, Lmax = float(L_valid.min()), float(L_valid.max())

    out = np.where(
        L < 0,
        0.5 * (L - Lmin) / (-Lmin + 1e-8),
        0.5 + 0.5 * L / (Lmax + 1e-8),
    )
    out = np.clip(out, 0.0, 1.0)
    return (255 * out).astype(np.uint8)


def height2laplacian(H, mask=None, mask_erode_px=4, clip_percentile=1.0):
    """Height map -> per-image zero-anchored curvature map (uint8, L=0 pinned
    to pixel 128). Local copy of Taxim/OpticalSimulation/simOptical.py's
    height2laplacian, kept in sync by hand rather than imported -- importing
    simOptical would pull in open3d/pyrender/trimesh, the heavy Taxim-mesh-
    simulation deps this module deliberately has none of (see module
    docstring). The normalization is per-image adaptive (uses only H's own
    gradient min/max), so it works unchanged regardless of height_map's units
    (meters here vs. Taxim's internal scale).

    :param mask: optional (H, W) bool/uint8; nonzero marks valid object-surface
        pixels (as opposed to background/out-of-view, e.g. `valid_depth_remap
        & (mask_crop > 0)` from ortho_project_raw). H has a hard step to 0
        outside this region (depth_sampled collapses to 0 there), which the
        double np.gradient + Gaussian blur turns into a spurious high-magnitude
        curvature ring for a few pixels inward of the true silhouette --
        left alone, that ring's huge magnitude dominates a single global
        min/max and compresses genuine interior curvature toward mid-gray.
        When `mask` is given, the interior (an eroded copy of `mask`, see
        `mask_erode_px`) and the boundary-ring-plus-background (everything
        else) are each normalized against their *own* min/max/percentile
        stats and then composited back together -- both regions keep their
        real curvature values (nothing is zeroed or discarded), they just
        no longer share one dynamic range.
    :param mask_erode_px: erosion radius in px, matching the ~4px spread of
        the double-gradient + 5x5 blur, i.e. how far inward of `mask` the
        boundary artifact reaches. Only used when `mask` is given.
    :param clip_percentile: use the [clip_percentile, 100-clip_percentile]
        percentiles of each region instead of its raw min()/max() for
        normalization, so a handful of outlier pixels can't dominate that
        region's own dynamic range. 0 recovers raw-min/max behavior.
    """
    gy, gx = np.gradient(H)
    L = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    L = cv2.GaussianBlur(L, (5, 5), 0)

    if mask is None:
        return _normalize_field(L, np.ones_like(L, dtype=bool), clip_percentile)

    mask_u8 = (np.asarray(mask) != 0).astype(np.uint8)
    k = 2 * mask_erode_px + 1
    interior = cv2.erode(mask_u8, np.ones((k, k), np.uint8)) != 0
    interior_encoded = _normalize_field(L, interior, clip_percentile)
    boundary_encoded = _normalize_field(L, ~interior, clip_percentile)
    return np.where(interior, interior_encoded, boundary_encoded)


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

# For depth_mode="diff_affine":
#   threshold = DIFF_AFFINE_INTERCEPT
#             + DIFF_AFFINE_SLOPE * max(diff - min(diff over segment), 0)
#
# WARNING -- this mode scores best on the pseudo-GT benchmark but does so
# degenerately, and is kept mainly as a documented reference point.
#
# Fitted on a train split of log/real_data_real_retrieval, it reaches held-out
# IoU 0.391 against 0.283 for the ARuCO baseline. But its median mask covers
# 99% of the valid region during contact: it has learned to switch the whole
# footprint on and off in time, not to place a mask in space. A trivial
# reference that marks the *entire* valid region whenever the diff exceeds a
# fixed threshold scores 0.391 / 0.154 -- indistinguishable from this mode's
# 0.391 / 0.155 -- so none of the gain comes from the height map.
# See test_scripts/sweep_render_mask.py and the report in
# log/render_mask_study.html.
#
# The diff is anchored to each segment's own no-contact minimum because
# absolute diff-from-blank levels shift with the blank frame, gel condition and
# lighting; a map fitted on raw values does not transfer between sessions.
# (Anchoring makes the fit session-relative but does not stop the saturation.)
DIFF_AFFINE_SLOPE     = 5.0
DIFF_AFFINE_INTERCEPT = -0.035


def render_mask_eval_positions(n_raw, num_frames, contact_window=None):
    """Normalized segment positions at which to sample the pressing-depth curve.

    With contact_window=None the mask spans the whole segment uniformly, which
    is what the pipeline has always done. But the shadow video it is paired
    with in `{i}_shadow_render_mask.mp4` is trim_and_resample'd to the contact
    window [cs_idx, ce_idx], which covers only ~64% of the segment on average
    (median 51%) -- so frame t of the two videos refers to a different instant
    for 71% of touches. Passing contact_window=(cs_idx, ce_idx) samples the
    same window and puts both videos on one clock.
    """
    if contact_window is None:
        return np.linspace(0.0, 1.0, num_frames)
    cs, ce = contact_window
    n = max(n_raw - 1, 1)
    return np.linspace(cs / n, ce / n, num_frames)


def make_render_mask_video(height_map_0, valid_depth_remap, mask_crop,
                           sensor_z_0, tvec_0, pose_contact_slice, num_frames,
                           render_mask_thres=RENDER_MASK_THRES_M,
                           render_mask_type="hard", mask_temperature=0.002,
                           depth_mode="aruco", diffs=None,
                           contact_window=None):
    """Generate per-frame render mask BGR images.

    pose_contact_slice: list of (rvec, tvec) tuples with tvec=None for missing poses.
    render_mask_type: "hard" (binary) or "soft" (sigmoid grayscale).

    depth_mode selects what drives the per-frame height threshold:
      "aruco"       RBF fit to the ARuCO pressing depth (the original, default)
      "diff_scaled" temporal shape from `diffs`, magnitude from the peak ARuCO
                    depth -- keeps the geometric scale but stops trusting the
                    marker frame-to-frame
      "diff_affine" threshold affine in `diffs` alone; ignores ARuCO and
                    render_mask_thres entirely. Scores best on the benchmark
                    but degenerates to a temporal on/off gate over the whole
                    footprint -- read the DIFF_AFFINE_* comment before using it.

    The diff modes exist because the marker turns out to carry no information
    about the per-touch threshold *level* (correlation with the oracle
    threshold across touches r = -0.08) even though that level is ~52% of the
    variance to explain, while the tactile diff-from-blank predicts both the
    level (r = +0.47) and the within-touch shape (r = +0.64). See
    test_scripts/analyze_oracle_thr.py.

    "diff_scaled" is the conservative choice: it keeps the geometric height
    threshold and only replaces the frame-to-frame marker jitter, improving
    held-out IoU 0.283 -> 0.336 without saturating.

    Note the diff modes make the mask depend on the tactile image, not on
    geometry alone -- appropriate if the mask is a contact-region annotation,
    but not if it is meant as tactile-independent supervision.

    diffs: per-frame diff-from-blank over the same segment as
    pose_contact_slice. Required for the diff modes.
    contact_window: optional (cs_idx, ce_idx); see render_mask_eval_positions.

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

    eval_x = render_mask_eval_positions(n_contact, num_frames, contact_window)

    if depth_mode != "aruco" and diffs is None:
        raise ValueError(f"depth_mode={depth_mode!r} requires `diffs`")

    # `thresholds` is absolute; `depths` (ARuCO modes) still adds render_mask_thres.
    thresholds = None
    if depth_mode == "aruco":
        if len(raw_x) >= 2:
            rbf = RBFInterpolator(
                np.array(raw_x)[:, None], np.array(raw_y),
                kernel="thin_plate_spline", smoothing=0.1)
            depths = rbf(eval_x[:, None])
        else:
            depths = np.zeros(num_frames)
    else:
        s = np.asarray(diffs, dtype=np.float64)
        s_i = np.interp(eval_x, np.arange(len(s)) / max(len(s) - 1, 1), s)
        if depth_mode == "diff_scaled":
            s0, smax = float(s[0]), float(np.max(s))
            shape = np.clip((s_i - s0) / (smax - s0), 0.0, None) \
                if smax - s0 > 1e-12 else np.zeros(num_frames)
            depths = (max(raw_y) if raw_y else 0.0) * shape
        elif depth_mode == "diff_affine":
            thresholds = DIFF_AFFINE_INTERCEPT + DIFF_AFFINE_SLOPE * np.clip(
                s_i - float(np.min(s)), 0.0, None)
            depths = np.zeros(num_frames)
        else:
            raise ValueError(f"unknown depth_mode: {depth_mode!r}")

    invalid = ~valid_depth_remap | (mask_crop == 0)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * MASK_OPEN_PX + 1, 2 * MASK_OPEN_PX + 1))
    frames_out = []
    for k, d in enumerate(depths):
        threshold = (float(thresholds[k]) if thresholds is not None
                     else float(d) + render_mask_thres)
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
