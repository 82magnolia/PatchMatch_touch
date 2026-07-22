"""Decomposed correspondence estimation: offset + linear (affine/homography).

Replaces the single-stage "fit one transform from one set of sparse matches"
approach (dinov3/dense_match.py's _fit_dense_field, including its rbf_affine /
rbf_homography thin-plate-spline variants) with a two-stage decomposition that
estimates each component at the scale where it is best conditioned:

  1. LINEAR stage, at the *match* scale (a wider physical footprint than the
     tactile video covers). A RANSAC affine/homography fit over sparse matches
     between the query and reference static images. The wide footprint gives
     the matcher far more object structure to work with, which is what makes
     the linear part well-conditioned. The fitted matrix is then conjugated
     into the video's coordinate space and its offset is removed, leaving a
     "zero-offset transform" that fixes the image centre.

  2. OFFSET stage, at the *video* scale. The zero-offset transform is applied
     to the reference static, and a second (independently selectable) matcher
     runs between that warped reference and the query static. The component-
     wise median of the resulting match displacements is the offset.

Why the split: a transform fitted at match scale s carries a translation
measured in match-scale pixels, which are r times coarser than the video's
(r = the physical footprint ratio, see compute_scale_ratio). Conjugating into
video coordinates multiplies that translation by r, amplifying whatever error
the match-scale fit had in it -- while leaving the linear part untouched
(exactly, for both affine and homography; see _rescale_transform). Estimating
the offset directly at video scale sidesteps that amplification entirely.

Coordinate/direction convention matches dinov3/dense_match.py: "left" is the
REFERENCE image (the one sampled from) and "right" is the QUERY image (which
defines the output grid), so the fitted matrix maps ref -> query and the NNF
(query -> ref) comes from its inverse.
"""

import numpy as np
import cv2

# Matchers usable for either stage. "dinov3" runs dinov3/dense_match.py's
# patch-feature matching (needs a weights path); the rest are image-matching-
# webui backends via imcui_match.py.
SPARSE_MATCHERS = ("dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                   "superpoint_lightglue", "sift_lightglue")

LINEAR_TRANSFORM_TYPES = ("affine", "homography")


# ---------------------------------------------------------------------------
# Sparse matching (backend dispatch)
# ---------------------------------------------------------------------------

def sparse_match(image_left, image_right, matcher,
                 dinov3_model=None, dinov3_weights=None,
                 num_points=100, stratify_threshold=20.0):
    """Run one sparse matcher on a (ref, query) float32 [0,1] (H, W, 3) pair.

    Returns (pts_l, pts_r): (N, 2) (row, col) arrays in each image's own pixel
    space -- the shared contract of dense_match._find_sparse_matches and
    imcui_match.compute_imcui_sparse_matches.
    """
    if matcher == "dinov3":
        if dinov3_weights is None:
            raise ValueError("matcher 'dinov3' requires --dinov3_weights.")
        from dinov3.dense_match import _find_sparse_matches, _load_model, MODEL_N_LAYERS
        from PIL import Image
        model, device = _load_model(dinov3_model, dinov3_weights)
        n_layers = MODEL_N_LAYERS[dinov3_model]
        pil_l = Image.fromarray((np.clip(image_left, 0, 1) * 255).astype(np.uint8))
        pil_r = Image.fromarray((np.clip(image_right, 0, 1) * 255).astype(np.uint8))
        return _find_sparse_matches(pil_l, pil_r, model, n_layers, device,
                                    num_points, stratify_threshold)

    from imcui_match import compute_imcui_sparse_matches
    return compute_imcui_sparse_matches(image_left, image_right, matcher)


# ---------------------------------------------------------------------------
# Matrix helpers (all 3x3 homogeneous, all ref -> query)
# ---------------------------------------------------------------------------

def _centre_shift(w, h):
    """(T, T_inv, cx, cy) where T moves the image centre to the origin."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    T = np.array([[1.0, 0.0, -cx],
                  [0.0, 1.0, -cy],
                  [0.0, 0.0, 1.0]])
    T_inv = np.array([[1.0, 0.0, cx],
                      [0.0, 1.0, cy],
                      [0.0, 0.0, 1.0]])
    return T, T_inv, cx, cy


def fit_linear(pts_l, pts_r, transform_type, reproj_threshold):
    """RANSAC affine or homography over sparse matches; returns (M, inliers, total).

    M is 3x3 homogeneous mapping ref(left) -> query(right), in whatever pixel
    space pts_l/pts_r were measured in.
    """
    if transform_type not in LINEAR_TRANSFORM_TYPES:
        raise ValueError(f"Unknown transform_type {transform_type!r}; "
                         f"must be one of {LINEAR_TRANSFORM_TYPES}.")
    pts_l_xy = pts_l[:, ::-1].astype(np.float32)   # cv2 wants (x, y)
    pts_r_xy = pts_r[:, ::-1].astype(np.float32)

    try:
        if transform_type == "affine":
            M, mask = cv2.estimateAffine2D(
                pts_l_xy, pts_r_xy, method=cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold)
            if M is not None:
                M = np.vstack([M, [0.0, 0.0, 1.0]])
        else:
            M, mask = cv2.findHomography(
                pts_l_xy, pts_r_xy, cv2.RANSAC, reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"{transform_type} RANSAC failed: {e}")
    if M is None or mask is None:
        raise RuntimeError(
            f"{transform_type} RANSAC failed - too few or degenerate matches "
            f"({len(pts_l)} total).")
    return M.astype(np.float64), int(mask.sum()), len(pts_l)


def _rescale_transform(M, r, w, h):
    """Conjugate a match-scale transform into video-scale coordinates.

    A video pixel p and its match-scale counterpart P are related by
    P = C + (p - C)/r (concentric canvases, same pixel dimensions, the match
    canvas covering r times the physical footprint). So the same geometric
    relation expressed on the video canvas is S^-1 M S with S that shrink.

    Writing M about the centre as [[A, t], [v^T, 1]], the conjugation gives
    [[A, r*t], [v^T/r, 1]]: the linear block A is unchanged, the translation
    scales by r, and the projective row shrinks by r (perspective accumulates
    over spatial extent, so it is r times weaker across an r times smaller
    canvas). Scaling v down also pushes the homography's vanishing line r times
    further out, which keeps the field well-conditioned over the video canvas.
    """
    if abs(r - 1.0) < 1e-9:
        return M
    _, _, cx, cy = _centre_shift(w, h)
    S = np.array([[1.0 / r, 0.0, cx * (1.0 - 1.0 / r)],
                  [0.0, 1.0 / r, cy * (1.0 - 1.0 / r)],
                  [0.0, 0.0, 1.0]])
    S_inv = np.array([[r, 0.0, cx * (1.0 - r)],
                      [0.0, r, cy * (1.0 - r)],
                      [0.0, 0.0, 1.0]])
    out = S_inv @ M @ S
    if abs(out[2, 2]) < 1e-12:
        raise np.linalg.LinAlgError("degenerate transform after scale conjugation")
    return out / out[2, 2]


def strip_offset(M, w, h):
    """Remove M's offset, leaving a transform that fixes the image centre.

    For a homography the centre's image is M_centred[:2, 2] / M_centred[2, 2],
    so normalising before zeroing the translation block is what actually pins
    the centre (not merely zeroing the raw entries).
    """
    T, T_inv, _, _ = _centre_shift(w, h)
    Mc = T @ M @ T_inv
    if abs(Mc[2, 2]) < 1e-12:
        raise np.linalg.LinAlgError("degenerate transform when stripping offset")
    Mc = Mc / Mc[2, 2]
    Mc[:2, 2] = 0.0
    return T_inv @ Mc @ T


def apply_offset(M, tx, ty):
    """Left-compose a pure translation onto M (applied in the query frame)."""
    T_t = np.array([[1.0, 0.0, tx],
                    [0.0, 1.0, ty],
                    [0.0, 0.0, 1.0]])
    return T_t @ M


def dense_field_from_matrix(M, h, w):
    """Evaluate M's inverse over the (h, w) query grid.

    Returns (src_row, src_col) float arrays: for each query pixel, where it
    comes from in the reference. Mirrors dense_match._dense_field_from_*'s
    return contract.
    """
    M_inv = np.linalg.inv(M)
    grid_col, grid_row = np.meshgrid(np.arange(w, dtype=np.float64),
                                     np.arange(h, dtype=np.float64))
    pts = np.stack([grid_col, grid_row, np.ones_like(grid_col)], axis=0).reshape(3, -1)
    src = M_inv @ pts
    denom = src[2:3]
    # A near-zero denominator means the query grid straddles the homography's
    # vanishing line; those pixels have no finite source. Flag rather than let
    # them come back as huge values that silently clip to the border.
    degenerate = np.abs(denom) < 1e-9
    denom = np.where(degenerate, 1e-9, denom)
    src = src[:2] / denom
    return src[1].reshape(h, w), src[0].reshape(h, w)


# ---------------------------------------------------------------------------
# Offset estimation (video scale)
# ---------------------------------------------------------------------------

OFFSET_METHODS = ("none", "median", "ransac")


def _ransac_translation(disp, threshold, max_candidates=1500):
    """Translation-only RANSAC over match displacements.

    The residual after the zero-offset warp is a pure translation by
    construction, so the model has 2 DOF and every single match is already a
    complete hypothesis. That makes exhaustive hypothesis testing cheap and
    deterministic: score each observed displacement by how many others fall
    within `threshold` of it, then average the winning consensus set.

    Unlike the median this rejects outliers outright rather than down-weighting
    them, which matters when a subset of matches is not just noisy but wrong
    (e.g. matches landing in the warped reference's empty border region).

    Returns (disp_mean (row, col), n_inliers).
    """
    n = len(disp)
    cand = disp
    if n > max_candidates:                    # cap the O(N^2) scoring pass
        sel = np.linspace(0, n - 1, max_candidates).astype(int)
        cand = disp[sel]
    d2 = ((cand[:, None, :] - disp[None, :, :]) ** 2).sum(-1)
    inliers = d2 <= threshold ** 2
    best = int(inliers.sum(axis=1).argmax())
    mask = inliers[best]
    return disp[mask].mean(axis=0), int(mask.sum())


def estimate_offset(warped_ref, query, matcher, method="median", threshold=8.0,
                    dinov3_model=None, dinov3_weights=None,
                    num_points=100, stratify_threshold=20.0):
    """Estimate the residual translation between a warped ref and the query.

    warped_ref has already had the zero-offset linear transform applied, so
    whatever systematic displacement remains between it and the query *is* the
    offset.

    method:
      "median"  component-wise median of all match displacements. No threshold
                to tune, but every match votes -- a large coherent block of
                wrong matches drags it.
      "ransac"  translation-only RANSAC (see _ransac_translation), which drops
                outliers instead of down-weighting them.

    Returns (tx, ty, n_matches, n_inliers).
    """
    pts_l, pts_r = sparse_match(warped_ref, query, matcher,
                                dinov3_model=dinov3_model, dinov3_weights=dinov3_weights,
                                num_points=num_points, stratify_threshold=stratify_threshold)
    if len(pts_l) == 0:
        raise RuntimeError("offset matcher returned zero matches")
    disp = pts_r - pts_l                      # (row, col), warped_ref -> query

    if method == "median":
        est, n_in = np.median(disp, axis=0), len(disp)
    elif method == "ransac":
        est, n_in = _ransac_translation(disp, threshold)
    else:
        raise ValueError(f"Unknown offset method {method!r}; must be one of {OFFSET_METHODS}")
    return float(est[1]), float(est[0]), len(pts_l), n_in


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_decomposed_nnf(ref_match, query_match, ref_video, query_video, ratio,
                           transform_type, reproj_threshold,
                           linear_matcher, offset_matcher, offset_method="median",
                           dinov3_model=None, dinov3_weights=None,
                           num_points=100, stratify_threshold=20.0):
    """Two-stage NNF: linear part at match scale, offset at video scale.

    ref_match/query_match : static pair at the match scale, full resolution
                            (never cropped -- the whole point is to give the
                            linear stage the wide footprint's extra structure).
    ref_video/query_video : static pair at the video's own scale.
    ratio                 : match footprint / video footprint (>= 1), from
                            compute_scale_ratio. 1.0 means both stages run on
                            the same images.

    Returns (nnf, info) with nnf an (H, W, 2) int32 array in the video's
    coordinate space, nnf[y, x] = (src_x, src_y), and info a dict of
    diagnostics (see keys assembled at the end).
    """
    h, w = query_video.shape[:2]

    # -- Stage 1: linear part, at match scale ------------------------------
    pts_l, pts_r = sparse_match(ref_match, query_match, linear_matcher,
                                dinov3_model=dinov3_model, dinov3_weights=dinov3_weights,
                                num_points=num_points, stratify_threshold=stratify_threshold)
    M_match, n_inliers, n_total = fit_linear(
        pts_l, pts_r, transform_type, reproj_threshold)

    M_video = _rescale_transform(M_match, ratio, w, h)
    M_zero = strip_offset(M_video, w, h)

    # -- Stage 2: offset, at video scale -----------------------------------
    # Warping the reference by the zero-offset transform puts it in the query's
    # orientation/shape but leaves it un-translated, so the residual the offset
    # matcher sees is the offset itself.
    warped_ref = cv2.warpPerspective(ref_video, M_zero, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    offset_status = "ok"
    n_offset_matches = n_offset_inliers = 0
    if offset_method == "none":
        # Ablation: keep only the linear part, always centred.
        tx, ty = 0.0, 0.0
        offset_status = "none (offset stage disabled)"
    else:
        try:
            tx, ty, n_offset_matches, n_offset_inliers = estimate_offset(
                warped_ref, query_video, offset_matcher,
                method=offset_method, threshold=reproj_threshold,
                dinov3_model=dinov3_model, dinov3_weights=dinov3_weights,
                num_points=num_points, stratify_threshold=stratify_threshold)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            tx, ty = 0.0, 0.0
            offset_status = f"failed ({type(e).__name__}: {e}); offset zeroed"

    # An offset larger than half the canvas means the reference's centre lands
    # outside the query region: the two touch patches do not overlap at all, so
    # there is no offset that makes them correspond. Zeroing keeps the linear
    # part (which is still informative) and hands a centred warp to the
    # downstream refinement network rather than smearing the border.
    if abs(tx) > (w - 1) / 2.0 or abs(ty) > (h - 1) / 2.0:
        offset_status = (f"out of bounds (tx={tx:.1f}, ty={ty:.1f}); offset zeroed")
        tx, ty = 0.0, 0.0

    M_final = apply_offset(M_zero, tx, ty)

    # -- Dense field -------------------------------------------------------
    src_row, src_col = dense_field_from_matrix(M_final, h, w)
    valid = ((src_col >= 0) & (src_col <= w - 1) &
             (src_row >= 0) & (src_row <= h - 1))

    nnf = np.zeros((h, w, 2), dtype=np.int32)
    nnf[..., 0] = np.clip(np.round(src_col), 0, w - 1)
    nnf[..., 1] = np.clip(np.round(src_row), 0, h - 1)

    info = {
        "transform_type": transform_type,
        "linear_matcher": linear_matcher,
        "offset_matcher": offset_matcher,
        "offset_method": offset_method,
        "ratio": ratio,
        "linear_inliers": n_inliers,
        "linear_matches": n_total,
        "offset_matches": n_offset_matches,
        "offset_inliers": n_offset_inliers,
        "offset": (tx, ty),
        "offset_status": offset_status,
        "valid_fraction": float(valid.mean()),
        "M_zero": M_zero,
        "M_final": M_final,
        "warped_ref": warped_ref,
    }
    return nnf, info
