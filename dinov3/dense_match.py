"""Gradio-free DINOv3 dense correspondence matching.

Factored out of app.py's matching logic (sparse patch matching + a fitted
geometric warp -- affine, homography, or thin-plate-spline RBF seeded by
either) so it can be imported from other pipelines (e.g.
main_retrieval_transfer_accel.py) without pulling in gradio/sklearn, and
without triggering app.py's module-level gr.Blocks() UI construction.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.interpolate import RBFInterpolator

PATCH_SIZE = 16
IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_N_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
}

_model_cache: dict = {}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, weights_path: str):
    cache_key = (model_name, weights_path)
    if cache_key not in _model_cache:
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DINOv3 weights not found: '{weights_path}'.")
        device = _get_device()
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=model_name,
            source="local",
            weights=weights_path,
        )
        model.eval()
        model.to(device)
        _model_cache[cache_key] = model
    return _model_cache[cache_key], _get_device()


def _resize(image: Image.Image) -> torch.Tensor:
    w, h = image.size
    h_p = IMAGE_SIZE // PATCH_SIZE
    w_p = round((w * IMAGE_SIZE) / (h * PATCH_SIZE))
    return TF.to_tensor(TF.resize(image, (h_p * PATCH_SIZE, w_p * PATCH_SIZE)))


def _extract_features(model, image: Image.Image, n_layers: int, device: str) -> torch.Tensor:
    img = _resize(image.convert("RGB"))
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float32):
            feats = model.get_intermediate_layers(img, n=range(n_layers), reshape=True, norm=True)
    return feats[-1].squeeze().detach().cpu()  # [D, H, W]


def _stratify_points(pts: torch.Tensor, threshold: float) -> np.ndarray:
    n = len(pts)
    sentinel = threshold + 1.0
    sq_norms = (pts * pts).sum(dim=1)
    dists = -2.0 * pts @ pts.T
    dists.add_(sq_norms[:, None]).add_(sq_norms[None, :])
    dists.fill_diagonal_(sentinel)
    keep = np.ones(n, dtype=bool)
    mask = (dists <= threshold).float()
    ones = torch.ones(n)
    counts = mask @ ones
    while counts.any():
        worst = int(counts.argmax())
        keep[worst] = False
        dists[worst, :] = sentinel
        dists[:, worst] = sentinel
        mask = (dists <= threshold).float()
        counts = mask @ ones
    return np.where(keep)[0]


def _nonzero_at_points(pil_image, points_rc):
    """Boolean (N,) array: True where the pixel at each (row, col) point in
    pil_image is non-zero (i.e. not background/invalid, per the all-zero-pixel
    convention used elsewhere in the pipeline, e.g. ref_static_mask).
    """
    arr = np.asarray(pil_image)  # H x W x 3 uint8
    h, w = arr.shape[:2]
    rows = np.clip(points_rc[:, 0].round().astype(int), 0, h - 1)
    cols = np.clip(points_rc[:, 1].round().astype(int), 0, w - 1)
    return np.any(arr[rows, cols] != 0, axis=-1)


def _find_sparse_matches(image_left, image_right, model, n_layers, device,
                         num_points, stratify_threshold):
    """DINOv3 patch matching: for each patch in image_left, find its best match in image_right.

    Candidate matches whose source or target point lands on an all-zero
    (background/invalid) pixel are dropped before stratification/sampling, so
    the downstream geometric fit isn't corrupted by background matches.

    Returns (pts_l, pts_r): (row, col) points in each image's original pixel space.
    """
    feat_l = _extract_features(model, image_left, n_layers, device)   # [D, H1, W1]
    feat_r = _extract_features(model, image_right, n_layers, device)  # [D, H2, W2]
    dim = feat_l.shape[0]

    feat_l_n = F.normalize(feat_l, p=2, dim=0)
    feat_r_n = F.normalize(feat_r, p=2, dim=0)

    heatmaps = torch.einsum(
        "k f, f h w -> k h w",
        feat_l_n.view(dim, -1).T,   # [N1, D]
        feat_r_n,                    # [D, H2, W2]
    )  # [N1, H2, W2]

    h1, w1 = feat_l.shape[1], feat_l.shape[2]
    h2, w2 = feat_r.shape[1], feat_r.shape[2]
    n1 = h1 * w1

    idx_l = torch.arange(n1)
    locs_l = (torch.stack([idx_l // w1, idx_l % w1], dim=-1).float() + 0.5) * PATCH_SIZE

    idx_r = heatmaps.flatten(-2).argmax(-1)  # [N1]
    locs_r = (torch.stack([idx_r // w2, idx_r % w2], dim=-1).float() + 0.5) * PATCH_SIZE

    scale_l = image_left.height / IMAGE_SIZE
    scale_r = image_right.height / IMAGE_SIZE

    valid = (_nonzero_at_points(image_left, (locs_l * scale_l).numpy())
             & _nonzero_at_points(image_right, (locs_r * scale_r).numpy()))
    if not np.any(valid):
        raise RuntimeError(
            "All DINOv3 matches land on all-zero (background) pixels in "
            "image_left/image_right; no valid foreground correspondence found.")
    valid_t = torch.from_numpy(valid)
    locs_l, locs_r = locs_l[valid_t], locs_r[valid_t]

    keep = _stratify_points(locs_l * scale_l, stratify_threshold ** 2)
    if len(keep) > num_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(keep, size=num_points, replace=False))

    pts_l = locs_l[keep].numpy() * scale_l   # [K, 2] (row, col), original px
    pts_r = locs_r[keep].numpy() * scale_r
    return pts_l, pts_r


TRANSFORM_TYPES = ("affine", "homography", "rbf_affine", "rbf_homography")
REFINE_LOSS_TYPES = ("l1", "l2", "huber", "gradient", "ncc")


# ---------------------------------------------------------------------------
# Photometric refinement of a RANSAC-fit affine/homography matrix
# ---------------------------------------------------------------------------

def _sobel_grad(img):
    """img: (1, C, H, W) torch float tensor. Returns (gx, gy), same shape,
    via a fixed 3x3 Sobel kernel applied per-channel."""
    c = img.shape[1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=img.device, dtype=img.dtype)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(img, kx, padding=1, groups=c)
    gy = F.conv2d(img, ky, padding=1, groups=c)
    return gx, gy


# Fixed worst-case per-pixel penalty charged to out-of-bounds samples in
# _photometric_loss, keyed by loss_type. Values in [0, 1] bound l1/l2/huber's
# own max possible per-pixel value at 1.0; gradient's Sobel-difference scale
# runs higher, so it gets a larger constant.
_OOB_PENALTY = {"l1": 1.0, "l2": 1.0, "huber": 1.0, "gradient": 2.0}


def _photometric_loss(pred, target, valid, loss_type, huber_delta=1.0):
    """pred/target: (1, C, H, W) in [0, 1], pred from a "zeros"-padded
    grid_sample. valid: (1, 1, H, W) bool, True where the sample fell inside
    image_left's bounds.

    Out-of-bounds locations are charged a fixed worst-case penalty (not
    excluded, not left at grid_sample's zero fill) so the optimizer can never
    lower the loss merely by shrinking coverage -- e.g. pushing most of the
    frame out of bounds so it goes unpenalized (excluding invalid pixels) or
    coincidentally matches the target via edge-repeat (a "border"-padded
    grid_sample). Only "ncc" (a whole-image statistic, not a per-pixel sum)
    is computed over valid pixels only instead.
    """
    if loss_type == "ncc":
        m = valid.expand_as(pred)
        p = pred[m] - pred[m].mean()
        t = target[m] - target[m].mean()
        ncc = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
        return 1.0 - ncc

    if loss_type == "l1":
        diff = (pred - target).abs()
    elif loss_type == "l2":
        diff = (pred - target) ** 2
    elif loss_type == "huber":
        diff = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    elif loss_type == "gradient":
        gx_p, gy_p = _sobel_grad(pred)
        gx_t, gy_t = _sobel_grad(target)
        diff = (gx_p - gx_t).abs() + (gy_p - gy_t).abs()
    else:
        raise ValueError(f"Unknown photometric refine loss_type: {loss_type!r}; "
                         f"must be one of {REFINE_LOSS_TYPES}.")

    valid_b = valid.expand_as(diff)
    diff = torch.where(valid_b, diff, torch.full_like(diff, _OOB_PENALTY[loss_type]))
    return diff.mean()


def _refine_transform_photometric(M_init, is_homography, image_left, image_right,
                                   loss_type, iters, lr, huber_delta=1.0):
    """Refine a RANSAC-fit affine (2x3) or homography (3x3) matrix via Adam
    against a dense photometric loss.

    M_init: maps left (x, y) -> right (x, y), the same convention
    cv2.estimateAffine2D(pts_l_xy, pts_r_xy) / cv2.findHomography(pts_l_xy,
    pts_r_xy) produce. image_left / image_right: float32 (H, W, 3) arrays in
    [0, 1] (same convention as compute_dinov3_nnf). Each step warps
    image_left into image_right's grid via the current (inverted) transform
    and a differentiable F.grid_sample, and minimizes the selected
    photometric loss against image_right.

    Returns a refined matrix, same shape/convention as M_init.
    """
    if loss_type not in REFINE_LOSS_TYPES:
        raise ValueError(f"Unknown photometric refine loss_type: {loss_type!r}; "
                         f"must be one of {REFINE_LOSS_TYPES}.")

    device = _get_device()
    h2, w2 = image_right.shape[:2]
    h1, w1 = image_left.shape[:2]

    left_t = torch.from_numpy(np.ascontiguousarray(image_left)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    right_t = torch.from_numpy(np.ascontiguousarray(image_right)).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # M3_init: M_init lifted to a full 3x3 (homography: as-is, normalized;
    # affine: padded with an identity third row so it composes the same way).
    M_init_t = torch.tensor(np.asarray(M_init, dtype=np.float64), dtype=torch.float32, device=device)
    if is_homography:
        M3_init = M_init_t / M_init_t[2, 2]
    else:
        M3_init = torch.eye(3, dtype=torch.float32, device=device)
        M3_init[:2, :] = M_init_t
    M3_init = M3_init.detach()

    # Parameterize as an incremental correction on top of M_init rather than
    # optimizing the raw matrix entries directly. This matters most for
    # homography: its projective row (H[2,0], H[2,1]) sits at a completely
    # different numerical scale than the affine block (they act as a
    # denominator, so an Adam step sized for e.g. translation is enormous
    # relative to them), which otherwise lets a single lr blow the fit up
    # into wild, unrecoverable distortion. `delta` starts at zero (so the
    # first iteration reproduces M_init exactly) and every entry is a
    # comparably-scaled multiplicative correction, regardless of transform_type.
    eye3 = torch.eye(3, dtype=torch.float32, device=device)
    delta = torch.zeros(3, 3, dtype=torch.float32, device=device, requires_grad=True)

    # dst (right) pixel grid; grid_col/grid_row have shape (h2, w2), matching
    # the numpy meshgrid convention used by _dense_field_from_affine/homography.
    grid_col, grid_row = torch.meshgrid(
        torch.arange(w2, dtype=torch.float32, device=device),
        torch.arange(h2, dtype=torch.float32, device=device),
        indexing="xy",
    )
    ones = torch.ones_like(grid_col)

    optimizer = torch.optim.Adam([delta], lr=lr)

    # Tracks the best matrix seen by loss value (including the RANSAC-init M
    # at iteration 0, since delta starts at zero) rather than just returning
    # wherever Adam ends up -- keeps refinement from ever regressing
    # photometric quality vs. no refinement even if later steps diverge.
    best_loss = None
    best_M = (M3_init if is_homography else M3_init[:2, :]).clone()

    for _ in range(iters):
        optimizer.zero_grad()
        M3 = M3_init @ (eye3 + delta)
        if is_homography:
            M = M3 / M3[2, 2]
            M_inv = torch.inverse(M)
            pts = torch.stack([grid_col, grid_row, ones], dim=0).reshape(3, -1)
            src = M_inv @ pts
            src_col = (src[0] / src[2]).reshape(h2, w2)
            src_row = (src[1] / src[2]).reshape(h2, w2)
        else:
            M = M3[:2, :]
            M_inv = torch.inverse(M3)[:2, :]
            src_col = M_inv[0, 0] * grid_col + M_inv[0, 1] * grid_row + M_inv[0, 2]
            src_row = M_inv[1, 0] * grid_col + M_inv[1, 1] * grid_row + M_inv[1, 2]

        # normalize src (in image_left's pixel space) to [-1, 1] for grid_sample
        norm_x = 2.0 * src_col / max(w1 - 1, 1) - 1.0
        norm_y = 2.0 * src_row / max(h1 - 1, 1) - 1.0
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, h2, w2, 2)

        warped = F.grid_sample(left_t, sample_grid, mode="bilinear",
                               padding_mode="zeros", align_corners=True)
        valid = ((norm_x >= -1) & (norm_x <= 1) & (norm_y >= -1) & (norm_y <= 1))
        valid = valid.unsqueeze(0).unsqueeze(0)  # (1, 1, h2, w2)

        loss = _photometric_loss(warped, right_t, valid, loss_type, huber_delta)
        loss_val = float(loss.detach())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best_M = M.detach().clone()

        loss.backward()
        optimizer.step()

    M_out = best_M.cpu().double().numpy()
    if is_homography:
        M_out = M_out / M_out[2, 2]
    return M_out


def _reprojection_inlier_mask(M, is_homography, pts_l_xy, pts_r_xy, reproj_threshold):
    """Same error metric RANSAC uses: forward-map pts_l_xy (left) through M
    and measure Euclidean distance to pts_r_xy (right); inlier iff distance
    <= reproj_threshold. M: (2, 3) affine or (3, 3) homography, left -> right
    (same convention cv2.estimateAffine2D/cv2.findHomography(pts_l_xy,
    pts_r_xy) produce).
    """
    M = np.asarray(M, dtype=np.float64)
    pts_l_xy = np.asarray(pts_l_xy, dtype=np.float64)
    ones = np.ones((len(pts_l_xy), 1), dtype=np.float64)
    pts_l_h = np.concatenate([pts_l_xy, ones], axis=1)  # [N, 3]
    pred = pts_l_h @ M.T  # [N, 3] homography or [N, 2] affine
    if is_homography:
        pred = pred[:, :2] / pred[:, 2:3]
    dist = np.linalg.norm(pred - pts_r_xy, axis=1)
    return dist <= reproj_threshold


def _dense_field_from_affine(pts_l_xy, pts_r_xy, h2, w2, reproj_threshold,
                             image_left=None, image_right=None, refine=False,
                             refine_loss="l1", refine_iters=100, refine_lr=1e-2,
                             refine_huber_delta=1.0):
    """RANSAC affine fit (optionally photometrically refined); returns the
    dense inverse (right -> left) field."""
    try:
        M, mask = cv2.estimateAffine2D(
            pts_l_xy, pts_r_xy, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"Affine RANSAC failed — too few or degenerate matches: {e}")
    if M is None or mask is None:
        raise RuntimeError("Affine RANSAC failed — too few or degenerate matches.")
    inlier_count = int(mask.sum())

    if refine:
        M = _refine_transform_photometric(
            M, False, image_left, image_right,
            refine_loss, refine_iters, refine_lr, refine_huber_delta)

    M_inv = cv2.invertAffineTransform(M)  # right(x, y) -> left(x, y), 2x3
    grid_col, grid_row = np.meshgrid(np.arange(w2, dtype=np.float64),
                                     np.arange(h2, dtype=np.float64))
    src_col = M_inv[0, 0] * grid_col + M_inv[0, 1] * grid_row + M_inv[0, 2]
    src_row = M_inv[1, 0] * grid_col + M_inv[1, 1] * grid_row + M_inv[1, 2]
    return src_row, src_col, inlier_count


def _dense_field_from_homography(pts_l_xy, pts_r_xy, h2, w2, reproj_threshold,
                                 image_left=None, image_right=None, refine=False,
                                 refine_loss="l1", refine_iters=100, refine_lr=1e-2,
                                 refine_huber_delta=1.0):
    """RANSAC homography fit (optionally photometrically refined); returns
    the dense inverse (right -> left) field."""
    try:
        H_mat, mask = cv2.findHomography(
            pts_l_xy, pts_r_xy, cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"Homography RANSAC failed — too few or degenerate matches: {e}")
    if H_mat is None or mask is None:
        raise RuntimeError("Homography RANSAC failed — too few or degenerate matches.")
    inlier_count = int(mask.sum())

    if refine:
        H_mat = _refine_transform_photometric(
            H_mat, True, image_left, image_right,
            refine_loss, refine_iters, refine_lr, refine_huber_delta)

    H_inv = np.linalg.inv(H_mat)
    grid_col, grid_row = np.meshgrid(np.arange(w2, dtype=np.float64),
                                     np.arange(h2, dtype=np.float64))
    ones = np.ones_like(grid_col)
    pts = np.stack([grid_col, grid_row, ones], axis=0).reshape(3, -1)  # [3, h2*w2]
    src = H_inv @ pts
    src = src[:2] / src[2:3]
    src_col = src[0].reshape(h2, w2)
    src_row = src[1].reshape(h2, w2)
    return src_row, src_col, inlier_count


def _dense_field_from_rbf(pts_l, pts_r, pts_l_xy, pts_r_xy, h2, w2,
                          transform_type, reproj_threshold,
                          image_left=None, image_right=None, refine=False,
                          refine_loss="l1", refine_iters=100, refine_lr=1e-2,
                          refine_huber_delta=1.0):
    """RANSAC (affine or homography, for inlier selection) + thin-plate-spline
    RBF dense warp fit on the inliers only.

    If refine, the RANSAC matrix is first photometrically refined, and the
    inlier mask is *recomputed* by reapplying the RANSAC reprojection-error
    test against the refined matrix instead of the original — RBF then
    interpolates raw displacements through this (possibly different) inlier
    set exactly as it does when refine=False.

    Returns (src_row, src_col, inlier_count, total): dense (h2, w2) arrays
    mapping each position in the "right" image's grid back to a source
    position in the "left" image, plus the RANSAC inlier count / total match
    count used to fit them.
    """
    is_homography = transform_type != "rbf_affine"
    try:
        if not is_homography:
            M_init, mask = cv2.estimateAffine2D(
                pts_l_xy, pts_r_xy, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
        else:
            M_init, mask = cv2.findHomography(
                pts_l_xy, pts_r_xy, cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"{transform_type} RANSAC (inlier selection for RBF) failed: {e}")
    if M_init is None or mask is None:
        raise RuntimeError(f"{transform_type} RANSAC (inlier selection for RBF) failed.")
    inlier_bool = mask.ravel().astype(bool)
    inlier_count = int(inlier_bool.sum())
    if inlier_count < 4:
        raise RuntimeError(f"Too few inliers ({inlier_count}) to fit RBF — need at least 4.")

    if refine:
        M_init = _refine_transform_photometric(
            M_init, is_homography, image_left, image_right,
            refine_loss, refine_iters, refine_lr, refine_huber_delta)
        inlier_bool = _reprojection_inlier_mask(
            M_init, is_homography, pts_l_xy, pts_r_xy, reproj_threshold)
        inlier_count = int(inlier_bool.sum())
        if inlier_count < 4:
            raise RuntimeError(
                f"Too few inliers ({inlier_count}) after photometric refinement "
                f"to fit RBF — need at least 4.")

    pts_l_in = pts_l[inlier_bool]   # [I, 2] (row, col)
    pts_r_in = pts_r[inlier_bool]   # [I, 2] (row, col)

    # Inverse mapping: for each right-image pixel, where does it come from in left?
    rbf = RBFInterpolator(
        pts_r_in,
        pts_l_in - pts_r_in,   # displacement: right -> left
        kernel="thin_plate_spline",
    )

    rows = np.arange(h2, dtype=np.float64)
    cols = np.arange(w2, dtype=np.float64)
    grid_col, grid_row = np.meshgrid(cols, rows)   # h2 x w2 each
    query_rc = np.column_stack([grid_row.ravel(), grid_col.ravel()])  # [h2*w2, 2]

    disp = rbf(query_rc)   # [h2*w2, 2] (delta_row, delta_col)
    src_row = (query_rc[:, 0] + disp[:, 0]).reshape(h2, w2)
    src_col = (query_rc[:, 1] + disp[:, 1]).reshape(h2, w2)
    return src_row, src_col, inlier_count


def _fit_dense_field(pts_l, pts_r, h2, w2, transform_type, reproj_threshold,
                     image_left=None, image_right=None, refine=False,
                     refine_loss="l1", refine_iters=100, refine_lr=1e-2,
                     refine_huber_delta=1.0):
    """Fit the requested transform and return the dense (right -> left) field.

    transform_type: one of TRANSFORM_TYPES — "affine", "homography" fit a
    single global transform over all matches (RANSAC-robust); "rbf_affine",
    "rbf_homography" use that same RANSAC fit only to select inliers, then
    interpolate a non-rigid thin-plate-spline warp through them (mirrors
    app.py's "Affine"/"Homography"/"RBF (Affine init)"/"RBF (Homography init)").

    If refine, the RANSAC-fit affine/homography matrix is additionally
    refined via dense photometric gradient descent (see
    _refine_transform_photometric) before being used — for "rbf_*" transform
    types this also reselects the RBF inlier set under the refined matrix
    (see _dense_field_from_rbf). Requires image_left/image_right when set.

    Returns (src_row, src_col, inlier_count, total).
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"Unknown transform_type: {transform_type!r}; "
                         f"must be one of {TRANSFORM_TYPES}.")
    if refine and (image_left is None or image_right is None):
        raise ValueError("photometric refine requires image_left/image_right.")

    pts_l_xy = pts_l[:, ::-1].astype(np.float32)  # cv2 uses (x, y) = (col, row)
    pts_r_xy = pts_r[:, ::-1].astype(np.float32)

    refine_kwargs = dict(image_left=image_left, image_right=image_right, refine=refine,
                         refine_loss=refine_loss, refine_iters=refine_iters,
                         refine_lr=refine_lr, refine_huber_delta=refine_huber_delta)
    if transform_type == "affine":
        src_row, src_col, inlier_count = _dense_field_from_affine(
            pts_l_xy, pts_r_xy, h2, w2, reproj_threshold, **refine_kwargs)
    elif transform_type == "homography":
        src_row, src_col, inlier_count = _dense_field_from_homography(
            pts_l_xy, pts_r_xy, h2, w2, reproj_threshold, **refine_kwargs)
    else:
        src_row, src_col, inlier_count = _dense_field_from_rbf(
            pts_l, pts_r, pts_l_xy, pts_r_xy, h2, w2, transform_type, reproj_threshold,
            **refine_kwargs)

    return src_row, src_col, inlier_count, len(pts_l)


def compute_dinov3_nnf(image_left, image_right, model_name, weights_path,
                       num_points=100, stratify_threshold=20.0, reproj_threshold=8.0,
                       transform_type="rbf_homography",
                       photometric_refine=False, photometric_refine_loss="l1",
                       photometric_refine_iters=100, photometric_refine_lr=1e-2,
                       photometric_refine_huber_delta=1.0):
    """Dense correspondence NNF via DINOv3 sparse matching + a fitted geometric warp.

    image_left / image_right: float32 (H, W, 3) arrays in [0, 1]. image_right
    defines the output grid (must be the *query* image); image_left is the
    source image sampled into that grid (must be the *reference* image) —
    this matches PatchMatch's NNF convention where
    nnf[query_y, query_x] = (ref_x, ref_y).

    transform_type: one of TRANSFORM_TYPES (default "rbf_homography", matching
    this module's original behavior).

    photometric_refine: if set, refines the fitted affine/homography matrix
    via dense photometric gradient descent before building the field (see
    dinov3/dense_match.py's _refine_transform_photometric / _fit_dense_field).

    Returns an (H_right, W_right, 2) int32 NNF, values clipped to image_left's
    bounds.
    """
    model, device = _load_model(model_name, weights_path)
    n_layers = MODEL_N_LAYERS[model_name]

    pil_left = Image.fromarray((np.clip(image_left, 0, 1) * 255).astype(np.uint8))
    pil_right = Image.fromarray((np.clip(image_right, 0, 1) * 255).astype(np.uint8))

    pts_l, pts_r = _find_sparse_matches(
        pil_left, pil_right, model, n_layers, device, num_points, stratify_threshold)

    h2, w2 = image_right.shape[:2]
    src_row, src_col, inlier_count, total = _fit_dense_field(
        pts_l, pts_r, h2, w2, transform_type, reproj_threshold,
        image_left=image_left, image_right=image_right,
        refine=photometric_refine, refine_loss=photometric_refine_loss,
        refine_iters=photometric_refine_iters, refine_lr=photometric_refine_lr,
        refine_huber_delta=photometric_refine_huber_delta)

    nnf = np.zeros((h2, w2, 2), dtype=np.int32)
    nnf[..., 0] = np.clip(np.round(src_col), 0, image_left.shape[1] - 1)
    nnf[..., 1] = np.clip(np.round(src_row), 0, image_left.shape[0] - 1)
    return nnf
