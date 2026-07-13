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


def _dense_field_from_affine(pts_l_xy, pts_r_xy, h2, w2, reproj_threshold):
    """RANSAC affine fit; returns the dense inverse (right -> left) field."""
    try:
        M, mask = cv2.estimateAffine2D(
            pts_l_xy, pts_r_xy, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"Affine RANSAC failed — too few or degenerate matches: {e}")
    if M is None or mask is None:
        raise RuntimeError("Affine RANSAC failed — too few or degenerate matches.")
    inlier_count = int(mask.sum())

    M_inv = cv2.invertAffineTransform(M)  # right(x, y) -> left(x, y), 2x3
    grid_col, grid_row = np.meshgrid(np.arange(w2, dtype=np.float64),
                                     np.arange(h2, dtype=np.float64))
    src_col = M_inv[0, 0] * grid_col + M_inv[0, 1] * grid_row + M_inv[0, 2]
    src_row = M_inv[1, 0] * grid_col + M_inv[1, 1] * grid_row + M_inv[1, 2]
    return src_row, src_col, inlier_count


def _dense_field_from_homography(pts_l_xy, pts_r_xy, h2, w2, reproj_threshold):
    """RANSAC homography fit; returns the dense inverse (right -> left) field."""
    try:
        H_mat, mask = cv2.findHomography(
            pts_l_xy, pts_r_xy, cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        raise RuntimeError(f"Homography RANSAC failed — too few or degenerate matches: {e}")
    if H_mat is None or mask is None:
        raise RuntimeError("Homography RANSAC failed — too few or degenerate matches.")
    inlier_count = int(mask.sum())

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
                          transform_type, reproj_threshold):
    """RANSAC (affine or homography, for inlier selection) + thin-plate-spline
    RBF dense warp fit on the inliers only.

    Returns (src_row, src_col, inlier_count, total): dense (h2, w2) arrays
    mapping each position in the "right" image's grid back to a source
    position in the "left" image, plus the RANSAC inlier count / total match
    count used to fit them.
    """
    try:
        if transform_type == "rbf_affine":
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


def _fit_dense_field(pts_l, pts_r, h2, w2, transform_type, reproj_threshold):
    """Fit the requested transform and return the dense (right -> left) field.

    transform_type: one of TRANSFORM_TYPES — "affine", "homography" fit a
    single global transform over all matches (RANSAC-robust); "rbf_affine",
    "rbf_homography" use that same RANSAC fit only to select inliers, then
    interpolate a non-rigid thin-plate-spline warp through them (mirrors
    app.py's "Affine"/"Homography"/"RBF (Affine init)"/"RBF (Homography init)").

    Returns (src_row, src_col, inlier_count, total).
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"Unknown transform_type: {transform_type!r}; "
                         f"must be one of {TRANSFORM_TYPES}.")

    pts_l_xy = pts_l[:, ::-1].astype(np.float32)  # cv2 uses (x, y) = (col, row)
    pts_r_xy = pts_r[:, ::-1].astype(np.float32)

    if transform_type == "affine":
        src_row, src_col, inlier_count = _dense_field_from_affine(
            pts_l_xy, pts_r_xy, h2, w2, reproj_threshold)
    elif transform_type == "homography":
        src_row, src_col, inlier_count = _dense_field_from_homography(
            pts_l_xy, pts_r_xy, h2, w2, reproj_threshold)
    else:
        src_row, src_col, inlier_count = _dense_field_from_rbf(
            pts_l, pts_r, pts_l_xy, pts_r_xy, h2, w2, transform_type, reproj_threshold)

    return src_row, src_col, inlier_count, len(pts_l)


def compute_dinov3_nnf(image_left, image_right, model_name, weights_path,
                       num_points=100, stratify_threshold=20.0, reproj_threshold=3.0,
                       transform_type="rbf_homography"):
    """Dense correspondence NNF via DINOv3 sparse matching + a fitted geometric warp.

    image_left / image_right: float32 (H, W, 3) arrays in [0, 1]. image_right
    defines the output grid (must be the *query* image); image_left is the
    source image sampled into that grid (must be the *reference* image) —
    this matches PatchMatch's NNF convention where
    nnf[query_y, query_x] = (ref_x, ref_y).

    transform_type: one of TRANSFORM_TYPES (default "rbf_homography", matching
    this module's original behavior).

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
        pts_l, pts_r, h2, w2, transform_type, reproj_threshold)

    nnf = np.zeros((h2, w2, 2), dtype=np.int32)
    nnf[..., 0] = np.clip(np.round(src_col), 0, image_left.shape[1] - 1)
    nnf[..., 1] = np.clip(np.round(src_row), 0, image_left.shape[0] - 1)
    return nnf
