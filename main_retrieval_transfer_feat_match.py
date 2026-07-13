"""
Retrieval-based touch video transfer using local feature matching.

For each query contact point, loads the top-1 retrieved reference from a
results.pkl (produced by retrieve_touch.py), computes a Nearest-Neighbor
Field (NNF) between their static modality images via one of several
correspondence backends (sparse matches -> RANSAC homography inliers ->
thin-plate-spline RBF dense warp), and uses that single NNF to warp every
frame of the query's touch video into the reference's coordinate layout.

--matcher selects the backend (default "dinov3"):
  dinov3                DINOv3 patch-feature matching (dinov3/dense_match.py,
                        requires --dinov3_weights, a gated checkpoint).
  disk_lightglue        DISK keypoints + LightGlue         (image-matching-webui)
  superpoint_superglue  SuperPoint keypoints + SuperGlue    (image-matching-webui)
  loftr                 LoFTR, an end-to-end dense matcher  (image-matching-webui)
  superpoint_lightglue  SuperPoint keypoints + LightGlue    (image-matching-webui)
  sift_lightglue        SIFT keypoints + LightGlue          (image-matching-webui)
See README.md for the one-time setup required for the image-matching-webui
backends (extra pip installs + a third-party clone), and imcui_match.py for
the implementation.

Unlike main_retrieval_transfer_accel.py, this script has no PatchMatch/CUDA
dependency at all -- the selected matcher is the entire correspondence
mechanism, computed once per query/ref pair and applied directly to every
touch frame (no iterative EM refinement, keyframe propagation, acceleration,
or downsampling).

Example usage (DINOv3, the default backend):
    python main_retrieval_transfer_feat_match.py \
        --query_dir log/gelsight_captures/session_01 \
        --ref_dir   log/gelsight_captures/session_01 \
        --retrieval_pkl log/touch_retrieval/results.pkl \
        --modality normal \
        --scale 1 \
        --video_type shadow \
        --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --save_dir log/transfer_feat_match

Example usage (an image-matching-webui backend instead):
    python main_retrieval_transfer_feat_match.py \
        --query_dir log/gelsight_captures/session_01 \
        --ref_dir   log/gelsight_captures/session_01 \
        --retrieval_pkl log/touch_retrieval/results.pkl \
        --modality normal \
        --scale 1 \
        --video_type shadow \
        --matcher superpoint_lightglue \
        --save_dir log/transfer_feat_match_spg_lg
"""

import argparse
import os
import pickle
from os import path as osp

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import lpips
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# ---------------------------------------------------------------------------
# Video I/O (mirrored from main_retrieval_transfer_accel.py)
# ---------------------------------------------------------------------------

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return frames, fps


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# Static image loading
# ---------------------------------------------------------------------------

def load_static_image(folder, idx, modality, scale):
    """Load a color-coded static modality JPG as float32 RGB in [0, 1].

    Used for visualization only.
    """
    if scale is not None:
        fname = f"{idx}_scale{scale:g}_{modality}.jpg"
    else:
        fname = f"{idx}_{modality}.jpg"
    path = osp.join(folder, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read static image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def load_static_raw(folder, idx, modality, scale):
    """Load static modality data for NNF computation.

    Modalities prefixed with "raw_" (e.g. "raw_normal", "raw_height") are loaded
    from the corresponding .npz file using the base name as the array key:
      raw_normal → {idx}[_scale{N}]_normal.npz, key "normal" → (H, W, 3) float32
      raw_height → {idx}[_scale{N}]_height.npz, key "height" → (H, W, 1) float32

    All other modalities are loaded from .jpg via load_static_image.
    """
    if not modality.startswith("raw_"):
        return load_static_image(folder, idx, modality, scale)

    base = modality[len("raw_"):]          # e.g. "raw_normal" → "normal"
    if scale is not None:
        fname = f"{idx}_scale{scale:g}_{base}.npz"
    else:
        fname = f"{idx}_{base}.npz"
    path = osp.join(folder, fname)
    if not osp.exists(path):
        raise FileNotFoundError(f"Cannot read raw modality file: {path}")
    arr = np.load(path)[base].astype(np.float32)
    if arr.ndim == 2:                      # height: (H, W) → (H, W, 1)
        arr = arr[..., np.newaxis]
    return arr


def build_combined_static(folder, idx, modalities, scale):
    """Load and channel-concatenate one or more static modality images.

    Modalities prefixed with "raw_" use .npz raw values; others use color-coded JPGs.
    """
    imgs = [load_static_raw(folder, idx, mod, scale) for mod in modalities]
    if len(imgs) == 1:
        return imgs[0]
    return np.concatenate(imgs, axis=-1).copy(order="C")


def compute_scale_ratio(base_scale, match_scale, convention):
    """Ratio of physical footprint (match canvas / base canvas).

    'render_scale' (GelSight real capture): physical FOV is proportional to
    the scale value, so ratio = match_scale / base_scale.
    'obj_scale_factor' (Taxim): physical FOV is fixed regardless of scale and
    a larger scale renders finer detail (i.e. covers less of the object per
    pixel), so ratio = base_scale / match_scale.
    """
    if convention == "render_scale":
        return match_scale / base_scale
    return base_scale / match_scale  # "obj_scale_factor"


def _prepare_init_scale_static(query_dir, ref_dir, query_idx, ref_idx, modalities,
                                base_scale, match_scale, convention):
    """Load a higher-res static pair and align it to the base canvas's FOV.

    Loads the match-scale static variant of the query/ref images, relates it
    back to the base (--scale) resolution's field of view via the physical
    ratio implied by `convention`, and returns (q_rs, r_rs, H, W): both
    arrays resized/cropped to the base resolution (H, W), ready for DINOv3
    matching.
    """
    q_hi = build_combined_static(query_dir, query_idx, modalities, match_scale)
    r_hi = build_combined_static(ref_dir, ref_idx, modalities, match_scale)
    if q_hi.shape != r_hi.shape:
        raise ValueError("match-scale query/ref static shape mismatch: "
                          f"{q_hi.shape} vs {r_hi.shape}")
    H, W = q_hi.shape[:2]
    r = compute_scale_ratio(base_scale, match_scale, convention)
    if r < 1.0:
        raise ValueError(
            f"match scale must cover at least as much physical area as --scale "
            f"(got ratio {r:.4f} < 1.0 for convention={convention!r}); a smaller "
            f"physical footprint can't seed the full base field of view.")

    if abs(r - 1.0) < 1e-6:
        return q_hi, r_hi, H, W

    # match canvas covers MORE physical area than base -> base FOV is a
    # center crop of the match canvas; crop then upsample to full res so
    # the resulting NNF is directly usable with no coordinate offset.
    ch, cw = max(1, round(H / r)), max(1, round(W / r))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    q_rs = cv2.resize(q_hi[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    r_rs = cv2.resize(r_hi[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    return q_rs, r_rs, H, W


def _load_query_ref_static_for_matching(query_dir, ref_dir, query_idx, ref_idx, modalities,
                                        base_scale, match_scale, convention, matcher_label):
    """Shared static-image loading/validation for every correspondence backend.

    If match_scale is None, loads directly at the base --scale. Otherwise
    reuses _prepare_init_scale_static to align a higher-res variant to the
    base canvas's field of view first. All of the matchers in this script
    (DINOv3 and the image-matching-webui backends) require RGB-like input,
    so the combined modalities must total exactly 3 channels.
    """
    if match_scale is None:
        q = build_combined_static(query_dir, query_idx, modalities, base_scale)
        r = build_combined_static(ref_dir, ref_idx, modalities, base_scale)
        if q.shape != r.shape:
            raise ValueError(f"query/ref static shape mismatch: {q.shape} vs {r.shape}")
    else:
        q, r, _, _ = _prepare_init_scale_static(
            query_dir, ref_dir, query_idx, ref_idx, modalities,
            base_scale, match_scale, convention)

    if q.shape[-1] != 3:
        raise ValueError(
            f"{matcher_label} matching requires modalities that combine to exactly "
            f"3 channels (got {q.shape[-1]}); pick a single RGB-like modality "
            f"(e.g. normal, raw_normal).")
    return q, r


def compute_dinov3_transfer_nnf(query_dir, ref_dir, query_idx, ref_idx, modalities,
                                base_scale, match_scale, convention,
                                dinov3_model, dinov3_weights,
                                num_points, stratify_threshold, reproj_threshold,
                                transform_type):
    """Compute the DINOv3 correspondence NNF used for the entire transfer.

    transform_type: one of "affine", "homography", "rbf_affine",
    "rbf_homography" (see dinov3/dense_match.py:TRANSFORM_TYPES).

    Returns a full (H, W, 2) int32 NNF in the base resolution's coordinate
    space.
    """
    q, r = _load_query_ref_static_for_matching(
        query_dir, ref_dir, query_idx, ref_idx, modalities,
        base_scale, match_scale, convention, "DINOv3")

    from dinov3.dense_match import compute_dinov3_nnf
    return compute_dinov3_nnf(
        image_left=r, image_right=q,  # left=REF (sampled), right=QUERY (output grid)
        model_name=dinov3_model, weights_path=dinov3_weights,
        num_points=num_points, stratify_threshold=stratify_threshold,
        reproj_threshold=reproj_threshold, transform_type=transform_type)


def compute_imcui_transfer_nnf(query_dir, ref_dir, query_idx, ref_idx, modalities,
                               base_scale, match_scale, convention,
                               matcher, reproj_threshold, transform_type):
    """Compute a correspondence NNF via one of image-matching-webui's local
    feature matchers (see imcui_match.py), as an alternative to DINOv3.

    matcher: one of imcui_match.METHOD_TO_ZOO_KEY's keys (e.g.
    "disk_lightglue", "superpoint_superglue", "loftr", "superpoint_lightglue",
    "sift_lightglue"). reproj_threshold/transform_type feed the same
    RANSAC-inlier-selection + geometric-fit stage the DINOv3 backend uses
    (dinov3/dense_match.py's _fit_dense_field), so the two backends' warps
    are directly comparable.

    Returns a full (H, W, 2) int32 NNF in the base resolution's coordinate
    space.
    """
    q, r = _load_query_ref_static_for_matching(
        query_dir, ref_dir, query_idx, ref_idx, modalities,
        base_scale, match_scale, convention, "IMCUI")

    from imcui_match import compute_imcui_nnf
    return compute_imcui_nnf(
        image_left=r, image_right=q,  # left=REF (sampled), right=QUERY (output grid)
        method=matcher, reproj_threshold=reproj_threshold, transform_type=transform_type)


# ---------------------------------------------------------------------------
# NNF reconstruction (mirrors PatchMatchSingle.reconstruct_avg, no PatchMatch dependency)
# ---------------------------------------------------------------------------

def reconstruct_avg(nnf, img, patch_size=1):
    """Warp img into the NNF's output grid via cv2.remap.

    nnf: (H, W, 2) int array, nnf[y, x] = (src_x, src_y) into img.
    """
    map_x = nnf[:, :, 0].astype(np.float32)
    map_y = nnf[:, :, 1].astype(np.float32)
    remapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    if patch_size > 1:
        return cv2.blur(remapped, (patch_size, patch_size))
    return remapped


# ---------------------------------------------------------------------------
# NNF figure
# ---------------------------------------------------------------------------

def _make_ref_color_grid(ref_shape):
    """Rainbow grid: hue sweeps diagonally from red (top-left) to purple (bottom-right)."""
    H_ref, W_ref = ref_shape[:2]
    xs = np.linspace(0, 1, W_ref, dtype=np.float32)
    ys = np.linspace(0, 1, H_ref, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hue = (xg + yg) / 2 * 0.75           # [0, 0.75]: red → yellow → green → blue → purple
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    return matplotlib.colors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))


def _make_nnf_warped(nnf, ref_color_grid):
    """Warp a canonical HSV position grid through the NNF.

    Reveals which reference regions are sampled at each query location.
    """
    return reconstruct_avg(nnf, ref_color_grid, patch_size=1)


def make_nnf_figure(query_idx, ref_idx, query_dir, ref_dir, modalities, scale,
                    nnf, ref_shape, save_dir):
    """Save a (M+1) × 2 diagnostic figure for one query entry.

    Rows 0..M-1 : [Query modality_i]  [Ref modality_i]
    Row M       : [NNF colormap]       [NNF warped]
    """
    M      = len(modalities)
    n_rows = M + 1
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(6, 3 * n_rows + 0.5),
                             squeeze=False)

    def read_rgb(folder, idx, mod):
        vis_mod = mod[len("raw_"):] if mod.startswith("raw_") else mod
        img = load_static_image(folder, idx, vis_mod, scale)   # always JPG, float32 [0,1]
        return img[:, :, :3]                                   # drop extra channels if any

    # -- Modality rows -------------------------------------------------------
    for row, mod in enumerate(modalities):
        for col, (folder, idx, label) in enumerate([
            (query_dir, query_idx, f"Query #{query_idx}"),
            (ref_dir,   ref_idx,   f"Ref #{ref_idx}"),
        ]):
            ax = axes[row, col]
            ax.imshow(read_rgb(folder, idx, mod))
            if row == 0:
                ax.set_title(label, fontsize=9,
                             fontweight="bold" if col == 0 else "normal")
            ax.text(-0.05, 0.5, mod, transform=ax.transAxes,
                    ha="right", va="center", rotation=90, fontsize=9)
            ax.axis("off")

    # -- NNF row -------------------------------------------------------------
    ref_color_grid = _make_ref_color_grid(ref_shape)
    nnf_imgs   = [_make_nnf_warped(nnf, ref_color_grid), ref_color_grid]
    nnf_titles = ["NNF warped", "ref color grid"]
    for col, (img, title) in enumerate(zip(nnf_imgs, nnf_titles)):
        ax = axes[M, col]
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        if col == 0:
            ax.text(-0.05, 0.5, "NNF", transform=ax.transAxes,
                    ha="right", va="center", rotation=90, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Query #{query_idx} → Ref #{ref_idx} — DINOv3 NNF ({', '.join(modalities)})",
                 fontsize=10)
    plt.tight_layout()

    out_path = osp.join(save_dir, f"{query_idx}_nnf.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Video frame evaluation function
# ---------------------------------------------------------------------------
def evaluate_video_metrics(frames_gt, frames_pred, lpips_model, device):
    """Calculates average MSE, PSNR, SSIM, and LPIPS given lists of GT and predicted frames."""
    # Match to the shorter length in case the number of frames differs
    num_frames = min(len(frames_gt), len(frames_pred))

    mse_sum, psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0, 0.0

    for i in range(num_frames):
        gt = frames_gt[i]      # [0, 1] float32 RGB
        pred = frames_pred[i]  # [0, 1] float32 RGB

        # Prevent resolution mismatch (resize prediction to match GT)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        # 1. MSE, PSNR
        mse = compute_mse(gt, pred)
        mse_sum += mse
        psnr_sum += compute_psnr(gt, pred, data_range=1.0) if mse > 0 else 100.0

        # 2. SSIM (Set channel_axis=-1 for multi-channel images)
        ssim_sum += compute_ssim(gt, pred, data_range=1.0, channel_axis=-1)

        # 3. LPIPS (Requires PyTorch Tensor [N, C, H, W] in [-1, 1] range)
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        with torch.no_grad():
            lpips_sum += lpips_model(gt_tensor, pred_tensor).item()

    return {
        "MSE": mse_sum / num_frames,
        "PSNR": psnr_sum / num_frames,
        "SSIM": ssim_sum / num_frames,
        "LPIPS": lpips_sum / num_frames
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transfer query touch videos to reference layout via a DINOv3 feature-match NNF."
    )
    parser.add_argument("--query_dir", required=True, type=str,
                        help="Folder with query touch data.")
    parser.add_argument("--ref_dir", required=True, type=str,
                        help="Folder with reference touch data.")
    parser.add_argument("--retrieval_pkl", required=True, type=str,
                        help="Path to results.pkl from retrieve_touch.py.")
    parser.add_argument("--modality", required=True, nargs="+",
                        choices=["color", "normal", "curvature", "height",
                                 "raw_normal", "raw_height", "shapeindex"],
                        help="Static modality(ies) used to compute the NNF. Must combine to "
                             "exactly 3 channels for DINOv3 (e.g. a single 'normal' or "
                             "'raw_normal'). Prefix with 'raw_' to load raw .npz values "
                             "instead of color-coded JPGs.")
    parser.add_argument("--video_type", required=True,
                        choices=["shadow", "sim"],
                        help="Touch video variant to transfer.")
    parser.add_argument("--scale", default=None, type=float,
                        help="Scale suffix for static images (e.g. 100 for Taxim, 0.5 for GelSight). "
                             "Omit to use base-resolution files. Tag formatted with :g (100.0→'100', 0.5→'0.5').")
    parser.add_argument("--matcher", default="dinov3",
                        choices=["dinov3", "disk_lightglue", "superpoint_superglue", "loftr",
                                 "superpoint_lightglue", "sift_lightglue"],
                        help="Correspondence backend used to compute the NNF (default: dinov3). "
                             "'dinov3' uses DINOv3 patch-feature matching (dinov3/dense_match.py, "
                             "requires --dinov3_weights). The other five run a local feature "
                             "matcher from image-matching-webui (imcui_match.py) instead -- see "
                             "README.md for setup. All backends share the same RANSAC-inlier-"
                             "selection + geometric-fit stage (dinov3/dense_match.py's "
                             "_fit_dense_field), configured uniformly via --reproj_threshold/"
                             "--transform_type regardless of --matcher.")
    parser.add_argument("--dinov3_match_scale", default=None, type=float,
                        help="Optional additional scale suffix used to compute the correspondence "
                             "(with --matcher dinov3 or any of the image-matching-webui backends) "
                             "from a higher-resolution static variant, aligned back to --scale's "
                             "field of view. Omit to match directly on the --scale images. Requires "
                             "--dinov3_match_scale_convention. (Named --dinov3_match_scale for "
                             "historical reasons; applies regardless of --matcher.)")
    parser.add_argument("--dinov3_match_scale_convention", default=None,
                        choices=["render_scale", "obj_scale_factor"],
                        help="How to interpret --scale/--dinov3_match_scale to compute the "
                             "physical-size ratio between them. 'render_scale' (GelSight real "
                             "capture): physical FOV is proportional to the scale value. "
                             "'obj_scale_factor' (Taxim): physical FOV is fixed and a larger "
                             "scale renders finer detail. Required when --dinov3_match_scale "
                             "is set.")
    parser.add_argument("--dinov3_model", default="dinov3_vitb16",
                        choices=["dinov3_vits16", "dinov3_vits16plus",
                                 "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"],
                        help="DINOv3 model variant (default: dinov3_vitb16). Only used with "
                             "--matcher dinov3.")
    parser.add_argument("--dinov3_weights", default=None, type=str,
                        help="Path to gated DINOv3 .pth weights. Required iff --matcher dinov3.")
    parser.add_argument("--dinov3_num_points", default=100, type=int,
                        help="Max sparse DINOv3 keypoints used to fit the RBF warp (default: 100). "
                             "Only used with --matcher dinov3.")
    parser.add_argument("--dinov3_stratify_threshold", default=20.0, type=float,
                        help="Spatial stratification threshold in px, avoids redundant nearby "
                             "keypoints (default: 20.0). Only used with --matcher dinov3.")
    parser.add_argument("--reproj_threshold", default=3.0, type=float,
                        help="RANSAC reprojection threshold in px, used to fit/select inliers "
                             "for --transform_type (default: 3.0). Applies to whichever "
                             "--matcher is selected -- every backend's sparse matches are fit "
                             "with the same dinov3/dense_match.py:_fit_dense_field stage.")
    parser.add_argument("--transform_type", default="rbf_homography",
                        choices=["affine", "homography", "rbf_affine", "rbf_homography"],
                        help="Geometric warp fitted from the selected --matcher's sparse "
                             "matches. 'affine'/'homography' fit a single global RANSAC-robust "
                             "transform over all matches. 'rbf_affine'/'rbf_homography' use that "
                             "same RANSAC fit only to select inliers, then interpolate a "
                             "non-rigid thin-plate-spline warp through them (default: "
                             "rbf_homography).")
    parser.add_argument("--save_dir", default="./log/transfer_feat_match", type=str,
                        help="Output directory for transferred videos.")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation mode (calculate metrics against GT video)")
    parser.add_argument("--no_nnf_figures", action="store_true",
                        help="Skip saving per-query NNF diagnostic figures.")
    args = parser.parse_args()

    if args.dinov3_match_scale is not None and args.dinov3_match_scale_convention is None:
        parser.error("--dinov3_match_scale requires --dinov3_match_scale_convention to be set.")
    if args.matcher == "dinov3" and args.dinov3_weights is None:
        parser.error("--dinov3_weights is required when --matcher dinov3 (the default).")

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load LPIPS model if evaluation mode is enabled
    # ------------------------------------------------------------------
    loss_fn_vgg = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval:
        print(f"Loading LPIPS model on {device}...")
        loss_fn_vgg = lpips.LPIPS(net='alex').to(device)

    # ------------------------------------------------------------------
    # Load retrieval results
    # ------------------------------------------------------------------
    with open(args.retrieval_pkl, "rb") as f:
        retrieval_results = pickle.load(f)
    print(f"Loaded {len(retrieval_results)} retrieval entries from: {args.retrieval_pkl}")

    # ------------------------------------------------------------------
    # Process each query
    # ------------------------------------------------------------------
    all_touch_metrics = {}  # query_idx -> metric dict; populated when --eval is set
    for entry in tqdm(retrieval_results, desc="Transferring"):
        query_idx = entry["query_idx"]
        ref_idx   = entry["topk_ref_indices"][0]  # top-1

        print(f"\nQuery {query_idx} → Reference {ref_idx}")

        # -- Load static images and build combined representation ----------
        try:
            query_static = build_combined_static(
                args.query_dir, query_idx, args.modality, args.scale)
            ref_static   = build_combined_static(
                args.ref_dir,   ref_idx,   args.modality, args.scale)
        except FileNotFoundError as e:
            print(f"  Skipping (missing static image): {e}")
            continue

        if query_static.shape != ref_static.shape:
            print(f"  Skipping: shape mismatch "
                  f"query={query_static.shape} ref={ref_static.shape}")
            continue

        # -- Compute the correspondence NNF (once per pair) ----------------
        # np.linalg.LinAlgError (e.g. a singular homography) is included alongside
        # the RANSAC/shape failures already raised as ValueError/RuntimeError --
        # any of these fall back to an identity NNF rather than skipping the
        # query outright, so a video is still produced for every query.
        try:
            if args.matcher == "dinov3":
                nnf = compute_dinov3_transfer_nnf(
                    args.query_dir, args.ref_dir, query_idx, ref_idx, args.modality,
                    args.scale, args.dinov3_match_scale, args.dinov3_match_scale_convention,
                    args.dinov3_model, args.dinov3_weights,
                    args.dinov3_num_points, args.dinov3_stratify_threshold,
                    args.reproj_threshold, args.transform_type)
            else:
                nnf = compute_imcui_transfer_nnf(
                    args.query_dir, args.ref_dir, query_idx, ref_idx, args.modality,
                    args.scale, args.dinov3_match_scale, args.dinov3_match_scale_convention,
                    args.matcher, args.reproj_threshold, args.transform_type)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"  {args.matcher} matching failed ({e}); falling back to identity transform.")
            h2, w2 = query_static.shape[:2]
            grid_col, grid_row = np.meshgrid(np.arange(w2), np.arange(h2))
            nnf = np.stack([grid_col, grid_row], axis=-1).astype(np.int32)

        # -- Load reference touch video ------------------------------------
        vid_path = osp.join(args.ref_dir, f"{ref_idx}_{args.video_type}.mp4")
        if not osp.exists(vid_path):
            print(f"  Skipping (missing touch video): {vid_path}")
            continue
        ref_frames, fps = read_video(vid_path)

        # -- Transfer each frame using the single DINOv3 NNF ---------------
        transferred = [reconstruct_avg(nnf, frame, patch_size=1)
                      for frame in tqdm(ref_frames, desc="Transferring frames", leave=False)]

        # -- NNF figure -------------------------------------------------------
        if not args.no_nnf_figures:
            fig_path = make_nnf_figure(
                query_idx=query_idx, ref_idx=ref_idx,
                query_dir=args.query_dir, ref_dir=args.ref_dir,
                modalities=args.modality,
                scale=args.dinov3_match_scale if args.dinov3_match_scale is not None else args.scale,
                nnf=nnf, ref_shape=ref_static.shape,
                save_dir=args.save_dir,
            )
            print(f"  Saved NNF figure: {fig_path}")

        # -- Save videos ------------------------------------------------------
        # Query touch video
        q_vid_path = osp.join(args.query_dir, f"{query_idx}_{args.video_type}.mp4")
        if osp.exists(q_vid_path):
            q_frames, q_fps = read_video(q_vid_path)
            write_video(osp.join(args.save_dir,
                                 f"{query_idx}_query_{args.video_type}.mp4"),
                        q_frames, q_fps)

        # Reference touch video
        write_video(osp.join(args.save_dir,
                             f"{query_idx}_ref_{args.video_type}.mp4"),
                    ref_frames, fps)

        # Transferred video
        out_path = osp.join(args.save_dir, f"{query_idx}_transferred.mp4")
        write_video(out_path, transferred, fps)
        print(f"  Saved: {out_path}")

        # Quantitative Evaluation Logic
        if args.eval:
            q_vid_path = osp.join(args.query_dir, f"{query_idx}_{args.video_type}.mp4")
            if not osp.exists(q_vid_path):
                print(f"  [Eval] Query(GT) video not found for eval: {q_vid_path}")
                continue

            q_frames, _ = read_video(q_vid_path)
            metrics = evaluate_video_metrics(q_frames, transferred, loss_fn_vgg, device)
            all_touch_metrics[query_idx] = metrics

            print("\n  [Evaluation Results]")
            print("  ---------------------------------------------------------")
            print(f"  MSE: {metrics['MSE']:.5f} | PSNR: {metrics['PSNR']:.2f} | "
                  f"SSIM: {metrics['SSIM']:.4f} | LPIPS: {metrics['LPIPS']:.4f}")
            print("MSE\tPSNR\tSSIM\tLPIPS")
            print(f"{metrics['MSE']:.5f}\t{metrics['PSNR']:.2f}\t{metrics['SSIM']:.4f}\t{metrics['LPIPS']:.4f}\n")

    # ------------------------------------------------------------------
    # Save metrics pkl
    # ------------------------------------------------------------------
    if args.eval and all_touch_metrics:
        metric_keys = ["MSE", "PSNR", "SSIM", "LPIPS"]
        avg = {k: sum(m[k] for m in all_touch_metrics.values()) / len(all_touch_metrics)
               for k in metric_keys}
        metrics_out = {"per_touch": all_touch_metrics, "average": avg}
        metrics_pkl_path = osp.join(args.save_dir, "metrics.pkl")
        with open(metrics_pkl_path, "wb") as f:
            pickle.dump(metrics_out, f)
        print(f"\nMetrics saved to: {metrics_pkl_path}")
        print(f"Average ({len(all_touch_metrics)} touch locations) — "
              f"MSE: {avg['MSE']:.5f} | PSNR: {avg['PSNR']:.2f} | "
              f"SSIM: {avg['SSIM']:.4f} | LPIPS: {avg['LPIPS']:.4f}")

    print(f"\nDone. Transferred videos saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
