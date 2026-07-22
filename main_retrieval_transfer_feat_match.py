"""
Retrieval-based touch video transfer using local feature matching.

For each query contact point, loads the top-1 retrieved reference from a
results.pkl (produced by retrieve_touch.py), computes a Nearest-Neighbor
Field (NNF) between their static modality images, and uses that single NNF to
warp every frame of the reference's touch video into the query's layout.

The correspondence is estimated as a DECOMPOSITION -- offset ∘ linear -- with
each component estimated at the scale where it is best conditioned
(decomposed_match.py):

  linear part  affine or homography, RANSAC-fit at --match_scale, whose wider
               physical footprint gives the matcher much more object structure
               to work with. Matcher selected by --matcher.
  offset part  estimated at --video_scale (the tactile video's own scale) as the
               median displacement between the zero-offset-warped reference
               and the query. Matcher selected by --offset_matcher.

Why split them: a translation fitted at the match scale is measured in pixels
that are `ratio` times coarser than the video's, so converting it into video
coordinates multiplies it -- and its error -- by that ratio, while leaving the
linear part exactly unchanged. Estimating the offset natively at the video
scale avoids the amplification entirely. When the offset would place the
reference's centre outside the query region (the two touch patches simply do
not overlap), it is zeroed, keeping the linear part and handing a centred warp
to the downstream refinement network.

--video_scale must name the scale the VIDEO was rendered at, since that is the space
the NNF is produced in. For GelSight that is 1 (the raw sensor FOV); for Taxim
it is the FIRST --obj_scale_factor given to gen_contact_video.py, because
videos there are always rendered from sims[0] while statics are written for
every requested scale.

The thin-plate-spline variants (rbf_affine / rbf_homography) and photometric
refinement have been removed from this script. dinov3/dense_match.py still
provides both for main_retrieval_transfer_accel.py.

Matchers available for either stage:
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
dependency at all -- the matchers are the entire correspondence mechanism,
computed once per query/ref pair and applied directly to every touch frame (no
iterative EM refinement, keyframe propagation, acceleration, or downsampling).

Example usage (real GelSight: video at scale 1, linear part fitted at scale 8):
    python main_retrieval_transfer_feat_match.py \
        --query_dir log/real_data_gt_retrieval/10 \
        --ref_dir   log/real_data_gt_retrieval/10 \
        --retrieval_pkl log/touch_retrieval/10/results.pkl \
        --modality curvature \
        --video_scale 1 --match_scale 8 --match_scale_convention render_scale \
        --video_type shadow \
        --matcher disk_lightglue --offset_matcher dinov3 \
        --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --save_dir log/transfer_feat_match

Example usage (Taxim: video at obj_scale_factor[0]=100, linear part at 25):
    python main_retrieval_transfer_feat_match.py \
        --query_dir Taxim/results/gen_contact_full_query_pseudo_mini/0 \
        --ref_dir   Taxim/results/gen_contact_full_pseudo_mini/0 \
        --retrieval_pkl log/touch_retrieval/0/results.pkl \
        --modality curvature \
        --video_scale 100 --match_scale 25 --match_scale_convention obj_scale_factor \
        --video_type shadow \
        --dinov3_weights dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --save_dir log/transfer_feat_match_pseudo_mini
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

from decomposed_match import SPARSE_MATCHERS, LINEAR_TRANSFORM_TYPES, OFFSET_METHODS

# cv2's own thread pool doesn't respect OMP_NUM_THREADS/MKL_NUM_THREADS, so it needs
# to be capped separately when several instances of this script run side by side
# (e.g. via run.sh's NUM_THREADS) to avoid CPU oversubscription.
if "NUM_THREADS" in os.environ:
    _num_threads = int(os.environ["NUM_THREADS"])
    cv2.setNumThreads(_num_threads)
    torch.set_num_threads(_num_threads)


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


def _check_rgb(img, label):
    if img.shape[-1] != 3:
        raise ValueError(
            f"{label} matching requires modalities that combine to exactly "
            f"3 channels (got {img.shape[-1]}); pick a single RGB-like modality "
            f"(e.g. normal, raw_normal).")
    return img


def load_static_pairs(query_dir, ref_dir, query_idx, ref_idx, modalities,
                      video_scale, match_scale, convention):
    """Load both static pairs the decomposed pipeline needs, plus their ratio.

    Returns (q_video, r_video, q_match, r_match, ratio):
      *_video : statics at --video_scale, the scale the tactile video lives at. The
                offset stage runs here, and the NNF is produced in this space.
      *_match : statics at --match_scale, full resolution and *uncropped* --
                the linear stage runs here precisely to get the wider
                footprint's extra structure. When --match_scale is omitted
                these are the same arrays as *_video and ratio is 1.0.
      ratio   : match footprint / video footprint (>= 1).

    Unlike the previous implementation, the match-scale pair is never cropped
    down to the video's field of view. Cropping discarded exactly the extra
    context the higher scale was loaded for (and, since every scale variant is
    rendered at the same pixel resolution, cost detail on top). The
    match->video reconciliation now happens on the fitted transform instead
    (decomposed_match._rescale_transform).
    """
    q_video = _check_rgb(build_combined_static(query_dir, query_idx, modalities, video_scale),
                         "static")
    r_video = _check_rgb(build_combined_static(ref_dir, ref_idx, modalities, video_scale),
                         "static")
    if q_video.shape != r_video.shape:
        raise ValueError(f"query/ref static shape mismatch: "
                         f"{q_video.shape} vs {r_video.shape}")

    if match_scale is None:
        return q_video, r_video, q_video, r_video, 1.0

    q_match = _check_rgb(build_combined_static(query_dir, query_idx, modalities, match_scale),
                         "static")
    r_match = _check_rgb(build_combined_static(ref_dir, ref_idx, modalities, match_scale),
                         "static")
    if q_match.shape != r_match.shape:
        raise ValueError("match-scale query/ref static shape mismatch: "
                         f"{q_match.shape} vs {r_match.shape}")
    if q_match.shape != q_video.shape:
        raise ValueError("match-scale and video-scale statics must share pixel "
                         f"dimensions (got {q_match.shape} vs {q_video.shape})")

    ratio = compute_scale_ratio(video_scale, match_scale, convention)
    if ratio < 1.0:
        raise ValueError(
            f"--match_scale must cover at least as much physical area as --video_scale "
            f"(got ratio {ratio:.4f} for convention={convention!r}); a smaller "
            f"physical footprint gives the linear stage less context, not more.")
    return q_video, r_video, q_match, r_match, ratio


def compute_transfer_nnf(query_dir, ref_dir, query_idx, ref_idx, modalities,
                         video_scale, match_scale, convention,
                         transform_type, reproj_threshold,
                         linear_matcher, offset_matcher, offset_method="median",
                         dinov3_model=None, dinov3_weights=None,
                         num_points=100, stratify_threshold=20.0):
    """Compute the correspondence NNF used for the entire transfer.

    Decomposes the warp into a linear part (affine or homography, RANSAC-fit
    at --match_scale, where the wider footprint gives the matcher more object
    structure) and an offset (estimated separately at --video_scale, the video's own
    scale, from matches between the zero-offset-warped reference and the
    query). See decomposed_match.py for the rationale and the algebra.

    Returns (nnf, info): an (H, W, 2) int32 NNF in the *video's* coordinate
    space, plus a diagnostics dict.
    """
    q_video, r_video, q_match, r_match, ratio = load_static_pairs(
        query_dir, ref_dir, query_idx, ref_idx, modalities,
        video_scale, match_scale, convention)

    from decomposed_match import compute_decomposed_nnf
    return compute_decomposed_nnf(
        ref_match=r_match, query_match=q_match,      # left=REF (sampled), right=QUERY (grid)
        ref_video=r_video, query_video=q_video, ratio=ratio,
        transform_type=transform_type, reproj_threshold=reproj_threshold,
        linear_matcher=linear_matcher, offset_matcher=offset_matcher,
        offset_method=offset_method,
        dinov3_model=dinov3_model, dinov3_weights=dinov3_weights,
        num_points=num_points, stratify_threshold=stratify_threshold)


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
# Local feature match figure (raw sparse matches, RANSAC inliers/outliers)
# ---------------------------------------------------------------------------

def _compute_sparse_matches_and_inliers(q, r, matcher, dinov3_model, dinov3_weights,
                                        num_points, stratify_threshold,
                                        reproj_threshold, transform_type):
    """Recompute the sparse matches (and RANSAC inlier mask) that fed the NNF.

    Mirrors the sparse-matching + inlier-selection stages of
    compute_dinov3_transfer_nnf/compute_imcui_transfer_nnf (and
    dinov3/dense_match.py's _fit_dense_field), but also returns the raw match
    points and the RANSAC inlier boolean mask, which the production NNF path
    doesn't need and so doesn't expose.

    Returns (pts_l, pts_r, inlier_mask, status, reason):
      pts_l/pts_r: (N, 2) (row, col) arrays in q/r's original pixel space, or
        None if sparse matching itself failed.
      inlier_mask: (N,) bool array, or None if fitting failed/wasn't reached.
      status: "ok" | "fit_fail" | "sparse_fail".
      reason: human-readable failure explanation, "" if status == "ok".
    """
    from decomposed_match import sparse_match
    try:
        pts_l, pts_r = sparse_match(
            r, q, matcher, dinov3_model=dinov3_model, dinov3_weights=dinov3_weights,
            num_points=num_points, stratify_threshold=stratify_threshold)
    except Exception as e:
        return None, None, None, "sparse_fail", f"{type(e).__name__}: {e}"

    # Same RANSAC call decomposed_match.fit_linear makes -- just also keeping the mask.
    pts_l_xy = pts_l[:, ::-1].astype(np.float32)
    pts_r_xy = pts_r[:, ::-1].astype(np.float32)
    min_needed = 3 if transform_type == "affine" else 4
    try:
        if transform_type == "affine":
            M_init, mask = cv2.estimateAffine2D(
                pts_l_xy, pts_r_xy, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
        else:
            M_init, mask = cv2.findHomography(
                pts_l_xy, pts_r_xy, cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    except cv2.error as e:
        return pts_l, pts_r, None, "fit_fail", f"cv2 error: {e}"
    if M_init is None or mask is None:
        return pts_l, pts_r, None, "fit_fail", "RANSAC returned None (degenerate configuration)"

    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_needed:
        return pts_l, pts_r, inlier_mask, "fit_fail", f"Only {n_inliers} inliers (need >= {min_needed})"
    return pts_l, pts_r, inlier_mask, "ok", ""


def _draw_match_panel(ref_img, query_img, pts_l, pts_r, inlier_mask, status, reason):
    """ref_img/query_img: float32 (H, W, 3) [0, 1]. Returns a BGR uint8 canvas
    with ref | query side by side, matches drawn as lines: green (RANSAC
    inlier), red (outlier), orange (matches found but the fit failed
    outright, e.g. too few inliers), colored border by status.
    """
    ref_u8 = cv2.cvtColor((np.clip(ref_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    q_u8 = cv2.cvtColor((np.clip(query_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    h, w = ref_u8.shape[:2]
    gap = 8
    canvas = np.full((h + 60, w * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[60:60 + h, 0:w] = ref_u8
    canvas[60:60 + h, w + gap:2 * w + gap] = q_u8
    off_x, off_y = w + gap, 60

    n_matches = 0 if pts_l is None else len(pts_l)
    n_inliers = 0 if inlier_mask is None else int(inlier_mask.sum())

    if pts_l is not None:
        for i in range(len(pts_l)):
            r1, c1 = pts_l[i]
            r2, c2 = pts_r[i]
            p1 = (int(round(c1)), int(round(r1)) + 60)
            p2 = (int(round(c2)) + off_x, int(round(r2)) + off_y)
            if status == "ok":
                color = (0, 180, 0) if inlier_mask[i] else (0, 0, 220)
            else:
                color = (0, 140, 255)  # orange: matches found but fit failed
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, color, -1, cv2.LINE_AA)

    border_color = {"ok": (0, 170, 0), "fit_fail": (0, 140, 255), "sparse_fail": (0, 0, 220)}[status]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_color, 6)

    header = f"matches={n_matches}  inliers={n_inliers}  status={status}"
    cv2.putText(canvas, header, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    if reason:
        cv2.putText(canvas, reason[:130], (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_match_figure(query_idx, ref_idx, q, r, matcher, dinov3_model, dinov3_weights,
                      num_points, stratify_threshold, reproj_threshold, transform_type,
                      save_dir):
    """Save a ref|query panel showing this query's raw sparse matches, colored
    by RANSAC inlier/outlier status -- unlike make_nnf_figure (which shows the
    final dense NNF), this exposes what the correspondence step actually
    found, including *why* a query fell back to identity (zero/degenerate
    matches vs. too few RANSAC inliers).
    """
    pts_l, pts_r, inlier_mask, status, reason = _compute_sparse_matches_and_inliers(
        q[..., :3], r[..., :3], matcher, dinov3_model, dinov3_weights,
        num_points, stratify_threshold, reproj_threshold, transform_type)
    canvas = _draw_match_panel(r[..., :3], q[..., :3], pts_l, pts_r, inlier_mask, status, reason)
    out_path = osp.join(save_dir, f"{query_idx}_matches.png")
    cv2.imwrite(out_path, canvas)
    return out_path, status


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
        description="Transfer reference touch videos into the query layout via a "
                    "decomposed (offset ∘ linear) feature-match NNF."
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
    parser.add_argument("--video_scale", "--scale", default=None, type=float,
                        dest="video_scale",
                        help="Scale suffix of the static images that share the tactile video's "
                             "coordinate space -- i.e. the scale the VIDEO was rendered at "
                             "(GelSight: 1, the raw sensor FOV; Taxim: the first "
                             "--obj_scale_factor passed to gen_contact_video.py, since videos "
                             "are always rendered from sims[0]). The offset stage runs here and "
                             "the output NNF lives in this space, so getting it wrong silently "
                             "misaligns every transfer. Omit to use base-resolution files. Tag "
                             "formatted with :g (100.0→'100', 0.5→'0.5'). (--scale is accepted "
                             "as a deprecated alias.)")
    parser.add_argument("--matcher", default="disk_lightglue",
                        choices=list(SPARSE_MATCHERS),
                        help="Matcher for the LINEAR stage, run at --match_scale (default: "
                             "disk_lightglue). 'dinov3' uses DINOv3 patch-feature matching "
                             "(requires --dinov3_weights); the rest are image-matching-webui "
                             "backends (imcui_match.py) -- see README.md for setup.")
    parser.add_argument("--offset_matcher", default="disk_lightglue",
                        choices=list(SPARSE_MATCHERS),
                        help="Matcher for the OFFSET stage, run at --video_scale between the "
                             "zero-offset-warped reference and the query (default: disk_lightglue; "
                             "it localises the residual translation better than dinov3 on both "
                             "sim and real data, where dinov3's patch-quantised displacements "
                             "under-shoot). Independent of --matcher: the two stages solve "
                             "different problems (wide-context geometry vs. a residual "
                             "translation) and are not necessarily best served by the same matcher.")
    parser.add_argument("--offset_method", default="median", choices=list(OFFSET_METHODS),
                        help="How the OFFSET stage turns the matcher's displacements into a "
                             "translation (default: median). 'median' takes the component-wise "
                             "median of every match. 'ransac' runs a translation-only RANSAC "
                             "(--reproj_threshold as the inlier radius), rejecting outliers "
                             "outright rather than down-weighting them. 'none' disables the "
                             "offset stage entirely, leaving the centred zero-offset linear "
                             "warp -- the ablation that isolates how much the offset is "
                             "actually contributing.")
    parser.add_argument("--match_scale", "--dinov3_match_scale", default=None, type=float,
                        dest="match_scale",
                        help="Scale suffix of the static variant used for the LINEAR stage -- "
                             "must cover at least as much physical area as --video_scale. The wider "
                             "footprint gives the matcher more object structure to fit the "
                             "linear part from; the fitted matrix is then conjugated back into "
                             "--video_scale's coordinate space (decomposed_match._rescale_transform), "
                             "so the images themselves are never cropped. Omit to run both "
                             "stages on the --video_scale images. Requires --match_scale_convention. "
                             "(--dinov3_match_scale is accepted as a deprecated alias.)")
    parser.add_argument("--match_scale_convention", "--dinov3_match_scale_convention",
                        default=None, dest="match_scale_convention",
                        choices=["render_scale", "obj_scale_factor"],
                        help="How to interpret --video_scale/--match_scale to compute the physical "
                             "footprint ratio between them. 'render_scale' (GelSight real "
                             "capture): physical FOV is proportional to the scale value. "
                             "'obj_scale_factor' (Taxim): the sensor FOV is fixed and a larger "
                             "scale renders finer detail. Required when --match_scale is set.")
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
    parser.add_argument("--reproj_threshold", default=8.0, type=float,
                        help="RANSAC reprojection threshold in px for the LINEAR stage's "
                             "inlier selection (default: 8.0). Measured in --match_scale "
                             "pixels, since that is where the linear fit runs. The offset "
                             "stage uses a median rather than RANSAC, so this does not "
                             "affect it.")
    parser.add_argument("--transform_type", default="homography",
                        choices=list(LINEAR_TRANSFORM_TYPES),
                        help="Linear component fitted at --match_scale (default: homography). "
                             "The full warp is always offset ∘ linear; the thin-plate-spline "
                             "variants (rbf_affine/rbf_homography) have been removed -- see "
                             "decomposed_match.py. Note dinov3/dense_match.py still provides "
                             "them for main_retrieval_transfer_accel.py.")
    parser.add_argument("--use_mask", action="store_true",
                        help="Composite transferred frames with the query's render mask video "
                             "(same convention/compositing as main_retrieval_transfer_accel.py's "
                             "--use_mask): output = mask * transferred + (1 - mask) * base_frame, "
                             "where base_frame is frame 0 of the reference touch video (assumed "
                             "pre-contact background). Requires {query_idx}_render_mask.mp4 in "
                             "--query_dir.")
    parser.add_argument("--save_dir", default="./log/transfer_feat_match", type=str,
                        help="Output directory for transferred videos.")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation mode (calculate metrics against GT video)")
    parser.add_argument("--no_nnf_figures", action="store_true",
                        help="Skip saving per-query NNF diagnostic figures.")
    parser.add_argument("--save_match_figures", action="store_true",
                        help="Save a per-query ref|query panel of the raw sparse feature "
                             "matches, colored by RANSAC inlier (green) / outlier (red) / "
                             "fit-failed (orange) -- unlike the NNF figure (final dense warp), "
                             "this shows what the correspondence step actually found, including "
                             "why a query fell back to identity. Disabled by default (recomputes "
                             "the sparse-matching stage, so adds runtime).")
    args = parser.parse_args()

    if args.match_scale is not None and args.match_scale_convention is None:
        parser.error("--match_scale requires --match_scale_convention to be set.")
    _active = [args.matcher] + ([] if args.offset_method == "none" else [args.offset_matcher])
    if "dinov3" in _active and args.dinov3_weights is None:
        parser.error("--dinov3_weights is required when either --matcher or --offset_matcher "
                     "is dinov3.")

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
    all_info = {}           # query_idx -> decomposed_match diagnostics dict
    n_offset_zeroed = 0
    for entry in tqdm(retrieval_results, desc="Transferring"):
        query_idx = entry["query_idx"]
        ref_idx   = entry["topk_ref_indices"][0]  # top-1

        print(f"\nQuery {query_idx} → Reference {ref_idx}")

        # -- Load static images and build combined representation ----------
        try:
            query_static = build_combined_static(
                args.query_dir, query_idx, args.modality, args.video_scale)
            ref_static   = build_combined_static(
                args.ref_dir,   ref_idx,   args.modality, args.video_scale)
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
        info = None
        try:
            nnf, info = compute_transfer_nnf(
                args.query_dir, args.ref_dir, query_idx, ref_idx, args.modality,
                args.video_scale, args.match_scale, args.match_scale_convention,
                args.transform_type, args.reproj_threshold,
                args.matcher, args.offset_matcher, args.offset_method,
                args.dinov3_model, args.dinov3_weights,
                args.dinov3_num_points, args.dinov3_stratify_threshold)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"  linear stage ({args.matcher}) failed ({e}); "
                  f"falling back to identity transform.")
            h2, w2 = query_static.shape[:2]
            grid_col, grid_row = np.meshgrid(np.arange(w2), np.arange(h2))
            nnf = np.stack([grid_col, grid_row], axis=-1).astype(np.int32)

        if info is not None:
            tx, ty = info["offset"]
            # "none" is the disabled-by-design ablation, not a failure to zero.
            if info["offset_status"] not in ("ok",) and info["offset_method"] != "none":
                n_offset_zeroed += 1
            print(f"  linear({args.matcher}, {args.transform_type}) "
                  f"{info['linear_inliers']}/{info['linear_matches']} inliers, ratio={info['ratio']:g} | "
                  f"offset({args.offset_matcher}/{args.offset_method}) ({tx:+.1f}, {ty:+.1f}) "
                  f"from {info['offset_inliers']}/{info['offset_matches']} matches "
                  f"[{info['offset_status']}] | "
                  f"valid {info['valid_fraction']*100:.1f}%")
            all_info[query_idx] = {k: v for k, v in info.items() if k != "warped_ref"}

        # -- Load reference touch video ------------------------------------
        vid_path = osp.join(args.ref_dir, f"{ref_idx}_{args.video_type}.mp4")
        if not osp.exists(vid_path):
            print(f"  Skipping (missing touch video): {vid_path}")
            continue
        ref_frames, fps = read_video(vid_path)

        # -- Transfer each frame using the single DINOv3 NNF ---------------
        transferred = [reconstruct_avg(nnf, frame, patch_size=1)
                      for frame in tqdm(ref_frames, desc="Transferring frames", leave=False)]

        # -- Optionally composite with the query's render mask -------------
        if args.use_mask:
            mask_path = osp.join(args.query_dir, f"{query_idx}_render_mask.mp4")
            if not osp.exists(mask_path):
                print(f"  Warning: mask video not found, ignoring mask: {mask_path}")
            else:
                mask_frames, _ = read_video(mask_path)
                # Mask may be stored as 3-channel grayscale; collapse to single channel
                if mask_frames[0].ndim == 3:
                    mask_frames = [f.mean(axis=-1, keepdims=True) for f in mask_frames]
                base_frame = ref_frames[0]  # pre-contact background from reference
                transferred = [
                    (mask_frames[i] if i < len(mask_frames) else mask_frames[-1]) * frame
                    + (1.0 - (mask_frames[i] if i < len(mask_frames) else mask_frames[-1])) * base_frame
                    for i, frame in enumerate(transferred)
                ]

        # -- NNF figure -------------------------------------------------------
        if not args.no_nnf_figures:
            fig_path = make_nnf_figure(
                query_idx=query_idx, ref_idx=ref_idx,
                query_dir=args.query_dir, ref_dir=args.ref_dir,
                modalities=args.modality,
                # The NNF now lives in the video's coordinate space regardless of
                # --match_scale, so the figure must show the --video_scale statics.
                scale=args.video_scale,
                nnf=nnf, ref_shape=ref_static.shape,
                save_dir=args.save_dir,
            )
            print(f"  Saved NNF figure: {fig_path}")

        # -- Match figure (linear stage's raw sparse matches, inliers/outliers) --
        if args.save_match_figures:
            try:
                _, _, q_match, r_match, _ = load_static_pairs(
                    args.query_dir, args.ref_dir, query_idx, ref_idx, args.modality,
                    args.video_scale, args.match_scale, args.match_scale_convention)
            except ValueError as e:
                print(f"  Skipping match figure (same load failure that triggered the "
                      f"identity fallback above): {e}")
            else:
                match_fig_path, match_status = make_match_figure(
                    query_idx=query_idx, ref_idx=ref_idx,
                    q=q_match, r=r_match, matcher=args.matcher,
                    dinov3_model=args.dinov3_model, dinov3_weights=args.dinov3_weights,
                    num_points=args.dinov3_num_points, stratify_threshold=args.dinov3_stratify_threshold,
                    reproj_threshold=args.reproj_threshold, transform_type=args.transform_type,
                    save_dir=args.save_dir,
                )
                print(f"  Saved match figure ({match_status}): {match_fig_path}")

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

    # ------------------------------------------------------------------
    # Save decomposition diagnostics
    # ------------------------------------------------------------------
    # How often the offset had to be zeroed is a direct read on retrieval
    # quality: it counts pairs whose contact points are further apart than the
    # sensor footprint, i.e. pairs with no overlap for any offset to find.
    if all_info:
        info_pkl_path = osp.join(args.save_dir, "decomposition.pkl")
        with open(info_pkl_path, "wb") as f:
            pickle.dump(all_info, f)
        valid = [i["valid_fraction"] for i in all_info.values()]
        print(f"\nDecomposition diagnostics saved to: {info_pkl_path}")
        print(f"Offset zeroed on {n_offset_zeroed}/{len(all_info)} queries "
              f"({100.0 * n_offset_zeroed / len(all_info):.1f}%) | "
              f"mean in-bounds NNF fraction {100.0 * sum(valid) / len(valid):.1f}%")

    print(f"\nDone. Transferred videos saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
