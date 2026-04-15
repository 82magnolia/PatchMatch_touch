"""
Retrieval-based touch video transfer using GPU PatchMatch.

For each query contact point, loads the top-1 retrieved reference from a
results.pkl (produced by retrieve_touch.py), computes a Nearest-Neighbor
Field (NNF) between their static modality images, and uses the NNF to warp
every frame of the query's touch video into the reference's coordinate layout.

Example usage:
    python main_retrieval_transfer.py \
        --query_dir Taxim/results/gen_contact \
        --ref_dir   Taxim/results/gen_contact \
        --retrieval_pkl log/touch_retrieval/results.pkl \
        --modality normal \
        --scale 25 \
        --video_type shadow \
        --save_dir log/transfer
"""

import argparse
import os
import pickle
from os import path as osp

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit  # noqa: F401 – initialises CUDA context
from tqdm import tqdm

import torch
import lpips
from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from PatchMatchCuda_single import PatchMatchSingle
from retrieve_touch import discover_files


# ---------------------------------------------------------------------------
# Video I/O (mirrored from demo_video.py)
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
        fname = f"{idx}_scale{scale}_{modality}.jpg"
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
        fname = f"{idx}_scale{scale}_{base}.npz"
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


def _make_nnf_warped(pm, ref_color_grid):
    """Warp a canonical HSV position grid through the NNF.

    Reveals which reference regions are sampled at each query location.
    """
    return pm.reconstruct_avg(ref_color_grid, patch_size=1)


def make_nnf_figure(query_idx, ref_idx, query_dir, ref_dir, modalities, scale,
                    pm, ref_shape, save_dir):
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
    nnf_imgs   = [_make_nnf_warped(pm, ref_color_grid), ref_color_grid]
    nnf_titles = ["NNF warped", "ref color grid"]
    for col, (img, title) in enumerate(zip(nnf_imgs, nnf_titles)):
        ax = axes[M, col]
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        if col == 0:
            ax.text(-0.05, 0.5, "NNF", transform=ax.transAxes,
                    ha="right", va="center", rotation=90, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Query #{query_idx} → Ref #{ref_idx} — NNF ({', '.join(modalities)})",
                 fontsize=10)
    plt.tight_layout()

    out_path = osp.join(save_dir, f"{query_idx}_nnf.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Contact mask
# ---------------------------------------------------------------------------

def compute_contact_mask(ref_frame, base_frame, threshold,
                         blur_sigma=3.0, morph_radius=5):
    """Robust binary mask of pixels where contact has occurred.

    Pipeline:
      1. Compute per-pixel L2 difference magnitude.
      2. Gaussian blur to suppress JPEG block artifacts.
      3. Threshold the blurred magnitude.
      4. Morphological open  (removes isolated noise blobs).
      5. Morphological close (fills holes inside the contact region).

    Returns float32 (H, W, 1).
    """
    diff = np.abs(ref_frame - base_frame)
    magnitude = np.linalg.norm(diff, axis=-1).astype(np.float32)  # (H, W)
    blurred = cv2.GaussianBlur(magnitude, (0, 0), blur_sigma)
    binary = (blurred > threshold).astype(np.uint8)
    k = morph_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary[..., np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# EM-style transfer
# ---------------------------------------------------------------------------

def em_transfer_frame(query_static, ref_static, ref_frame, init_estimate,
                      em_iters, patch_size, pm_iters,
                      ref_contact_mask=None, ref_static_mask=None,
                      init_nnf=None, use_accel=False):
    """EM-style single-frame transfer.

    Iteratively refines correspondences by combining static + current-touch
    channels into the NNF computation.

    E-step: NNF via PatchMatchSingle(concat(query_static, estimate),
                                     concat(ref_static,   ref_frame))
    M-step: estimate = reconstruct_avg(ref_frame) → output in query space

    Returns (transferred_frame, final_pm).
    """
    estimate = init_estimate.copy()
    init_orig = init_estimate  # fixed reference for non-contact regions
    max_radius = int(max(query_static.shape[:2]))
    pm = None
    current_nnf = init_nnf  # <-- Start with the passed NNF (None if it's the first)
    for em_step in range(em_iters):
        query_combined = np.concatenate([query_static, estimate], axis=-1).copy(order="C")
        ref_combined   = np.concatenate([ref_static,   ref_frame], axis=-1).copy(order="C")
        
        if current_nnf is None or not use_accel:
            current_iters = pm_iters
            current_radius = max_radius
            pm = PatchMatchSingle(query_combined, ref_combined, patch_size=patch_size, init_nnf=None)
        else:
            # If NNF already exists (from a previous frame or previous EM step), perform only fine-tuning
            current_iters = max(1, pm_iters // 3)  # Reduce iterations to 1/3
            current_radius = max(5, int(max_radius * 0.1)) # Reduce search radius to 10%
            pm = PatchMatchSingle(query_combined, ref_combined, patch_size=patch_size, init_nnf=current_nnf) # Pass the initial NNF

        pm.propagate(iters=current_iters, rand_search_radius=current_radius)
        
        # Save the updated NNF for the next EM step
        if use_accel:
            current_nnf = pm.nnf
        else:
            current_nnf = None

        estimate = pm.reconstruct_avg(ref_frame, patch_size=1)
        valid = np.ones((estimate.shape[0], estimate.shape[1], 1), dtype=np.float32)
        if ref_contact_mask is not None:
            valid = valid * ref_contact_mask
        if ref_static_mask is not None:
            valid = valid * ref_static_mask
        # valid is in reference coordinates; warp to query coordinates via NNF
        valid_q = (np.atleast_3d(pm.reconstruct_avg(valid, patch_size=1))[..., :1] > 0.5).astype(np.float32)
        estimate = valid_q * estimate + (1.0 - valid_q) * init_orig
    return estimate, pm, valid_q

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
        mse_sum += compute_mse(gt, pred)
        psnr_sum += compute_psnr(gt, pred, data_range=1.0)
        
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
        description="Transfer query touch videos to reference layout via PatchMatch NNF."
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
                        help="Static modality(ies) used to compute the NNF. "
                             "Multiple modalities are channel-concatenated. "
                             "Prefix with 'raw_' to load raw .npz values instead "
                             "of color-coded JPGs (e.g. raw_normal, raw_height).")
    parser.add_argument("--video_type", required=True,
                        choices=["shadow", "sim"],
                        help="Touch video variant to transfer.")
    parser.add_argument("--scale", default=None, type=int,
                        help="Scale suffix in mm for static images (e.g. 25). "
                             "Omit to use base-resolution files.")
    parser.add_argument("--use_mask", action="store_true",
                        help="Composite transferred frames with the query mask video. "
                             "Requires {idx}_mask.mp4 in --query_dir.")
    parser.add_argument("--save_dir", default="./log/transfer", type=str,
                        help="Output directory for transferred videos.")
    parser.add_argument("--patch_size", default=3, type=int,
                        help="PatchMatch patch size (default: 3).")
    parser.add_argument("--iters", default=10, type=int,
                        help="PatchMatch propagation iterations (default: 10).")
    parser.add_argument("--em", action="store_true",
                        help="Use EM-style iterative synthesis (combines static + "
                             "current touch estimate in each NNF computation).")
    parser.add_argument("--em_iters", default=3, type=int,
                        help="Number of EM iterations per frame (default: 3). "
                             "Only used when --em is set.")
    parser.add_argument("--use_ref_contact_mask", action="store_true",
                        help="Gate reconstruction to contact regions detected in the "
                             "reference video. Non-contact pixels are kept as the "
                             "pre-contact base frame.")
    parser.add_argument("--ref_contact_threshold", default=0.05, type=float,
                        help="Threshold on blurred ||ref_frame - base_frame|| to define "
                             "the reference contact mask (default: 0.05). Only used when "
                             "--use_ref_contact_mask is set.")
    parser.add_argument("--ref_contact_blur_sigma", default=3.0, type=float,
                        help="Gaussian blur sigma applied to the difference magnitude "
                             "before thresholding, suppressing JPEG block artifacts "
                             "(default: 3.0). Only used when --use_ref_contact_mask is set.")
    parser.add_argument("--ref_contact_morph_radius", default=5, type=int,
                        help="Radius of the elliptical structuring element used for "
                             "morphological open+close on the contact mask (default: 5). "
                             "Only used when --use_ref_contact_mask is set.")
    parser.add_argument("--use_ref_static_mask", action="store_true",
                        help="Prevent overwriting estimates for pixels where ref_static "
                             "is zero (e.g. background / invalid regions).")
    parser.add_argument("--use_accel", action="store_true",
                        help="Enable acceleration mode (inter-frame NNF transfer)")
    parser.add_argument("--em_iters_subseq", default=-1, type=int,
                        help="Number of reduced EM iterations for subsequent frames (default: 10). "
                             "Only used when --em is set.")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation mode (calculate metrics against GT video)")
    parser.add_argument("--use_keyframe", action="store_true",
                             help="Use keyframe to propagate the NNF forwards and backwards from it")
    parser.add_argument("--use_video_concate", default=None, type=int,
                             help="concatenate video")
    args = parser.parse_args()

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
    # Build a lookup: ref_idx -> path for each modality (for quick access)
    # ------------------------------------------------------------------
    ref_idx_to_path = {}
    for mod in args.modality:
        entries = discover_files(args.ref_dir, mod, args.scale)
        ref_idx_to_path[mod] = {idx: p for idx, p in entries}

    # ------------------------------------------------------------------
    # Process each query
    # ------------------------------------------------------------------
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

        # -- Compute NNF (standard) or defer to EM loop below ----------------
        if not args.em:
            max_radius = int(max(query_static.shape[:2]))
            pm = PatchMatchSingle(query_static, ref_static, patch_size=args.patch_size)
            pm.propagate(iters=args.iters, rand_search_radius=max_radius)
            fig_pm = pm

        # -- Load reference touch video ------------------------------------
        vid_path = osp.join(args.ref_dir, f"{ref_idx}_{args.video_type}.mp4")
        if not osp.exists(vid_path):
            print(f"  Skipping (missing touch video): {vid_path}")
            continue
        ref_frames, fps = read_video(vid_path)

        # -- Optionally load mask video (from query dir) -------------------
        mask_frames = None
        if args.use_mask:
            mask_path = osp.join(args.query_dir, f"{query_idx}_render_mask.mp4")  # NOTE: Render mask is obtained from heightmap thresholding
            if not osp.exists(mask_path):
                print(f"  Warning: mask video not found, ignoring mask: {mask_path}")
            else:
                mask_frames, _ = read_video(mask_path)
                # Mask may be stored as 3-channel grayscale; collapse to single channel
                if mask_frames[0].ndim == 3:
                    mask_frames = [f.mean(axis=-1, keepdims=True) for f in mask_frames]

        base_frame = ref_frames[0]  # pre-contact background from reference

        # Mask out ref_static zero regions (invalid / background pixels)
        ref_static_mask = None
        if args.use_ref_static_mask:
            ref_static_mask = (ref_static != 0).any(axis=-1, keepdims=True).astype(np.float32)

        # -- Transfer each frame ------------------------------------------
        transferred = []
        ref_contact_masks = []
        valid_q_frames = []
        if args.use_keyframe:
            assert args.em  # Keyframes are only used for EM
            fig_pm = None
            transferred = [None] * len(ref_frames)
            valid_q_frames = [None] * len(ref_frames)
            ref_contact_masks = [None] * len(ref_frames)

            # Pre-compute contact masks & find the Anchor frame
            deformation_scores = []
            for i, ref_frame in enumerate(ref_frames):
                if args.use_ref_contact_mask:
                    mask = compute_contact_mask(
                        ref_frame, base_frame, args.ref_contact_threshold,
                        args.ref_contact_blur_sigma, args.ref_contact_morph_radius)
                    ref_contact_masks[i] = mask
                    deformation_scores.append(np.sum(mask))
                else:
                    # Fallback metric: L1 difference if contact mask is not used
                    diff = np.abs(ref_frame - base_frame)
                    deformation_scores.append(np.sum(diff))
                    
            anchor_idx = int(np.argmax(deformation_scores))
            print(f"  [Info] Anchor frame selected: {anchor_idx}/{len(ref_frames)-1} (Max deformation)")

            # Helper function to process a single frame inside the loop
            def process_frame(i, current_em_iters, pass_nnf, init_frame):
                output, last_pm, valid_q = em_transfer_frame(
                    query_static, ref_static, ref_frames[i], init_frame,
                    current_em_iters, args.patch_size, args.iters,
                    ref_contact_mask=ref_contact_masks[i],
                    ref_static_mask=ref_static_mask,
                    init_nnf=pass_nnf,
                    use_accel=args.use_accel)
                
                # Apply render mask if provided
                if mask_frames is not None:
                    mask = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
                    output = mask * output + (1.0 - mask) * base_frame
                    
                transferred[i] = output
                valid_q_frames[i] = valid_q
                return last_pm

            # Process Anchor Frame (Full Search)
            anchor_pm = process_frame(
                i=anchor_idx, 
                current_em_iters=args.em_iters, 
                pass_nnf=None, 
                init_frame=base_frame)
            
            fig_pm = anchor_pm  # Save best NNF for the visualization figure

            # Forward Propagation (Anchor -> End)
            prev_nnf = anchor_pm.nnf
            if args.em_iters_subseq == -1:
                current_em_iters = args.em_iters
            else:
                current_em_iters = args.em_iters_subseq
            for i in tqdm(range(anchor_idx + 1, len(ref_frames)), desc="Forward Pass", leave=False):
                pass_nnf = prev_nnf if args.use_accel else None
                last_pm = process_frame(
                    i=i, 
                    current_em_iters=current_em_iters, 
                    pass_nnf=pass_nnf, 
                    init_frame=transferred[i - 1])
                prev_nnf = last_pm.nnf

            # Backward Propagation (Anchor -> Start)
            prev_nnf = anchor_pm.nnf
            for i in tqdm(range(anchor_idx - 1, -1, -1), desc="Backward Pass", leave=False):
                pass_nnf = prev_nnf if args.use_accel else None
                # init_frame uses the temporally adjacent frame we just processed (i + 1)
                last_pm = process_frame(
                    i=i, 
                    current_em_iters=current_em_iters, 
                    pass_nnf=pass_nnf, 
                    init_frame=transferred[i + 1])
                prev_nnf = last_pm.nnf
        elif args.use_video_concate:
            fig_pm = None
            prev_nnf = None
            
            # If -1, treat the entire video as a single chunk. If positive, set as the number of frames per chunk (n)
            chunk_size = len(ref_frames) if args.use_video_concate == -1 else args.use_video_concate
            
            # Total number of chunks required to group exactly by chunk_size
            num_offsets = (len(ref_frames) + chunk_size - 1) // chunk_size

            # Pre-allocate arrays since we are processing frames out of temporal order
            transferred = [None] * len(ref_frames)
            valid_q_frames = [None] * len(ref_frames)
            ref_contact_masks = [None] * len(ref_frames)
            
            for offset in tqdm(range(num_offsets), desc="Transferring interleaved chunks", leave=False):
                # Generate interleaved indices (e.g., if stride=5 -> [0, 5, 10, 15...], [1, 6, 11, 16...])
                chunk_indices = list(range(offset, len(ref_frames), num_offsets))
                if not chunk_indices:
                    continue
                
                chunk_frames = [ref_frames[idx] for idx in chunk_indices]
                
                # Calculate NNF (using the first frame of the current interleaved chunk)
                first_idx = chunk_indices[0]
                current_ref_frame = chunk_frames[0]
                
                # Use the previously transferred frame as init (temporally adjacent)
                init = ref_frames[0] if first_idx == 0 else transferred[first_idx - 1]
                
                ref_contact_mask = None
                if args.use_ref_contact_mask:
                    ref_contact_mask = compute_contact_mask(
                        current_ref_frame, base_frame, args.ref_contact_threshold,
                        args.ref_contact_blur_sigma, args.ref_contact_morph_radius)
                
                # Determine EM iterations (full iterations for the first offset, reduced for subsequent)
                if args.em_iters_subseq == -1 or offset == 0:
                    current_em_iters = args.em_iters
                else:
                    current_em_iters = args.em_iters_subseq
                
                pass_nnf = prev_nnf if args.use_accel else None
                
                if args.em:
                    # Generate the NNF map for the current chunk
                    _, pm, _ = em_transfer_frame(
                        query_static, ref_static, current_ref_frame, init,
                        current_em_iters, args.patch_size, args.iters,
                        ref_contact_mask=ref_contact_mask,
                        ref_static_mask=ref_static_mask,
                        init_nnf=pass_nnf,
                        use_accel=args.use_accel)
                
                if offset == 0:
                    fig_pm = pm
                prev_nnf = pm.nnf
                
                # Concatenate frames -> Shape: (H, W, 3 * C)
                concat_ref = np.concatenate(chunk_frames, axis=-1) 
                
                chunk_valids = []
                for idx in chunk_indices:
                    frame = ref_frames[idx]
                    valid = np.ones((frame.shape[0], frame.shape[1], 1), dtype=np.float32)
                    
                    if args.use_ref_contact_mask:
                        c_mask = compute_contact_mask(
                            frame, base_frame, args.ref_contact_threshold,
                            args.ref_contact_blur_sigma, args.ref_contact_morph_radius)
                        ref_contact_masks[idx] = c_mask
                        valid = valid * c_mask
                        
                    if ref_static_mask is not None:
                        valid = valid * ref_static_mask
                        
                    chunk_valids.append(valid)
                
                # Concatenate masks -> Shape: (H, W, 1 * C)
                concat_valid = np.concatenate(chunk_valids, axis=-1)
                
                # Reconstruct entirely via PatchMatch
                concat_out = pm.reconstruct_avg(concat_ref, patch_size=1)
                concat_valid_q = pm.reconstruct_avg(concat_valid, patch_size=1)
                
                # Split back and insert into the correct temporal positions
                out_frames = np.split(concat_out, len(chunk_frames), axis=-1)
                valid_q_frames_chunk = np.split(concat_valid_q, len(chunk_frames), axis=-1)
                
                for j, (out_f, valid_q_f) in enumerate(zip(out_frames, valid_q_frames_chunk)):
                    global_i = chunk_indices[j]
                    
                    valid_q = (np.atleast_3d(valid_q_f) > 0.5).astype(np.float32)
                    valid_q_frames[global_i] = valid_q
                    
                    # Composite with the query background
                    final_out = valid_q * out_f + (1.0 - valid_q) * base_frame
                    
                    # Composite with the mask video if provided
                    if mask_frames is not None:
                        mask = mask_frames[global_i] if global_i < len(mask_frames) else mask_frames[-1]
                        final_out = mask * final_out + (1.0 - mask) * base_frame
                        
                    # Place the transferred frame in its correct chronological spot
                    transferred[global_i] = final_out
        else:
            if args.em:
                fig_pm = None
                prev_nnf = None  # <-- Add a variable to pass NNF between frames
                for i, ref_frame in enumerate(tqdm(ref_frames, desc="Transferring frames", leave=False)):
                    init = ref_frames[0] if i == 0 else transferred[-1]
                    ref_contact_mask = None
                    if args.use_ref_contact_mask:
                        ref_contact_mask = compute_contact_mask(
                            ref_frame, base_frame, args.ref_contact_threshold,
                            args.ref_contact_blur_sigma, args.ref_contact_morph_radius)
                        ref_contact_masks.append(ref_contact_mask)
                    
                    # Full EM iterations for the first frame, fewer for subsequent frames
                    if args.em_iters_subseq == -1 or i == 0:
                        current_em_iters = args.em_iters
                    else:
                        current_em_iters = args.em_iters_subseq
                    # Pass prev_nnf
                    if args.use_accel is False:
                        prev_nnf = None
                    output, last_pm, valid_q = em_transfer_frame(
                        query_static, ref_static, ref_frame, init,
                        current_em_iters, args.patch_size, args.iters,
                        ref_contact_mask=ref_contact_mask,
                        ref_static_mask=ref_static_mask,
                        init_nnf=prev_nnf,
                        use_accel=args.use_accel)      # <-- Passed here
                    prev_nnf = last_pm.nnf
                    if valid_q is not None:
                        valid_q_frames.append(valid_q)
                    if i == 0:
                        fig_pm = last_pm  # pm from frame-0 final EM iter for figure
                    if mask_frames is not None:
                        mask = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
                        output = mask * output + (1.0 - mask) * base_frame
                    transferred.append(output)
            else:
                for i, frame in enumerate(tqdm(ref_frames, desc="Transferring frames", leave=False)):
                    output = pm.reconstruct_avg(frame, patch_size=1)
                    valid = np.ones((output.shape[0], output.shape[1], 1), dtype=np.float32)
                    if args.use_ref_contact_mask:
                        ref_contact_mask = compute_contact_mask(frame, base_frame,
                                                                args.ref_contact_threshold)
                        ref_contact_masks.append(ref_contact_mask)
                        valid = valid * ref_contact_mask
                    if ref_static_mask is not None:
                        valid = valid * ref_static_mask
                    # valid is in reference coordinates; warp to query coordinates via NNF
                    valid_q = (np.atleast_3d(pm.reconstruct_avg(valid, patch_size=1))[..., :1] > 0.5).astype(np.float32)
                    valid_q_frames.append(valid_q)
                    output = valid_q * output + (1.0 - valid_q) * base_frame
                    if mask_frames is not None:
                        mask = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
                        output = mask * output + (1.0 - mask) * base_frame
                    transferred.append(output)

        # -- NNF figure -------------------------------------------------------
        fig_path = make_nnf_figure(
            query_idx=query_idx, ref_idx=ref_idx,
            query_dir=args.query_dir, ref_dir=args.ref_dir,
            modalities=args.modality, scale=args.scale,
            pm=fig_pm, ref_shape=ref_static.shape,
            save_dir=args.save_dir,
        )
        print(f"  Saved NNF figure: {fig_path}")

        # -- Save individual keyframe images ----------------------------------
        # Ensure anchor_idx exists
        if args.use_keyframe:
            def save_rgb_image(img_array, save_path):
                """Helper to save a [0, 1] RGB float32 array to an image file."""
                # Clip to [0, 1] safely, convert to uint8, and swap RGB to BGR for cv2
                img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(save_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # Transferred image at key frame
            save_rgb_image(transferred[anchor_idx], 
                        osp.join(args.save_dir, f"{query_idx}_keyframe_transferred.png"))

            # Original reference touch image at key frame
            save_rgb_image(ref_frames[anchor_idx], 
                        osp.join(args.save_dir, f"{query_idx}_keyframe_ref_touch.png"))      

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
        suffix = "_em" if args.em else ""
        out_path = osp.join(args.save_dir, f"{query_idx}_transferred{suffix}.mp4")
        write_video(out_path, transferred, fps)
        print(f"  Saved: {out_path}")

        # valid_q video (query-space composite mask)
        if valid_q_frames:
            write_video(osp.join(args.save_dir, f"{query_idx}_valid_q.mp4"),
                        [np.repeat(m, 3, axis=-1) for m in valid_q_frames], fps)

        # Reference contact mask video
        if ref_contact_masks:
            mask_vid = [np.repeat(m, 3, axis=-1) for m in ref_contact_masks]
            write_video(osp.join(args.save_dir, f"{query_idx}_ref_contact_mask.mp4"),
                        mask_vid, fps)
            
        # Qutive Evaluation Logic
        if args.eval:
            q_vid_path = osp.join(args.query_dir, f"{query_idx}_{args.video_type}.mp4")
            if not osp.exists(q_vid_path):
                print(f"  [Eval] Query(GT) video not found for eval: {q_vid_path}")
                continue
            
            # Load GT video (Query)
            q_frames, _ = read_video(q_vid_path)
            
            # Evaluate the result from the accelerated code (transferred)
            metrics_accel = evaluate_video_metrics(q_frames, transferred, loss_fn_vgg, device)
            
            print("\n  [Evaluation Results]")
            print("  ---------------------------------------------------------")
            print(f"  [Accel Code]  MSE: {metrics_accel['MSE']:.5f} | PSNR: {metrics_accel['PSNR']:.2f} | "
                  f"SSIM: {metrics_accel['SSIM']:.4f} | LPIPS: {metrics_accel['LPIPS']:.4f}")

            print("MSE\tPSNR\tSSIM\tLPIPS")
            print(f"{metrics_accel['MSE']:.5f}\t{metrics_accel['PSNR']:.2f}\t{metrics_accel['SSIM']:.4f}\t{metrics_accel['LPIPS']:.4f}\n")

    print(f"\nDone. Transferred videos saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
