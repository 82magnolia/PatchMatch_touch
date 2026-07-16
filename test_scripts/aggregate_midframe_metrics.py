"""Aggregates MSE/PSNR/SSIM/LPIPS across all sessions under a
transfer_pipeline.py output directory, using only each touch video's middle
frame (temporal index min(len(query_gt), len(transferred)) // 2) instead of
averaging over every frame like metrics.pkl (main_retrieval_transfer_feat_match.py's
evaluate_video_metrics) does -- so results aren't recoverable from the
existing metrics.pkl files and must be recomputed directly from the videos.

Seeks directly to the middle/first frame via cv2.VideoCapture's
CAP_PROP_POS_FRAMES instead of decoding every frame, since only two frames
per video (mid, and frame 0 when --masked) are needed.

--masked restricts all four metrics to the contact region only, via
compute_contact_mask (copied here rather than imported from
main_retrieval_transfer_accel.py, since that module's import chain triggers
pycuda.autoinit + a compiled PatchMatchCuda_single CUDA extension load --
heavy and irrelevant side effects for what's otherwise a pure numpy/cv2
utility function). The mask is built from the query ground-truth video
(mid frame vs. its own frame 0, i.e. pre-contact background) and applied to
both the ground-truth and transferred mid frames, since the transferred
frame already lives in the query's coordinate grid (per compute_dinov3_nnf's
NNF convention). SSIM is masked via skimage's full=True per-pixel map;
LPIPS (not natively pixel-maskable with this repo's non-spatial lpips.LPIPS)
is approximated by zeroing the non-contact background in both frames before
scoring -- a common practical masked-LPIPS approximation.

Example usage:
    python test_scripts/aggregate_midframe_metrics.py \
        --dir log/transfer_pipeline_real_data_gt_retrieval_sift_lightglue \
        --masked
"""

import argparse
import os
import re
from os import path as osp

import cv2
import numpy as np
import torch
import lpips

TRANSFERRED_RE = re.compile(r"^(\d+)_transferred\.mp4$")
METRIC_KEYS = ["MSE", "PSNR", "SSIM", "LPIPS"]

from skimage.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def compute_contact_mask(ref_frame, base_frame, threshold,
                         blur_sigma=3.0, morph_radius=5):
    """Robust binary mask of pixels where contact has occurred.

    Copied verbatim from main_retrieval_transfer_accel.py (see that file for
    the canonical version) -- pipeline: L2 diff magnitude -> Gaussian blur ->
    threshold -> morphological open (denoise) -> morphological close (fill
    holes). Returns float32 (H, W, 1).
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


def read_frame_at(path, idx):
    """Seek directly to frame idx (float32 RGB in [0, 1]), or None if the
    video has fewer frames / fails to open."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        return None, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, count - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, count
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0, count



def frame_count(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def mid_frame_metrics(gt_path, pred_path, lpips_model, device,
                      masked=False, contact_threshold=0.05,
                      blur_sigma=3.0, morph_radius=5):
    n_gt = frame_count(gt_path)
    n_pred = frame_count(pred_path)
    n = min(n_gt, n_pred)
    if n <= 0:
        return None
    mid = n // 2

    gt, _ = read_frame_at(gt_path, mid)
    pred, _ = read_frame_at(pred_path, mid)
    if gt is None or pred is None:
        return None
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mask_frac = None
    if masked:
        gt_frame0, _ = read_frame_at(gt_path, 0)
        if gt_frame0 is None or gt_frame0.shape != gt.shape:
            return None
        mask = compute_contact_mask(gt, gt_frame0, contact_threshold, blur_sigma, morph_radius)
        mask_bool = mask[..., 0] > 0.5
        mask_frac = float(mask_bool.mean())
        if not mask_bool.any():
            return None  # no detected contact in the mid frame -- not a meaningful sample

        mse = float(((gt - pred) ** 2)[mask_bool].mean())
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
        _, ssim_map = compute_ssim(gt, pred, data_range=1.0, channel_axis=-1, full=True)
        ssim = float(ssim_map[mask_bool].mean())

        gt_masked = gt * mask
        pred_masked = pred * mask
        gt_t = torch.from_numpy(gt_masked).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        pred_t = torch.from_numpy(pred_masked).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    else:
        mse = compute_mse(gt, pred)
        psnr = compute_psnr(gt, pred, data_range=1.0) if mse > 0 else 100.0
        ssim = compute_ssim(gt, pred, data_range=1.0, channel_axis=-1)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

    with torch.no_grad():
        lpips_val = lpips_model(gt_t, pred_t).item()

    result = {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips_val}
    if mask_frac is not None:
        result["contact_frac"] = mask_frac
    return result


def find_cases(root):
    """(session_dir, query_idx) pairs discovered from */transfer/*_transferred.mp4."""
    cases = []
    for dirpath, _, filenames in os.walk(root):
        if osp.basename(dirpath) != "transfer":
            continue
        for fname in filenames:
            m = TRANSFERRED_RE.match(fname)
            if m:
                cases.append((dirpath, int(m.group(1))))
    return sorted(cases)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate MSE/PSNR/SSIM/LPIPS using only each video's middle frame.")
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--video_type", default="shadow", type=str)
    parser.add_argument("--masked", action="store_true",
                        help="Restrict all four metrics to the contact region only, via "
                             "compute_contact_mask on the query GT video's mid frame vs. its "
                             "own frame 0 (pre-contact background).")
    parser.add_argument("--contact_threshold", default=0.05, type=float,
                        help="compute_contact_mask threshold (default: 0.05, matching "
                             "main_retrieval_transfer_accel.py's --ref_contact_threshold default).")
    parser.add_argument("--blur_sigma", default=3.0, type=float)
    parser.add_argument("--morph_radius", default=5, type=int)
    args = parser.parse_args()

    cases = find_cases(args.dir)
    if not cases:
        print(f"No transferred videos found under: {args.dir}")
        return
    print(f"Found {len(cases)} touch location(s) under: {args.dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    all_metrics = []
    skipped_no_contact = 0
    for transfer_dir, query_idx in cases:
        gt_path = osp.join(transfer_dir, f"{query_idx}_query_{args.video_type}.mp4")
        pred_path = osp.join(transfer_dir, f"{query_idx}_transferred.mp4")
        if not (osp.exists(gt_path) and osp.exists(pred_path)):
            continue
        m = mid_frame_metrics(gt_path, pred_path, lpips_model, device,
                              masked=args.masked, contact_threshold=args.contact_threshold,
                              blur_sigma=args.blur_sigma, morph_radius=args.morph_radius)
        if m:
            all_metrics.append(m)
        elif args.masked:
            skipped_no_contact += 1

    if not all_metrics:
        print("No usable (query, transferred) video pairs found.")
        return

    n = len(all_metrics)
    avg = {k: sum(m[k] for m in all_metrics) / n for k in METRIC_KEYS}
    label = "Masked mid-frame-only" if args.masked else "Mid-frame-only"
    print(f"\n{'='*60}")
    print(f"{label} average over {n} touch locations")
    if args.masked:
        avg_frac = sum(m["contact_frac"] for m in all_metrics) / n
        print(f"(skipped {skipped_no_contact} with no detected contact region; "
             f"avg contact area = {avg_frac * 100:.1f}% of frame)")
    print(f"{'='*60}")
    print(f"  MSE  : {avg['MSE']:.5f}")
    print(f"  PSNR : {avg['PSNR']:.2f}")
    print(f"  SSIM : {avg['SSIM']:.4f}")
    print(f"  LPIPS: {avg['LPIPS']:.4f}")
    print(f"{'='*60}")
    print("\nTSV:")
    print("MSE\tPSNR\tSSIM\tLPIPS")
    print(f"{avg['MSE']:.5f}\t{avg['PSNR']:.2f}\t{avg['SSIM']:.4f}\t{avg['LPIPS']:.4f}")


if __name__ == "__main__":
    main()
