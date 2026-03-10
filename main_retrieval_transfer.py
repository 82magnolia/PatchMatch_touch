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
import numpy as np
import pycuda.autoinit  # noqa: F401 – initialises CUDA context
from tqdm import tqdm

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
    """Load a static modality image as float32 RGB in [0, 1]."""
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


def build_combined_static(folder, idx, modalities, scale):
    """Load and channel-concatenate one or more static modality images."""
    imgs = [load_static_image(folder, idx, mod, scale) for mod in modalities]
    if len(imgs) == 1:
        return imgs[0]
    return np.concatenate(imgs, axis=-1).copy(order="C")


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
                        choices=["color", "normal", "curvature", "height"],
                        help="Static modality(ies) used to compute the NNF. "
                             "Multiple modalities are channel-concatenated.")
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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

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

        # -- Compute NNF: maps query coords → ref coords -------------------
        max_radius = int(max(query_static.shape[:2]))
        pm = PatchMatchSingle(query_static, ref_static, patch_size=args.patch_size)  # Finds a mapping f from ref -> img: nearest pixel in img for each pixel in ref
        pm.propagate(iters=args.iters, rand_search_radius=max_radius)

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

        # -- Transfer each frame ------------------------------------------
        transferred = []
        for i, frame in enumerate(ref_frames):
            output = pm.reconstruct_avg(frame, patch_size=1)

            if mask_frames is not None:
                mask = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
                output = mask * output + (1.0 - mask) * base_frame

            transferred.append(output)

        # -- Save output video --------------------------------------------
        out_path = osp.join(args.save_dir, f"{query_idx}_transferred.mp4")
        write_video(out_path, transferred, fps)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Transferred videos saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
