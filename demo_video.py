import argparse
import numpy as np
from PIL import Image
import time
import os
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PatchMatchCuda import PatchMatch
from PatchMatchCuda_single import PatchMatchSingle
from tqdm import trange
import cv2


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
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for frame in frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_a", default=None, type=str)
    parser.add_argument("--vid_b", default=None, type=str)
    parser.add_argument("--img_a", default=None, type=str, help="Static image to use instead of vid_a[0] for NNF computation")
    parser.add_argument("--img_b", default=None, type=str, help="Static image to use instead of vid_b[0] for NNF computation")
    parser.add_argument("--vid_a_prime", required=True, type=str)
    parser.add_argument("--vid_b_prime", required=True, type=str)
    parser.add_argument("--vid_mask", default=None, type=str, help="Mask of objects(b_prime) in contact with gel")
    parser.add_argument("--save_dir", default="./log/result", type=str)
    parser.add_argument("--pm_ver", default="double", help="Type of patchmatch algorithm to use", type=str)
    args = parser.parse_args()

    if args.vid_a is None and args.img_a is None:
        parser.error("--img_a is required when --vid_a is not provided")
    if args.vid_b is None and args.img_b is None:
        parser.error("--img_b is required when --vid_b is not provided")

    print("Loading videos...")
    vid_a = read_video(args.vid_a)[0] if args.vid_a is not None else None
    vid_b = read_video(args.vid_b)[0] if args.vid_b is not None else None
    vid_a_prime, fps = read_video(args.vid_a_prime)
    vid_b_prime_gt, _ = read_video(args.vid_b_prime)
    if args.vid_mask is not None:
        vid_mask, _ = read_video(args.vid_mask)

    if vid_a is not None:
        assert len(vid_a) == len(vid_a_prime), \
            "vid_a and vid_a_prime must have the same number of frames."
    if vid_b is not None:
        assert len(vid_b) == len(vid_a_prime), \
            "vid_b and vid_a_prime must have the same number of frames."

    reconstructed_frames = []
    base_frame = vid_a_prime[0]     # base_frame -> gelsight background image

    for i in trange(len(vid_a_prime)):
        print(f"Processing frame {i+1}/{len(vid_a_prime)}")

        if i == 0:  # Find PatchMatch only for initial frame
            def _load_img(path):
                bgr = cv2.imread(path)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            img = _load_img(args.img_a) if args.img_a is not None else vid_a[i]
            ref = _load_img(args.img_b) if args.img_b is not None else vid_b[i]

            # Initialize patchmatch
            if args.pm_ver == "double":
                pm = PatchMatch(ref, ref, img, img, patch_size=3)  # Finds a mapping f from ref -> img: nearest pixel in img for each pixel in ref
            elif args.pm_ver == "single":
                pm = PatchMatchSingle(ref, img, patch_size=3)  # Finds a mapping f from ref -> img: nearest pixel in img for each pixel in ref
            else:
                raise NotImplementedError("Other PatchMatch versions not supported")

            # Find NNF
            max_radius = max(img.shape)  # Set maximum random search radius as image size
            pm.propagate(iters=10, rand_search_radius=max_radius)

        img_prime = vid_a_prime[i]
        ref_prime = pm.reconstruct_avg(img_prime, patch_size=1)  # Uses f and reads off from img_prime to create ref_prime

        if args.vid_mask is not None:
            mask = vid_mask[i]
            final_frame = (mask * ref_prime) + ((1.0 - mask) * base_frame)
        else:
            final_frame = ref_prime

        reconstructed_frames.append(final_frame)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Save original videos
    print("Saving input videos...")
    if vid_a is not None:
        write_video(os.path.join(args.save_dir, "vid_a.mp4"), vid_a, fps)
    if vid_b is not None:
        write_video(os.path.join(args.save_dir, "vid_b.mp4"), vid_b, fps)
    write_video(os.path.join(args.save_dir, "vid_a_prime.mp4"), vid_a_prime, fps)
    write_video(os.path.join(args.save_dir, "vid_b_prime_gt.mp4"), vid_b_prime_gt, fps)

    # Save reconstructed video
    print("Saving reconstructed video...")
    write_video(os.path.join(args.save_dir, "vid_b_prime.mp4"), reconstructed_frames, fps)

    print("All videos saved under:", args.save_dir)
