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
    parser.add_argument("--vid_a", required=True, type=str)
    parser.add_argument("--vid_b", required=True, type=str)
    parser.add_argument("--vid_a_prime", required=True, type=str)
    parser.add_argument("--vid_b_prime", required=True, type=str)
    parser.add_argument("--save_dir", default="./log/result", type=str)
    parser.add_argument("--pm_ver", default="double", help="Type of patchmatch algorithm to use", type=str)
    args = parser.parse_args()

    print("Loading videos...")
    vid_a, fps = read_video(args.vid_a)
    vid_b, _ = read_video(args.vid_b)
    vid_a_prime, _ = read_video(args.vid_a_prime)
    vid_b_prime_gt, _ = read_video(args.vid_b_prime)

    assert len(vid_a) == len(vid_b) == len(vid_a_prime), \
        "All input videos must have the same number of frames."

    reconstructed_frames = []

    for i in trange(len(vid_a)):
        print(f"Processing frame {i+1}/{len(vid_a)}")

        if i == 0:  # Find PatchMatch only for initial frame
            img = vid_a[i]
            ref = vid_b[i]

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

        reconstructed_frames.append(ref_prime)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Save original videos
    print("Saving input videos...")
    write_video(os.path.join(args.save_dir, "vid_a.mp4"), vid_a, fps)
    write_video(os.path.join(args.save_dir, "vid_b.mp4"), vid_b, fps)
    write_video(os.path.join(args.save_dir, "vid_a_prime.mp4"), vid_a_prime, fps)
    write_video(os.path.join(args.save_dir, "vid_b_prime_gt.mp4"), vid_b_prime_gt, fps)

    # Save reconstructed video
    print("Saving reconstructed video...")
    write_video(os.path.join(args.save_dir, "vid_b_prime.mp4"), reconstructed_frames, fps)

    print("All videos saved under:", args.save_dir)
