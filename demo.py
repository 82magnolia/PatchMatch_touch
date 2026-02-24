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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_a", required=True, type=str)
    parser.add_argument("--img_b", required=True, type=str)
    parser.add_argument("--img_a_prime", required=True, type=str)
    parser.add_argument("--img_b_prime", required=True, type=str)
    parser.add_argument("--save_dir", default="./log/result", type=str)
    parser.add_argument("--pm_ver", default="double", help="Type of patchmatch algorithm to use", type=str)
    args = parser.parse_args()

    img = (np.array(Image.open(args.img_a)) / 255).astype(np.float32)
    ref = (np.array(Image.open(args.img_b)) / 255).astype(np.float32)
    img_prime = (np.array(Image.open(args.img_a_prime)) / 255).astype(np.float32)
    ref_prime_gt = (np.array(Image.open(args.img_b_prime)) / 255).astype(np.float32)
    ref_prime = np.zeros_like(img_prime)

    max_radius = max(img.shape)  # Set maximum random search radius as image size

    # Initialize patchmatch
    if args.pm_ver == "double":
        pm = PatchMatch(ref, ref, img, img, patch_size=3)  # Finds a mapping f from ref -> img: nearest pixel in img for each pixel in ref
    elif args.pm_ver == "single":
        pm = PatchMatchSingle(ref, img, patch_size=3)  # Finds a mapping f from ref -> img: nearest pixel in img for each pixel in ref
    else:
        raise NotImplementedError("Other PatchMatch versions not supported")

    start = time.time()
    pm.propagate(iters=10, rand_search_radius=max_radius)
    end = time.time()
    print(end - start)
    ref_prime = pm.reconstruct_avg(img_prime, patch_size=1)  # Uses f and reads off from img_prime to create ref_prime

    # Un-normalize images
    img = (img * 255).astype(np.uint8)
    ref = (ref * 255).astype(np.uint8)
    img_prime = (img_prime * 255).astype(np.uint8)
    ref_prime = (ref_prime * 255).astype(np.uint8)
    ref_prime_gt = (ref_prime_gt * 255).astype(np.uint8)

    Image.fromarray(ref_prime).show()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    Image.fromarray(img).save(os.path.join(args.save_dir, "img_a.png"))
    Image.fromarray(ref).save(os.path.join(args.save_dir, "img_b.png"))
    Image.fromarray(img_prime).save(os.path.join(args.save_dir, "img_a_prime.png"))
    Image.fromarray(ref_prime).save(os.path.join(args.save_dir, "img_b_prime.png"))
    Image.fromarray(ref_prime_gt).save(os.path.join(args.save_dir, "img_b_prime_gt.png"))
