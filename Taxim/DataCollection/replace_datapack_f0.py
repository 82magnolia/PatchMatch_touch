"""
Replace the background frame (f0) in a Taxim dataPack.npz with a blank frame
captured from a real GelSight sensor, keeping all other keys unchanged.
"""
import argparse
from os import path as osp

import cv2
import numpy as np

import sys
sys.path.append("..")
import Basics.sensorParams as psp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_datapack", required=True, type=str,
                         help="Path to the existing dataPack.npz to copy other keys from")
    parser.add_argument("--blank_jpg", required=True, type=str,
                         help="Path to a .jpg blank (no-contact) frame captured from the real sensor")
    parser.add_argument("--out_datapack", required=True, type=str,
                         help="Path to write the new dataPack.npz")
    args = parser.parse_args()

    assert osp.exists(args.in_datapack), f"{args.in_datapack} does not exist"
    assert osp.exists(args.blank_jpg), f"{args.blank_jpg} does not exist"

    data = dict(np.load(args.in_datapack, allow_pickle=True))

    # cv2.imread returns BGR uint8, matching the convention of the existing f0
    f0 = cv2.imread(args.blank_jpg)
    if f0 is None:
        raise ValueError(f"Failed to read image: {args.blank_jpg}")

    if f0.shape[0] != psp.h or f0.shape[1] != psp.w:
        print(f"Resizing blank frame from {f0.shape[:2]} to {(psp.h, psp.w)}")
        f0 = cv2.resize(f0, (psp.w, psp.h))

    data["f0"] = f0

    np.savez(args.out_datapack, **data)
    print(f"Wrote {args.out_datapack} with new f0 of shape {f0.shape}")


if __name__ == "__main__":
    main()
