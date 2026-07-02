import argparse
import os
import os.path as osp
from glob import glob

import cv2


def frame_number(fn):
    base = osp.basename(fn)
    name, _ = osp.splitext(base)
    try:
        return int(name.split("_")[-1])
    except Exception:
        return 10**9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--target_w", type=int, default=640)
    parser.add_argument("--target_h", type=int, default=480)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.ppm"]
    files = []
    for ext in exts:
        files.extend(glob(osp.join(args.src, ext)))

    files = sorted(files, key=frame_number)

    print("num files:", len(files))

    for fn in files:
        img = cv2.imread(fn)
        if img is None:
            print("[SKIP] cannot read:", fn)
            continue

        out = cv2.resize(
            img,
            (args.target_w, args.target_h),
            interpolation=cv2.INTER_CUBIC,
        )

        out_fn = osp.join(args.dst, osp.basename(fn))
        cv2.imwrite(out_fn, out)
        print("saved:", out_fn, out.shape)


if __name__ == "__main__":
    main()
