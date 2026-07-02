import argparse
from glob import glob
from os import path as osp
import os
import cv2
import numpy as np


def extract_frame_number(filename):
    base = osp.basename(filename)
    name, _ = osp.splitext(base)
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 10**9


def find_images(data_path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.ppm"]
    files = []
    for ext in exts:
        files.extend(glob(osp.join(data_path, ext)))
    return sorted(files, key=extract_frame_number)


def find_background(files):
    valid_bg_names = {
        "frame_0.jpg",
        "frame_0.jpeg",
        "frame_0.png",
        "frame_0.ppm",
        "ball0.jpeg",
    }

    for i, fn in enumerate(files):
        if osp.basename(fn) in valid_bg_names:
            return i, fn

    raise FileNotFoundError(
        "Background image not found. Expected frame_0.jpg, frame_0.png, "
        "frame_0.ppm, frame_0.jpeg, or ball0.jpeg."
    )


def read_image(path, target_size=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img


class Annotator:
    def __init__(self, files, bg_id, bg_img, data_path, target_size=None):
        self.files = files
        self.bg_id = bg_id
        self.bg_img = bg_img
        self.data_path = data_path
        self.target_size = target_size

        self.imgs = []
        self.touch_centers = []
        self.touch_radius = []
        self.names = []

        self.idx = 0
        self.cx = None
        self.cy = None
        self.radius = 50
        self.step = 4
        self.dragging = False

    def reset_circle(self, img):
        h, w = img.shape[:2]
        self.cx = w // 2
        self.cy = h // 2
        self.radius = max(10, min(h, w) // 6)

    def draw_overlay(self, img):
        display_img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = display_img.copy()
        overlay = display_img.copy()

        cv2.circle(
            overlay,
            (int(self.cx), int(self.cy)),
            int(self.radius),
            (0, 255, 0),
            -1,
        )

        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.circle(out, (int(self.cx), int(self.cy)), int(self.radius), (0, 255, 0), 2)
        cv2.circle(out, (int(self.cx), int(self.cy)), 3, (0, 0, 255), -1)

        text1 = f"file: {osp.basename(self.files[self.idx])}"
        text2 = f"center=({self.cx},{self.cy}) radius={self.radius} step={self.step}"
        text3 = "ENTER/s: save | k: skip | arrows/wasd: move | m/p or -/+: radius | f/c: step | q/ESC: quit"

        cv2.putText(out, text1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, text3, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return out

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cx = x
            self.cy = y
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cx = x
            self.cy = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.cx = x
            self.cy = y
            self.dragging = False

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.radius += 1
            else:
                self.radius = max(1, self.radius - 1)

    def save_current(self, img):
        fn = self.files[self.idx]
        self.imgs.append(img.copy())
        self.touch_centers.append([float(self.cx), float(self.cy)])
        self.touch_radius.append(float(self.radius))
        self.names.append(osp.basename(fn))

        print(
            f"[SAVE] {osp.basename(fn)} "
            f"center=({self.cx}, {self.cy}), radius={self.radius}"
        )

    def save_npz(self):
        out_fn = osp.join(self.data_path, "dataPack.npz")

        np.savez(
            out_fn,
            f0=self.bg_img,
            imgs=np.array(self.imgs),
            touch_center=np.array(self.touch_centers),
            touch_radius=np.array(self.touch_radius),
            names=np.array(self.names),
            img_size=self.bg_img.shape,
        )

        print(f"\nSaved: {out_fn}")
        print(f"num annotated frames: {len(self.imgs)}")
        print(f"background shape: {self.bg_img.shape}")

    def run(self):
        cv2.namedWindow("Taxim OpenCV Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Taxim OpenCV Calibration", self.mouse_callback)

        while self.idx < len(self.files):
            if self.idx == self.bg_id:
                self.idx += 1
                continue

            img = read_image(self.files[self.idx], self.target_size)
            if img is None:
                print(f"[WARN] cannot read {self.files[self.idx]}")
                self.idx += 1
                continue

            self.reset_circle(img)

            while True:
                vis = self.draw_overlay(img)
                cv2.imshow("Taxim OpenCV Calibration", vis)
                key = cv2.waitKey(30) & 0xFF

                if key in [27, ord("q")]:
                    self.save_npz()
                    cv2.destroyAllWindows()
                    return

                elif key in [13, ord("s")]:
                    self.save_current(img)
                    self.idx += 1
                    break

                elif key == ord("k"):
                    print(f"[SKIP] {osp.basename(self.files[self.idx])}")
                    self.idx += 1
                    break

                elif key in [ord("a"), 81]:
                    self.cx -= self.step
                elif key in [ord("d"), 83]:
                    self.cx += self.step
                elif key in [ord("w"), 82]:
                    self.cy -= self.step
                elif key in [ord("x"), 84]:
                    self.cy += self.step

                elif key in [ord("m"), ord("-"), ord("_")]:
                    self.radius = max(1, self.radius - 1)
                elif key in [ord("p"), ord("+"), ord("=")]:
                    self.radius += 1

                elif key == ord("f"):
                    self.step = max(1, self.step // 2)
                elif key == ord("c"):
                    self.step *= 2

        self.save_npz()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=True, help="Path to raw tactile image folder")
    parser.add_argument("--target_w", type=int, default=None, help="Optional output width for annotation and saved frames")
    parser.add_argument("--target_h", type=int, default=None, help="Optional output height for annotation and saved frames")
    args = parser.parse_args()

    target_size = None
    if args.target_w is not None and args.target_h is not None:
        target_size = (args.target_w, args.target_h)

    files = find_images(args.data_path)
    if len(files) == 0:
        raise RuntimeError(f"No image files found in {args.data_path}")

    bg_id, bg_fn = find_background(files)
    bg_img = read_image(bg_fn, target_size)

    if bg_img is None:
        raise RuntimeError(f"Cannot read background image: {bg_fn}")

    print(f"Background: {bg_fn}")
    if target_size is not None:
        print(f"Resize during annotation: {bg_img.shape[1]}x{bg_img.shape[0]}")
    print(f"Total images: {len(files)}")
    print("Controls:")
    print("  left click / drag: set contact center")
    print("  ENTER or s: save current annotation")
    print("  k: skip")
    print("  arrows or a/d/w/x: move circle")
    print("  m/p or -/+: decrease/increase radius")
    print("  f/c: decrease/increase moving step")
    print("  q or ESC: save and quit")

    app = Annotator(files, bg_id, bg_img, args.data_path, target_size)
    app.run()


if __name__ == "__main__":
    main()
