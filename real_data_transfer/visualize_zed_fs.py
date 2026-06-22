"""
ZED 2i stereo viewer with FoundationStereo / Fast-FoundationStereo depth estimation.

Live mode:
  Streams ZED left RGB + ZED surface normals side-by-side.
  Press 'c' to capture a stereo pair and run FoundationStereo inference.
  Results appear in a second window (RGB | FS depth | FS normals) and an
  Open3D point cloud viewer (non-blocking, in a background thread).

Controls:
  c — capture stereo frame and run inference
  q — quit
"""

import sys
import os
import argparse
import threading
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_FS_REPO_DIRS = {
    "foundation_stereo":      "FoundationStereo",
    "fast_foundation_stereo": "Fast-FoundationStereo",
}

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found. Install via the ZED SDK installer.")

try:
    import torch
except ImportError:
    sys.exit("torch not found. Install with: pip install torch")

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not found. Install with: pip install open3d")

from visualize_zed_normal_sim import build_camera, normals_to_colormap, DEPTH_MODES

# ── display constants ──────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 640, 360


# ── helpers ───────────────────────────────────────────────────────────────────

def normals_from_xyz(xyz_map: np.ndarray) -> np.ndarray:
    """Estimate surface normals from an (H,W,3) xyz map via finite differences."""
    dx = np.roll(xyz_map, -1, axis=1) - xyz_map
    dy = np.roll(xyz_map, -1, axis=0) - xyz_map
    n = np.cross(dx, dy)
    norms = np.linalg.norm(n, axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(norms > 1e-8, n / norms, 0.0)
    return n.astype(np.float32)


def normals_xyz_to_bgr(n: np.ndarray) -> np.ndarray:
    """(H,W,3) float normals in [-1,1] → (H,W,3) uint8 BGR colormap."""
    rgb = ((n + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR


def depth_to_bgr(depth: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """(H,W) float depth in metres → (H,W,3) uint8 TURBO colormap."""
    valid = np.isfinite(depth) & (depth > z_near) & (depth < z_far)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        d = depth[valid]
        d_norm = ((d - z_near) / (z_far - z_near)).clip(0, 1)
        u8 = (d_norm * 255).astype(np.uint8)
        tmp = np.zeros(depth.shape, dtype=np.uint8)
        tmp[valid] = u8
        vis = cv2.applyColorMap(tmp, cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    return vis


def _show_pcd(xyz_map: np.ndarray, color_rgb: np.ndarray,
              z_near: float, z_far: float) -> None:
    """Display an Open3D point cloud in a background thread."""
    pts = xyz_map.reshape(-1, 3)
    clr = color_rgb.reshape(-1, 3)
    valid = (np.isfinite(pts).all(axis=1)
             & (pts[:, 2] > z_near) & (pts[:, 2] < z_far))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        (clr[valid].astype(np.float64) / 255.0).clip(0, 1))
    o3d.visualization.draw_geometries([pcd], window_name="FoundationStereo Point Cloud")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_type: str, model_dir: str, valid_iters: int, max_disp: int):
    """Load FoundationStereo or Fast-FoundationStereo model onto GPU."""
    repo_dir = os.path.join(_SCRIPT_DIR, _FS_REPO_DIRS[model_type])
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    if model_type == "fast_foundation_stereo":
        import yaml
        cfg_path = os.path.join(os.path.dirname(model_dir), "cfg.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        from omegaconf import OmegaConf
        args = OmegaConf.create(cfg)
        model = torch.load(model_dir, map_location="cpu", weights_only=False)
        model.args.valid_iters = valid_iters
        model.args.max_disp = max_disp
    else:
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        cfg_path = os.path.join(os.path.dirname(model_dir), "cfg.yaml")
        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = valid_iters
        args = cfg
        model = FoundationStereo(args)
        ckpt = torch.load(model_dir, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, model_type: str, left_rgb: np.ndarray, right_rgb: np.ndarray,
                  intr: dict, baseline: float, scale: float,
                  valid_iters: int, z_near: float, z_far: float):
    """Run FoundationStereo inference on a rectified stereo pair.

    Args:
        left_rgb / right_rgb: (H,W,3) uint8 RGB images
        intr: camera intrinsics dict (fx,fy,cx,cy)
        baseline: stereo baseline in metres
        scale: downscale factor for inference (≤1)

    Returns:
        disp:    (H,W) float32 disparity at original resolution
        depth:   (H,W) float32 depth in metres
        xyz_map: (H,W,3) float32 3D points in camera frame
    """
    from core.utils.utils import InputPadder

    H, W = left_rgb.shape[:2]
    if scale != 1.0:
        nw, nh = int(W * scale), int(H * scale)
        l = cv2.resize(left_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        r = cv2.resize(right_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    else:
        l, r, nw, nh = left_rgb, right_rgb, W, H

    img0 = torch.as_tensor(l).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(r).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0p, img1p = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(True):
        if model_type == "fast_foundation_stereo":
            disp_t = model.forward(img0p, img1p, iters=valid_iters, test_mode=True,
                                   optimize_build_volume="pytorch1")
        else:
            disp_t = model.forward(img0p, img1p, iters=valid_iters, test_mode=True)

    disp_small = padder.unpad(disp_t.float()).cpu().numpy().squeeze()  # (nh, nw)

    disp = disp_small.astype(np.float32).clip(0, None)

    sx = nw / W
    K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                  [0, intr["fy"] * sx, intr["cy"] * sx],
                  [0, 0, 1]], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > 0, K[0, 0] * baseline / disp, np.nan).astype(np.float32)
    depth[(depth < z_near) | (depth > z_far)] = np.nan

    from Utils import depth2xyzmap
    xyz_map = depth2xyzmap(depth, K)
    return disp, depth, xyz_map


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ZED 2i viewer with FoundationStereo / Fast-FoundationStereo depth."
    )
    p.add_argument("--model_dir", required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--model_type", default="fast_foundation_stereo",
                   choices=["foundation_stereo", "fast_foundation_stereo"],
                   help="Which model to use (default: fast_foundation_stereo)")
    p.add_argument("--valid_iters", type=int, default=None,
                   help="Refinement iterations (default: 8 for fast, 32 for standard)")
    p.add_argument("--max_disp", type=int, default=192,
                   help="Maximum disparity for Fast-FoundationStereo (default: 192)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Image downscale factor for inference, ≤1 (default: 1.0)")
    p.add_argument("--depth_mode", choices=list(DEPTH_MODES.keys()),
                   default="neural_plus",
                   help="ZED depth mode for live normal display (default: neural_plus)")
    p.add_argument("--zed_confidence", type=int, default=95,
                   help="ZED depth confidence 0-100 (default: 95)")
    p.add_argument("--z_near", type=float, default=0.1,
                   help="Near depth clip in metres (default: 0.1)")
    p.add_argument("--z_far", type=float, default=3.0,
                   help="Far depth clip in metres (default: 3.0)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.valid_iters is None:
        args.valid_iters = 8 if args.model_type == "fast_foundation_stereo" else 32

    print(f"Loading {args.model_type} from {args.model_dir} ...")
    model = load_model(args.model_type, args.model_dir,
                       args.valid_iters, args.max_disp)
    print("Model ready.")

    cam, rt, intr = build_camera(args.depth_mode, args.zed_confidence)
    print(f"ZED opened. fx={intr['fx']:.1f}  baseline will be read from calibration.")

    # Stereo baseline from ZED calibration (returned in metres)
    info = cam.get_camera_information()
    baseline = info.camera_configuration.calibration_parameters.get_camera_baseline()
    print(f"Baseline: {baseline*1000:.1f} mm")

    image_l_sl  = sl.Mat()
    image_r_sl  = sl.Mat()
    normals_sl  = sl.Mat()

    LIVE_WIN   = "ZED: RGB | Normals  (c=capture  q=quit)"
    RESULT_WIN = "FoundationStereo: RGB | Depth | Normals"

    try:
        while True:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(image_l_sl, sl.VIEW.LEFT)
            cam.retrieve_image(image_r_sl, sl.VIEW.RIGHT)
            cam.retrieve_measure(normals_sl, sl.MEASURE.NORMALS)

            left_bgr   = image_l_sl.get_data()[:, :, :3].copy()
            right_bgr  = image_r_sl.get_data()[:, :, :3].copy()
            normals_np = normals_sl.get_data().copy()

            # Live display: RGB | ZED normals
            c_d = cv2.resize(left_bgr, (DISPLAY_W, DISPLAY_H))
            n_d = cv2.resize(normals_to_colormap(normals_np), (DISPLAY_W, DISPLAY_H))
            live = np.hstack([c_d, n_d])
            cv2.putText(live, "c=capture  q=quit  |  RGB (left)  ZED normals (right)",
                        (10, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)
            cv2.imshow(LIVE_WIN, live)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("c"):
                # Freeze and run inference
                banner = live.copy()
                cv2.putText(banner, "Running FoundationStereo... please wait",
                            (10, DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 255), 2)
                cv2.imshow(LIVE_WIN, banner)
                cv2.waitKey(1)

                # FoundationStereo expects RGB input
                left_rgb  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

                disp, depth, xyz_map = run_inference(
                    model, args.model_type, left_rgb, right_rgb,
                    intr, baseline, args.scale, args.valid_iters,
                    args.z_near, args.z_far)

                # Build normals from xyz_map
                n_fs  = normals_from_xyz(xyz_map)
                n_bgr = normals_xyz_to_bgr(n_fs)

                # 3-panel result display
                d_bgr = depth_to_bgr(depth, args.z_near, args.z_far)
                r_rgb = cv2.resize(left_bgr,  (DISPLAY_W, DISPLAY_H))
                r_dep = cv2.resize(d_bgr,     (DISPLAY_W, DISPLAY_H))
                r_nrm = cv2.resize(n_bgr,     (DISPLAY_W, DISPLAY_H))
                cv2.putText(r_dep, f"FS depth [{args.z_near:.1f}-{args.z_far:.1f}m]",
                            (6, DISPLAY_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 200, 200), 1)
                cv2.putText(r_nrm, "FS normals",
                            (6, DISPLAY_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 200, 200), 1)
                result_img = np.hstack([r_rgb, r_dep, r_nrm])
                cv2.imshow(RESULT_WIN, result_img)

                # Open3D point cloud in background thread
                threading.Thread(
                    target=_show_pcd,
                    args=(xyz_map, left_rgb, args.z_near, args.z_far),
                    daemon=True,
                ).start()

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
