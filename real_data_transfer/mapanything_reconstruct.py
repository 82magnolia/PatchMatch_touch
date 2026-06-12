"""
MapAnything 3D reconstruction from turntable captures.

Reads the output of capture_turntable.py (intrinsics.json, poses.json, and
per-capture RGB/mask images), runs MapAnything inference with the known
metric poses and intrinsics, and saves:
  - Per-frame depth maps  ({idx:03d}_depth_ma.npy, _depth_ma_vis.png)
  - Per-frame confidence  ({idx:03d}_conf.npy)  [optional, --save_conf]
  - Combined colored point cloud  (pointcloud.ply)
  - Refined camera poses  (poses_ma.json)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

# MapAnything lives as a local package under real_data_transfer/map-anything.
# Add it to sys.path so it can be imported without installing into the env.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MA_DIR = os.path.join(_SCRIPT_DIR, "map-anything")
if _MA_DIR not in sys.path:
    sys.path.insert(0, _MA_DIR)

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_captures(capture_dir: str, use_masked: bool) -> list[dict]:
    """Return a sorted list of capture records that have valid poses."""
    intr_path = os.path.join(capture_dir, "intrinsics.json")
    poses_path = os.path.join(capture_dir, "poses.json")

    if not os.path.exists(intr_path):
        sys.exit(f"intrinsics.json not found in {capture_dir}")
    if not os.path.exists(poses_path):
        sys.exit(f"poses.json not found in {capture_dir}")

    with open(intr_path) as f:
        intr = json.load(f)

    K = np.array([
        [intr["fx"],       0.0, intr["cx"]],
        [      0.0, intr["fy"], intr["cy"]],
        [      0.0,       0.0,       1.0],
    ], dtype=np.float32)

    with open(poses_path) as f:
        records = json.load(f)

    captures = []
    for rec in records:
        T = rec.get("T_world_from_cam")
        if T is None:
            print(f"  Skipping pick {rec['pick_idx']} — no T_world_from_cam")
            continue

        idx = rec["pick_idx"]
        prefix = os.path.join(capture_dir, f"{idx:03d}")

        rgb_key = "_rgb_masked.png" if use_masked else "_rgb.png"
        rgb_path = prefix + rgb_key
        mask_path = prefix + "_mask.png"

        if not os.path.exists(rgb_path):
            print(f"  Skipping pick {idx} — {rgb_path} not found")
            continue

        captures.append({
            "idx": idx,
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "K": K.copy(),
            "T_world_from_cam": np.array(T, dtype=np.float64),
        })

    captures.sort(key=lambda c: c["idx"])
    return captures


def depth_to_colormap(depth_m: np.ndarray) -> np.ndarray:
    """Convert a float32 depth map (metres) to a JET BGR visualisation."""
    valid = depth_m > 0
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    d_min = depth_m[valid].min()
    d_max = depth_m[valid].max()
    norm = np.zeros_like(depth_m, dtype=np.float32)
    if d_max > d_min:
        norm[valid] = (depth_m[valid] - d_min) / (d_max - d_min)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Save a coloured point cloud to a PLY file using plyfile."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        sys.exit("plyfile not installed. Run: pip install plyfile")

    assert xyz.shape[1] == 3 and rgb.shape[1] == 3
    r = rgb[:, 0].astype(np.uint8)
    g = rgb[:, 1].astype(np.uint8)
    b = rgb[:, 2].astype(np.uint8)
    verts = np.empty(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = r, g, b
    PlyData([PlyElement.describe(verts, "vertex")]).write(path)
    print(f"Saved point cloud ({len(xyz):,} pts) → {path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_views(captures: list[dict]) -> list[dict]:
    """Build raw view dicts (before preprocess_inputs resizing)."""
    views = []
    for cap in captures:
        bgr = cv2.imread(cap["rgb_path"])
        if bgr is None:
            print(f"  Warning: could not read {cap['rgb_path']}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        views.append({
            "img": rgb,                                    # (H, W, 3) uint8
            "intrinsics": torch.from_numpy(cap["K"]),      # (3, 3) float32
            "camera_poses": torch.from_numpy(             # (4, 4) float64
                cap["T_world_from_cam"].astype(np.float32)
            ),
            "is_metric_scale": True,
            "_cap": cap,   # stash for output alignment; stripped before infer
        })
    return views


def run_reconstruction(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load captures ────────────────────────────────────────────────────────
    captures = load_captures(args.capture_dir, use_masked=not args.no_mask)
    if not captures:
        sys.exit("No valid captures found (all missing poses or images).")
    print(f"Loaded {len(captures)} captures from {args.capture_dir}")

    # ── load model ───────────────────────────────────────────────────────────
    if args.apache:
        model_name = "facebook/map-anything-apache"
    else:
        model_name = args.model
    print(f"Loading model: {model_name} ...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    print("Model ready.")

    # ── build + preprocess views ─────────────────────────────────────────────
    raw_views = build_views(captures)
    caps_ordered = [v.pop("_cap") for v in raw_views]  # extract stashed caps

    print("Preprocessing inputs ...")
    views = preprocess_inputs(raw_views)  # resizes to 518px patch grid

    # ── inference ────────────────────────────────────────────────────────────
    print("Running MapAnything inference ...")
    with torch.inference_mode():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            ignore_pose_inputs=False,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=True,
        )
    print("Inference complete.")

    # ── save outputs ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    all_xyz = []
    all_rgb = []
    poses_ma = []

    for i, (pred, cap) in enumerate(zip(outputs, caps_ordered)):
        idx = cap["idx"]
        prefix = os.path.join(args.output_dir, f"{idx:03d}")

        # --- depth (Z-depth in camera frame, metres) -------------------------
        depth_z = pred["depth_z"][0].squeeze(-1)   # (H_inf, W_inf) tensor
        H_orig = cap["K"][1, 2] * 2   # approximate — actual size from image
        bgr_check = cv2.imread(cap["rgb_path"])
        H_orig, W_orig = bgr_check.shape[:2]

        # Resize depth back to original capture resolution
        depth_np = depth_z.float().cpu().numpy()
        depth_resized = cv2.resize(
            depth_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        np.save(f"{prefix}_depth_ma.npy", depth_resized)
        cv2.imwrite(f"{prefix}_depth_ma_vis.png", depth_to_colormap(depth_resized))

        # --- confidence ------------------------------------------------------
        if args.save_conf:
            conf = pred["conf"][0].cpu().numpy().astype(np.float32)
            conf_resized = cv2.resize(conf, (W_orig, H_orig),
                                      interpolation=cv2.INTER_LINEAR)
            np.save(f"{prefix}_conf.npy", conf_resized)

        # --- refined camera pose from MapAnything ----------------------------
        T_ma = pred.get("camera_poses")
        T_ma_np = (T_ma[0].cpu().numpy().tolist()
                   if T_ma is not None else cap["T_world_from_cam"].tolist())
        intr_ma = pred.get("intrinsics")
        K_ma = (intr_ma[0].cpu().numpy().tolist()
                if intr_ma is not None else cap["K"].tolist())
        poses_ma.append({
            "pick_idx": idx,
            "T_world_from_cam": T_ma_np,
            "intrinsics": K_ma,
        })

        # --- colored point cloud contribution --------------------------------
        mask_path = cap["mask_path"]
        obj_mask = None
        if os.path.exists(mask_path):
            obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0  # (H_orig, W_orig)

        # Use MapAnything's intrinsics + poses for world-frame unprojection
        K_tensor = (intr_ma[0] if intr_ma is not None
                    else torch.from_numpy(cap["K"]).to(device))
        T_tensor = (T_ma[0] if T_ma is not None
                    else torch.from_numpy(cap["T_world_from_cam"].astype(np.float32)).to(device))

        pts3d_world, valid = depthmap_to_world_frame(
            depth_z.to(device), K_tensor.to(device), T_tensor.to(device)
        )  # (H_inf, W_inf, 3), (H_inf, W_inf)

        # Bring mask to inference resolution for filtering
        H_inf, W_inf = depth_z.shape
        valid_np = valid.cpu().numpy()
        if obj_mask is not None:
            obj_mask_inf = cv2.resize(
                obj_mask.astype(np.uint8), (W_inf, H_inf),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            valid_np = valid_np & obj_mask_inf

        pts_np = pts3d_world.cpu().numpy().reshape(-1, 3)
        valid_flat = valid_np.reshape(-1)
        pts_filtered = pts_np[valid_flat]

        # Match colors: load original RGB, resize to inference res
        color_rgb = cv2.cvtColor(cv2.imread(cap["rgb_path"]), cv2.COLOR_BGR2RGB)
        color_inf = cv2.resize(color_rgb, (W_inf, H_inf),
                               interpolation=cv2.INTER_LINEAR)
        colors_flat = color_inf.reshape(-1, 3)[valid_flat]

        all_xyz.append(pts_filtered)
        all_rgb.append(colors_flat)

        print(f"  [{i+1}/{len(caps_ordered)}] pick {idx:03d} — "
              f"{valid_flat.sum():,} pts")

    # ── save poses_ma.json ───────────────────────────────────────────────────
    poses_path = os.path.join(args.output_dir, "poses_ma.json")
    with open(poses_path, "w") as f:
        json.dump(poses_ma, f, indent=2)
    print(f"Saved refined poses → {poses_path}")

    # ── save point cloud ─────────────────────────────────────────────────────
    if all_xyz:
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        if args.max_pts > 0 and len(xyz) > args.max_pts:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(xyz), args.max_pts, replace=False)
            xyz, rgb = xyz[keep], rgb[keep]
            print(f"Downsampled to {args.max_pts:,} pts")

        save_ply(os.path.join(args.output_dir, "pointcloud.ply"), xyz, rgb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MapAnything 3D reconstruction from turntable captures"
    )
    p.add_argument("--capture_dir", required=True,
                   help="Directory written by capture_turntable.py")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: <capture_dir>/mapanything_out)")
    p.add_argument("--model", default="facebook/map-anything",
                   help="HuggingFace model ID or local path")
    p.add_argument("--apache", action="store_true",
                   help="Use facebook/map-anything-apache (Apache 2.0 license)")
    p.add_argument("--no_mask", action="store_true",
                   help="Use unmasked RGB (_rgb.png) instead of _rgb_masked.png")
    p.add_argument("--save_conf", action="store_true",
                   help="Also save confidence maps as {idx}_conf.npy")
    p.add_argument("--max_pts", type=int, default=500000,
                   help="Max points in combined cloud (0 = no limit, default 500000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.capture_dir, "mapanything_out")
    run_reconstruction(args)
