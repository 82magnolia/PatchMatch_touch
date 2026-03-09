import argparse
import os
import sys
from os import path as osp

import cv2
import numpy as np
import open3d as o3d
import pyrender
from matplotlib import colormaps
from tqdm import tqdm

sys.path.append("..")
from simOptical import height2laplacian, mesh_simulator


def build_press_depth_arr(mode, press_min, press_max, num_step):
    if mode == "continuous_press":
        return np.linspace(press_min, press_max, num_step)
    else:  # back_forth_press
        fwd = np.linspace(press_min, press_max, num_step // 2)
        bwd = np.linspace(press_max, press_min, num_step - num_step // 2 + 1)
        return np.concatenate([fwd, bwd[1:]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", required=True, type=str, help="Path to .obj file")
    parser.add_argument("--contact_ply", required=True, type=str, help="Path to .ply file with contact points")
    parser.add_argument("--mode", default="continuous_press", choices=["continuous_press", "back_forth_press"])
    parser.add_argument("--depth_range_info", default=[0.0, 1.0, 100.], type=float, nargs=3,
                        metavar=("MIN_DEPTH", "MAX_DEPTH", "NUM_STEPS"))
    parser.add_argument("--obj_scale_factor", default=25.0, type=float,
                        help="Target max-axis length in mm for scaling the object (used with --scale_method)")
    parser.add_argument("--scale_method", default="max_len", choices=["max_len"],
                        help="Scaling method: 'max_len' normalizes longest axis to --obj_scale_factor mm")
    parser.add_argument("--override_hw", default=None, type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--contact_theta", default=0.0, type=float)
    parser.add_argument("--save_dir", default="../results/gen_contact", type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Paths
    file_path = osp.dirname(osp.abspath(args.obj_path))
    obj = osp.basename(args.obj_path)
    data_folder = osp.join(osp.dirname(__file__), "..", "calibs")
    gelpad_model_path = osp.join(data_folder, "gelmap5.npy")

    # Load contact points from .ply
    pcd = o3d.io.read_point_cloud(args.contact_ply)
    contact_points = np.asarray(pcd.points)
    print(f"Loaded {len(contact_points)} contact point(s) from {args.contact_ply}")

    # Compute scale factor from contact point bounding box
    if args.scale_method == "max_len":
        long_axis = (contact_points.max(axis=0) - contact_points.min(axis=0)).argmax()
        axis_len = contact_points[:, long_axis].max() - contact_points[:, long_axis].min()
        obj_scale_factor = args.obj_scale_factor / axis_len
    else:
        raise NotImplementedError(f"Scale method '{args.scale_method}' not supported")

    # Build press depth array
    press_min, press_max, num_step = args.depth_range_info
    num_step = int(num_step)
    press_depth_arr = build_press_depth_arr(args.mode, press_min, press_max, num_step)

    # Create shared renderers (must be created once for off-screen rendering)
    override_hw = tuple(args.override_hw) if args.override_hw is not None else None
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.override_hw[1] if args.override_hw else 640,
        viewport_height=args.override_hw[0] if args.override_hw else 480,
    )
    normal_renderer = pyrender.OffscreenRenderer(
        viewport_width=args.override_hw[1] if args.override_hw else 640,
        viewport_height=args.override_hw[0] if args.override_hw else 480,
    )

    # Instantiate simulator once, reusing renderers
    sim = mesh_simulator(
        data_folder, file_path, obj,
        obj_scale_factor=obj_scale_factor,
        override_hw=override_hw,
        renderer=renderer,
        normal_renderer=normal_renderer,
    )

    # Per-contact-point loop
    for idx, contact_point in enumerate(tqdm(contact_points, desc="Contact points")):
        sim_video = None
        shadow_video = None
        mask_video = None

        for press_idx, press_depth in enumerate(tqdm(press_depth_arr, desc=f"  [{idx}] Press", leave=False)):
            height_map, gel_map, render_contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = \
                sim.generateHeightMap(gelpad_model_path, press_depth, dx=0, dy=0,
                                      contact_point=contact_point,
                                      contact_theta=args.contact_theta)
            heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, render_contact_mask)
            sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)

            sim_img_u8 = sim_img.astype(np.uint8)
            shadow_img_u8 = shadow_sim_img.astype(np.uint8)
            h_out, w_out = sim_img_u8.shape[:2]

            if press_idx == 0:
                # Init video writers
                sim_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_sim.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out))
                shadow_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_shadow.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out))
                mask_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_mask.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out), isColor=False)

                # First-frame static outputs
                # Color
                raw_color_img = (raw_color_map * 255).astype(np.uint8)
                cv2.imwrite(osp.join(args.save_dir, f"{idx}_color.jpg"),
                            cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))

                # Normal image + raw
                raw_normal_img = (vis_raw_normal_map * 255).astype(np.uint8)
                cv2.imwrite(osp.join(args.save_dir, f"{idx}_normal.jpg"),
                            cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))
                np.savez_compressed(osp.join(args.save_dir, f"{idx}_normal.npz"), normal=raw_normal_map)

                # Height image + raw
                norm_height = colormaps.get_cmap("viridis")(
                    (raw_height_map - raw_height_map.min()) /
                    (raw_height_map.max() - raw_height_map.min() + 1e-6))
                height_img = (norm_height * 255).astype(np.uint8)
                cv2.imwrite(osp.join(args.save_dir, f"{idx}_height.jpg"),
                            cv2.cvtColor(height_img, cv2.COLOR_RGB2BGR))
                np.savez_compressed(osp.join(args.save_dir, f"{idx}_height.npz"), height=raw_height_map)

                # Curvature
                cv2.imwrite(osp.join(args.save_dir, f"{idx}_curvature.jpg"),
                            height2laplacian(raw_height_map))

            # Write video frames
            sim_video.write(cv2.cvtColor(sim_img_u8, cv2.COLOR_RGB2BGR))
            shadow_video.write(cv2.cvtColor(shadow_img_u8, cv2.COLOR_RGB2BGR))
            mask_video.write((contact_mask * 255).astype(np.uint8))

        if sim_video is not None:
            sim_video.release()
            shadow_video.release()
            mask_video.release()

    renderer.delete()
    normal_renderer.delete()
    print(f"Done. Results saved to: {args.save_dir}")
