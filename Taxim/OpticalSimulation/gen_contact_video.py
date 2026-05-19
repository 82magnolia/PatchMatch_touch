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
from simOptical import height2laplacian, mesh_simulator, height2shapeindex


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
    parser.add_argument("--obj_scale_factor", default=[25.0], type=float, nargs="+",
                        help="Target max-axis length(s) in mm for scaling the object (used with --scale_method); "
                             "multiple values render static outputs at each scale")
    parser.add_argument("--scale_method", default="max_len", choices=["max_len"],
                        help="Scaling method: 'max_len' normalizes longest axis to --obj_scale_factor mm")
    parser.add_argument("--override_hw", default=None, type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--contact_theta", default=0.0, type=float)
    parser.add_argument("--rand_contact_theta", action="store_true",
                        help="Sample a random contact_theta from [0, 2*pi] for each contact point")
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

    # Compute scale factors from contact point bounding box
    if args.scale_method == "max_len":
        long_axis = (contact_points.max(axis=0) - contact_points.min(axis=0)).argmax()
        axis_len = contact_points[:, long_axis].max() - contact_points[:, long_axis].min()
        obj_scale_factors = [scale_target_mm / axis_len for scale_target_mm in args.obj_scale_factor]
    else:
        raise NotImplementedError(f"Scale method '{args.scale_method}' not supported")

    # Build press depth array
    press_min, press_max, num_step = args.depth_range_info
    num_step = int(num_step)
    press_depth_arr = build_press_depth_arr(args.mode, press_min, press_max, num_step)

    # Create shared renderer (must be created once for off-screen rendering)
    override_hw = tuple(args.override_hw) if args.override_hw is not None else None
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.override_hw[1] if args.override_hw else 640,
        viewport_height=args.override_hw[0] if args.override_hw else 480,
    )

    # Instantiate one simulator per scale factor, all sharing the same renderer
    sims = [
        mesh_simulator(
            data_folder, file_path, obj,
            obj_scale_factor=scale_factor,
            override_hw=override_hw,
            renderer=renderer,
        )
        for scale_factor in obj_scale_factors
    ]

    # Per-contact-point loop
    for idx, contact_point in enumerate(tqdm(contact_points, desc="Contact points")):
        contact_theta = np.random.uniform(0, 2 * np.pi) if args.rand_contact_theta else args.contact_theta
        sim_video = None
        shadow_video = None
        mask_video = None
        render_mask_video = None

        for press_idx, press_depth in enumerate(tqdm(press_depth_arr, desc=f"  [{idx}] Press", leave=False)):
            height_map, gel_map, render_contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = \
                sims[0].generateHeightMap(gelpad_model_path, press_depth, dx=0, dy=0,
                                          contact_point=contact_point,
                                          contact_theta=contact_theta)
            heightMap, contact_mask, contact_height = sims[0].deformApprox(press_depth, height_map, gel_map, render_contact_mask)
            sim_img, shadow_sim_img = sims[0].simulating(heightMap, contact_mask, contact_height, shadow=True)

            sim_img_u8 = sim_img.astype(np.uint8)
            shadow_img_u8 = shadow_sim_img.astype(np.uint8)
            h_out, w_out = sim_img_u8.shape[:2]

            if press_idx == 0:
                # Init video writers (using primary scale)
                sim_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_sim.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out))
                shadow_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_shadow.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out))
                mask_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_mask.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out), isColor=False)
                render_mask_video = cv2.VideoWriter(
                    osp.join(args.save_dir, f"{idx}_render_mask.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 5., (w_out, h_out), isColor=False)

                # Static outputs at each scale (RGB, normal, height, curvature)
                for scale_mm, sim_s, (scale_color_map, scale_normal_map, scale_vis_normal_map, scale_height_map) in zip(
                    args.obj_scale_factor,
                    sims,
                    [(raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map)]
                    + [extra_sim.generateHeightMap(gelpad_model_path, press_depth, dx=0, dy=0,
                                                   contact_point=contact_point,
                                                   contact_theta=contact_theta)[3:]
                       for extra_sim in sims[1:]],
                ):
                    tag = f"{idx}_scale{int(scale_mm)}"

                    raw_color_img = (scale_color_map * 255).astype(np.uint8)
                    cv2.imwrite(osp.join(args.save_dir, f"{tag}_color.jpg"),
                                cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))

                    raw_normal_img = (scale_vis_normal_map * 255).astype(np.uint8)
                    cv2.imwrite(osp.join(args.save_dir, f"{tag}_normal.jpg"),
                                cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))
                    np.savez_compressed(osp.join(args.save_dir, f"{tag}_normal.npz"), normal=scale_normal_map)

                    norm_height = colormaps.get_cmap("viridis")(
                        (scale_height_map - scale_height_map.min()) /
                        (scale_height_map.max() - scale_height_map.min() + 1e-6))
                    height_img = (norm_height * 255).astype(np.uint8)
                    cv2.imwrite(osp.join(args.save_dir, f"{tag}_height.jpg"),
                                cv2.cvtColor(height_img, cv2.COLOR_RGB2BGR))
                    np.savez_compressed(osp.join(args.save_dir, f"{tag}_height.npz"), height=scale_height_map)

                    cv2.imwrite(osp.join(args.save_dir, f"{tag}_curvature.jpg"),
                                height2laplacian(scale_height_map))
                    cv2.imwrite(osp.join(args.save_dir, f"{tag}_shapeindex.jpg"),
                                height2shapeindex(scale_height_map))

            # Write video frames
            sim_video.write(cv2.cvtColor(sim_img_u8, cv2.COLOR_RGB2BGR))
            shadow_video.write(cv2.cvtColor(shadow_img_u8, cv2.COLOR_RGB2BGR))
            mask_video.write((contact_mask * 255).astype(np.uint8))
            render_mask_video.write((render_contact_mask * 255).astype(np.uint8))

        if sim_video is not None:
            sim_video.release()
            shadow_video.release()
            mask_video.release()
            render_mask_video.release()

    renderer.delete()
    print(f"Done. Results saved to: {args.save_dir}")
