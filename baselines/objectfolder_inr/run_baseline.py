#!/usr/bin/env python3
"""ObjectFolder INR baseline using ArUco pose conditioning and Taxim rendering."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
OBJECTFOLDER_RENDERER_ROOT = BASELINE_ROOT / "vendor" / "objectfolder"
DEFAULT_TAXIM_CALIBRATION = OBJECTFOLDER_RENDERER_ROOT / "calibs"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from objectfolder_inr.contracts import discover_indices, output_names, resolve_pairs
from objectfolder_inr.data import condition_for_touch
from objectfolder_inr.pose import (
    load_aruco_conditions,
    load_sensor_offset,
    load_sim_contact_points,
    sim_conditions,
)


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser(description=__doc__)
    item.add_argument("--ref_dir", type=Path, required=True)
    item.add_argument("--query_dir", type=Path, required=True)
    item.add_argument("--save_dir", type=Path, required=True)
    item.add_argument("--scale", type=float, nargs="+", default=[100.0])
    item.add_argument(
        "--video_type", choices=["shadow", "sim", "tactile_normal"], default="shadow"
    )
    item.add_argument(
        "--retrieval_mode",
        choices=["dinov3", "tsv", "sim_gt_retrieval", "real_gt_retrieval"],
        default="dinov3",
    )
    item.add_argument("--tsv", type=Path)
    item.add_argument("--retrieval_modality", default="normal")
    item.add_argument("--dino_weights", type=Path)
    item.add_argument("--query_indices", type=int, nargs="+")

    assets = item.add_argument_group("ObjectFolder assets and pose metadata")
    assets.add_argument("--checkpoint", type=Path, required=False)
    assets.add_argument("--object_file", type=Path)
    assets.add_argument("--object_mesh", type=Path)
    assets.add_argument("--normalization_stats", type=Path)
    assets.add_argument("--contact_points", type=Path)
    assets.add_argument(
        "--pose_source", choices=["auto", "aruco", "sim"], default="auto"
    )
    assets.add_argument(
        "--sensor_offset_file",
        type=Path,
        default=PROJECT_ROOT / "log" / "gelsight_sensor_offset.json",
        help=(
            "GelSight marker-to-gel calibration JSON produced by "
            "real_data_transfer/calibrate_sensor_offset.py."
        ),
    )
    assets.add_argument(
        "--taxim_calibration",
        type=Path,
        default=DEFAULT_TAXIM_CALIBRATION,
        help="ObjectFolder Taxim calibration (default: vendored calibration).",
    )
    assets.add_argument("--allow_index_coordinate_fallback", action="store_true")

    training = item.add_argument_group("TouchNet training")
    training.add_argument("--train_if_missing", action="store_true")
    training.add_argument("--train_only", action="store_true")
    training.add_argument("--levels", type=int, default=10)
    training.add_argument("--network_depth", type=int, default=8)
    training.add_argument("--network_width", type=int, default=256)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--samples_per_touch", type=int, default=4096)
    training.add_argument("--batch_size", type=int, default=4096)
    training.add_argument("--learning_rate", type=float, default=5e-4)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cuda")

    execution = item.add_argument_group("Execution")
    execution.add_argument("--inference_batch_size", type=int, default=16384)
    execution.add_argument("--inr_height", type=int, default=120)
    execution.add_argument("--inr_width", type=int, default=160)
    execution.add_argument("--sim_press_min_mm", type=float, default=0.0)
    execution.add_argument("--sim_press_max_mm", type=float, default=10.0)
    execution.add_argument("--dry_run", action="store_true")
    execution.add_argument("--skip_eval", action="store_true")
    execution.add_argument("--debug_images", action="store_true")
    return item


def select_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "ObjectFolder_INR requested CUDA, but torch.cuda.is_available() is "
            "false. Run on a CUDA host or explicitly pass --device cpu for a "
            "slow diagnostic run."
        )
    return torch.device(name)


def video_metadata_without_tactile_gt(folder: Path, index: int) -> tuple[int, int, int, float]:
    """Use the geometry render-mask stream, never query tactile RGB."""
    import cv2

    path = folder / f"{index}_render_mask.mp4"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing query geometry metadata video: {path}. It is needed only for "
            "frame count, resolution, and FPS; query tactile RGB is not opened."
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(capture.get(cv2.CAP_PROP_FPS)) or 5.0,
    )
    capture.release()
    if min(result[:3]) < 1:
        raise RuntimeError(f"Invalid video metadata in {path}: {result}")
    return result


def write_mapping(save_dir: Path, pairs: list[tuple[int, int]], mode: str) -> Path | None:
    if mode == "tsv":
        return None
    name = "odd_to_even.tsv" if mode == "real_gt_retrieval" else "identity.tsv"
    path = save_dir / name
    path.write_text(
        "query\tref\n" + "".join(f"{query}\t{reference}\n" for query, reference in pairs)
    )
    return path


def save_retrieval(save_dir: Path, pairs: list[tuple[int, int]]) -> Path:
    rows = [
        {
            "query_idx": query,
            "topk_ref_indices": [reference],
            "topk_similarities": None,
        }
        for query, reference in pairs
    ]
    target = save_dir / "retrieval" / "results.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(rows, stream)
    return target


def run_dinov3(args) -> list[tuple[int, int]]:
    if args.dino_weights is None:
        raise ValueError("--dino_weights is required for retrieval_mode=dinov3")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "retrieve_touch.py"),
        "--ref_dir",
        str(args.ref_dir),
        "--query_dir",
        str(args.query_dir),
        "--modality",
        args.retrieval_modality,
        "--scale",
        f"{args.scale[0]:g}",
        "--retrieval_mode",
        "dinov3",
        "--top_k",
        "1",
        "--dinov3_weights",
        str(args.dino_weights),
        "--save_dir",
        str(args.save_dir / "retrieval"),
        "--no_figures",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    with (args.save_dir / "retrieval" / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    return [(int(row["query_idx"]), int(row["topk_ref_indices"][0])) for row in rows]


def get_conditions(
    args,
    query_idx: int,
    frame_count: int,
    pose_source: str,
    marker_to_contact: np.ndarray,
    inplane_offset_deg: float,
):
    if pose_source == "aruco":
        return load_aruco_conditions(
            args.query_dir,
            query_idx,
            frame_count,
            marker_to_contact=marker_to_contact,
            inplane_offset_deg=inplane_offset_deg,
        )
    if args.contact_points is None and not args.allow_index_coordinate_fallback:
        raise FileNotFoundError(
            "Simulation inference requires --contact_points. Coordinates were used "
            "by Taxim generation but are not stored in the rendered Dataset directory."
        )
    if args.contact_points is None:
        point = np.array([float(query_idx), 0.0, 0.0])
    else:
        points = load_sim_contact_points(args.contact_points)
        if query_idx >= len(points):
            raise IndexError(f"No contact point {query_idx} in {args.contact_points}")
        point = points[query_idx]
    return sim_conditions(
        point,
        frame_count,
        press_min_mm=args.sim_press_min_mm,
        press_max_mm=args.sim_press_max_mm,
    )


def train_if_requested(
    args,
    training_ref_indices: list[int],
    pose_source: str,
    device,
    marker_to_contact: np.ndarray,
    inplane_offset_deg: float,
):
    from objectfolder_inr.training import train_checkpoint

    if args.object_file is not None and args.object_file.exists() and not args.train_only:
        return
    if args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required for training; inference accepts --checkpoint "
            "or a legacy --object_file"
        )
    if args.checkpoint.exists() and not args.train_only:
        return
    if not args.train_if_missing and not args.train_only:
        raise FileNotFoundError(
            f"Missing checkpoint {args.checkpoint}; provide one or use --train_if_missing"
        )
    train_checkpoint(
        ref_dir=args.ref_dir,
        ref_indices=training_ref_indices,
        checkpoint=args.checkpoint,
        scale=args.scale[0] if args.scale else None,
        pose_source=pose_source,
        contact_points=args.contact_points,
        allow_index_coordinate_fallback=args.allow_index_coordinate_fallback,
        marker_to_contact=marker_to_contact,
        inplane_offset_deg=inplane_offset_deg,
        sensor_offset_file=args.sensor_offset_file,
        levels=args.levels,
        network_depth=args.network_depth,
        network_width=args.network_width,
        epochs=args.epochs,
        samples_per_touch=args.samples_per_touch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )


def resolved_config(args, pairs, pose_source: str, sensor_offset: dict | None) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            config[key] = [str(item.resolve()) for item in value]
    config["pairs"] = pairs
    config["resolved_pose_source"] = pose_source
    config["sensor_offset"] = sensor_offset
    if sensor_offset is not None:
        config["marker_to_contact_camera_rule"] = "R_marker @ [x, y, -z] + tvec"
    config["target_representation"] = "pseudo-height"
    config["query_tactile_policy"] = "evaluation-only"
    return config


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.ref_dir = args.ref_dir.resolve()
    args.query_dir = args.query_dir.resolve()
    args.save_dir = args.save_dir.resolve()
    args.sensor_offset_file = args.sensor_offset_file.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    scale = args.scale[0] if args.scale else None
    ref_indices = discover_indices(args.ref_dir, scale)
    query_indices = discover_indices(args.query_dir, scale)
    if args.query_indices:
        requested = set(args.query_indices)
        query_indices = [index for index in query_indices if index in requested]

    if args.retrieval_mode == "dinov3" and not args.dry_run:
        pairs = run_dinov3(args)
    elif args.retrieval_mode == "dinov3":
        pairs = []
    else:
        pairs = resolve_pairs(args.retrieval_mode, ref_indices, query_indices, args.tsv)
        write_mapping(args.save_dir, pairs, args.retrieval_mode)
        save_retrieval(args.save_dir, pairs)

    pose_source = args.pose_source
    if pose_source == "auto":
        pose_source = "aruco" if args.retrieval_mode == "real_gt_retrieval" else "sim"
    if pose_source == "aruco":
        try:
            marker_to_contact, inplane_offset_deg, sensor_offset = load_sensor_offset(
                args.sensor_offset_file
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        marker_to_contact = np.zeros(3, dtype=np.float64)
        inplane_offset_deg = 0.0
        sensor_offset = None
    config = resolved_config(args, pairs, pose_source, sensor_offset)
    (args.save_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    import cv2
    import torch

    device = select_device(args.device)
    training_ref_indices = (
        sorted({reference for _, reference in pairs})
        if args.retrieval_mode == "real_gt_retrieval"
        else ref_indices
    )
    train_if_requested(
        args,
        training_ref_indices,
        pose_source,
        device,
        marker_to_contact,
        inplane_offset_deg,
    )
    if args.train_only:
        print(f"Saved checkpoint: {args.checkpoint}")
        return

    from objectfolder_inr.rendering import Renderer, predict_height
    from objectfolder_inr.training import load_checkpoint

    model_path = args.checkpoint or args.object_file
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Inference requires an existing --checkpoint or --object_file")
    model, checkpoint = load_checkpoint(model_path, device)
    if args.normalization_stats is not None:
        if args.normalization_stats.suffix == ".json":
            stats = json.loads(args.normalization_stats.read_text())
        else:
            with np.load(args.normalization_stats) as archive:
                stats = {key: archive[key] for key in archive.files}
        checkpoint["feature_min"] = np.asarray(stats["feature_min"], dtype=np.float32)
        checkpoint["feature_max"] = np.asarray(stats["feature_max"], dtype=np.float32)
    feature_min = torch.as_tensor(checkpoint["feature_min"], dtype=torch.float32, device=device)
    feature_max = torch.as_tensor(checkpoint["feature_max"], dtype=torch.float32, device=device)
    normalization_mode = checkpoint.get("normalization_mode", "signed_unit")
    renderer = Renderer(
        args.video_type,
        args.taxim_calibration.resolve() if args.taxim_calibration else None,
        OBJECTFOLDER_RENDERER_ROOT,
    )
    transfer_dir = args.save_dir / "transfer"
    pose_dir = args.save_dir / "pose_features"
    debug_dir = args.save_dir / "debug"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for query_idx, reference_idx in pairs:
        frame_count, width, height, fps = video_metadata_without_tactile_gt(
            args.query_dir, query_idx
        )
        conditions = get_conditions(
            args,
            query_idx,
            frame_count,
            pose_source,
            marker_to_contact,
            inplane_offset_deg,
        )
        names = output_names(query_idx, args.video_type)
        prediction_path = transfer_dir / names["prediction"]
        writer = cv2.VideoWriter(
            str(prediction_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {prediction_path}")
        height_at_peak = None
        frame_at_zero = None
        frame_at_peak = None
        peak_depth = -1.0
        for frame_idx, condition in enumerate(conditions):
            normalized_height = predict_height(
                model,
                condition,
                args.inr_height,
                args.inr_width,
                feature_min,
                feature_max,
                args.inference_batch_size,
                device,
                normalization_mode,
            )
            rendered_frame = renderer.render(
                normalized_height, condition.displacement, (width, height)
            )
            writer.write(rendered_frame)
            if frame_idx == 0:
                frame_at_zero = rendered_frame.copy()
            if abs(condition.displacement) >= peak_depth:
                height_at_peak = normalized_height
                frame_at_peak = rendered_frame.copy()
                peak_depth = abs(condition.displacement)
        writer.release()

        np.savez_compressed(
            pose_dir / f"{query_idx}.npz",
            xyz=np.stack([item.xyz for item in conditions]),
            theta=np.array([item.theta for item in conditions]),
            phi=np.array([item.phi for item in conditions]),
            displacement=np.array([item.displacement for item in conditions]),
            source=pose_source,
        )
        if args.debug_images and height_at_peak is not None:
            cv2.imwrite(
                str(debug_dir / f"{query_idx}_predicted_height.png"),
                np.clip(height_at_peak * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(str(debug_dir / f"{query_idx}_depth0.png"), frame_at_zero)
            cv2.imwrite(str(debug_dir / f"{query_idx}_peak.png"), frame_at_peak)

        reference_video = args.ref_dir / f"{reference_idx}_{args.video_type}.mp4"
        query_video = args.query_dir / f"{query_idx}_{args.video_type}.mp4"
        if not reference_video.exists():
            raise FileNotFoundError(f"Missing reference video: {reference_video}")
        if not query_video.exists():
            raise FileNotFoundError(f"Missing evaluation query video: {query_video}")
        shutil.copy2(reference_video, transfer_dir / names["reference"])
        # Byte-copy only; query tactile content is not decoded during prediction.
        shutil.copy2(query_video, transfer_dir / names["query"])
        print(f"[inference] query={query_idx} ref={reference_idx} -> {prediction_path}")

    if not args.skip_eval:
        from transfer_pipeline import _evaluate_videos

        # Use one writable shared cache instead of downloading AlexNet per run.
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "log" / "model_cache" / "torch"))
        _evaluate_videos(
            pred_dir=str(transfer_dir),
            query_dir=str(args.query_dir),
            video_type=args.video_type,
            pred_glob="*_transferred.mp4",
            query_stem_fn=lambda index: f"{index}_{args.video_type}.mp4",
            out_pkl=str(transfer_dir / "metrics.pkl"),
        )


if __name__ == "__main__":
    main()
