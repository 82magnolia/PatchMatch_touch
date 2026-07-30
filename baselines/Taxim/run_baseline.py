#!/usr/bin/env python3
"""Direct per-frame Taxim baseline for PatchMatch_touch."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from taxim_baseline.contracts import (
    RETRIEVAL_MODES,
    VIDEO_TYPES,
    discover_indices,
    output_names,
    resolve_pairs,
)
from taxim_baseline.depth import aruco_pressing_depths, back_forth_depths
from taxim_baseline.geometry import (
    load_real_geometry,
    load_real_pose_geometry,
    load_sim_height,
)
from taxim_baseline.media import VideoSink, require_cv2, video_info
from taxim_baseline.renderer import TaximHeightRenderer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ref_dir", required=True, type=Path)
    result.add_argument("--query_dir", required=True, type=Path)
    result.add_argument("--save_dir", required=True, type=Path)
    result.add_argument("--scale", nargs="+", type=float, default=[100.0])
    result.add_argument("--video_type", choices=VIDEO_TYPES, default="shadow")
    result.add_argument(
        "--retrieval_mode", choices=RETRIEVAL_MODES, default="sim_gt_retrieval"
    )
    result.add_argument("--tsv", type=Path)
    result.add_argument("--retrieval_modality", default="normal")
    result.add_argument("--dino_weights", type=Path)
    result.add_argument("--query_indices", nargs="+", type=int)
    result.add_argument("--data_mode", choices=["auto", "sim", "real"], default="auto")

    result.add_argument(
        "--calibration",
        type=Path,
        default=BASELINE_ROOT / "calibs" / "gelsight_pseudo_mini",
        help="Directory with dataPack.npz, polycalib.npz, and shadowTable.npz.",
    )
    result.add_argument(
        "--gel_map",
        type=Path,
        help="Gel height map; defaults to <calibration>/gelmap5.npy.",
    )
    result.add_argument("--background", type=Path)
    result.add_argument(
        "--mesh",
        type=Path,
        help="Optional source mesh provenance; derived per-query height NPZ is used at runtime.",
    )
    result.add_argument(
        "--sensor_offset_file",
        type=Path,
        default=PROJECT_ROOT / "log" / "gelsight_sensor_offset.json",
        help=(
            "GelSight marker-to-gel calibration JSON produced by "
            "real_data_transfer/calibrate_sensor_offset.py."
        ),
    )
    result.add_argument("--depth_sign", type=float, default=1.0)
    result.add_argument("--depth_offset_mm", type=float, default=0.0)
    result.add_argument("--depth_scale", type=float, default=1.0)
    result.add_argument(
        "--real_geometry_mode",
        choices=["full_pose", "legacy_scalar"],
        default="full_pose",
        help="Full pose rerasterizes object_cache with marker-to-gel calibration.",
    )
    result.add_argument(
        "--pose_inpaint_method", choices=["telea", "nearest", "ns"], default="telea"
    )
    result.add_argument("--interp_max_gap", type=int, default=3)
    result.add_argument("--smoothing_sigma", type=float, default=1.0)
    result.add_argument("--clip_min_mm", type=float, default=0.0)
    result.add_argument("--clip_max_mm", type=float, default=10.0)
    result.add_argument(
        "--surface_offset_mm",
        type=float,
        default=-5.0,
        help="Saved real height-map contact threshold offset (repository default: -5 mm).",
    )
    result.add_argument("--sim_press_min_mm", type=float, default=0.0)
    result.add_argument("--sim_press_max_mm", type=float, default=10.0)
    result.add_argument("--timing_suffix", default="render_mask")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--dry_run", action="store_true")
    result.add_argument("--skip_eval", action="store_true")
    result.add_argument("--debug_images", action="store_true")
    return result


def run_dinov3(args, directory: Path) -> list[tuple[int, int]]:
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
        str(directory),
        "--no_figures",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    with (directory / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    return [
        (int(row["query_idx"]), int(row["topk_ref_indices"][0]))
        for row in rows
        if row["topk_ref_indices"]
    ]


def save_pairing(save_dir: Path, pairs: list[tuple[int, int]], mode: str) -> None:
    retrieval = save_dir / "retrieval"
    retrieval.mkdir(parents=True, exist_ok=True)
    with (retrieval / "results.pkl").open("wb") as stream:
        pickle.dump(
            [
                {
                    "query_idx": query,
                    "topk_ref_indices": [reference],
                    "topk_similarities": None,
                }
                for query, reference in pairs
            ],
            stream,
        )
    if mode in ("sim_gt_retrieval", "real_gt_retrieval"):
        target = save_dir / (
            "odd_to_even.tsv" if mode == "real_gt_retrieval" else "identity.tsv"
        )
        target.write_text(
            "query\tref\n"
            + "".join(f"{query}\t{reference}\n" for query, reference in pairs)
        )


def resolve_mode(args) -> str:
    if args.data_mode != "auto":
        return args.data_mode
    return "real" if args.retrieval_mode == "real_gt_retrieval" else "sim"


def serializable_config(args, pairs, mode, records, sensor_offset):
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "pairs": pairs,
            "resolved_data_mode": mode,
            "per_query": records,
            "sensor_offset": sensor_offset,
            "query_tactile_used_for_prediction": False,
        }
    )
    return config


def main() -> None:
    args = parser().parse_args()
    for key in ("ref_dir", "query_dir", "save_dir", "calibration"):
        setattr(args, key, getattr(args, key).resolve())
    for key in (
        "tsv",
        "dino_weights",
        "gel_map",
        "background",
        "mesh",
        "sensor_offset_file",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.gel_map = args.gel_map or args.calibration / "gelmap5.npy"
    if not args.ref_dir.is_dir() or not args.query_dir.is_dir():
        raise SystemExit(f"Reference/query directory missing: {args.ref_dir}, {args.query_dir}")
    for path, label in (
        (args.mesh, "mesh"),
        (args.background, "background"),
    ):
        if path is not None and not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    mode = resolve_mode(args)
    sensor_offset = None
    if mode == "real":
        if not args.sensor_offset_file.is_file():
            raise SystemExit(
                f"Missing GelSight sensor offset JSON: {args.sensor_offset_file}"
            )
        try:
            sensor_offset = json.loads(args.sensor_offset_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Cannot read GelSight sensor offset JSON "
                f"{args.sensor_offset_file}: {exc}"
            ) from exc
        required = (
            "offset_x_m",
            "offset_y_m",
            "offset_z_m",
            "offset_theta_deg",
        )
        missing = [key for key in required if key not in sensor_offset]
        if missing:
            raise SystemExit(
                f"{args.sensor_offset_file} is missing calibration keys {missing}"
            )
        sensor_offset = {key: float(sensor_offset[key]) for key in required}
        if sensor_offset["offset_z_m"] < 0:
            raise SystemExit(
                f"{args.sensor_offset_file}: offset_z_m must be the positive "
                "marker-face to gel-tip distance"
            )
    if args.clip_max_mm < args.clip_min_mm:
        raise SystemExit("--clip_max_mm must be >= --clip_min_mm")
    np.random.seed(args.seed)

    refs = discover_indices(args.ref_dir, args.scale[0], args.retrieval_modality)
    queries = discover_indices(args.query_dir, args.scale[0], args.retrieval_modality)
    retrieval_dir = args.save_dir / "retrieval"
    if args.retrieval_mode == "dinov3":
        pairs = [] if args.dry_run else run_dinov3(args, retrieval_dir)
    else:
        try:
            pairs = resolve_pairs(args.retrieval_mode, refs, queries, args.tsv)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.query_indices is not None and pairs:
        requested = set(args.query_indices)
        pairs = [pair for pair in pairs if pair[0] in requested]
        missing = sorted(requested - {query for query, _ in pairs})
        if missing:
            raise SystemExit(f"Requested query indices are unavailable: {missing}")
    if not pairs and not (args.dry_run and args.retrieval_mode == "dinov3"):
        raise SystemExit("No pairs resolved")

    prepared = {}
    records = {}
    for query_idx, _ in pairs:
        timing = args.query_dir / f"{query_idx}_{args.timing_suffix}.mp4"
        try:
            count, width, height, fps = video_info(timing)
            if mode == "sim":
                geometry, geometry_path = load_sim_height(
                    args.query_dir, query_idx, args.scale[0]
                )
                depths = back_forth_depths(
                    count, args.sim_press_min_mm, args.sim_press_max_mm
                )
                depth_meta = {
                    "source": "known Taxim back_forth_press schedule",
                    "depth_min_mm": float(depths.min()),
                    "depth_max_mm": float(depths.max()),
                }
                valid = None
            else:
                if args.real_geometry_mode == "full_pose":
                    geometry, depth_meta = load_real_pose_geometry(
                        args.query_dir,
                        query_idx,
                        count,
                        sensor_offset=sensor_offset,
                        sensor_offset_file=args.sensor_offset_file,
                        max_gap=args.interp_max_gap,
                        inpaint_method=args.pose_inpaint_method,
                    )
                    valid = None
                    geometry_path = Path(depth_meta["object_cache"])
                    depths = np.asarray(
                        [
                            (
                                ((surface < args.surface_offset_mm / 1000.0) & mask)
                                .sum()
                                / max(int(mask.sum()), 1)
                            )
                            for surface, mask in geometry
                        ],
                        dtype=np.float64,
                    )
                    depth_meta["contact_fraction_min"] = float(depths.min())
                    depth_meta["contact_fraction_max"] = float(depths.max())
                else:
                    geometry, valid, geometry_path = load_real_geometry(
                        args.query_dir, query_idx
                    )
                    depths, depth_meta = aruco_pressing_depths(
                        args.query_dir,
                        query_idx,
                        count,
                        sign=args.depth_sign,
                        offset_mm=args.depth_offset_mm,
                        scale=args.depth_scale,
                        max_gap=args.interp_max_gap,
                        smoothing_sigma=args.smoothing_sigma,
                        clip_min_mm=args.clip_min_mm,
                        clip_max_mm=args.clip_max_mm,
                    )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        prepared[query_idx] = (timing, count, width, height, fps, geometry, valid, depths)
        records[str(query_idx)] = {
            "timing_video": str(timing),
            "geometry_path": str(geometry_path),
            "frame_count": count,
            "width": width,
            "height": height,
            "fps": fps,
            "depth": depth_meta,
        }

    config = serializable_config(args, pairs, mode, records, sensor_offset)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_pairing(args.save_dir, pairs, args.retrieval_mode)
    (args.save_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    transfer = args.save_dir / "transfer"
    depth_dir = args.save_dir / "depth"
    debug_dir = args.save_dir / "debug"
    transfer.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    if args.debug_images:
        debug_dir.mkdir(exist_ok=True)
    cv2 = require_cv2()

    renderers = {}
    for query_idx, reference_idx in pairs:
        timing, count, width, height, fps, geometry, valid, depths = prepared[query_idx]
        key = (height, width)
        if key not in renderers:
            try:
                renderers[key] = TaximHeightRenderer(
                    BASELINE_ROOT,
                    args.calibration,
                    args.gel_map,
                    height,
                    width,
                    args.background,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                raise SystemExit(str(exc)) from exc
        renderer = renderers[key]
        names = output_names(query_idx, args.video_type)
        sink = VideoSink(transfer / names["prediction"], width, height, fps)
        peak = int(np.argmax(depths))
        debug_frames = {}
        for frame_idx, depth_mm in enumerate(depths):
            if mode == "sim":
                frame, deformed, mask = renderer.render_sim(
                    geometry, float(depth_mm), args.video_type
                )
            elif args.real_geometry_mode == "full_pose":
                surface, surface_valid = geometry[frame_idx]
                frame, deformed, mask = renderer.render_real_pose(
                    surface,
                    surface_valid,
                    args.surface_offset_mm,
                    args.video_type,
                )
            else:
                frame, deformed, mask = renderer.render_real(
                    geometry,
                    valid,
                    float(depth_mm),
                    args.surface_offset_mm,
                    args.video_type,
                )
            sink.write(frame)
            if frame_idx in (0, peak, count - 1):
                debug_frames[frame_idx] = (frame.copy(), deformed.copy(), mask.copy())
        sink.close()
        payload = (
            {"contact_fraction": depths, "source": "real_full_pose"}
            if mode == "real" and args.real_geometry_mode == "full_pose"
            else {"depth_mm": depths, "source": mode}
        )
        np.savez_compressed(depth_dir / f"{query_idx}.npz", **payload)

        reference_video = args.ref_dir / f"{reference_idx}_{args.video_type}.mp4"
        query_video = args.query_dir / f"{query_idx}_{args.video_type}.mp4"
        for path, label in (
            (reference_video, "reference tactile video"),
            (query_video, "query tactile video for post-prediction packaging/evaluation"),
        ):
            if not path.is_file():
                raise SystemExit(f"Missing {label}: {path}")
        shutil.copy2(reference_video, transfer / names["reference"])
        shutil.copy2(query_video, transfer / names["query"])
        if args.debug_images:
            for frame_idx, (frame, deformed, mask) in debug_frames.items():
                stem = f"{query_idx}_frame{frame_idx:03}"
                cv2.imwrite(str(debug_dir / f"{stem}_render.png"), frame)
                cv2.imwrite(
                    str(debug_dir / f"{stem}_height.png"),
                    cv2.normalize(deformed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                )
                cv2.imwrite(
                    str(debug_dir / f"{stem}_mask.png"), mask.astype(np.uint8) * 255
                )

    if not args.skip_eval:
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from transfer_pipeline import _evaluate_videos
        except ImportError as exc:
            raise SystemExit(
                "Evaluation needs torch, lpips, and scikit-image; install environment.yml "
                "or pass --skip_eval."
            ) from exc
        _evaluate_videos(
            pred_dir=str(transfer),
            query_dir=str(args.query_dir),
            video_type=args.video_type,
            pred_glob="*_transferred.mp4",
            query_stem_fn=lambda index: f"{index}_{args.video_type}.mp4",
            out_pkl=str(transfer / "metrics.pkl"),
        )


if __name__ == "__main__":
    main()
