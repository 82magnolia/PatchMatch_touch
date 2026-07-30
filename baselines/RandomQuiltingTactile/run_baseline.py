#!/usr/bin/env python3
"""RandomQuiltingTactile baseline with 2D fallback and full TDF/Taxim mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from rqt.contracts import discover_indices, output_names, resolve_pairs


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for video execution. Activate the PatchMatch/Taxim "
            "environment or install opencv-python. --dry_run works without it."
        ) from exc
    return cv2


def video_info(path: Path):
    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 5.0
    capture.release()
    if count < 1 or width < 1 or height < 1:
        raise RuntimeError(f"Video has invalid metadata: {path}")
    return count, width, height, fps


def read_frame(path: Path, index: int):
    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return frame


def most_contact_frame(ref_dir: Path, ref_idx: int, ref_video: Path) -> int:
    """Inspect reference data only; prefer its explicit render-mask sequence."""
    cv2 = require_cv2()
    mask_path = ref_dir / f"{ref_idx}_render_mask.mp4"
    if mask_path.exists():
        capture = cv2.VideoCapture(str(mask_path))
        scores = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            scores.append(float(frame.mean()))
        capture.release()
        if scores:
            return max(range(len(scores)), key=scores.__getitem__)

    capture = cv2.VideoCapture(str(ref_video))
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Cannot read reference video: {ref_video}")
    blank_path = ref_dir / "blank_frame.jpg"
    blank = cv2.imread(str(blank_path)) if blank_path.exists() else first
    scores = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        resized = cv2.resize(blank, (frame.shape[1], frame.shape[0]))
        scores.append(float(cv2.absdiff(frame, resized).mean()))
    capture.release()
    return 1 + max(range(len(scores)), key=scores.__getitem__) if scores else 0


def write_repeated_video(image, output: Path, count: int, width: int, height: int, fps: float):
    cv2 = require_cv2()
    frame = cv2.resize(image, (width, height))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")
    for _ in range(count):
        writer.write(frame)
    writer.release()


def write_mapping(save_dir: Path, pairs, mode: str):
    name = "odd_to_even.tsv" if mode == "real_gt_retrieval" else "identity.tsv"
    if mode == "tsv":
        return None
    path = save_dir / name
    path.write_text(
        "query\tref\n" + "".join(f"{query}\t{reference}\n" for query, reference in pairs)
    )
    return path


def save_retrieval(save_dir: Path, pairs):
    retrieval = [
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
        pickle.dump(retrieval, stream)
    return target


def run_dinov3(args, save_dir: Path):
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
        str(save_dir / "retrieval"),
        "--no_figures",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    with (save_dir / "retrieval" / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    return [(int(row["query_idx"]), int(row["topk_ref_indices"][0])) for row in rows]


def cache_key(args, reference: int) -> str:
    values = {
        "object": args.object_id,
        "reference": reference,
        "block": args.quilt_block,
        "overlap": args.quilt_overlap,
        "tolerance": args.quilt_tolerance,
        "max_candidates": args.quilt_max_candidates,
        "config": str(args.tdf_config),
        "checkpoint": str(args.tdf_checkpoint),
        "seed": args.seed,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def full_prediction(args, quilt_path: Path, query_idx: int, reference: int, cache_dir: Path):
    """Run the imported normal->TDF->view normal->height->Taxim sequence."""
    required = {
        "--tdf_config": args.tdf_config,
        "--object_mesh": args.object_mesh,
        "--query_view_path": args.query_view_path,
        "--objectfolder_object_dir": args.objectfolder_object_dir,
        "--objectfile_checkpoint": args.objectfile_checkpoint,
        "--taxim_calibration": args.taxim_calibration,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError("Full mode requires " + ", ".join(missing))
    absent = [
        f"{flag}={value}"
        for flag, value in required.items()
        if flag != "--query_view_path" and not Path(value).exists()
    ]
    if args.tdf_checkpoint is not None and not args.tdf_checkpoint.exists():
        absent.append(f"--tdf_checkpoint={args.tdf_checkpoint}")
    if absent:
        raise FileNotFoundError("Full-mode assets do not exist: " + ", ".join(absent))
    key = cache_key(args, reference)
    texture_name = f"rqt_{key}"
    tdf_root = args.tdf_root.resolve()
    texture_dir = tdf_root / "data" / "tactile_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    staged_texture = texture_dir / f"{texture_name}_tactile_texture_map_2_normal.png"
    shutil.copy2(quilt_path, staged_texture)
    trained_mesh = tdf_root / "logs" / key / f"{key}.obj"

    if not trained_mesh.exists():
        if not args.train_if_missing:
            raise FileNotFoundError(
                f"Missing cached TDF mesh {trained_mesh}; rerun with --train_if_missing"
            )
        train_command = [
            args.tdf_python,
            str(tdf_root / "main.py"),
            "--config",
            str(args.tdf_config.resolve()),
            f"save_path={key}",
            f"mesh={args.object_mesh.resolve()}",
            f"tactile_texture_object={texture_name}",
        ]
        if args.tdf_checkpoint is not None:
            train_command.append(f"load={args.tdf_checkpoint.resolve()}")
        subprocess.run(
            train_command,
            cwd=tdf_root,
            check=True,
        )

    query_view = Path(str(args.query_view_path).format(query_idx=query_idx)).resolve()
    if not query_view.exists():
        raise FileNotFoundError(f"Missing query view point cloud: {query_view}")
    shared_outputs = BASELINE_ROOT / "outputs"
    shared_outputs.mkdir(exist_ok=True)
    subprocess.run(
        [
            args.tdf_python,
            str(tdf_root / "vis_render.py"),
            str(trained_mesh),
            "--mode",
            "viewspace_normal",
            "--num_azimuth",
            "1",
            "--view_path",
            str(query_view),
            "--scale",
            str(args.tdf_render_scale),
            "--save",
            str(shared_outputs),
        ],
        cwd=tdf_root,
        check=True,
    )
    subprocess.run(
        [
            args.tdf_python,
            str(tdf_root / "tactile_simul.py"),
            "--obj_number",
            key,
        ],
        cwd=tdf_root,
        check=True,
    )
    object_dir = args.objectfolder_object_dir.resolve()
    subprocess.run(
        [
            args.objectfolder_python,
            str(BASELINE_ROOT / "ObjectFolder" / "model_train.py"),
            "--mode",
            "eval",
            "-obj_path",
            str(object_dir),
            "--object_model",
            args.object_model,
            "--sample_ply",
            args.object_sample_ply,
            "--object_file_path",
            str(args.objectfile_checkpoint.resolve()),
            "--TDF_num",
            key,
            "-obj_scale_factor",
            str(args.object_scale_factor),
        ],
        cwd=BASELINE_ROOT / "ObjectFolder",
        env={**os.environ, "RQT_CALIB_DIR": str(args.taxim_calibration.resolve())},
        check=True,
    )
    prediction = shared_outputs / "sim_img.png"
    if not prediction.exists():
        raise RuntimeError(f"Taxim did not produce {prediction}")
    cached = cache_dir / f"full_{query_idx}.png"
    shutil.copy2(prediction, cached)
    return cached


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_dir", required=True, type=Path)
    parser.add_argument("--query_dir", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--scale", nargs="+", type=float, default=[100.0])
    parser.add_argument("--video_type", choices=["shadow", "sim", "tactile_normal"], default="shadow")
    parser.add_argument(
        "--retrieval_mode",
        choices=["dinov3", "tsv", "sim_gt_retrieval", "real_gt_retrieval"],
        default="sim_gt_retrieval",
    )
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--retrieval_modality", default="normal")
    parser.add_argument("--dino_weights", type=Path)
    parser.add_argument(
        "--query_indices",
        nargs="+",
        type=int,
        help="Optional query-index subset to run after retrieval/pairing.",
    )
    parser.add_argument("--pipeline_mode", choices=["fallback", "full"], default="fallback")
    parser.add_argument("--quilt_block", type=int, default=30)
    parser.add_argument("--quilt_overlap", type=int, default=6)
    parser.add_argument("--quilt_tolerance", type=float, default=0.1)
    parser.add_argument("--quilt_max_candidates", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--object_id", default="unknown")
    parser.add_argument("--tdf_root", type=Path, default=BASELINE_ROOT / "TactileDreamFusion")
    parser.add_argument("--tdf_config", type=Path)
    parser.add_argument("--tdf_checkpoint", type=Path)
    parser.add_argument("--tdf_python", default=sys.executable)
    parser.add_argument("--object_mesh", type=Path)
    parser.add_argument("--query_view_path", type=Path)
    parser.add_argument("--tdf_render_scale", type=float, default=1.5)
    parser.add_argument("--taxim_calibration", type=Path)
    parser.add_argument("--objectfolder_object_dir", type=Path)
    parser.add_argument("--objectfile_checkpoint", type=Path)
    parser.add_argument("--object_model", default="ObjectFile.pth")
    parser.add_argument("--object_sample_ply", default="contact_point.ply")
    parser.add_argument("--object_scale_factor", type=float, default=700.0)
    parser.add_argument("--objectfolder_python", default=sys.executable)
    parser.add_argument("--train_if_missing", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.ref_dir, args.query_dir, args.save_dir = (
        args.ref_dir.resolve(),
        args.query_dir.resolve(),
        args.save_dir.resolve(),
    )
    for label, folder in (("reference", args.ref_dir), ("query", args.query_dir)):
        if not folder.is_dir():
            raise SystemExit(f"{label.capitalize()} directory does not exist: {folder}")
    if args.retrieval_mode == "dinov3" and not args.dino_weights:
        raise SystemExit("--dino_weights is required for retrieval_mode=dinov3")

    ref_indices = discover_indices(args.ref_dir, args.scale[0], args.retrieval_modality)
    query_indices = discover_indices(args.query_dir, args.scale[0], args.retrieval_modality)
    if args.retrieval_mode == "dinov3":
        if args.dry_run:
            pairs = []
        else:
            pairs = run_dinov3(args, args.save_dir)
    else:
        pairs = resolve_pairs(args.retrieval_mode, ref_indices, query_indices, args.tsv)
    if args.query_indices is not None and pairs:
        requested = set(args.query_indices)
        pairs = [(query, reference) for query, reference in pairs if query in requested]
        missing = sorted(requested - {query for query, _ in pairs})
        if missing:
            raise SystemExit(
                "Requested query indices are absent from the resolved pairs: "
                + ", ".join(map(str, missing))
            )

    resolved = vars(args).copy()
    resolved.update(
        {
            "ref_dir": str(args.ref_dir),
            "query_dir": str(args.query_dir),
            "save_dir": str(args.save_dir),
            "pairs": pairs,
        }
    )
    for key, value in list(resolved.items()):
        if isinstance(value, Path):
            resolved[key] = str(value)
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True)
    )
    write_mapping(args.save_dir, pairs, args.retrieval_mode)
    if args.retrieval_mode != "dinov3":
        save_retrieval(args.save_dir, pairs)
    transfer_dir = args.save_dir / "transfer"
    cache_dir = args.save_dir / "cache"
    transfer_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    cv2 = require_cv2()
    from rqt.quilting import quilt

    for query_idx, ref_idx in pairs:
        names = output_names(query_idx, args.video_type)
        ref_video = args.ref_dir / f"{ref_idx}_{args.video_type}.mp4"
        query_video = args.query_dir / f"{query_idx}_{args.video_type}.mp4"
        query_mask = args.query_dir / f"{query_idx}_render_mask.mp4"
        for path, purpose in (
            (ref_video, "reference tactile video"),
            (query_video, "query tactile video (evaluation only)"),
            (query_mask, "query render mask used for output timing"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {purpose}: {path}")

        frame_count, width, height, fps = video_info(query_mask)
        selected = most_contact_frame(args.ref_dir, ref_idx, ref_video)
        preferred = args.ref_dir / f"{ref_idx}_tactile_normal.mp4"
        patch_video = preferred if preferred.exists() else ref_video
        patch = read_frame(patch_video, selected)
        synthesized = quilt(
            patch,
            (height, width),
            block=args.quilt_block,
            overlap=args.quilt_overlap,
            tolerance=args.quilt_tolerance,
            seed=args.seed,
            max_candidates=args.quilt_max_candidates,
        )
        quilt_path = cache_dir / f"ref_{ref_idx}_quilted.png"
        cv2.imwrite(str(quilt_path), synthesized)
        prediction_image = quilt_path
        if args.pipeline_mode == "full":
            prediction_image = full_prediction(
                args, quilt_path, query_idx, ref_idx, cache_dir
            )
        image = cv2.imread(str(prediction_image))
        if image is None:
            raise RuntimeError(f"Cannot read generated image: {prediction_image}")
        write_repeated_video(
            image,
            transfer_dir / names["prediction"],
            frame_count,
            width,
            height,
            fps,
        )
        shutil.copy2(ref_video, transfer_dir / names["reference"])
        shutil.copy2(query_video, transfer_dir / names["query"])
        if args.debug_images:
            debug_dir = args.save_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{query_idx}_selected_ref.png"), patch)
            shutil.copy2(quilt_path, debug_dir / f"{query_idx}_quilted.png")

    if not args.skip_eval:
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from transfer_pipeline import _evaluate_videos
        except ImportError as exc:
            raise SystemExit(
                "Evaluation dependencies are unavailable. Activate the PatchMatch "
                "environment (torch, lpips, scikit-image) or use --skip_eval."
            ) from exc
        _evaluate_videos(
            pred_dir=str(transfer_dir),
            query_dir=str(args.query_dir),
            video_type=args.video_type,
            pred_glob="*_transferred.mp4",
            query_stem_fn=lambda idx: f"{idx}_{args.video_type}.mp4",
            out_pkl=str(transfer_dir / "metrics.pkl"),
        )


if __name__ == "__main__":
    main()
