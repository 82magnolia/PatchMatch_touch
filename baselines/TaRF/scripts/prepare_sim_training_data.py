#!/usr/bin/env python3
"""Pair Taxim simulation RGB/height views with peak-contact tactile targets."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--roots", nargs="+", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--limit", type=int)
    result.add_argument(
        "--split-mode",
        choices=("object_buckets", "rebot_finetune"),
        default="object_buckets",
        help=(
            "object_buckets uses the original modulo split; rebot_finetune "
            "uses sim IDs 21-100 for training and holds out IDs 1-20."
        ),
    )
    result.add_argument(
        "--target-video-type",
        choices=("shadow", "tactile_normal"),
        default="shadow",
        help="Video modality from which to extract the peak-contact target.",
    )
    return result


def split_for_object(object_id: str, split_mode: str) -> str | None:
    value = int(object_id)
    if split_mode == "rebot_finetune":
        if 21 <= value <= 100:
            return "train"
        if 1 <= value <= 10:
            return "val"
        if 11 <= value <= 20:
            return "test"
        return None
    bucket = (int(object_id) - 1) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


def peak_contact_frame(mask_video: Path, touch_video: Path):
    masks = cv2.VideoCapture(str(mask_video))
    best_index, best_area, index = 0, -1, 0
    while True:
        ok, frame = masks.read()
        if not ok:
            break
        area = int((frame[..., 0] > 127).sum())
        if area > best_area:
            best_index, best_area = index, area
        index += 1
    masks.release()
    touch = cv2.VideoCapture(str(touch_video))
    touch.set(cv2.CAP_PROP_POS_FRAMES, best_index)
    ok, frame = touch.read()
    touch.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {best_index} from {touch_video}")
    return frame, best_index


def build_record(task):
    root_index, root, object_dir, touch_index, target_dir, target_video_type = task
    prefix = object_dir / str(touch_index)
    required = {
        "rgb_40_50": prefix.with_name(f"{touch_index}_scale25_color.jpg"),
        "height_40_50": prefix.with_name(f"{touch_index}_scale25_height.npz"),
        "rgb_0_40": prefix.with_name(f"{touch_index}_scale100_color.jpg"),
        "height_0_40": prefix.with_name(f"{touch_index}_scale100_height.npz"),
    }
    mask = prefix.with_name(f"{touch_index}_mask.mp4")
    video = prefix.with_name(f"{touch_index}_{target_video_type}.mp4")
    missing = [path for path in [*required.values(), mask, video] if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(map(str, missing)))
    sample_id = f"r{root_index}_o{object_dir.name}_t{touch_index}"
    target = target_dir / f"{sample_id}.jpg"
    if not target.is_file():
        frame, peak_index = peak_contact_frame(mask, video)
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Cannot write {target}")
    else:
        peak_index = None
    return {
        "id": sample_id,
        "source_root": str(root.resolve()),
        "object_id": object_dir.name,
        "touch_index": touch_index,
        "peak_frame": peak_index,
        "target_video_type": target_video_type,
        **{key: str(value.resolve()) for key, value in required.items()},
        "touch": str(target.resolve()),
    }


def main() -> None:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target_dir = args.output / "touch"
    target_dir.mkdir(exist_ok=True)
    tasks = []
    for root_index, root in enumerate(args.roots):
        for object_dir in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        ):
            if split_for_object(object_dir.name, args.split_mode) is None:
                continue
            indices = sorted(
                int(path.name.split("_", 1)[0])
                for path in object_dir.glob("*_scale100_color.jpg")
            )
            tasks.extend(
                (
                    root_index,
                    root,
                    object_dir,
                    index,
                    target_dir,
                    args.target_video_type,
                )
                for index in indices
            )
    if args.limit is not None:
        tasks = tasks[: args.limit]
    manifest = {"train": [], "val": [], "test": []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(build_record, tasks):
            split = split_for_object(record["object_id"], args.split_mode)
            if split is not None:
                manifest[split].append(record)
    manifest["metadata"] = {
        "roots": [str(root.resolve()) for root in args.roots],
        "pairing": {"40_50": "scale25", "0_40": "scale100"},
        "target_video_type": args.target_video_type,
        "target": (
            f"maximum-contact frame from {args.target_video_type}.mp4 "
            "selected by mask.mp4"
        ),
        "counts": {split: len(manifest[split]) for split in ("train", "val", "test")},
        "split_mode": args.split_mode,
    }
    if args.split_mode == "rebot_finetune":
        manifest["metadata"]["object_ids"] = {
            "train": list(range(21, 101)),
            "val": list(range(1, 11)),
            "test": list(range(11, 21)),
        }
    if args.target_video_type == "tactile_normal":
        first = tasks[0]
        _, _, object_dir, touch_index, _, target_video_type = first
        video = object_dir / f"{touch_index}_{target_video_type}.mp4"
        capture = cv2.VideoCapture(str(video))
        ok, background = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot read no-contact frame from {video}")
        background_path = args.output / "tactile_normal_background.jpg"
        if not cv2.imwrite(str(background_path), background):
            raise RuntimeError(f"Cannot write {background_path}")
        manifest["metadata"]["background"] = str(background_path.resolve())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["metadata"], indent=2))


if __name__ == "__main__":
    main()
