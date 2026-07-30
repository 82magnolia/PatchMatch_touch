"""Dependency-free PatchMatch input/output contracts."""

from __future__ import annotations

import csv
import re
from pathlib import Path

METRIC_KEYS = ("MSE", "PSNR", "SSIM", "LPIPS")
VIDEO_TYPES = ("shadow", "sim", "tactile_normal")
RETRIEVAL_MODES = ("dinov3", "tsv", "sim_gt_retrieval", "real_gt_retrieval")


def discover_indices(folder: Path, scale: float | None, modality: str = "normal") -> list[int]:
    tag = "" if scale is None else f"_scale{scale:g}"
    still = re.compile(rf"^(\d+){re.escape(tag)}_{re.escape(modality)}\.(?:jpg|png)$")
    video = re.compile(r"^(\d+)_(?:shadow|sim|tactile_normal|render_mask)\.mp4$")
    images = {
        int(match.group(1))
        for path in folder.iterdir()
        if path.is_file() and (match := still.match(path.name))
    }
    if images:
        return sorted(images)
    return sorted(
        {
            int(match.group(1))
            for path in folder.iterdir()
            if path.is_file() and (match := video.match(path.name))
        }
    )


def load_tsv(path: Path) -> list[tuple[int, int]]:
    pairs = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query", "ref"]:
            raise ValueError(f"{path} must have header query<TAB>ref")
        for row in reader:
            refs = [part.strip() for part in row["ref"].split(",") if part.strip()]
            if not refs:
                raise ValueError(f"Query {row['query']} has no reference in {path}")
            pairs.append((int(row["query"]), int(refs[0])))
    return pairs


def resolve_pairs(
    mode: str,
    ref_indices: list[int],
    query_indices: list[int],
    tsv: Path | None = None,
) -> list[tuple[int, int]]:
    refs, queries = set(ref_indices), set(query_indices)
    if mode == "sim_gt_retrieval":
        pairs = [(index, index) for index in sorted(refs & queries)]
    elif mode == "real_gt_retrieval":
        pairs = [
            (index, index - 1)
            for index in sorted(queries)
            if index % 2 == 1 and index - 1 in refs
        ]
    elif mode == "tsv":
        if tsv is None:
            raise ValueError("--tsv is required for retrieval_mode=tsv")
        pairs = [(q, r) for q, r in load_tsv(tsv) if q in queries and r in refs]
    else:
        raise ValueError(f"Pair resolution for {mode!r} requires retrieval results")
    if not pairs:
        raise ValueError(f"No valid query/reference pairs for {mode!r}")
    return pairs


def output_names(query_idx: int, video_type: str) -> dict[str, str]:
    if video_type not in VIDEO_TYPES:
        raise ValueError(video_type)
    return {
        "prediction": f"{query_idx}_transferred.mp4",
        "reference": f"{query_idx}_ref_{video_type}.mp4",
        "query": f"{query_idx}_query_{video_type}.mp4",
    }


def metric_payload(per_touch: dict[int, dict[str, float]]) -> dict:
    for index, values in per_touch.items():
        missing = set(METRIC_KEYS) - set(values)
        if missing:
            raise ValueError(f"Metrics for {index} are missing: {sorted(missing)}")
    average = {
        key: (
            sum(float(values[key]) for values in per_touch.values()) / len(per_touch)
            if per_touch
            else 0.0
        )
        for key in METRIC_KEYS
    }
    return {"per_touch": per_touch, "average": average}

