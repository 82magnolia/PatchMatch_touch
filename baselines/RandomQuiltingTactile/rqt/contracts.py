"""Dependency-free baseline contracts used by the runner and unit tests."""

from __future__ import annotations

import csv
import re
from pathlib import Path

METRIC_KEYS = ("MSE", "PSNR", "SSIM", "LPIPS")


def format_scale(scale: float | None) -> str:
    return "" if scale is None else f"_scale{scale:g}"


def discover_indices(folder: Path, scale: float | None, modality: str = "normal") -> list[int]:
    """Discover touch indices without importing the heavyweight retrieval stack."""
    tag = re.escape(format_scale(scale))
    image_pattern = re.compile(rf"^(\d+){tag}_{re.escape(modality)}\.jpg$")
    video_pattern = re.compile(r"^(\d+)_(?:shadow|sim|tactile_normal)\.mp4$")
    image_indices = {
        int(match.group(1))
        for path in folder.iterdir()
        if (match := image_pattern.match(path.name))
    }
    if image_indices:
        return sorted(image_indices)
    return sorted(
        {
            int(match.group(1))
            for path in folder.iterdir()
            if (match := video_pattern.match(path.name))
        }
    )


def load_tsv(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query", "ref"]:
            raise ValueError(f"{path} must have the header: query<TAB>ref")
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
    """Resolve the single retrieved reference used by this baseline."""
    refs, queries = set(ref_indices), set(query_indices)
    if mode == "sim_gt_retrieval":
        pairs = [(idx, idx) for idx in sorted(refs & queries)]
    elif mode == "real_gt_retrieval":
        pairs = [(idx, idx - 1) for idx in sorted(queries) if idx % 2 == 1 and idx - 1 in refs]
    elif mode == "tsv":
        if tsv is None:
            raise ValueError("--tsv is required for retrieval_mode=tsv")
        pairs = [(q, r) for q, r in load_tsv(tsv) if q in queries and r in refs]
    else:
        raise ValueError(f"Pair resolution for {mode!r} requires retrieval results")
    if not pairs:
        raise ValueError(f"No valid query/reference pairs for retrieval mode {mode!r}")
    return pairs


def output_names(query_idx: int, video_type: str) -> dict[str, str]:
    return {
        "prediction": f"{query_idx}_transferred.mp4",
        "reference": f"{query_idx}_ref_{video_type}.mp4",
        "query": f"{query_idx}_query_{video_type}.mp4",
    }


def repeated_frames(frame, count: int):
    if count < 1:
        raise ValueError("Frame count must be positive")
    return [frame for _ in range(count)]


def metric_payload(per_touch: dict) -> dict:
    """Build the exact schema consumed by PatchMatch_touch/parse_metrics.py."""
    for query_idx, values in per_touch.items():
        missing = set(METRIC_KEYS) - set(values)
        if missing:
            raise ValueError(f"Metrics for {query_idx} are missing: {sorted(missing)}")
    average = {
        key: (
            sum(float(values[key]) for values in per_touch.values()) / len(per_touch)
            if per_touch
            else 0.0
        )
        for key in METRIC_KEYS
    }
    return {"per_touch": per_touch, "average": average}
