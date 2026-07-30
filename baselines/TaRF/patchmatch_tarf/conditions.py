"""Resolve and preprocess TaRF RGB, depth, and sensor-background conditions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueryConditions:
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    background_path: Path
    duplicated_single_view: bool = False

    def as_dict(self) -> dict:
        return {
            "rgb_paths": [str(path) for path in self.rgb_paths],
            "depth_paths": [str(path) for path in self.depth_paths],
            "background_path": str(self.background_path),
            "duplicated_single_view": self.duplicated_single_view,
        }


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def _manifest_entry(manifest: Path, query_idx: int) -> tuple[list[Path], list[Path]]:
    data = json.loads(manifest.read_text())
    entry = data.get(str(query_idx), data.get(query_idx))
    if entry is None:
        raise FileNotFoundError(f"Condition manifest {manifest} has no query {query_idx}")
    base = manifest.parent

    def paths(key: str) -> list[Path]:
        return [
            (base / value).resolve() if not Path(value).is_absolute() else Path(value)
            for value in entry.get(key, [])
        ]

    return paths("rgb"), paths("depth")


def resolve_conditions(
    query_dir: Path,
    query_idx: int,
    scale: float,
    background_path: Path,
    conditions_dir: Path | None = None,
    manifest: Path | None = None,
) -> QueryConditions:
    """Resolve two RGB/depth views without ever consulting query tactile videos."""
    if not background_path.is_file():
        raise FileNotFoundError(f"Sensor background does not exist: {background_path}")
    root = conditions_dir.resolve() if conditions_dir else query_dir.resolve()
    if manifest:
        rgbs, depths = _manifest_entry(manifest.resolve(), query_idx)
    else:
        scale_text = f"{scale:g}"
        per_query = root / str(query_idx)
        rgb_view_candidates = [
            per_query / "rgb" / "40_50.png",
            per_query / "rgb" / "40_50.jpg",
            root / "rgb" / f"{query_idx}_40_50.png",
        ]
        rgb_second_candidates = [
            per_query / "rgb" / "0_40.png",
            per_query / "rgb" / "0_40.jpg",
            root / "rgb" / f"{query_idx}_0_40.png",
        ]
        depth_view_candidates = [
            per_query / "depth" / "40_50.npy",
            per_query / "depth" / "40_50.npz",
            root / "depth" / f"{query_idx}_40_50.npy",
        ]
        depth_second_candidates = [
            per_query / "depth" / "0_40.npy",
            per_query / "depth" / "0_40.npz",
            root / "depth" / f"{query_idx}_0_40.npy",
        ]
        flat_rgb = [
            root / f"{query_idx}_scale{scale_text}_color.jpg",
            root / f"{query_idx}_scale{scale_text}_color.png",
            root / f"{query_idx}_color.jpg",
            root / f"{query_idx}_rgb.png",
        ]
        flat_depth = [
            root / f"{query_idx}_scale{scale_text}_height.npz",
            root / f"{query_idx}_scale{scale_text}_height.npy",
            root / f"{query_idx}_depth.npy",
            root / f"{query_idx}_depth.npz",
            root / f"{query_idx}_height.npz",
            root / f"{query_idx}_height.jpg",
            root / f"{query_idx}_height.png",
        ]
        first_rgb = _first_existing(rgb_view_candidates) or _first_existing(flat_rgb)
        second_rgb = _first_existing(rgb_second_candidates)
        first_depth = _first_existing(depth_view_candidates) or _first_existing(flat_depth)
        second_depth = _first_existing(depth_second_candidates)
        rgbs = [path for path in (first_rgb, second_rgb) if path]
        depths = [path for path in (first_depth, second_depth) if path]

    missing = []
    if not rgbs:
        missing.append(
            f"RGB: {query_idx}_scale{scale:g}_color.jpg, {query_idx}_color.jpg, "
            f"or {query_idx}/rgb/{{40_50,0_40}}.png"
        )
    if not depths:
        missing.append(
            f"depth: {query_idx}_scale{scale:g}_height.npz, {query_idx}_depth.npy, "
            f"{query_idx}_height.jpg, or {query_idx}/depth/{{40_50,0_40}}.npy"
        )
    absent = [str(path) for path in [*rgbs, *depths] if not path.is_file()]
    if missing or absent:
        detail = "; ".join(missing + ([f"manifest paths absent: {', '.join(absent)}"] if absent else []))
        raise FileNotFoundError(
            f"Missing TaRF query conditions for query {query_idx} under {root}. "
            f"Accepted inputs: {detail}. Query tactile GT is intentionally not a fallback."
        )
    duplicated = len(rgbs) == 1 or len(depths) == 1
    while len(rgbs) < 2:
        rgbs.append(rgbs[0])
    while len(depths) < 2:
        depths.append(depths[0])
    if len(rgbs) != len(depths):
        raise ValueError(
            f"Query {query_idx} has {len(rgbs)} RGB views but {len(depths)} depth views"
        )
    return QueryConditions(tuple(rgbs[:2]), tuple(depths[:2]), background_path, duplicated)


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        for key in ("depth", "height", "arr_0"):
            if key in archive:
                array = archive[key]
                break
        else:
            if len(archive.files) != 1:
                raise ValueError(f"Cannot choose a depth array from {path}: {archive.files}")
            array = archive[archive.files[0]]
    elif path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is needed to read a depth image") from exc
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read depth image: {path}")
        # Real-data height previews are 8-bit normalized maps. Mapping to the
        # upstream 0..5 m conditioning range preserves their full contrast.
        array = image.astype(np.float32) / 255.0 * 5.0
    else:
        raise ValueError(f"Depth must be .npy, .npz, or an 8-bit image: {path}")
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Depth array must be HxW after squeeze, got {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Depth contains NaN/Inf: {path}")
    return array
