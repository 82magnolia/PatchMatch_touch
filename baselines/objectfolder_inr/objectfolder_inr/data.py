"""Reference-only training data loaders for the ObjectFolder INR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose import PoseCondition, load_aruco_conditions, load_sim_contact_points


def height_path(folder: Path, touch_idx: int, scale: float | None) -> Path | None:
    candidates = []
    if scale is not None:
        candidates.append(folder / f"{touch_idx}_scale{scale:g}_height.npz")
    candidates.extend(
        (
            folder / f"{touch_idx}_height.npz",
            folder / f"{touch_idx}_contact_data.npz",
        )
    )
    return next((path for path in candidates if path.exists()), None)


def load_pseudo_height(folder: Path, touch_idx: int, scale: float | None) -> np.ndarray:
    path = height_path(folder, touch_idx, scale)
    if path is None:
        raise FileNotFoundError(
            f"No numeric height target for touch {touch_idx} in {folder}; expected "
            f"{touch_idx}_scale<scale>_height.npz, {touch_idx}_height.npz, or "
            f"{touch_idx}_contact_data.npz"
        )
    with np.load(path) as archive:
        if "height" in archive.files:
            height = archive["height"]
        elif "height_map_0" in archive.files:
            height = archive["height_map_0"]
        else:
            raise KeyError(f"{path} contains no 'height' or 'height_map_0'")
    height = np.nan_to_num(np.asarray(height, dtype=np.float32))
    low, high = np.percentile(height, (1.0, 99.0))
    if high - low < 1e-8:
        return np.zeros_like(height, dtype=np.float32)
    return np.clip((height - low) / (high - low), 0.0, 1.0).astype(np.float32)


def condition_for_touch(
    folder: Path,
    touch_idx: int,
    *,
    pose_source: str,
    contact_points: Path | None,
    allow_index_coordinate_fallback: bool,
    marker_to_contact: np.ndarray,
    inplane_offset_deg: float,
) -> PoseCondition:
    if pose_source == "aruco":
        conditions = load_aruco_conditions(
            folder,
            touch_idx,
            50,
            marker_to_contact=marker_to_contact,
            inplane_offset_deg=inplane_offset_deg,
        )
        return max(conditions, key=lambda item: item.displacement)

    if contact_points is not None:
        points = load_sim_contact_points(contact_points)
        if touch_idx >= len(points):
            raise IndexError(
                f"Touch {touch_idx} has no coordinate in {contact_points} "
                f"({len(points)} points)"
            )
        return PoseCondition(points[touch_idx], 0.0, 0.0, 0.0)

    if not allow_index_coordinate_fallback:
        raise FileNotFoundError(
            "Simulation ObjectFolder requires --contact_points from the Taxim "
            "generation run. The shared rendered directory does not encode xyz. "
            "Use --allow_index_coordinate_fallback only for a smoke test."
        )
    # Deliberately explicit, deterministic smoke-test surrogate.
    return PoseCondition(np.array([float(touch_idx), 0.0, 0.0]), 0.0, 0.0, 0.0)


def raw_condition_features(condition: PoseCondition) -> np.ndarray:
    return np.array(
        [
            *condition.xyz,
            condition.theta,
            np.cos(condition.phi),
            np.sin(condition.phi),
            condition.displacement,
        ],
        dtype=np.float32,
    )


def feature_bounds(conditions: list[PoseCondition]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack([raw_condition_features(item) for item in conditions])
    minimum = np.concatenate((raw.min(axis=0), [0.0, 0.0])).astype(np.float32)
    maximum = np.concatenate((raw.max(axis=0), [1.0, 1.0])).astype(np.float32)
    # Keep displacement useful at inference even if static reference targets
    # were captured at a single depth.
    minimum[6] = min(minimum[6], -0.012)
    maximum[6] = max(maximum[6], 0.012)
    same = np.isclose(minimum, maximum)
    minimum[same] -= 0.5
    maximum[same] += 0.5
    return minimum, maximum
