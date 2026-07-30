"""Known simulation schedules and calibrated ArUco-to-pressing-depth conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def back_forth_depths(count: int, minimum_mm: float = 0.0, maximum_mm: float = 10.0):
    """Match Taxim/gen_contact_video.py's back_forth_press schedule."""
    if count < 1:
        raise ValueError("Frame count must be positive")
    forward = np.linspace(minimum_mm, maximum_mm, count // 2)
    backward = np.linspace(maximum_mm, minimum_mm, count - count // 2 + 1)
    return np.concatenate([forward, backward[1:]])[:count]


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill bounded NaN runs no longer than ``max_gap``."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1:
        raise ValueError("Expected a one-dimensional depth sequence")
    missing = ~np.isfinite(result)
    start = None
    for index in range(len(result) + 1):
        active = index < len(result) and missing[index]
        if active and start is None:
            start = index
        if start is not None and not active:
            end = index - 1
            length = end - start + 1
            if (
                length <= max_gap
                and start > 0
                and index < len(result)
                and np.isfinite(result[start - 1])
                and np.isfinite(result[index])
            ):
                result[start:index] = np.linspace(
                    result[start - 1], result[index], length + 2
                )[1:-1]
            start = None
    return result


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    return np.interp(target, source, values)


def aruco_pressing_depths(
    folder: Path,
    touch_idx: int,
    count: int,
    *,
    sign: float = 1.0,
    offset_mm: float = 0.0,
    scale: float = 1.0,
    max_gap: int = 3,
    smoothing_sigma: float = 1.0,
    clip_min_mm: float = 0.0,
    clip_max_mm: float = 10.0,
    align_contact_window: bool = False,
) -> tuple[np.ndarray, dict]:
    """Use dot(sensor_z_0, tvec_i - tvec_0), the repository convention."""
    pose_path = folder / f"{touch_idx}_pose_contact.npz"
    contact_path = folder / f"{touch_idx}_contact_data.npz"
    if not pose_path.is_file() or not contact_path.is_file():
        raise FileNotFoundError(
            f"Real Taxim needs {pose_path.name} and {contact_path.name}; "
            "depth is never inferred from query tactile images."
        )
    with np.load(pose_path) as pose:
        tvecs = np.asarray(pose["tvecs"], dtype=np.float64)
    with np.load(contact_path) as contact:
        origin = np.asarray(contact["tvec_0"], dtype=np.float64)
        normal = np.asarray(contact["sensor_z_0"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    raw_m = np.full(len(tvecs), np.nan, dtype=np.float64)
    valid = np.isfinite(tvecs).all(axis=1)
    raw_m[valid] = (tvecs[valid] - origin) @ normal
    filled_m = interpolate_short_gaps(raw_m, max_gap)

    # pose_contact.npz is already the capture's trimmed contact slice. The
    # cs_idx/ce_idx values in meta.json index the original frame buffer and
    # therefore must not normally be applied a second time.
    start, end = 0, len(filled_m) - 1
    meta_path = folder / f"{touch_idx}_meta.json"
    if align_contact_window and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        start = max(0, int(meta.get("cs_idx", start)))
        end = min(len(filled_m) - 1, int(meta.get("ce_idx", end)))
        if end >= start:
            filled_m = filled_m[start : end + 1]
    if not np.isfinite(filled_m).all():
        bad = int((~np.isfinite(filled_m)).sum())
        raise ValueError(
            f"ArUco sequence {touch_idx} retains {bad} missing samples after filling "
            f"only gaps <= {max_gap}. Increase --interp_max_gap only if justified."
        )
    depth_mm = sign * filled_m * 1000.0
    depth_mm = depth_mm * scale + offset_mm
    endpoint_depths = np.clip(
        depth_mm[[0, -1]], clip_min_mm, clip_max_mm
    )
    if smoothing_sigma > 0 and len(depth_mm) > 1:
        depth_mm = gaussian_filter1d(depth_mm, sigma=smoothing_sigma, mode="nearest")
    depth_mm = np.clip(depth_mm, clip_min_mm, clip_max_mm)
    # Smoothing must not move the calibrated contact-boundary poses. In the
    # standard captures this keeps the transferred video at depth 0 initially.
    depth_mm[0], depth_mm[-1] = endpoint_depths
    depth_mm = _resample(depth_mm, count)
    metadata = {
        "formula": "clip((sign * dot(sensor_z_0, tvec_i - tvec_0) * 1000) * scale + offset_mm)",
        "pose_path": str(pose_path),
        "contact_path": str(contact_path),
        "raw_count": len(raw_m),
        "valid_raw_count": int(valid.sum()),
        "contact_window": [start, end],
        "depth_min_mm": float(depth_mm.min()),
        "depth_max_mm": float(depth_mm.max()),
    }
    return depth_mm, metadata
