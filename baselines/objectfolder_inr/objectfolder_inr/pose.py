"""Convert saved ArUco marker poses into ObjectFolder TouchNet conditions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Values used when the shared real dataset was captured.
DEFAULT_MARKER_TO_CONTACT_M = np.array([0.0, -0.0033, -0.0512], dtype=np.float64)
DEFAULT_INPLANE_OFFSET_DEG = -6.6


def load_sensor_offset(path: Path) -> tuple[np.ndarray, float, dict]:
    """Load capture calibration and convert positive gel distance to local -Z."""
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read GelSight sensor offset {path}: {exc}") from exc
    required = (
        "offset_x_m",
        "offset_y_m",
        "offset_z_m",
        "offset_theta_deg",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"GelSight sensor offset {path} is missing {missing}")
    resolved = {key: float(values[key]) for key in required}
    if resolved["offset_z_m"] < 0:
        raise ValueError(
            f"{path}: offset_z_m must be the positive marker-face to gel-tip distance"
        )
    marker_to_contact = np.array(
        [
            resolved["offset_x_m"],
            resolved["offset_y_m"],
            -resolved["offset_z_m"],
        ],
        dtype=np.float64,
    )
    return marker_to_contact, resolved["offset_theta_deg"], resolved


@dataclass(frozen=True)
class PoseCondition:
    xyz: np.ndarray
    theta: float
    phi: float
    displacement: float

    def serializable(self) -> dict:
        data = asdict(self)
        data["xyz"] = self.xyz.tolist()
        return data


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation without requiring OpenCV in unit tests."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rvec / angle
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def condition_from_pose(
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    marker_to_contact: np.ndarray = DEFAULT_MARKER_TO_CONTACT_M,
    origin_tvec: np.ndarray | None = None,
    origin_normal: np.ndarray | None = None,
    inplane_offset_deg: float = DEFAULT_INPLANE_OFFSET_DEG,
) -> PoseCondition:
    """Derive contact coordinate, inclination, azimuth, and indentation.

    The sensor contact normal is the marker's negative Z axis. Indentation is
    marker motion projected on the contact-start sensor normal.
    """
    rotation = rodrigues(rvec)
    contact = rotation @ np.asarray(marker_to_contact, dtype=np.float64) + tvec
    normal = -(rotation[:, 2])
    normal /= max(np.linalg.norm(normal), 1e-12)
    theta = float(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
    phi = (
        0.0
        if np.linalg.norm(normal[:2]) < 1e-12
        else float(
            np.arctan2(normal[1], normal[0]) + np.deg2rad(inplane_offset_deg)
        )
    )
    displacement = 0.0
    if origin_tvec is not None:
        axis = normal if origin_normal is None else np.asarray(origin_normal, dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        displacement = float(np.dot(axis, np.asarray(tvec) - np.asarray(origin_tvec)))
    return PoseCondition(contact, theta, phi, displacement)


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    x = np.arange(len(values))
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if not valid.any():
            raise ValueError("ArUco pose sequence has no valid marker detections")
        values[:, column] = np.interp(x, x[valid], values[valid, column])
    return values


def resample_pose_sequence(rvecs: np.ndarray, tvecs: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("Frame count must be positive")
    rvecs = _interpolate_missing(rvecs)
    tvecs = _interpolate_missing(tvecs)
    source = np.linspace(0.0, 1.0, len(rvecs))
    target = np.linspace(0.0, 1.0, count)
    out_r = np.stack([np.interp(target, source, rvecs[:, i]) for i in range(3)], axis=1)
    out_t = np.stack([np.interp(target, source, tvecs[:, i]) for i in range(3)], axis=1)
    return out_r, out_t


def load_aruco_conditions(
    folder: Path,
    touch_idx: int,
    count: int,
    *,
    marker_to_contact: np.ndarray = DEFAULT_MARKER_TO_CONTACT_M,
    inplane_offset_deg: float = DEFAULT_INPLANE_OFFSET_DEG,
) -> list[PoseCondition]:
    pose_path = folder / f"{touch_idx}_pose_contact.npz"
    if not pose_path.exists():
        raise FileNotFoundError(
            f"Missing ArUco poses: {pose_path}. Real ObjectFolder inference requires "
            "{idx}_pose_contact.npz from the capture/reprocessing pipeline."
        )
    with np.load(pose_path) as pose:
        raw_rvecs = pose["rvecs"]
        raw_tvecs = pose["tvecs"]
    meta_path = folder / f"{touch_idx}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        start = max(0, int(meta.get("cs_idx", 0)))
        end = min(len(raw_rvecs) - 1, int(meta.get("ce_idx", len(raw_rvecs) - 1)))
        if end >= start:
            raw_rvecs = raw_rvecs[start : end + 1]
            raw_tvecs = raw_tvecs[start : end + 1]
    rvecs, tvecs = resample_pose_sequence(raw_rvecs, raw_tvecs, count)

    contact_path = folder / f"{touch_idx}_contact_data.npz"
    if contact_path.exists():
        with np.load(contact_path) as contact:
            origin_tvec = np.asarray(contact["tvec_0"], dtype=np.float64)
            origin_normal = np.asarray(contact["sensor_z_0"], dtype=np.float64)
    else:
        origin_tvec = tvecs[0]
        origin_normal = -(rodrigues(rvecs[0])[:, 2])

    conditions = [
        condition_from_pose(
            rvec,
            tvec,
            marker_to_contact=marker_to_contact,
            origin_tvec=origin_tvec,
            origin_normal=origin_normal,
            inplane_offset_deg=inplane_offset_deg,
        )
        for rvec, tvec in zip(rvecs, tvecs)
    ]
    # The stored origin can be the first valid pose or the peak pose, depending
    # on marker visibility. Re-anchor to the aligned contact-window start and
    # clamp withdrawal motion: post-contact marker travel can be centimetres
    # and must not be mistaken for gel indentation.
    baseline = conditions[0].displacement
    return [
        PoseCondition(item.xyz, item.theta, item.phi, max(item.displacement - baseline, 0.0))
        for item in conditions
    ]


def load_sim_contact_points(path: Path) -> np.ndarray:
    """Load contact coordinates from NPY/NPZ or an ASCII PLY."""
    if path.suffix == ".npy":
        points = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive.files else archive.files[0]
        points = archive[key]
    elif path.suffix == ".json":
        points = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    elif path.suffix == ".ply":
        lines = path.read_text().splitlines()
        try:
            end = lines.index("end_header")
        except ValueError as exc:
            raise ValueError(f"{path} is not an ASCII PLY") from exc
        points = np.asarray(
            [[float(part) for part in line.split()[:3]] for line in lines[end + 1 :] if line.strip()]
        )
    else:
        raise ValueError("--contact_points supports .npy, .npz, .json, or ASCII .ply")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Contact points must have shape (N,3+), got {points.shape}")
    return points[:, :3]


def sim_conditions(
    point: np.ndarray,
    count: int,
    press_min_mm: float = 0.0,
    press_max_mm: float = 10.0,
    theta: float = 0.0,
    phi: float = 0.0,
) -> list[PoseCondition]:
    forward = np.linspace(press_min_mm, press_max_mm, count // 2)
    backward = np.linspace(press_max_mm, press_min_mm, count - count // 2 + 1)
    depths_m = np.concatenate([forward, backward[1:]])[:count] / 1000.0
    return [PoseCondition(np.asarray(point), theta, phi, float(d)) for d in depths_m]
