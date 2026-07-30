"""Load simulation and real geometry without touching query tactile RGB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_npz_array(path: Path, preferred: tuple[str, ...]) -> np.ndarray:
    with np.load(path) as archive:
        for key in preferred:
            if key in archive:
                return np.asarray(archive[key], dtype=np.float32)
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]], dtype=np.float32)
        raise ValueError(f"Cannot choose array from {path}: {archive.files}")


def load_sim_height(folder: Path, touch_idx: int, scale: float) -> tuple[np.ndarray, Path]:
    candidates = [
        folder / f"{touch_idx}_scale{scale:g}_height.npz",
        folder / f"{touch_idx}_height.npz",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing known simulation geometry for query {touch_idx}. Accepted: "
            + ", ".join(candidate.name for candidate in candidates)
        )
    height = load_npz_array(path, ("height", "height_map", "arr_0")).squeeze()
    if height.ndim != 2 or not np.isfinite(height).all():
        raise ValueError(f"Simulation height must be finite HxW: {path}, got {height.shape}")
    return height, path


def load_real_geometry(folder: Path, touch_idx: int):
    path = folder / f"{touch_idx}_contact_data.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real geometry: {path}. Run the repository real-data preprocessing first."
        )
    with np.load(path) as data:
        required = ("height_map_0", "valid_depth_remap", "mask_crop")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        height_m = np.asarray(data["height_map_0"], dtype=np.float32)
        valid = np.asarray(data["valid_depth_remap"], dtype=bool)
        valid &= np.asarray(data["mask_crop"]) > 0
    if height_m.shape != valid.shape:
        raise ValueError(f"Real height/mask shape mismatch in {path}")
    return height_m, valid, path


def _resample_full_poses(rvecs, tvecs, count: int, max_gap: int):
    """Fill only short internal gaps, then SLERP rotations onto output time."""
    from scipy.spatial.transform import Rotation, Slerp

    rvecs = np.asarray(rvecs, dtype=np.float64)
    tvecs = np.asarray(tvecs, dtype=np.float64)
    valid = np.isfinite(rvecs).all(axis=1) & np.isfinite(tvecs).all(axis=1)
    missing = ~valid
    start = None
    for index in range(len(valid) + 1):
        if index < len(valid) and missing[index] and start is None:
            start = index
        if start is not None and (index == len(valid) or not missing[index]):
            length = index - start
            bounded = start > 0 and index < len(valid)
            if length > max_gap or not bounded:
                raise ValueError(
                    f"Full-pose ArUCo sequence has an unfillable gap of {length} "
                    f"frame(s) at [{start}, {index - 1}]"
                )
            start = None
    source = np.flatnonzero(valid).astype(np.float64)
    if len(source) < 2:
        raise ValueError("Full-pose Taxim needs at least two valid ArUCo poses")
    target = np.linspace(0.0, len(valid) - 1, count)
    rotations = Slerp(source, Rotation.from_rotvec(rvecs[valid]))(target).as_rotvec()
    translations = np.column_stack(
        [np.interp(target, source, tvecs[valid, axis]) for axis in range(3)]
    )
    return rotations, translations


def load_real_pose_geometry(
    folder: Path,
    touch_idx: int,
    count: int,
    *,
    sensor_offset: dict,
    sensor_offset_file: Path,
    max_gap=3,
    inpaint_method="telea",
):
    """Rerasterize the static RGB-D object from every offset-aware gel pose."""
    import cv2

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from real_data_transfer._gelsight_processing import (
        _make_Rz,
        _rotate_rvec_z,
        ortho_project_raw,
    )

    pose_path = folder / f"{touch_idx}_pose_contact.npz"
    meta_path = folder / f"{touch_idx}_meta.json"
    seg_meta_path = folder / f"{touch_idx}_seg_meta.json"
    intrinsics_path = folder / "intrinsics.json"
    required = (pose_path, meta_path, intrinsics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Full-pose real Taxim is missing: " + ", ".join(missing))
    meta = json.loads(meta_path.read_text())
    seg_meta = json.loads(seg_meta_path.read_text()) if seg_meta_path.is_file() else {}
    view_idx = int(seg_meta.get("view_idx", 0))
    cache_path = folder / f"object_cache_{view_idx}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Full-pose real Taxim needs {cache_path}")

    with np.load(pose_path) as pose:
        rvecs = np.asarray(pose["rvecs"], dtype=np.float64)
        tvecs = np.asarray(pose["tvecs"], dtype=np.float64)
    start = max(0, int(meta.get("cs_idx", 0)))
    end = min(len(rvecs) - 1, int(meta.get("ce_idx", len(rvecs) - 1)))
    if end < start:
        raise ValueError(f"Invalid contact window [{start}, {end}] in {meta_path}")
    rvecs, tvecs = _resample_full_poses(
        rvecs[start : end + 1], tvecs[start : end + 1], count, max_gap
    )

    with np.load(cache_path) as cache:
        normals = np.asarray(cache["normals"])
        color = np.asarray(cache["color"])
        mask = np.asarray(cache["mask"])
        depth = np.asarray(cache["depth"])
    intrinsics = json.loads(intrinsics_path.read_text())
    # Pass the saved capture calibration in its native convention:
    # positive marker-face -> gel-tip Z. ortho_project_raw applies local -Z.
    capture_offset = (
        float(sensor_offset["offset_x_m"]),
        float(sensor_offset["offset_y_m"]),
        float(sensor_offset["offset_z_m"]),
    )
    inplane_offset_deg = float(sensor_offset["offset_theta_deg"])
    align = _make_Rz(90)
    frames = []
    failed = []
    for frame_index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        result = ortho_project_raw(
            normals,
            color,
            mask,
            depth,
            intrinsics,
            inpaint_method,
            rvec=_rotate_rvec_z(rvec, align),
            tvec=tvec,
            offset=capture_offset,
            theta_deg=float(inplane_offset_deg),
        )
        if result is None:
            frames.append(
                (
                    np.zeros((240, 320), dtype=np.float32),
                    np.zeros((240, 320), dtype=bool),
                )
            )
            failed.append(frame_index)
        else:
            surface = np.asarray(result[5], dtype=np.float32)
            valid = np.asarray(result[7], dtype=bool) & (np.asarray(result[8]) > 0)
            frames.append((surface, valid))
    metadata = {
        "mode": "full_pose_offset_projection",
        "pose_path": str(pose_path),
        "object_cache": str(cache_path),
        "intrinsics": str(intrinsics_path),
        "contact_window": [start, end],
        "sensor_offset_file": str(sensor_offset_file),
        "sensor_offset": {
            key: float(value) for key, value in sensor_offset.items()
        },
        "marker_to_contact_camera_rule": "R_marker @ [x, y, -z] + tvec",
        "inplane_offset_deg": float(inplane_offset_deg),
        "failed_projection_frames": failed,
    }
    return frames, metadata
