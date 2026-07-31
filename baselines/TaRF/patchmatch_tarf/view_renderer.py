"""Build TaRF's two fixed-standoff RGB-D views around a query touch target."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


VIEW_SPECS = {
    "40_50": {"standoff_m": 0.45, "fov_deg": 50.0},
    # Original TaRF zoom condition: 5 cm target distance, 40.86-degree FOV.
    "0_40": {"standoff_m": 0.05, "fov_deg": 40.86},
}
GELSIGHT_FOV_M = (0.0186, 0.0143)
PREPARED_SCALES = {
    "sim": {"40_50": 25.0, "0_40": 100.0},
    "real": {"40_50": 4.0, "0_40": 1.0},
}
REAL_HEIGHT_CUTOFF_M = 0.050
TAXIM_HEIGHT_PIXEL_M = 0.0295e-3


def _rz(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def marker_contact_target(
    raw_rotation: np.ndarray,
    tvec: np.ndarray,
    sensor_offset: dict,
) -> np.ndarray:
    """Apply the capture pipeline's positive-Z calibration convention."""
    marker_to_contact_m = np.array(
        [
            float(sensor_offset["offset_x_m"]),
            float(sensor_offset["offset_y_m"]),
            -float(sensor_offset["offset_z_m"]),
        ],
        dtype=np.float64,
    )
    return np.asarray(raw_rotation, dtype=np.float64) @ marker_to_contact_m + np.asarray(
        tvec, dtype=np.float64
    )


def _inpaint(array: np.ndarray, missing: np.ndarray) -> np.ndarray:
    if not missing.any() or missing.all():
        return array
    if array.ndim == 2:
        return cv2.inpaint(array.astype(np.float32), missing.astype(np.uint8), 5, cv2.INPAINT_NS)
    channels = [
        cv2.inpaint(array[..., channel], missing.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        for channel in range(array.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _perspective_splat(
    points: np.ndarray,
    colors: np.ndarray,
    target: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    forward: np.ndarray,
    standoff_m: float,
    fov_deg: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    origin = target - standoff_m * forward
    relative = points - origin
    x = relative @ x_axis
    y = relative @ y_axis
    z = relative @ forward
    focal = size / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.rint(focal * x / z + (size - 1) / 2.0).astype(np.int64)
        py = np.rint(focal * y / z + (size - 1) / 2.0).astype(np.int64)
    valid = (
        np.isfinite(z)
        & (z > 1e-4)
        & (px >= 0)
        & (px < size)
        & (py >= 0)
        & (py < size)
    )
    if not valid.any():
        raise ValueError("No RGB-D points project into the TaRF virtual camera")
    px, py, z, colors = px[valid], py[valid], z[valid], colors[valid]
    linear = py * size + px
    order = np.argsort(z)
    linear_sorted = linear[order]
    _, first = np.unique(linear_sorted, return_index=True)
    chosen = order[first]
    image = np.zeros((size * size, 3), dtype=np.uint8)
    depth = np.zeros(size * size, dtype=np.float32)
    image[linear[chosen]] = colors[chosen]
    depth[linear[chosen]] = z[chosen]
    image = image.reshape(size, size, 3)
    depth = depth.reshape(size, size)
    missing = depth <= 0
    coverage = float(np.mean(~missing))
    if coverage < 0.001:
        raise ValueError(f"TaRF virtual-view coverage is only {coverage:.3%}")
    image = _inpaint(image, missing)
    depth = _inpaint(depth, missing)
    return image, depth, coverage


def _real_views(
    folder: Path,
    query_idx: int,
    output: Path,
    sensor_offset_file: Path,
    size: int,
) -> dict:
    meta_path = folder / f"{query_idx}_meta.json"
    seg_meta_path = folder / f"{query_idx}_seg_meta.json"
    intrinsics_path = folder / "intrinsics.json"
    missing = [
        path
        for path in (meta_path, intrinsics_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Fixed-view real TaRF is missing: " + ", ".join(map(str, missing)))
    meta = json.loads(meta_path.read_text())
    seg_meta = json.loads(seg_meta_path.read_text()) if seg_meta_path.is_file() else {}
    try:
        sensor_offset = json.loads(sensor_offset_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read GelSight sensor offset {sensor_offset_file}: {exc}"
        ) from exc
    required_offset_keys = (
        "offset_x_m",
        "offset_y_m",
        "offset_z_m",
        "offset_theta_deg",
    )
    missing_offset_keys = [
        key for key in required_offset_keys if key not in sensor_offset
    ]
    if missing_offset_keys:
        raise ValueError(
            f"GelSight sensor offset {sensor_offset_file} is missing "
            f"{missing_offset_keys}"
        )
    offset_x_m = float(sensor_offset["offset_x_m"])
    offset_y_m = float(sensor_offset["offset_y_m"])
    offset_z_m = float(sensor_offset["offset_z_m"])
    inplane_offset_deg = float(sensor_offset["offset_theta_deg"])
    if offset_z_m < 0:
        raise ValueError(
            f"{sensor_offset_file}: offset_z_m must be the positive marker-face "
            "to gel-tip distance used by real_data_transfer"
        )
    cache_path = folder / f"object_cache_{int(seg_meta.get('view_idx', 0))}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Fixed-view real TaRF needs {cache_path}")

    with np.load(cache_path) as cache:
        color = np.asarray(cache["color"], dtype=np.uint8)
        depth = np.asarray(cache["depth"], dtype=np.float64)
    intrinsics = json.loads(intrinsics_path.read_text())
    rows, cols = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    points = np.column_stack(
        (
            (cols[valid] - intrinsics["cx"]) / intrinsics["fx"] * z,
            (rows[valid] - intrinsics["cy"]) / intrinsics["fy"] * z,
            z,
        )
    )
    colors = color[valid]

    raw_rvec = np.asarray(meta["rvec"], dtype=np.float64)
    raw_rotation, _ = cv2.Rodrigues(raw_rvec)
    aligned_rotation = raw_rotation @ _rz(90.0)
    sensor_rotation = aligned_rotation @ np.diag([1.0, -1.0, -1.0]) @ _rz(
        -inplane_offset_deg
    )
    # Mirror real_data_transfer._gelsight_processing.ortho_project_raw exactly:
    # the JSON stores positive marker-face -> gel-tip Z, while the gel lies on
    # the marker's local -Z side. The contact translation is expressed in the
    # raw marker frame, not the 90-degree-aligned sensor sampling frame.
    target = marker_contact_target(raw_rotation, meta["tvec"], sensor_offset)
    x_axis = sensor_rotation[:, 0]
    y_axis = sensor_rotation[:, 1]
    forward = sensor_rotation[:, 2]

    records = {}
    for name, spec in VIEW_SPECS.items():
        rgb, view_depth, coverage = _perspective_splat(
            points,
            colors,
            target,
            x_axis,
            y_axis,
            forward,
            spec["standoff_m"],
            spec["fov_deg"],
            size,
        )
        rgb_path = output / "rgb" / f"{name}.png"
        depth_path = output / "depth" / f"{name}.npy"
        cv2.imwrite(str(rgb_path), rgb)
        np.save(depth_path, view_depth)
        records[name] = {
            **spec,
            "requested_standoff_m": spec["standoff_m"],
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "coverage_before_inpaint": coverage,
        }
    return {
        "mode": "real_rgbd_aruco_original_tarf_views",
        "target_camera_xyz_m": target.tolist(),
        "sensor_offset_file": str(sensor_offset_file),
        "sensor_offset": {
            key: float(sensor_offset[key]) for key in required_offset_keys
        },
        "marker_to_contact_camera_rule": "R_marker @ [x, y, -z] + tvec",
        "inplane_offset_deg": inplane_offset_deg,
        "source_cache": str(cache_path),
        "source_intrinsics": str(intrinsics_path),
        "views": records,
    }


def _crop_for_footprint(array: np.ndarray, source_extent: tuple[float, float], footprint: float):
    height, width = array.shape[:2]
    crop_width = max(2, min(width, round(width * footprint / source_extent[0])))
    crop_height = max(2, min(height, round(height * footprint / source_extent[1])))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return array[top : top + crop_height, left : left + crop_width]


def _center_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return array[top : top + side, left : left + side]


def _scale_tag(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else f"{scale:g}"


def _resize_prepared_rgb(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read prepared TaRF RGB view: {path}")
    return cv2.resize(_center_square(image), (size, size), interpolation=cv2.INTER_CUBIC)


def _decode_viridis_height(path: Path) -> np.ndarray:
    """Invert the real-data `applyColorMap(VIRIDIS)` height preview."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read prepared TaRF height view: {path}")
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)
    lut = lut[:, 0].astype(np.int32)
    pixels = image.reshape(-1, 3).astype(np.int32)
    indices = np.empty(len(pixels), dtype=np.uint8)
    # Chunking avoids a full H*W*256 temporary for every query.
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start : start + 8192]
        distance = np.sum((chunk[:, None, :] - lut[None, :, :]) ** 2, axis=2)
        indices[start : start + len(chunk)] = np.argmin(distance, axis=1)
    indentation = indices.reshape(image.shape[:2]).astype(np.float32) / 255.0
    return -indentation * REAL_HEIGHT_CUTOFF_M


def _prepared_depth(path: Path, *, real: bool, standoff_m: float, size: int) -> np.ndarray:
    if real:
        relative_height = _decode_viridis_height(path)
    else:
        with np.load(path) as archive:
            elevation = np.asarray(archive["height"], dtype=np.float32)
        # Taxim stores height in GelSight calibration pixels, not millimetres.
        elevation = (elevation - float(np.nanmin(elevation))) * TAXIM_HEIGHT_PIXEL_M
        # Greater object height is closer to the virtual RGB-D camera.
        relative_height = -elevation
    relative_height = cv2.resize(
        _center_square(relative_height), (size, size), interpolation=cv2.INTER_LINEAR
    )
    return np.clip(standoff_m + relative_height, 0.0, 5.0).astype(np.float32)


def _prepared_scaled_views(
    folder: Path,
    query_idx: int,
    output: Path,
    size: int,
    *,
    domain: str,
    sensor_offset_file: Path | None = None,
) -> dict:
    records = {}
    for name in ("40_50", "0_40"):
        scale = PREPARED_SCALES[domain][name]
        tag = _scale_tag(scale)
        color_path = folder / f"{query_idx}_scale{tag}_color.jpg"
        height_path = folder / (
            f"{query_idx}_scale{tag}_height.jpg"
            if domain == "real"
            else f"{query_idx}_scale{tag}_height.npz"
        )
        missing = [path for path in (color_path, height_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Prepared {domain} TaRF {name} view is missing: "
                + ", ".join(map(str, missing))
            )
        rgb = _resize_prepared_rgb(color_path, size)
        depth = _prepared_depth(
            height_path,
            real=domain == "real",
            standoff_m=VIEW_SPECS[name]["standoff_m"],
            size=size,
        )
        rgb_path = output / "rgb" / f"{name}.png"
        depth_path = output / "depth" / f"{name}.npy"
        cv2.imwrite(str(rgb_path), rgb)
        np.save(depth_path, depth)
        records[name] = {
            **VIEW_SPECS[name],
            "prepared_scale": scale,
            "source_rgb": str(color_path),
            "source_height": str(height_path),
            "rgb": str(rgb_path),
            "depth": str(depth_path),
        }
    metadata = {
        "mode": f"{domain}_prepared_multiscale_tarf_views",
        "view_order": ["40_50", "0_40"],
        "scale_mapping": {
            name: PREPARED_SCALES[domain][name] for name in ("40_50", "0_40")
        },
        "views": records,
    }
    if domain == "real" and sensor_offset_file is not None:
        metadata["sensor_offset_file"] = str(sensor_offset_file)
        metadata["alignment"] = (
            "Prepared scale images are already centered using the saved "
            "ArUco-to-gel calibration."
        )
    return metadata


def _sim_views(folder: Path, query_idx: int, output: Path, size: int) -> dict:
    return _prepared_scaled_views(folder, query_idx, output, size, domain="sim")


def prepare_fixed_view_conditions(
    query_dir: Path,
    query_idx: int,
    output_root: Path,
    *,
    sensor_offset_file: Path | None = None,
    size=480,
) -> tuple[Path, dict]:
    """Create original-TaRF `40_50` and `0_40` conditions for one query."""
    output = output_root / str(query_idx)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    if (query_dir / "intrinsics.json").is_file():
        if sensor_offset_file is None:
            raise ValueError(
                "Real TaRF fixed views require --sensor_offset_file"
            )
        metadata = _prepared_scaled_views(
            query_dir,
            query_idx,
            output,
            int(size),
            domain="real",
            sensor_offset_file=Path(sensor_offset_file),
        )
    else:
        metadata = _sim_views(query_dir, query_idx, output, int(size))
    (output / "view_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return output_root, metadata
