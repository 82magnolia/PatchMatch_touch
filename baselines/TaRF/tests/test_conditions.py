from pathlib import Path

import cv2
import numpy as np
import pytest

from patchmatch_tarf.conditions import load_depth, resolve_conditions
from patchmatch_tarf.view_renderer import (
    marker_contact_target,
    prepare_fixed_view_conditions,
)


def _image(path: Path):
    assert cv2.imwrite(str(path), np.full((20, 30, 3), 127, np.uint8))


def test_flat_static_modalities_are_accepted_and_duplicated(tmp_path: Path):
    _image(tmp_path / "4_scale100_color.jpg")
    _image(tmp_path / "background.jpg")
    np.savez(tmp_path / "4_scale100_height.npz", height=np.ones((20, 30)))
    condition = resolve_conditions(
        tmp_path, 4, 100, tmp_path / "background.jpg"
    )
    assert len(condition.rgb_paths) == len(condition.depth_paths) == 2
    assert condition.rgb_paths[0] == condition.rgb_paths[1]
    assert condition.duplicated_single_view
    assert load_depth(condition.depth_paths[0]).shape == (20, 30)


def test_missing_conditions_never_fall_back_to_tactile_video(tmp_path: Path):
    _image(tmp_path / "background.jpg")
    (tmp_path / "4_shadow.mp4").touch()
    with pytest.raises(FileNotFoundError, match="Query tactile GT is intentionally not a fallback"):
        resolve_conditions(tmp_path, 4, 100, tmp_path / "background.jpg")


def test_real_height_preview_is_a_supported_depth_condition(tmp_path: Path):
    _image(tmp_path / "5_scale1_color.jpg")
    _image(tmp_path / "background.jpg")
    assert cv2.imwrite(
        str(tmp_path / "5_height.jpg"), np.full((20, 30), 255, np.uint8)
    )
    condition = resolve_conditions(tmp_path, 5, 1, tmp_path / "background.jpg")
    assert load_depth(condition.depth_paths[0]).max() == pytest.approx(5.0)


def test_sim_fixed_views_use_original_tarf_view_geometry(
    tmp_path: Path,
):
    query = tmp_path / "query"
    query.mkdir()
    color = np.zeros((120, 160, 3), dtype=np.uint8)
    color[:, :80] = (0, 0, 255)
    assert cv2.imwrite(str(query / "0_scale25_color.jpg"), color)
    np.savez(
        query / "0_scale25_height.npz",
        height=np.linspace(0, 20, 120 * 160).reshape(120, 160),
    )
    output = tmp_path / "conditions"
    _, metadata = prepare_fixed_view_conditions(query, 0, output, size=64)
    close = np.load(output / "0/depth/0_40.npy")
    context = np.load(output / "0/depth/40_50.npy")
    assert close.shape == context.shape == (64, 64)
    assert float(close.min()) == pytest.approx(0.05, abs=1e-3)
    assert float(context.min()) == pytest.approx(0.45, abs=1e-3)
    assert metadata["views"]["0_40"]["fov_deg"] == pytest.approx(40.86)


def test_marker_target_matches_capture_offset_sign_and_axes():
    offset = {
        "offset_x_m": 0.001,
        "offset_y_m": -0.0033,
        "offset_z_m": 0.0512,
    }
    target = marker_contact_target(
        np.eye(3), np.array([0.1, 0.2, 0.3]), offset
    )
    assert target == pytest.approx([0.101, 0.1967, 0.2488])
