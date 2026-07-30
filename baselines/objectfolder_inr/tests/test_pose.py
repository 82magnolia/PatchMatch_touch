import numpy as np
import pytest

from objectfolder_inr.pose import (
    DEFAULT_MARKER_TO_CONTACT_M,
    condition_from_pose,
    load_sensor_offset,
    resample_pose_sequence,
)


def test_identity_pose_to_objectfolder_features():
    tvec = np.array([0.1, 0.2, 0.3])
    result = condition_from_pose(np.zeros(3), tvec, inplane_offset_deg=0.0)
    np.testing.assert_allclose(result.xyz, tvec + DEFAULT_MARKER_TO_CONTACT_M)
    assert result.theta == pytest.approx(0.0)
    assert result.phi == pytest.approx(0.0)


def test_depth_is_translation_along_initial_sensor_normal():
    result = condition_from_pose(
        np.zeros(3),
        np.array([0.0, 0.0, 0.29]),
        origin_tvec=np.array([0.0, 0.0, 0.30]),
        origin_normal=np.array([0.0, 0.0, -1.0]),
        inplane_offset_deg=0.0,
    )
    assert result.displacement == pytest.approx(0.01)


def test_missing_pose_interpolation_and_resampling():
    rvecs = np.array([[0.0, 0.0, 0.0], [np.nan] * 3, [0.0, 0.0, 0.2]])
    tvecs = np.array([[0.0, 0.0, 1.0], [np.nan] * 3, [0.0, 0.0, 0.8]])
    out_r, out_t = resample_pose_sequence(rvecs, tvecs, 5)
    assert out_r.shape == (5, 3)
    assert out_t[2, 2] == pytest.approx(0.9)


def test_sensor_offset_json_uses_capture_positive_z_convention(tmp_path):
    path = tmp_path / "offset.json"
    path.write_text(
        '{"offset_x_m": 0.001, "offset_y_m": -0.0033, '
        '"offset_z_m": 0.0512, "offset_theta_deg": -6.6}'
    )
    marker_to_contact, theta, resolved = load_sensor_offset(path)
    assert marker_to_contact == pytest.approx([0.001, -0.0033, -0.0512])
    assert theta == pytest.approx(-6.6)
    assert resolved["offset_z_m"] == pytest.approx(0.0512)
