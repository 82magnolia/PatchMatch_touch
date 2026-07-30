import numpy as np
import pytest

from taxim_baseline.geometry import _resample_full_poses


def test_full_pose_resampling_uses_rotation_slerp():
    rvecs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]])
    tvecs = np.array([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]])
    rotations, translations = _resample_full_poses(rvecs, tvecs, 3, max_gap=1)
    np.testing.assert_allclose(rotations[1], [0.0, 0.0, np.pi / 4], atol=1e-7)
    np.testing.assert_allclose(translations[1], [0.005, 0.0, 0.0], atol=1e-7)


def test_full_pose_rejects_long_missing_marker_gap():
    rvecs = np.array(
        [[0.0, 0.0, 0.0], [np.nan] * 3, [np.nan] * 3, [0.0, 0.0, 0.1]]
    )
    tvecs = np.array(
        [[0.0, 0.0, 0.0], [np.nan] * 3, [np.nan] * 3, [0.0, 0.0, 0.001]]
    )
    with pytest.raises(ValueError, match="unfillable gap"):
        _resample_full_poses(rvecs, tvecs, 5, max_gap=1)
