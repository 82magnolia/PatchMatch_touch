import json

import numpy as np
import pytest

from taxim_baseline.depth import (
    aruco_pressing_depths,
    back_forth_depths,
    interpolate_short_gaps,
)


def test_back_forth_starts_and_ends_at_zero():
    values = back_forth_depths(50, 0.0, 10.0)
    assert len(values) == 50
    assert values[0] == pytest.approx(0.0)
    assert values.max() == pytest.approx(10.0)
    assert values[-1] == pytest.approx(0.0)


def test_only_short_bounded_pose_gaps_are_interpolated():
    short = interpolate_short_gaps(np.array([0.0, np.nan, 2.0]), 1)
    np.testing.assert_allclose(short, [0.0, 1.0, 2.0])
    long = interpolate_short_gaps(np.array([0.0, np.nan, np.nan, 3.0]), 1)
    assert np.isnan(long[1:3]).all()


def test_aruco_coordinate_projection_calibration_and_resampling(tmp_path):
    tvecs = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.001], [np.nan] * 3, [0.0, 0.0, 0.003]]
    )
    np.savez(tmp_path / "1_pose_contact.npz", tvecs=tvecs)
    np.savez(
        tmp_path / "1_contact_data.npz",
        tvec_0=np.zeros(3),
        sensor_z_0=np.array([0.0, 0.0, 2.0]),
    )
    (tmp_path / "1_meta.json").write_text(json.dumps({"cs_idx": 0, "ce_idx": 3}))
    depth, metadata = aruco_pressing_depths(
        tmp_path,
        1,
        7,
        sign=1,
        scale=2,
        offset_mm=0.5,
        max_gap=1,
        smoothing_sigma=0,
        clip_min_mm=0,
        clip_max_mm=5,
    )
    np.testing.assert_allclose(depth[[0, 2, 4, 6]], [0.5, 2.5, 4.5, 5.0])
    assert metadata["valid_raw_count"] == 3
