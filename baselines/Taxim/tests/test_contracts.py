from pathlib import Path

import pytest

from taxim_baseline.contracts import metric_payload, output_names, resolve_pairs


def test_sim_identity_and_real_odd_to_previous_even():
    assert resolve_pairs("sim_gt_retrieval", [0, 1, 3], [0, 2, 3]) == [(0, 0), (3, 3)]
    assert resolve_pairs("real_gt_retrieval", [0, 2, 4], [1, 3, 5]) == [
        (1, 0),
        (3, 2),
        (5, 4),
    ]


def test_tsv_uses_first_reference(tmp_path: Path):
    mapping = tmp_path / "pairs.tsv"
    mapping.write_text("query\tref\n7\t4,2\n")
    assert resolve_pairs("tsv", [2, 4], [7], mapping) == [(7, 4)]


def test_output_names_and_metric_schema():
    assert output_names(5, "shadow") == {
        "prediction": "5_transferred.mp4",
        "reference": "5_ref_shadow.mp4",
        "query": "5_query_shadow.mp4",
    }
    payload = metric_payload(
        {5: {"MSE": 1.0, "PSNR": 2.0, "SSIM": 3.0, "LPIPS": 4.0}}
    )
    assert payload["average"]["SSIM"] == pytest.approx(3.0)
