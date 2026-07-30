from pathlib import Path

import pytest

from objectfolder_inr.contracts import metric_payload, output_names, resolve_pairs


def test_pairing_modes():
    assert resolve_pairs("sim_gt_retrieval", [0, 1, 2], [1, 2, 3]) == [(1, 1), (2, 2)]
    assert resolve_pairs("real_gt_retrieval", [0, 2, 4], [1, 3, 5]) == [
        (1, 0),
        (3, 2),
        (5, 4),
    ]


def test_tsv_and_output_names(tmp_path: Path):
    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("query\tref\n7\t2,3\n")
    assert resolve_pairs("tsv", [2, 3], [7], tsv) == [(7, 2)]
    assert output_names(7, "shadow") == {
        "prediction": "7_transferred.mp4",
        "reference": "7_ref_shadow.mp4",
        "query": "7_query_shadow.mp4",
    }


def test_metrics_schema():
    values = {"MSE": 0.1, "PSNR": 10.0, "SSIM": 0.8, "LPIPS": 0.2}
    payload = metric_payload({1: values, 3: values})
    assert payload["average"] == values
    with pytest.raises(ValueError):
        metric_payload({1: {"MSE": 0.1}})
