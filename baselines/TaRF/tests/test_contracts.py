from pathlib import Path

import pytest

from patchmatch_tarf.contracts import metric_payload, output_names, resolve_pairs


def test_sim_and_real_pairing():
    assert resolve_pairs("sim_gt_retrieval", [0, 2, 4], [0, 1, 2]) == [(0, 0), (2, 2)]
    assert resolve_pairs("real_gt_retrieval", [0, 1, 2, 3], [0, 1, 2, 3]) == [
        (1, 0),
        (3, 2),
    ]


def test_tsv_uses_first_reference(tmp_path: Path):
    mapping = tmp_path / "pairs.tsv"
    mapping.write_text("query\tref\n7\t3,2\n")
    assert resolve_pairs("tsv", [2, 3], [7], mapping) == [(7, 3)]


def test_output_names_and_metric_schema():
    assert output_names(7, "shadow") == {
        "prediction": "7_transferred.mp4",
        "reference": "7_ref_shadow.mp4",
        "query": "7_query_shadow.mp4",
    }
    payload = metric_payload(
        {7: {"MSE": 0.1, "PSNR": 10.0, "SSIM": 0.5, "LPIPS": 0.2}}
    )
    assert payload["average"]["SSIM"] == pytest.approx(0.5)

