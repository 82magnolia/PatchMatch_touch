import importlib.util
from pathlib import Path

import cv2
import numpy as np

from img2touch.ldm.patchmatch_sim_data import (
    PatchMatchSimDataset,
    height_to_depth_condition,
)


def _preparation_module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_sim_training_data.py"
    spec = importlib.util.spec_from_file_location("prepare_sim_training_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reference_even_query_odd_split_contract():
    split = _preparation_module().split_for_sample
    assert split(0, "1", 0) == "train"
    assert split(0, "1", 6) == "train"
    assert split(0, "1", 1) is None
    assert split(1, "1", 1) == "val"
    assert split(1, "1", 5) == "val"
    assert split(1, "1", 3) == "test"
    assert split(1, "1", 7) == "test"
    assert split(1, "1", 0) is None
    assert split(0, "951", 0) is None


def test_height_is_converted_to_inverse_camera_depth(tmp_path: Path):
    path = tmp_path / "height.npz"
    np.savez(
        path,
        height=np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32),
    )
    depth = height_to_depth_condition(str(path), "0_40", 2)[..., 0]
    assert depth[0, 1] < depth[0, 0]


def test_dataset_returns_upstream_two_view_contract(tmp_path: Path):
    image = np.full((12, 16, 3), 127, np.uint8)
    paths = {}
    for name in ("rgb25", "rgb100", "touch", "background"):
        path = tmp_path / f"{name}.jpg"
        assert cv2.imwrite(str(path), image)
        paths[name] = str(path)
    for name in ("height25", "height100"):
        path = tmp_path / f"{name}.npz"
        np.savez(path, height=np.ones((12, 16), dtype=np.float32))
        paths[name] = str(path)
    record = {
        "rgb_40_50": paths["rgb25"],
        "height_40_50": paths["height25"],
        "rgb_0_40": paths["rgb100"],
        "height_0_40": paths["height100"],
        "touch": paths["touch"],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        __import__("json").dumps({"train": [record], "val": [record], "test": [record]})
    )
    dataset = PatchMatchSimDataset(
        str(manifest), "train", paths["background"], size=16
    )
    sample = dataset[0]
    assert sample["image"].shape == (16, 16, 3)
    assert sample["aux"].shape == (16, 16, 11)
