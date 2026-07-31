"""Paired PatchMatch_touch simulation data for TaRF fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


TAXIM_HEIGHT_PIXEL_M = 0.0295e-3
VIEW_DEPTH_M = {"40_50": 0.45, "0_40": 0.05}


def _center_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return array[top : top + side, left : left + side]


def _rgb(path: str, size: int, *, rotate_ccw: bool = False) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(_center_square(image), cv2.COLOR_BGR2RGB)
    if rotate_ccw:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return image.astype(np.float32) / 127.5 - 1.0


def height_to_depth_condition(path: str, view: str, size: int) -> np.ndarray:
    """Convert Taxim elevation pixels to TaRF's normalized camera-depth channel.

    Taxim's saved height is an elevation: larger values are closer to the
    virtual camera. TaRF conditions on camera depth, so elevation is subtracted
    from the view's nominal standoff and then normalized exactly like upstream
    real-time inference: clip to 0..5 m and map to -1..1.
    """
    with np.load(path) as archive:
        height = np.asarray(archive["height"], dtype=np.float32)
    elevation_m = (height - float(np.nanmin(height))) * TAXIM_HEIGHT_PIXEL_M
    depth_m = np.clip(VIEW_DEPTH_M[view] - elevation_m, 0.0, 5.0)
    depth_m = cv2.resize(
        _center_square(depth_m), (size, size), interpolation=cv2.INTER_LINEAR
    )
    return (depth_m / 5.0 * 2.0 - 1.0)[..., None].astype(np.float32)


class PatchMatchSimDataset(Dataset):
    """Return `image` tactile targets and 11-channel `aux` TaRF conditions."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        background_path: str,
        size: int = 256,
        **_ignored,
    ):
        manifest = json.loads(Path(manifest_path).read_text())
        if split not in manifest:
            raise KeyError(f"{manifest_path} has no {split!r} split")
        self.records = manifest[split]
        self.background_path = str(Path(background_path))
        self.size = int(size)
        if not self.records:
            raise ValueError(f"{manifest_path} split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        record = self.records[index]
        context_rgb = _rgb(record["rgb_40_50"], self.size)
        context_depth = height_to_depth_condition(
            record["height_40_50"], "40_50", self.size
        )
        close_rgb = _rgb(record["rgb_0_40"], self.size)
        close_depth = height_to_depth_condition(
            record["height_0_40"], "0_40", self.size
        )
        background = _rgb(self.background_path, self.size, rotate_ccw=True)
        target = _rgb(record["touch"], self.size)
        return {
            "image": target,
            "aux": np.concatenate(
                [context_rgb, context_depth, close_rgb, close_depth, background],
                axis=-1,
            ).astype(np.float32),
        }
