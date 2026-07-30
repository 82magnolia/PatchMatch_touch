"""Height inference and Taxim/normal rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .model import build_pixel_features


@torch.inference_mode()
def predict_height(
    model,
    condition,
    height: int,
    width: int,
    feature_min: torch.Tensor,
    feature_max: torch.Tensor,
    batch_size: int,
    device: torch.device,
    normalization_mode: str = "signed_unit",
) -> np.ndarray:
    features = build_pixel_features(
        condition,
        height,
        width,
        feature_min,
        feature_max,
        device=device,
        normalization_mode=normalization_mode,
    )
    chunks = [
        model(features[start : start + batch_size]).cpu()
        for start in range(0, len(features), batch_size)
    ]
    return torch.cat(chunks).reshape(height, width).numpy()


def height_to_normal_bgr(height: np.ndarray) -> np.ndarray:
    import cv2

    dy, dx = np.gradient(height.astype(np.float32))
    normal = np.dstack((-dx, -dy, np.ones_like(height)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
    rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class Renderer:
    def __init__(self, video_type: str, calibration: Path | None, legacy_root: Path):
        self.video_type = video_type
        self.taxim = None
        if video_type in {"shadow", "sim"}:
            if calibration is None:
                raise ValueError("--taxim_calibration is required for shadow/sim rendering")
            required = ("polycalib.npz", "dataPack.npz", "depth_bg.npy", "real_bg.npy")
            missing = [name for name in required if not (calibration / name).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Taxim calibration {calibration} is missing: {', '.join(missing)}"
                )
            sys.path.insert(0, str(legacy_root))
            from taxim_render import TaximRender

            self.taxim = TaximRender(str(calibration))

    def render(self, normalized_height: np.ndarray, displacement_m: float, size: tuple[int, int]) -> np.ndarray:
        import cv2

        width, height = size
        if abs(float(displacement_m)) <= 1e-9:
            if self.video_type == "tactile_normal":
                # RGB normal (0,0,1) converted to BGR.
                flat = np.empty((height, width, 3), dtype=np.uint8)
                flat[...] = (255, 128, 128)
                return flat
            return cv2.resize(
                np.clip(self.taxim.real_bg, 0, 255).astype(np.uint8),
                (width, height),
            )
        if self.video_type == "tactile_normal":
            return cv2.resize(height_to_normal_bgr(normalized_height), (width, height))

        # Legacy ObjectFolder TouchNet predicts depths in this calibrated range.
        depth_m = 0.0339 + np.clip(normalized_height, 0.0, 1.0) * (0.04 - 0.0339)
        depth_m = cv2.resize(depth_m.astype(np.float32), (160, 120))
        _, _, tactile = self.taxim.render(depth_m, abs(float(displacement_m)))
        frame = np.clip(tactile, 0, 255).astype(np.uint8)
        return cv2.resize(frame, (width, height))
