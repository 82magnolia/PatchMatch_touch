"""Adapted ObjectFolder TouchNet: positional encoding plus a NeRF-style MLP."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim: int = 9, levels: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.register_buffer("frequencies", 2.0 ** torch.arange(levels))
        self.output_dim = input_dim * (1 + 2 * levels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = [values]
        for frequency in self.frequencies:
            encoded.extend((torch.sin(values * frequency), torch.cos(values * frequency)))
        return torch.cat(encoded, dim=-1)


class TouchNet(nn.Module):
    def __init__(self, levels: int = 10, depth: int = 8, width: int = 256):
        super().__init__()
        self.encoder = PositionalEncoder(9, levels)
        self.depth = depth
        self.skip = depth // 2
        layers = []
        for index in range(depth):
            input_width = self.encoder.output_dim if index == 0 else width
            if index > 0 and index - 1 == self.skip:
                input_width += self.encoder.output_dim
            layers.append(nn.Linear(input_width, width))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        hidden = encoded
        for index, layer in enumerate(self.layers):
            hidden = functional.relu(layer(hidden), inplace=True)
            if index == self.skip:
                hidden = torch.cat((hidden, encoded), dim=-1)
        return torch.sigmoid(self.output(hidden))


def build_pixel_features(
    condition,
    height: int,
    width: int,
    feature_min: torch.Tensor,
    feature_max: torch.Tensor,
    *,
    device: torch.device,
    normalization_mode: str = "signed_unit",
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    xyz = torch.as_tensor(condition.xyz, dtype=torch.float32, device=device)
    base = torch.tensor(
        [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            condition.theta,
            __import__("math").cos(condition.phi),
            __import__("math").sin(condition.phi),
            condition.displacement,
        ],
        dtype=torch.float32,
        device=device,
    )
    features = torch.empty((height, width, 9), dtype=torch.float32, device=device)
    features[..., :7] = base
    features[..., 7] = xx
    features[..., 8] = yy
    denominator = torch.clamp(feature_max - feature_min, min=1e-8)
    normalized = (features - feature_min) / denominator
    if normalization_mode == "signed_unit":
        normalized = normalized * 2.0 - 1.0
    elif normalization_mode == "legacy_objectfolder":
        # Original ObjectFolder leaves cos(phi)/sin(phi) in [-1,1] while the
        # other seven channels are min/max normalized to [0,1].
        normalized[..., 4:6] = features[..., 4:6]
    else:
        raise ValueError(f"Unknown normalization mode: {normalization_mode}")
    return normalized.reshape(-1, 9)
