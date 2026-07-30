"""Train/fine-tune the adapted ObjectFolder TouchNet on reference heights."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from .data import (
    condition_for_touch,
    feature_bounds,
    load_pseudo_height,
    raw_condition_features,
)
from .model import TouchNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_checkpoint(
    *,
    ref_dir: Path,
    ref_indices: list[int],
    checkpoint: Path,
    scale: float | None,
    pose_source: str,
    contact_points: Path | None,
    allow_index_coordinate_fallback: bool,
    marker_to_contact: np.ndarray,
    inplane_offset_deg: float,
    sensor_offset_file: Path,
    levels: int,
    network_depth: int,
    network_width: int,
    epochs: int,
    samples_per_touch: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    conditions = [
        condition_for_touch(
            ref_dir,
            index,
            pose_source=pose_source,
            contact_points=contact_points,
            allow_index_coordinate_fallback=allow_index_coordinate_fallback,
            marker_to_contact=marker_to_contact,
            inplane_offset_deg=inplane_offset_deg,
        )
        for index in ref_indices
    ]
    targets = [load_pseudo_height(ref_dir, index, scale) for index in ref_indices]
    feature_min_np, feature_max_np = feature_bounds(conditions)
    feature_min = torch.from_numpy(feature_min_np).to(device)
    feature_max = torch.from_numpy(feature_max_np).to(device)

    model = TouchNet(levels=levels, depth=network_depth, width=network_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)

    all_features, all_targets = [], []
    for condition, target in zip(conditions, targets):
        height, width = target.shape
        count = min(samples_per_touch, height * width)
        flat_indices = generator.choice(height * width, size=count, replace=False)
        yy, xx = np.unravel_index(flat_indices, (height, width))
        base = np.repeat(raw_condition_features(condition)[None, :], count, axis=0)
        pixels = np.stack(
            (xx.astype(np.float32) / max(width - 1, 1), yy.astype(np.float32) / max(height - 1, 1)),
            axis=1,
        )
        all_features.append(np.concatenate((base, pixels), axis=1))
        all_targets.append(target[yy, xx, None])

    features = torch.from_numpy(np.concatenate(all_features).astype(np.float32)).to(device)
    target_values = torch.from_numpy(np.concatenate(all_targets).astype(np.float32)).to(device)
    features = (features - feature_min) / torch.clamp(feature_max - feature_min, min=1e-8)
    features = features * 2.0 - 1.0

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(features), device=device)
        loss_sum = 0.0
        for start in range(0, len(features), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(features[indices])
            loss = functional.mse_loss(prediction, target_values[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(indices)
        print(f"[train] epoch {epoch + 1}/{epochs} mse={loss_sum / len(features):.7f}")

    payload = {
        "format": "patchmatch_touch_objectfolder_inr_v1",
        "model_state_dict": model.state_dict(),
        "model": {"levels": levels, "depth": network_depth, "width": network_width},
        "feature_min": feature_min_np,
        "feature_max": feature_max_np,
        "normalization_mode": "signed_unit",
        "target_representation": "pseudo-height",
        "training": {
            "ref_dir": str(ref_dir.resolve()),
            "ref_indices": ref_indices,
            "scale": scale,
            "pose_source": pose_source,
            "contact_points": str(contact_points.resolve()) if contact_points else None,
            "sensor_offset_file": str(sensor_offset_file.resolve()),
            "epochs": epochs,
            "samples_per_touch": samples_per_touch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return payload


def load_checkpoint(path: Path, device: torch.device) -> tuple[TouchNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") == "patchmatch_touch_objectfolder_inr_v1":
        config = payload["model"]
        model = TouchNet(
            levels=int(config["levels"]),
            depth=int(config["depth"]),
            width=int(config["width"]),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload

    if "TouchNet" not in payload:
        raise ValueError(
            f"{path} is neither an adapted checkpoint nor a legacy ObjectFile.pth"
        )
    legacy = payload["TouchNet"]
    state = {
        key.removeprefix("module."): value
        for key, value in legacy["model_state_dict"].items()
    }
    first_weight = state["pts_linears.0.weight"]
    width = int(first_weight.shape[0])
    layer_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("pts_linears.") and key.endswith(".weight")
    }
    model = TouchNet(levels=10, depth=max(layer_ids) + 1, width=width).to(device)
    converted = {}
    for key, value in state.items():
        if key.startswith("pts_linears."):
            converted["layers." + key.removeprefix("pts_linears.")] = value
        elif key.startswith("output_linear."):
            converted["output." + key.removeprefix("output_linear.")] = value
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if set(missing) != {"encoder.frequencies"} or unexpected:
        raise ValueError(
            f"Unexpected legacy TouchNet state layout; missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    xyz_min = np.asarray(legacy["xyz_min"], dtype=np.float32)
    xyz_max = np.asarray(legacy["xyz_max"], dtype=np.float32)
    if xyz_min.ndim == 0:
        xyz_min = np.repeat(xyz_min, 3)
    if xyz_max.ndim == 0:
        xyz_max = np.repeat(xyz_max, 3)
    converted_payload = {
        "format": "legacy_objectfolder_objectfile",
        "feature_min": np.concatenate(
            (xyz_min, [0.0, -1.0, -1.0, 0.0005, 0.0, 0.0])
        ).astype(np.float32),
        "feature_max": np.concatenate(
            (xyz_max, [np.deg2rad(15.0), 1.0, 1.0, 0.002, 1.0, 1.0])
        ).astype(np.float32),
        "normalization_mode": "legacy_objectfolder",
        "target_representation": "ObjectFolder depth",
    }
    return model, converted_payload
