"""TaRF image-to-touch generation and an explicit checkpoint-free smoke backend."""

from __future__ import annotations

import contextlib
import random
import sys
from pathlib import Path

import numpy as np

from .conditions import QueryConditions, load_depth


def validate_tarf_assets(
    config: Path,
    diffusion_ckpt: Path | None,
    first_stage_ckpt: Path | None,
    ranking_rgb_ckpt: Path | None,
    ranking_tac_ckpt: Path | None,
) -> None:
    required_assets = {
        "--config": config,
        "--diffusion_ckpt": diffusion_ckpt,
        "--ranking_rgb_enc_ckpt": ranking_rgb_ckpt,
        "--ranking_tac_enc_ckpt": ranking_tac_ckpt,
    }
    missing = [
        f"{flag}={path}"
        for flag, path in required_assets.items()
        if path is None or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The TaRF backend requires the official diffusion and two ranking "
            "checkpoints. Missing: "
            + ", ".join(missing)
            + ". Download the pretrained models documented in baselines/TaRF/README.md, "
            "or use --smoke_test only to validate plumbing."
        )
    if first_stage_ckpt is not None and not first_stage_ckpt.is_file():
        raise FileNotFoundError(f"--first_stage_ckpt does not exist: {first_stage_ckpt}")


def _center_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    size = min(height, width)
    top, left = (height - size) // 2, (width - size) // 2
    return array[top : top + size, left : left + size]


def _load_rgb_tensor(path: Path, torch, cv2):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read RGB condition: {path}")
    image = cv2.cvtColor(_center_square(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
    return tensor.permute(2, 0, 1).unsqueeze(0), image


def _load_depth_tensor(path: Path, torch, cv2, multiplier: float, clip_max: float):
    depth = _center_square(load_depth(path)) * multiplier
    depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
    depth = np.clip(depth, 0.0, clip_max) / clip_max * 2.0 - 1.0
    return torch.from_numpy(depth.astype(np.float32)).view(1, 1, 256, 256)


def _load_background_tensor(path: Path, torch, cv2):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read sensor background: {path}")
    image = cv2.cvtColor(_center_square(image), cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


class RankingEncoder:
    """Source-compatible ResNet-50 projection encoder."""

    @staticmethod
    def build(torch, torchvision):
        nn = torch.nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.resnet = torchvision.models.resnet50(weights=None)
                self.resnet.fc = nn.Linear(2048, 32)

            def forward(self, batch):
                features = self.resnet(batch)
                return torch.nn.functional.normalize(features, dim=1)

        return Model()


def _load_state(torch, path: Path):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    return payload


class TaRFGenerator:
    """Deterministic, one-shot adaptation of TaRF's real-time estimator."""

    def __init__(
        self,
        source_root: Path,
        config: Path,
        diffusion_ckpt: Path,
        first_stage_ckpt: Path | None,
        ranking_rgb_ckpt: Path,
        ranking_tac_ckpt: Path,
        n_samples: int = 8,
        ddim_steps: int = 200,
        guidance_scale: float = 7.5,
        ddim_eta: float = 0.0,
        seed: int = 42,
        device: str = "cuda",
        depth_multiplier: float = 1.0,
        depth_clip_max: float = 5.0,
    ):
        validate_tarf_assets(
            config, diffusion_ckpt, first_stage_ckpt, ranking_rgb_ckpt, ranking_tac_ckpt
        )
        if n_samples < 1 or ddim_steps < 1:
            raise ValueError("n_samples and ddim_steps must be positive")
        import cv2
        import torch
        import torchvision
        from omegaconf import OmegaConf

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "TaRF inference requested CUDA, but torch.cuda.is_available() is false. "
                "Run on a CUDA host or explicitly use --device cpu (very slow)."
            )
        self.cv2, self.torch = cv2, torch
        self.device = torch.device(device)
        self.n_samples, self.ddim_steps = n_samples, ddim_steps
        self.guidance_scale, self.ddim_eta, self.seed = guidance_scale, ddim_eta, seed
        self.depth_multiplier, self.depth_clip_max = depth_multiplier, depth_clip_max

        img2touch = source_root / "img2touch"
        sys.path.insert(0, str(img2touch))
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.util import instantiate_from_config

        cfg = OmegaConf.load(str(config))
        cfg.model.params.ckpt_path = None
        # The official img2touch Lightning checkpoint contains all
        # `first_stage_model.*` parameters. A separate KL checkpoint is only
        # needed for distributions that stripped those weights.
        cfg.model.params.first_stage_config.params.ckpt_path = (
            str(first_stage_ckpt) if first_stage_ckpt is not None else None
        )
        model = instantiate_from_config(cfg.model)
        state = _load_state(torch, diffusion_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing_first_stage = [
            key for key in missing if key.startswith("first_stage_model.")
        ]
        if first_stage_ckpt is None and missing_first_stage:
            raise RuntimeError(
                "The diffusion checkpoint does not embed complete KL first-stage "
                f"weights ({len(missing_first_stage)} keys missing); provide "
                "--first_stage_ckpt."
            )
        if missing:
            print(f"[TaRF] diffusion checkpoint missing {len(missing)} model keys")
        if unexpected:
            print(f"[TaRF] diffusion checkpoint has {len(unexpected)} unexpected keys")
        self.model = model.to(self.device).eval()
        self.sampler = DDIMSampler(self.model)

        self.rgb_encoder = RankingEncoder.build(torch, torchvision)
        self.tac_encoder = RankingEncoder.build(torch, torchvision)
        self.rgb_encoder.load_state_dict(_load_state(torch, ranking_rgb_ckpt), strict=True)
        self.tac_encoder.load_state_dict(_load_state(torch, ranking_tac_ckpt), strict=True)
        self.rgb_encoder = self.rgb_encoder.to(self.device).eval()
        self.tac_encoder = self.tac_encoder.to(self.device).eval()

    def _condition(self, conditions: QueryConditions):
        parts, first_rgb = [], None
        for rgb_path, depth_path in zip(conditions.rgb_paths, conditions.depth_paths):
            rgb, rgb_image = _load_rgb_tensor(rgb_path, self.torch, self.cv2)
            if first_rgb is None:
                first_rgb = rgb_image
            parts.extend(
                [
                    rgb,
                    _load_depth_tensor(
                        depth_path,
                        self.torch,
                        self.cv2,
                        self.depth_multiplier,
                        self.depth_clip_max,
                    ),
                ]
            )
        parts.append(
            _load_background_tensor(conditions.background_path, self.torch, self.cv2)
        )
        return self.torch.cat(parts, dim=1).to(self.device), first_rgb

    def _ranking_input(self, images: np.ndarray):
        torch = self.torch
        tensor = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(
            tensor, size=(128, 128), mode="bilinear", align_corners=False
        ).to(self.device)

    def generate(self, conditions: QueryConditions, query_idx: int):
        torch = self.torch
        seed = self.seed + int(query_idx)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        condition, first_rgb = self._condition(conditions)
        prompts = condition.repeat(self.n_samples, 1, 1, 1)
        autocast = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast, self.model.ema_scope():
            learned = self.model.get_learned_conditioning(prompts)
            shape = (self.model.channels, self.model.image_size, self.model.image_size)
            latent, _ = self.sampler.sample(
                S=self.ddim_steps,
                conditioning=learned,
                batch_size=self.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=None,
                eta=self.ddim_eta,
                x_T=None,
            )
            decoded = self.model.decode_first_stage(latent)
            candidates = (
                torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                * 255.0
            ).astype(np.uint8)
            rgb_rank = np.rot90(first_rgb, k=-1).copy()[None]
            tac_rank = np.rot90(candidates, k=-1, axes=(1, 2)).copy()
            rgb_features = self.rgb_encoder(self._ranking_input(rgb_rank))
            tac_features = self.tac_encoder(self._ranking_input(tac_rank))
            scores = (rgb_features @ tac_features.T).squeeze(0).cpu().numpy()
        selected = int(np.argmax(scores))
        return candidates[selected], candidates, scores.astype(float).tolist(), selected


class SmokeGenerator:
    """Checkpoint-free pipeline check; deliberately not a scientific TaRF result."""

    def __init__(self, n_samples: int = 4, seed: int = 42):
        self.n_samples, self.seed = n_samples, seed

    def generate(self, conditions: QueryConditions, query_idx: int):
        import cv2

        rgb = cv2.imread(str(conditions.rgb_paths[0]), cv2.IMREAD_COLOR)
        background = cv2.imread(str(conditions.background_path), cv2.IMREAD_COLOR)
        if rgb is None or background is None:
            raise RuntimeError("Cannot read smoke-test RGB/background conditions")
        rgb = cv2.resize(_center_square(rgb), (256, 256), interpolation=cv2.INTER_CUBIC)
        background = cv2.resize(
            _center_square(background), (256, 256), interpolation=cv2.INTER_CUBIC
        )
        depth = _center_square(load_depth(conditions.depth_paths[0]))
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
        depth -= depth.min()
        depth /= max(float(depth.max()), 1e-6)
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        magnitude = cv2.normalize(
            cv2.magnitude(grad_x, grad_y), None, 0.0, 1.0, cv2.NORM_MINMAX
        )
        rng = np.random.default_rng(self.seed + int(query_idx))
        candidates, scores = [], []
        luminance = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for index in range(self.n_samples):
            gain = 0.35 + 0.15 * index
            tint = rgb.astype(np.float32) - rgb.mean(axis=(0, 1), keepdims=True)
            contact = depth[..., None] * tint * 0.18 + magnitude[..., None] * np.array(
                [35.0, -20.0, 28.0], dtype=np.float32
            )
            noise = rng.normal(0.0, 1.5, background.shape)
            candidate = np.clip(background + gain * contact + noise, 0, 255).astype(np.uint8)
            candidates.append(candidate)
            tactile_luma = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
            tactile_luma = (tactile_luma - tactile_luma.mean()) / (tactile_luma.std() + 1e-6)
            target = (luminance - luminance.mean()) / (luminance.std() + 1e-6)
            scores.append(float(np.mean(target * tactile_luma)))
        selected = int(np.argmax(scores))
        candidates_rgb = np.stack(
            [cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB) for candidate in candidates]
        )
        return candidates_rgb[selected], candidates_rgb, scores, selected
