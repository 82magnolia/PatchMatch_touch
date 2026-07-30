"""Height-map adapter around the checked-in Taxim optical/deformation methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class TaximHeightRenderer:
    """Reuse Taxim's simulator methods without requiring mesh rasterization."""

    def __init__(
        self,
        taxim_root: Path,
        calibration_dir: Path,
        gel_map_path: Path,
        output_height: int,
        output_width: int,
        background_path: Path | None = None,
    ):
        import cv2

        required = ("polycalib.npz", "dataPack.npz", "shadowTable.npz")
        missing = [name for name in required if not (calibration_dir / name).is_file()]
        if missing or not gel_map_path.is_file():
            raise FileNotFoundError(
                f"Taxim assets missing under {calibration_dir}: "
                + ", ".join(missing + ([] if gel_map_path.is_file() else [str(gel_map_path)]))
            )
        sys.path.insert(0, str(taxim_root))
        sys.path.insert(0, str(taxim_root / "OpticalSimulation"))
        from Basics.CalibData import CalibData
        import Basics.params as params
        import Basics.sensorParams as sensor_params
        from simOptical import height_map_to_normals, simulator

        self.cv2 = cv2
        self.params = params
        self.sensor_params = sensor_params
        self.simulator_class = simulator
        self.height_map_to_normals = height_map_to_normals
        self.height = int(output_height)
        self.width = int(output_width)
        self.height_pixel_mm = float(sensor_params.pixmm)
        self.gel_map_path = gel_map_path

        core = simulator.__new__(simulator)
        core.psp_h, core.psp_w = self.height, self.width
        core.psp_mm = sensor_params.pixmm * sensor_params.h / self.height
        core.height_psp_mm = sensor_params.pixmm
        core.calib_data = CalibData(str(calibration_dir / "polycalib.npz"))
        with np.load(calibration_dir / "dataPack.npz", allow_pickle=True) as data:
            core.f0 = np.asarray(data["f0"])
        if background_path is not None:
            if not background_path.is_file():
                raise FileNotFoundError(f"Background does not exist: {background_path}")
            if background_path.suffix.lower() == ".npy":
                core.f0 = np.asarray(np.load(background_path, allow_pickle=True))
            else:
                loaded = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
                if loaded is None:
                    raise ValueError(f"Cannot read background: {background_path}")
                core.f0 = loaded
        if core.f0.shape[:2] != (self.height, self.width):
            core.f0 = cv2.resize(core.f0, (self.width, self.height))
        core.bg_proc = simulator.processInitialFrame(core)
        shadow = np.load(calibration_dir / "shadowTable.npz", allow_pickle=True)
        core.shadow_depth = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        core.direction = shadow["shadowDirections"]
        core.shadowTable = shadow["shadowTable"]
        self.core = core

        gel = np.asarray(np.load(gel_map_path), dtype=np.float32)
        if gel.shape != (self.height, self.width):
            gel = cv2.resize(gel, (self.width, self.height))
        self.gel = cv2.GaussianBlur(
            gel, (params.kernel_size, params.kernel_size), 0
        )

    def _format(self, deformed, contact_mask, contact_height, modality: str):
        cv2 = self.cv2
        if modality == "tactile_normal":
            normal = self.height_map_to_normals(deformed)
            rgb = (
                np.clip((normal + 1.0) * 0.5, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        else:
            sim, shadow = self.core.simulating(
                deformed, contact_mask, contact_height, shadow=(modality == "shadow")
            )
            rgb = shadow if modality == "shadow" else sim
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def render_sim(self, raw_height: np.ndarray, pressing_depth_mm: float, modality: str):
        cv2 = self.cv2
        height = np.asarray(raw_height, dtype=np.float32)
        if height.shape != (self.height, self.width):
            height = cv2.resize(height, (self.width, self.height))
        footprint = height > 0
        press_pixels = float(pressing_depth_mm) / self.height_pixel_mm
        gel = -self.gel + (float(self.gel.max()) + float(height.max()) - press_pixels)
        contact = (height > gel) & footprint
        interacted = np.where(contact, height, gel).astype(np.float32)
        deformed, mask, contact_height = self.core.deformApprox(
            float(pressing_depth_mm), interacted, gel, contact
        )
        return self._format(deformed, mask, contact_height, modality), deformed, mask

    def render_real(
        self,
        surface_height_m: np.ndarray,
        valid_mask: np.ndarray,
        pressing_depth_mm: float,
        surface_offset_mm: float,
        modality: str,
    ):
        cv2 = self.cv2
        surface = np.asarray(surface_height_m, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=np.uint8)
        if surface.shape != (self.height, self.width):
            surface = cv2.resize(surface, (self.width, self.height))
            valid = cv2.resize(
                valid, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        valid = valid.astype(bool)
        threshold_m = (float(pressing_depth_mm) + float(surface_offset_mm)) / 1000.0
        penetration_mm = np.maximum(threshold_m - surface, 0.0) * 1000.0
        penetration_mm[~valid] = 0.0
        height_pixels = penetration_mm / self.height_pixel_mm
        contact = (penetration_mm > 0.0) & valid
        flat_gel = np.zeros_like(height_pixels, dtype=np.float32)
        deformed, mask, contact_height = self.core.deformApprox(
            max(float(pressing_depth_mm), 0.0),
            height_pixels.astype(np.float32),
            flat_gel,
            contact,
        )
        return self._format(deformed, mask, contact_height, modality), deformed, mask

    def render_real_pose(
        self,
        surface_height_m: np.ndarray,
        valid_mask: np.ndarray,
        surface_offset_mm: float,
        modality: str,
    ):
        """Render a surface rerasterized in the current offset-aware gel frame."""
        cv2 = self.cv2
        surface = np.asarray(surface_height_m, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=np.uint8)
        if surface.shape != (self.height, self.width):
            surface = cv2.resize(surface, (self.width, self.height))
            valid = cv2.resize(
                valid, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        valid = valid.astype(bool)
        object_height = -surface * 1000.0 / self.height_pixel_mm
        gel = (
            float(self.gel.max())
            - self.gel
            - float(surface_offset_mm) / self.height_pixel_mm
        ).astype(np.float32)
        contact = (object_height > gel) & valid
        interacted = np.where(contact, object_height, gel).astype(np.float32)
        penetration = np.maximum(object_height - gel, 0.0)
        pressing_mm = float(penetration[contact].max() * self.height_pixel_mm) \
            if contact.any() else 0.0
        deformed, mask, contact_height = self.core.deformApprox(
            pressing_mm, interacted, gel, contact
        )
        return self._format(deformed, mask, contact_height, modality), deformed, mask


def validate_modality(modality: str) -> None:
    if modality not in ("shadow", "sim", "tactile_normal"):
        raise ValueError(f"Unsupported Taxim modality: {modality}")
