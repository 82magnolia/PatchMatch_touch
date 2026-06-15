"""
Camera abstraction for RealSense D435i and ZED 2i.

Both classes expose the same interface:
  cam.grab()        → bool  (False on failure)
  cam.color_bgr     → (H,W,3) uint8
  cam.depth_mm      → (H,W)   uint16, millimetres
  cam.intrinsics    → dict with fx, fy, cx, cy, width, height,
                               camera_matrix (3x3), dist_coeffs (5,)

Usage:
  from camera_utils import build_camera
  cam = build_camera(args)   # args.camera in {"realsense","zed"}, args.depth_mode for ZED
  try:
      while True:
          if not cam.grab(): continue
          color = cam.color_bgr
          depth = cam.depth_mm
  finally:
      cam.close()
"""

import sys
import numpy as np

CAPTURE_W, CAPTURE_H = 1280, 720

DEPTH_MODES_ZED = {
    "performance": None,
    "quality":     None,
    "ultra":       None,
    "neural":      None,
    "neural_plus": None,
}


# ── base ──────────────────────────────────────────────────────────────────────

class CameraBase:
    def grab(self) -> bool:
        raise NotImplementedError

    @property
    def color_bgr(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def depth_mm(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def intrinsics(self) -> dict:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


# ── RealSense ─────────────────────────────────────────────────────────────────

class RealSenseCamera(CameraBase):
    def __init__(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            sys.exit("pyrealsense2 not found. Install with: pip install pyrealsense2")

        self._rs = rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            sys.exit("No RealSense device found. Check USB connection.")
        dev = devices[0]
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"Found device: {name}  (serial {serial})")

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, CAPTURE_W, CAPTURE_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, CAPTURE_W, CAPTURE_H, rs.format.z16, 30)
        profile = pipeline.start(cfg)

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
        temporal.set_option(rs.option.filter_smooth_delta, 40)
        self._temporal = temporal

        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self._intr = {
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "camera_matrix": np.array(
                [[intr.fx, 0, intr.ppx],
                 [0, intr.fy, intr.ppy],
                 [0, 0, 1]], dtype=np.float64),
            "dist_coeffs": np.array(intr.coeffs, dtype=np.float64),
        }

        self._color_bgr = None
        self._depth_mm = None

    def grab(self) -> bool:
        frameset = self._pipeline.wait_for_frames()
        aligned = self._align.process(frameset)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            return False
        df = self._temporal.process(df)
        self._color_bgr = np.asanyarray(cf.get_data())
        self._depth_mm = np.asanyarray(df.get_data())   # uint16 mm
        return True

    @property
    def color_bgr(self) -> np.ndarray:
        return self._color_bgr

    @property
    def depth_mm(self) -> np.ndarray:
        return self._depth_mm

    @property
    def intrinsics(self) -> dict:
        return self._intr

    def close(self):
        self._pipeline.stop()


# ── ZED 2i ────────────────────────────────────────────────────────────────────

class ZEDCamera(CameraBase):
    def __init__(self, depth_mode: str = "neural_plus"):
        try:
            import pyzed.sl as sl
        except ImportError:
            sys.exit("pyzed not found. Install with: pip install pyzed")

        self._sl = sl

        mode_map = {
            "performance": sl.DEPTH_MODE.PERFORMANCE,
            "quality":     sl.DEPTH_MODE.QUALITY,
            "ultra":       sl.DEPTH_MODE.ULTRA,
            "neural":      sl.DEPTH_MODE.NEURAL,
            "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
        }
        if depth_mode not in mode_map:
            sys.exit(f"Unknown ZED depth mode: {depth_mode}. "
                     f"Choose from {list(mode_map)}")

        cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = mode_map[depth_mode]
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.3
        init.depth_maximum_distance = 3.0
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            sys.exit(f"Failed to open ZED camera: {status}")

        info = cam.get_camera_information()
        model = info.camera_model
        serial = info.serial_number
        print(f"Found device: {model}  (serial {serial})  depth_mode={depth_mode}")

        self._cam = cam
        self._rt = sl.RuntimeParameters()
        self._rt.enable_fill_mode = False

        self._image_sl = sl.Mat()
        self._depth_sl = sl.Mat()

        # ZED depth is already rectified; dist_coeffs are effectively zero
        lc = info.camera_configuration.calibration_parameters.left_cam
        fx, fy = lc.fx, lc.fy
        cx, cy = lc.cx, lc.cy
        w = info.camera_configuration.resolution.width
        h = info.camera_configuration.resolution.height
        self._intr = {
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "width": w, "height": h,
            "camera_matrix": np.array(
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float64),
            "dist_coeffs": np.zeros(5, dtype=np.float64),
        }

        self._color_bgr = None
        self._depth_mm = None

    def grab(self) -> bool:
        if self._cam.grab(self._rt) != self._sl.ERROR_CODE.SUCCESS:
            return False
        self._cam.retrieve_image(self._image_sl, self._sl.VIEW.LEFT)
        self._cam.retrieve_measure(self._depth_sl, self._sl.MEASURE.DEPTH)

        self._color_bgr = self._image_sl.get_data()[:, :, :3].copy()  # drop alpha

        depth_m = self._depth_sl.get_data()   # float32, metres, NaN for invalid
        depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
        self._depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
        return True

    @property
    def color_bgr(self) -> np.ndarray:
        return self._color_bgr

    @property
    def depth_mm(self) -> np.ndarray:
        return self._depth_mm

    @property
    def intrinsics(self) -> dict:
        return self._intr

    def close(self):
        self._cam.close()


# ── factory ───────────────────────────────────────────────────────────────────

def build_camera(args) -> CameraBase:
    """Build the camera requested by args.camera (and args.depth_mode for ZED)."""
    if args.camera == "zed":
        depth_mode = getattr(args, "depth_mode", "neural_plus") or "neural_plus"
        return ZEDCamera(depth_mode=depth_mode)
    return RealSenseCamera()
