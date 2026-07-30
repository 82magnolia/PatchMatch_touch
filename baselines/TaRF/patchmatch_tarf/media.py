"""Video helpers that keep prediction timing independent of query tactile pixels."""

from __future__ import annotations

from pathlib import Path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; activate the TaRF conda environment") from exc
    return cv2


def video_info(path: Path) -> tuple[int, int, int, float]:
    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open timing video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 5.0
    capture.release()
    if min(count, width, height) < 1:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return count, width, height, fps


def write_repeated_video(image, path: Path, count: int, width: int, height: int, fps: float):
    cv2 = require_cv2()
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    for _ in range(count):
        writer.write(frame)
    writer.release()

