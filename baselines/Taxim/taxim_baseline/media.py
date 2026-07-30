"""Video metadata and output helpers."""

from __future__ import annotations

from pathlib import Path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; activate the Taxim environment") from exc
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


class VideoSink:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        cv2 = require_cv2()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.size = (width, height)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video: {path}")

    def write(self, frame):
        cv2 = require_cv2()
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def close(self):
        self.writer.release()

