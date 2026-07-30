from pathlib import Path

import cv2
import numpy as np

from patchmatch_tarf.media import video_info, write_repeated_video


def test_repeated_video_preserves_requested_timing(tmp_path: Path):
    path = tmp_path / "prediction.mp4"
    image = np.full((12, 16, 3), (10, 80, 190), np.uint8)
    write_repeated_video(image, path, count=6, width=32, height=24, fps=7.0)
    count, width, height, fps = video_info(path)
    assert (count, width, height) == (6, 32, 24)
    assert abs(fps - 7.0) < 0.1
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == 6
    assert np.array_equal(frames[0], frames[-1])

