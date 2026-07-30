import numpy as np

from taxim_baseline.media import VideoSink, require_cv2, video_info


def test_video_sink_preserves_frame_contract(tmp_path):
    path = tmp_path / "out.mp4"
    sink = VideoSink(path, 32, 24, 7.0)
    for value in range(5):
        sink.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    sink.close()
    count, width, height, fps = video_info(path)
    assert (count, width, height) == (5, 32, 24)
    assert fps > 0

    capture = require_cv2().VideoCapture(str(path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 5
