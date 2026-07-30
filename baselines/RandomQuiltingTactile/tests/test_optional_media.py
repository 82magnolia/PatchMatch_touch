import tempfile
import unittest
from pathlib import Path


try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


@unittest.skipIf(np is None, "NumPy is not installed in the lightweight test environment")
class QuiltingTests(unittest.TestCase):
    def test_quilting_shape_and_seed(self):
        from rqt.quilting import quilt

        source = np.arange(18 * 18 * 3, dtype=np.uint8).reshape(18, 18, 3)
        first = quilt(source, (20, 24), block=8, overlap=2, tolerance=0.1, seed=7)
        second = quilt(source, (20, 24), block=8, overlap=2, tolerance=0.1, seed=7)
        self.assertEqual(first.shape, (20, 24, 3))
        self.assertTrue(np.array_equal(first, second))


@unittest.skipIf(cv2 is None, "OpenCV is not installed in the lightweight test environment")
class VideoTests(unittest.TestCase):
    def test_repeated_video_metadata(self):
        from run_baseline import video_info, write_repeated_video

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "repeat.mp4"
            frame = np.full((10, 12, 3), 127, dtype=np.uint8)
            write_repeated_video(frame, output, count=4, width=12, height=10, fps=5.0)
            count, width, height, fps = video_info(output)
            self.assertEqual((count, width, height), (4, 12, 10))
            self.assertGreater(fps, 0)


if __name__ == "__main__":
    unittest.main()
