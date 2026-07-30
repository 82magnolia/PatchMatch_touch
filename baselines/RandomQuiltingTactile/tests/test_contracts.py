import tempfile
import unittest
from pathlib import Path

from rqt.contracts import metric_payload, output_names, repeated_frames, resolve_pairs


class ContractTests(unittest.TestCase):
    def test_sim_identity_pairing(self):
        self.assertEqual(
            resolve_pairs("sim_gt_retrieval", [0, 1, 2], [1, 2, 3]),
            [(1, 1), (2, 2)],
        )

    def test_real_odd_to_previous_even_pairing(self):
        self.assertEqual(
            resolve_pairs("real_gt_retrieval", [0, 1, 2, 3], [0, 1, 2, 3]),
            [(1, 0), (3, 2)],
        )

    def test_tsv_uses_first_retrieved_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            path.write_text("query\tref\n4\t2,3\n")
            self.assertEqual(resolve_pairs("tsv", [2, 3], [4], path), [(4, 2)])

    def test_output_names(self):
        self.assertEqual(
            output_names(7, "shadow"),
            {
                "prediction": "7_transferred.mp4",
                "reference": "7_ref_shadow.mp4",
                "query": "7_query_shadow.mp4",
            },
        )

    def test_video_repetition_contract(self):
        frame = object()
        frames = repeated_frames(frame, 4)
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(item is frame for item in frames))

    def test_metrics_schema(self):
        values = {"MSE": 1, "PSNR": 2, "SSIM": 3, "LPIPS": 4}
        payload = metric_payload({9: values})
        self.assertEqual(set(payload), {"per_touch", "average"})
        self.assertEqual(payload["average"], values)

    def test_query_subset_filter_shape(self):
        pairs = resolve_pairs("sim_gt_retrieval", [0, 1, 2], [0, 1, 2])
        selected = [(query, reference) for query, reference in pairs if query in {1}]
        self.assertEqual(selected, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
