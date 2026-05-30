from __future__ import annotations

import unittest

from ego_vla.video import compute_sampled_frame_indices


class VideoSamplingTests(unittest.TestCase):
    def test_samples_by_rate(self) -> None:
        indices = compute_sampled_frame_indices(total_frames=30, fps=30.0, sample_rate_hz=5.0)
        self.assertEqual(indices, [0, 6, 12, 18, 24])

    def test_non_positive_sample_rate_keeps_all_frames(self) -> None:
        indices = compute_sampled_frame_indices(total_frames=4, fps=30.0, sample_rate_hz=0)
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_max_frames_caps_output(self) -> None:
        indices = compute_sampled_frame_indices(
            total_frames=100,
            fps=25.0,
            sample_rate_hz=5.0,
            max_frames=3,
        )
        self.assertEqual(indices, [0, 5, 10])

    def test_invalid_fps_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_sampled_frame_indices(total_frames=10, fps=0.0, sample_rate_hz=5.0)


if __name__ == "__main__":
    unittest.main()
