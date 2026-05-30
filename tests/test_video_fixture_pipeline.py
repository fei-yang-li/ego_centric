from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ego_vla.config import ProcessingConfig
from ego_vla.pipeline import EgoVlaPipeline


class VideoFixturePipelineTests(unittest.TestCase):
    def test_repository_mp4_fixture_processes_without_pose_backend(self) -> None:
        fixture = Path(__file__).with_name("81cb47c22f510a2e2734c9a7069baa75.mp4")
        self.assertTrue(fixture.exists(), "Expected repository MP4 fixture to exist")

        config = ProcessingConfig()
        config.pose.backend = "none"
        config.video.sample_rate_hz = 5.0
        config.video.max_frames = 3

        with tempfile.TemporaryDirectory() as tmp:
            metadata = EgoVlaPipeline(config).process(fixture, Path(tmp) / "dataset")
            output = metadata["output"]
            records_path = Path(output["records_path"])
            frames_dir = Path(output["frames_dir"])

            self.assertEqual(output["record_count"], 3)
            self.assertTrue(records_path.exists())
            self.assertTrue((frames_dir / "000000.jpg").exists())


if __name__ == "__main__":
    unittest.main()
