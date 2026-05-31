from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ego_vla.config import ProcessingConfig
from ego_vla.pipeline import EgoVlaPipeline

FIXTURE = Path(__file__).with_name("81cb47c22f510a2e2734c9a7069baa75.mp4")


class PipelineStageTests(unittest.TestCase):
    def test_default_pipeline_writes_no_segments(self) -> None:
        config = ProcessingConfig()
        config.pose.backend = "none"
        config.video.sample_rate_hz = 5.0
        config.video.max_frames = 4

        with tempfile.TemporaryDirectory() as tmp:
            metadata = EgoVlaPipeline(config).process(FIXTURE, Path(tmp) / "ds")
            output = metadata["output"]
            self.assertEqual(output["segment_count"], 0)
            self.assertIsNone(output["segments_path"])
            self.assertFalse((Path(tmp) / "ds" / "segments.jsonl").exists())
            self.assertEqual(metadata["stages"]["pose"], "none")
            self.assertEqual(metadata["stages"]["actions"], "none")
            self.assertEqual(metadata["calibration"]["backend"], "none")

    def test_segment_caption_pipeline_writes_segments(self) -> None:
        config = ProcessingConfig()
        config.pose.backend = "none"
        config.video.sample_rate_hz = 5.0
        config.video.max_frames = 10
        config.actions.backend = "segment_caption"
        config.actions.extra = {
            "segmenter": "fixed_window",
            "captioner": "template",
            "window_sec": 1.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds"
            metadata = EgoVlaPipeline(config).process(FIXTURE, dataset_dir)
            output = metadata["output"]

            self.assertGreater(output["segment_count"], 0)
            segments_path = dataset_dir / "segments.jsonl"
            self.assertTrue(segments_path.exists())
            self.assertEqual(metadata["stages"]["actions"], "segment_caption")

            segments = [json.loads(line) for line in segments_path.read_text().splitlines()]
            self.assertEqual(len(segments), output["segment_count"])
            first = segments[0]
            self.assertTrue(first["segment_id"].startswith(first["episode_id"]))
            self.assertTrue(first["caption"])
            self.assertEqual(first["source"]["segmenter"], "fixed_window")

    def test_frame_records_preserve_not_computed_defaults(self) -> None:
        config = ProcessingConfig()
        config.pose.backend = "none"
        config.video.max_frames = 2

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds"
            EgoVlaPipeline(config).process(FIXTURE, dataset_dir)
            first = json.loads((dataset_dir / "records.jsonl").read_text().splitlines()[0])
            self.assertEqual(first["depth"]["status"], "not_computed")
            self.assertEqual(first["ego_motion"]["status"], "not_computed")
            self.assertEqual(first["provenance"]["calibration_backend"], "none")


if __name__ == "__main__":
    unittest.main()
