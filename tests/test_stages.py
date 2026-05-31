from __future__ import annotations

import unittest
from pathlib import Path

from ego_vla.config import ProcessingConfig
from ego_vla.schemas import SegmentRecord, VideoMetadata, VLAFrameRecord
from ego_vla.stages import (
    StageContext,
    available_backends,
    build_clip_processor,
    build_frame_estimator,
    build_pre_processor,
    build_record_processors,
)


def _record(frame_id: str, frame_index: int, timestamp_sec: float) -> VLAFrameRecord:
    return VLAFrameRecord(
        dataset_name="unit",
        episode_id="episode",
        frame_id=frame_id,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        observation={"rgb": {"path": f"frames/{frame_index:06d}.jpg"}},
        camera={"coordinate_frame": "camera"},
    )


def _context(config: ProcessingConfig) -> StageContext:
    return StageContext(
        config=config,
        video_path=Path("video.mp4"),
        video_metadata=VideoMetadata(
            source_path="video.mp4",
            width=96,
            height=64,
            fps=30.0,
            frame_count=90,
            duration_sec=3.0,
        ),
        output_dir=Path("/tmp/does-not-need-to-exist"),
        frames_dir=Path("/tmp/does-not-need-to-exist/frames"),
    )


class StageBuilderTests(unittest.TestCase):
    def test_available_backends_include_builtins(self) -> None:
        backends = available_backends()
        self.assertIn("none", backends["calibration"])
        self.assertIn("mediapipe", backends["pose"])
        self.assertIn("segment_caption", backends["actions"])
        self.assertIn("fixed_window", backends["segmenter"])
        self.assertIn("template", backends["captioner"])

    def test_default_builders_select_none_backends(self) -> None:
        config = ProcessingConfig()
        config.pose.backend = "none"  # default is mediapipe, which needs system libs
        self.assertEqual(build_pre_processor(config).backend_name, "none")
        self.assertEqual(build_frame_estimator(config).backend_name, "none")
        processors = build_record_processors(config)
        self.assertEqual([p.backend_name for p in processors], ["none", "none", "none"])
        self.assertEqual(build_clip_processor(config).backend_name, "none")

    def test_calibration_forwards_config_intrinsics(self) -> None:
        config = ProcessingConfig()
        config.camera.intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 48.0, "cy": 32.0}
        calibration = build_pre_processor(config).run(_context(config))
        self.assertEqual(calibration.status, "provided")
        self.assertEqual(calibration.intrinsics["fx"], 100.0)
        self.assertTrue(calibration.metric_scale)

    def test_record_processors_mark_not_computed(self) -> None:
        config = ProcessingConfig()
        records = [_record("episode/000000", 0, 0.0)]
        ctx = _context(config)
        for processor in build_record_processors(config):
            processor.run(records, ctx)
        self.assertEqual(records[0].depth.status, "not_computed")
        self.assertEqual(records[0].ego_motion.status, "not_computed")

    def test_segment_caption_clip_processor_produces_segments(self) -> None:
        config = ProcessingConfig()
        config.actions.backend = "segment_caption"
        config.actions.extra = {
            "segmenter": "fixed_window",
            "captioner": "template",
            "window_sec": 1.0,
        }
        records = [
            _record(f"episode/{i:06d}", i, i * 0.5) for i in range(6)
        ]  # 0.0 .. 2.5s
        clip = build_clip_processor(config)
        segments = clip.run(records, _context(config))
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(isinstance(s, SegmentRecord) for s in segments))
        self.assertTrue(all(s.caption for s in segments))
        self.assertEqual(segments[0].source["segmenter"], "fixed_window")
        self.assertEqual(segments[0].source["captioner"], "template")


if __name__ == "__main__":
    unittest.main()
