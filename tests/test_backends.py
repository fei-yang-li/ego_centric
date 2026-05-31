from __future__ import annotations

import unittest
from pathlib import Path

from ego_vla.backends.captioning import TemplateCaptioner
from ego_vla.backends.segmentation import FixedWindowSegmenter, NoneSegmenter
from ego_vla.config import ProcessingConfig
from ego_vla.schemas import HandPoseEstimate, Landmark, VideoMetadata, VLAFrameRecord
from ego_vla.stages import StageContext


def _record(index: int, timestamp_sec: float, with_hand: bool = False) -> VLAFrameRecord:
    record = VLAFrameRecord(
        dataset_name="unit",
        episode_id="episode",
        frame_id=f"episode/{index:06d}",
        frame_index=index,
        timestamp_sec=timestamp_sec,
        observation={"rgb": {"path": f"frames/{index:06d}.jpg"}},
        camera={"coordinate_frame": "camera"},
    )
    if with_hand:
        record.hands = {
            "left": HandPoseEstimate(
                side="left",
                source="unit",
                coordinate_frame="image_normalized",
                landmarks=[Landmark(name="wrist", x=0.3, y=0.8)],
            )
        }
    return record


def _context() -> StageContext:
    config = ProcessingConfig()
    return StageContext(
        config=config,
        video_path=Path("video.mp4"),
        video_metadata=VideoMetadata("video.mp4", 96, 64, 30.0, 90, 3.0),
        output_dir=Path("/tmp/x"),
        frames_dir=Path("/tmp/x/frames"),
    )


class SegmentationTests(unittest.TestCase):
    def test_none_segmenter_returns_empty(self) -> None:
        segmenter = NoneSegmenter({}, ProcessingConfig())
        self.assertEqual(segmenter.segment([_record(0, 0.0)], _context()), [])

    def test_fixed_window_partitions_by_time(self) -> None:
        records = [_record(i, i * 0.5) for i in range(6)]  # 0.0, 0.5 .. 2.5s
        segmenter = FixedWindowSegmenter({"window_sec": 1.0}, ProcessingConfig())
        segments = segmenter.segment(records, _context())
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].frame_ids, ["episode/000000", "episode/000001"])
        self.assertEqual(segments[0].start_sec, 0.0)
        self.assertEqual(segments[0].end_sec, 0.5)
        self.assertEqual(segments[1].frame_ids, ["episode/000002", "episode/000003"])

    def test_fixed_window_overlap_with_stride(self) -> None:
        records = [_record(i, i * 0.5) for i in range(5)]  # 0..2.0s
        segmenter = FixedWindowSegmenter(
            {"window_sec": 1.0, "stride_sec": 0.5}, ProcessingConfig()
        )
        segments = segmenter.segment(records, _context())
        self.assertGreater(len(segments), 3)

    def test_fixed_window_rejects_bad_params(self) -> None:
        with self.assertRaises(ValueError):
            FixedWindowSegmenter({"window_sec": 0}, ProcessingConfig())


class CaptionerTests(unittest.TestCase):
    def test_template_caption_reports_hand_frames(self) -> None:
        records = [_record(0, 0.0, with_hand=True), _record(1, 0.5, with_hand=False)]
        segmenter = FixedWindowSegmenter({"window_sec": 2.0}, ProcessingConfig())
        segments = segmenter.segment(records, _context())
        self.assertEqual(len(segments), 1)
        captioner = TemplateCaptioner({}, ProcessingConfig())
        captioner.caption(segments[0], records, _context())
        self.assertIn("hands visible in 1/2 frames", segments[0].caption)
        self.assertEqual(segments[0].status, "heuristic")
        self.assertEqual(segments[0].source["captioner"], "template")


if __name__ == "__main__":
    unittest.main()
