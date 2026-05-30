from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ego_vla.visualization import render_pose_overlay_video, summarize_records, write_dataset_report


class VisualizationTests(unittest.TestCase):
    def test_summary_counts_pose_and_hand_records(self) -> None:
        records = [
            {
                "body_pose_2d": {"landmarks": [{"name": "left_wrist", "x": 0.1, "y": 0.2}]},
                "body_pose_3d": {"landmarks": [{"name": "left_wrist", "x": 0.1, "y": 0.2}]},
                "hands": {"left": {"landmarks": [{"name": "wrist", "x": 0.2, "y": 0.3}]}},
                "warnings": ["ok"],
                "provenance": {"pose_backend": "unit"},
            },
            {"warnings": ["missing"], "provenance": {"pose_backend": "none"}},
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["body_pose_2d_frames"], 1)
        self.assertEqual(summary["body_pose_3d_frames"], 1)
        self.assertEqual(summary["hand_frames"], 1)
        self.assertEqual(summary["pose_backend_counts"]["unit"], 1)

    def test_writes_html_report_and_pose_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            frames_dir = dataset_dir / "frames"
            frames_dir.mkdir(parents=True)
            self._write_test_frames(frames_dir)
            self._write_test_records(dataset_dir)

            report = write_dataset_report(dataset_dir, max_records=2)
            report_path = Path(report["report_path"])
            self.assertTrue(report_path.exists())
            self.assertIn("episode/000000", report_path.read_text(encoding="utf-8"))

            overlay = render_pose_overlay_video(dataset_dir, max_frames=2, fps=5.0)
            overlay_path = Path(overlay["video_path"])
            self.assertTrue(overlay_path.exists())
            self.assertGreater(overlay_path.stat().st_size, 0)
            self.assertEqual(overlay["frames_rendered"], 2)
            self.assertEqual(overlay["body_pose_frames"], 2)
            self.assertEqual(overlay["hand_pose_frames"], 1)

    def _write_test_frames(self, frames_dir: Path) -> None:
        import cv2
        import numpy as np

        for index in range(2):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :, 1] = 60 + index * 80
            cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), frame)

    def _write_test_records(self, dataset_dir: Path) -> None:
        records = [
            _record(
                frame_id="episode/000000",
                image_path="frames/000000.jpg",
                timestamp_sec=0.0,
                include_hand=True,
            ),
            _record(
                frame_id="episode/000001",
                image_path="frames/000001.jpg",
                timestamp_sec=0.2,
                include_hand=False,
            ),
        ]
        with (dataset_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle)
                handle.write("\n")
        (dataset_dir / "metadata.json").write_text(
            json.dumps({"dataset": {"name": "unit"}}),
            encoding="utf-8",
        )


def _record(
    frame_id: str,
    image_path: str,
    timestamp_sec: float,
    include_hand: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_name": "unit",
        "episode_id": "episode",
        "frame_id": frame_id,
        "frame_index": 0,
        "timestamp_sec": timestamp_sec,
        "observation": {
            "rgb": {
                "path": image_path,
                "width": 96,
                "height": 64,
                "encoding": "jpg",
            }
        },
        "camera": {"coordinate_frame": "camera"},
        "body_pose_2d": {
            "source": "unit",
            "coordinate_frame": "image_normalized",
            "landmarks": [
                {"name": "left_shoulder", "x": 0.2, "y": 0.2, "visibility": 1.0},
                {"name": "right_shoulder", "x": 0.8, "y": 0.2, "visibility": 1.0},
                {"name": "left_elbow", "x": 0.25, "y": 0.5, "visibility": 1.0},
                {"name": "left_wrist", "x": 0.3, "y": 0.8, "visibility": 1.0},
            ],
        },
        "hands": {},
        "warnings": [],
        "provenance": {"pose_backend": "unit"},
    }
    if include_hand:
        record["hands"] = {
            "left": {
                "side": "left",
                "source": "unit",
                "coordinate_frame": "image_normalized_relative_z",
                "landmarks": [
                    {"name": "wrist", "x": 0.3, "y": 0.8},
                    {"name": "index_finger_mcp", "x": 0.35, "y": 0.7},
                    {"name": "index_finger_tip", "x": 0.4, "y": 0.55},
                ],
            }
        }
    return record


if __name__ == "__main__":
    unittest.main()
