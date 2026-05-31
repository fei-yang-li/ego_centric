from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ego_vla.backends._common import (
    IoUTracker,
    compose_pose,
    encode_depth_png16,
    pinhole_intrinsics,
    rotation_matrix_to_quaternion,
)
from ego_vla.backends.depth import DepthMapProcessor
from ego_vla.backends.objects import DetectionTrackProcessor
from ego_vla.config import ProcessingConfig
from ego_vla.schemas import VideoMetadata, VLAFrameRecord
from ego_vla.stages import (
    CALIBRATION_BACKENDS,
    DEPTH_BACKENDS,
    EGO_MOTION_BACKENDS,
    OBJECT_BACKENDS,
    StageContext,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def _record(index: int, output_dir: Path) -> VLAFrameRecord:
    return VLAFrameRecord(
        dataset_name="unit",
        episode_id="episode",
        frame_id=f"episode/{index:06d}",
        frame_index=index,
        timestamp_sec=index * 0.1,
        observation={"rgb": {"path": f"frames/{index:06d}.jpg"}},
        camera={"coordinate_frame": "camera"},
    )


def _context(output_dir: Path, width: int = 16, height: int = 12) -> StageContext:
    return StageContext(
        config=ProcessingConfig(),
        video_path=Path("video.mp4"),
        video_metadata=VideoMetadata("video.mp4", width, height, 30.0, 30, 1.0),
        output_dir=output_dir,
        frames_dir=output_dir / "frames",
    )


def _write_frame(output_dir: Path, index: int, width: int = 16, height: int = 12) -> None:
    import cv2

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 50 + index * 20, dtype=np.uint8)
    cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), image)


class GeometryHelperTests(unittest.TestCase):
    def test_quaternion_of_identity(self) -> None:
        self.assertEqual(rotation_matrix_to_quaternion(np.eye(3)), [0.0, 0.0, 0.0, 1.0])

    def test_quaternion_of_z_rotation(self) -> None:
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quaternion = rotation_matrix_to_quaternion(rot)
        self.assertAlmostEqual(quaternion[2], 0.70710678, places=5)
        self.assertAlmostEqual(quaternion[3], 0.70710678, places=5)

    def test_compose_pose_translates_in_world(self) -> None:
        rot_prev = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        _, t_world = compose_pose(rot_prev, np.zeros(3), np.eye(3), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(t_world, [0.0, 1.0, 0.0], atol=1e-9)

    def test_pinhole_intrinsics_center(self) -> None:
        intr = pinhole_intrinsics(1280, 720, 70.0)
        self.assertEqual(intr["cx"], 640.0)
        self.assertEqual(intr["cy"], 360.0)
        self.assertGreater(intr["fx"], 0.0)


class IoUTrackerTests(unittest.TestCase):
    def test_persistent_id_for_overlapping_box(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3)
        first = tracker.update([{"label": "cup", "bbox_xyxy": [0, 0, 10, 10]}])
        second = tracker.update([{"label": "cup", "bbox_xyxy": [1, 1, 11, 11]}])
        self.assertEqual(first, second)

    def test_new_id_for_disjoint_box(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3)
        first = tracker.update([{"label": "cup", "bbox_xyxy": [0, 0, 10, 10]}])
        second = tracker.update([{"label": "cup", "bbox_xyxy": [100, 100, 110, 110]}])
        self.assertNotEqual(first, second)


class DepthEncodingTests(unittest.TestCase):
    def test_metric_depth_to_millimetres(self) -> None:
        encoded, meta = encode_depth_png16(np.array([[1.0, 2.0]]), metric=True)
        self.assertEqual(encoded.tolist(), [[1000, 2000]])
        self.assertFalse(meta["normalized"])

    def test_relative_depth_normalized(self) -> None:
        encoded, meta = encode_depth_png16(np.array([[0.0, 1.0]]), metric=False)
        self.assertEqual(encoded.max(), 65535)
        self.assertTrue(meta["normalized"])


class DepthProcessorTests(unittest.TestCase):
    def test_writes_depth_map_and_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_frame(output_dir, 0)
            record = _record(0, output_dir)

            def fake_infer(rgb: np.ndarray) -> np.ndarray:
                return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

            processor = DepthMapProcessor(fake_infer, "fake_depth", metric=True)
            processor.run([record], _context(output_dir))

            self.assertEqual(record.depth.backend, "fake_depth")
            self.assertEqual(record.depth.status, "ok")
            self.assertTrue(record.depth.metric_scale)
            self.assertTrue((output_dir / record.depth.path).exists())


class ObjectProcessorTests(unittest.TestCase):
    def test_detection_track_processor_assigns_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            records = []
            for i in range(2):
                _write_frame(output_dir, i)
                records.append(_record(i, output_dir))

            def detector(frame_bgr, frame_index):
                return [{"label": "cup", "bbox_xyxy": [1, 1, 8, 8], "confidence": 0.9}]

            processor = DetectionTrackProcessor(detector, IoUTracker(), "fake_objects")
            ctx = _context(output_dir)
            processor.run(records, ctx)

            self.assertEqual(records[0].objects[0].label, "cup")
            self.assertEqual(records[0].objects[0].track_id, records[1].objects[0].track_id)
            self.assertEqual(ctx.extra["objects"]["backend"], "fake_objects")


class CalibrationTests(unittest.TestCase):
    def test_pinhole_prior_backend(self) -> None:
        config = ProcessingConfig()
        config.calibration.backend = "pinhole_prior"
        config.calibration.extra = {"horizontal_fov_deg": 90.0}
        with tempfile.TemporaryDirectory() as tmp:
            calibration = CALIBRATION_BACKENDS.create(
                "pinhole_prior", config.calibration, config
            ).run(_context(Path(tmp), width=1000, height=500))
        self.assertEqual(calibration.status, "approximate")
        self.assertEqual(calibration.intrinsics["cx"], 500.0)

    def test_checkerboard_requires_images_dir(self) -> None:
        config = ProcessingConfig()
        with self.assertRaises(RuntimeError):
            CALIBRATION_BACKENDS.create("opencv_checkerboard", config.calibration, config)


class MissingDependencyTests(unittest.TestCase):
    def _assert_build_raises(self, registry, name) -> None:
        config = ProcessingConfig()
        section = config.depth  # any BackendConfig works for these factories
        with self.assertRaises(RuntimeError):
            registry.create(name, section, config)

    def test_external_ego_motion_backends_raise(self) -> None:
        for name in ("dpvo", "droid_slam", "orbslam3"):
            self._assert_build_raises(EGO_MOTION_BACKENDS, name)

    @unittest.skipIf(HAS_TORCH and HAS_TRANSFORMERS, "depth deps installed")
    def test_depth_anything_requires_deps(self) -> None:
        self._assert_build_raises(DEPTH_BACKENDS, "depth_anything_v2")

    @unittest.skipIf(HAS_ULTRALYTICS, "ultralytics installed")
    def test_yolo_requires_deps(self) -> None:
        self._assert_build_raises(OBJECT_BACKENDS, "yolo")

    @unittest.skipIf(HAS_TORCH and HAS_TRANSFORMERS, "transformers installed")
    def test_grounding_dino_requires_deps(self) -> None:
        self._assert_build_raises(OBJECT_BACKENDS, "grounding_dino")


if __name__ == "__main__":
    unittest.main()
