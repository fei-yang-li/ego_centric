"""Multi-stage pipeline abstractions.

The dataset builder runs in ordered stages so that estimators operating at
different time scales can be composed without changing the per-frame record
contract:

1. ``PreProcessor``    -- runs once per video (e.g. camera calibration).
2. ``FrameEstimator``  -- runs per sampled frame (e.g. body/hand pose).
3. ``RecordProcessor`` -- runs over the full list of frame records, in place
   (e.g. depth per frame, object detection + tracking, SLAM/VO trajectory).
4. ``ClipProcessor``   -- runs over the full list of frame records and returns
   clip-level ``SegmentRecord`` objects (temporal action segmentation +
   captioning/labeling).

Each stage type owns a ``Registry``; built-in backends register themselves when
``ego_vla.backends`` is imported. Selecting a backend is a config concern, so
new capabilities are added by registering a factory rather than editing the
orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ego_vla.config import ProcessingConfig
from ego_vla.registry import Registry
from ego_vla.schemas import (
    CameraCalibration,
    FrameEstimate,
    SegmentRecord,
    VideoMetadata,
    VLAFrameRecord,
)


@dataclass(slots=True)
class StageContext:
    """Shared, read-mostly context handed to every stage."""

    config: ProcessingConfig
    video_path: Path
    video_metadata: VideoMetadata
    output_dir: Path
    frames_dir: Path
    calibration: CameraCalibration | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_frame_path(self, record: VLAFrameRecord) -> Path:
        """Absolute path to a record's RGB frame on disk."""
        rgb = record.observation.get("rgb", {}) if record.observation else {}
        rel = rgb.get("path")
        if not rel:
            raise ValueError(f"Record {record.frame_id} has no observation.rgb.path")
        return self.output_dir / rel


@runtime_checkable
class PreProcessor(Protocol):
    backend_name: str

    def run(self, ctx: StageContext) -> CameraCalibration:
        ...


@runtime_checkable
class FrameEstimator(Protocol):
    backend_name: str

    def estimate(
        self, frame_bgr: object, frame_index: int, timestamp_sec: float
    ) -> FrameEstimate:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class RecordProcessor(Protocol):
    backend_name: str

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class ClipProcessor(Protocol):
    backend_name: str

    def run(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Turns a frame-record sequence into clip boundaries (no labels)."""

    name: str

    def segment(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        ...


@runtime_checkable
class Captioner(Protocol):
    """Fills a segment's caption/instruction/label in place."""

    name: str

    def caption(
        self, segment: SegmentRecord, records: list[VLAFrameRecord], ctx: StageContext
    ) -> None:
        ...


CALIBRATION_BACKENDS: Registry[PreProcessor] = Registry("calibration")
POSE_BACKENDS: Registry[FrameEstimator] = Registry("pose")
DEPTH_BACKENDS: Registry[RecordProcessor] = Registry("depth")
EGO_MOTION_BACKENDS: Registry[RecordProcessor] = Registry("ego_motion")
OBJECT_BACKENDS: Registry[RecordProcessor] = Registry("objects")
ACTION_BACKENDS: Registry[ClipProcessor] = Registry("actions")
SEGMENTERS: Registry[Segmenter] = Registry("segmenter")
CAPTIONERS: Registry[Captioner] = Registry("captioner")


_BACKENDS_LOADED = False


def _ensure_backends_loaded() -> None:
    """Import built-in backends so their registrations run (idempotent)."""
    global _BACKENDS_LOADED
    if not _BACKENDS_LOADED:
        import ego_vla.backends  # noqa: F401  (import for side-effect: registration)

        _BACKENDS_LOADED = True


def build_pre_processor(config: ProcessingConfig) -> PreProcessor:
    _ensure_backends_loaded()
    return CALIBRATION_BACKENDS.create(config.calibration.backend, config.calibration, config)


def build_frame_estimator(config: ProcessingConfig) -> FrameEstimator:
    _ensure_backends_loaded()
    return POSE_BACKENDS.create(config.pose.backend, config.pose)


def build_record_processors(config: ProcessingConfig) -> list[RecordProcessor]:
    """Record processors run in a fixed order: depth -> objects -> ego_motion.

    Ego-motion runs last so a future SLAM/VO backend can consume depth and
    object tracks if it wants to.
    """
    _ensure_backends_loaded()
    return [
        DEPTH_BACKENDS.create(config.depth.backend, config.depth, config),
        OBJECT_BACKENDS.create(config.objects.backend, config.objects, config),
        EGO_MOTION_BACKENDS.create(config.ego_motion.backend, config.ego_motion, config),
    ]


def build_clip_processor(config: ProcessingConfig) -> ClipProcessor:
    _ensure_backends_loaded()
    return ACTION_BACKENDS.create(config.actions.backend, config.actions, config)


def available_backends() -> dict[str, list[str]]:
    """Map of stage name -> registered backend names (for discovery/CLI)."""
    _ensure_backends_loaded()
    return {
        "calibration": CALIBRATION_BACKENDS.available(),
        "pose": POSE_BACKENDS.available(),
        "depth": DEPTH_BACKENDS.available(),
        "objects": OBJECT_BACKENDS.available(),
        "ego_motion": EGO_MOTION_BACKENDS.available(),
        "actions": ACTION_BACKENDS.available(),
        "segmenter": SEGMENTERS.available(),
        "captioner": CAPTIONERS.available(),
    }
