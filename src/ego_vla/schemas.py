from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Landmark:
    name: str
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class PoseEstimate:
    source: str
    coordinate_frame: str
    landmarks: list[Landmark] = field(default_factory=list)
    confidence: float | None = None
    status: str = "ok"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HandPoseEstimate:
    side: str
    source: str
    coordinate_frame: str
    landmarks: list[Landmark] = field(default_factory=list)
    confidence: float | None = None
    status: str = "ok"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ObjectEstimate:
    label: str
    bbox_xyxy: list[float]
    confidence: float | None = None
    track_id: str | None = None
    mask_path: str | None = None


@dataclass(slots=True)
class DenseSignal:
    path: str | None = None
    backend: str = "none"
    status: str = "not_computed"
    coordinate_frame: str | None = None
    metric_scale: bool | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EgoMotionEstimate:
    backend: str = "none"
    status: str = "not_computed"
    coordinate_frame: str = "world"
    translation_xyz: list[float] | None = None
    quaternion_xyzw: list[float] | None = None
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionEstimate:
    label: str | None = None
    controls: dict[str, float] = field(default_factory=dict)
    language_instruction: str | None = None
    backend: str = "none"
    status: str = "not_computed"
    confidence: float | None = None


@dataclass(slots=True)
class CameraCalibration:
    """Per-video camera calibration produced by a pre-processing stage.

    Calibration is the metric anchor for depth and SLAM/VO: without it those
    backends can only produce up-to-scale estimates.
    """

    backend: str = "none"
    status: str = "not_computed"
    coordinate_frame: str = "camera"
    intrinsics: dict[str, float] | None = None
    distortion: list[float] | None = None
    image_width: int | None = None
    image_height: int | None = None
    metric_scale: bool | None = None
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class SegmentRecord:
    """A clip-level record: a contiguous slice of frames with an optional
    action label, language instruction, and free-text caption.

    Segments are produced by a clip stage (temporal segmentation + captioning)
    and written to ``segments.jsonl`` alongside the per-frame ``records.jsonl``.
    """

    segment_id: str
    episode_id: str
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    frame_ids: list[str] = field(default_factory=list)
    key_frame_ids: list[str] = field(default_factory=list)
    label: str | None = None
    instruction: str | None = None
    caption: str | None = None
    objects: list[str] = field(default_factory=list)
    confidence: float | None = None
    status: str = "ok"
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class FrameEstimate:
    body_pose_3d: PoseEstimate | None = None
    body_pose_2d: PoseEstimate | None = None
    hands: dict[str, HandPoseEstimate] = field(default_factory=dict)
    objects: list[ObjectEstimate] = field(default_factory=list)
    depth: DenseSignal = field(default_factory=DenseSignal)
    ego_motion: EgoMotionEstimate = field(default_factory=EgoMotionEstimate)
    action: ActionEstimate = field(default_factory=ActionEstimate)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoMetadata:
    source_path: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_sec: float | None


@dataclass(slots=True)
class VLAFrameRecord:
    dataset_name: str
    episode_id: str
    frame_id: str
    frame_index: int
    timestamp_sec: float
    observation: dict[str, Any]
    camera: dict[str, Any]
    body_pose_3d: PoseEstimate | None = None
    body_pose_2d: PoseEstimate | None = None
    hands: dict[str, HandPoseEstimate] = field(default_factory=dict)
    objects: list[ObjectEstimate] = field(default_factory=list)
    depth: DenseSignal = field(default_factory=DenseSignal)
    ego_motion: EgoMotionEstimate = field(default_factory=EgoMotionEstimate)
    action: ActionEstimate = field(default_factory=ActionEstimate)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def json_schema_example() -> dict[str, Any]:
    record = VLAFrameRecord(
        dataset_name="egocentric_vla_debug",
        episode_id="debug_video",
        frame_id="debug_video/000000",
        frame_index=0,
        timestamp_sec=0.0,
        observation={
            "rgb": {
                "path": "frames/000000.jpg",
                "width": 1920,
                "height": 1080,
                "encoding": "jpg",
            },
            "source_video": "debug_video.mp4",
        },
        camera={
            "intrinsics": None,
            "coordinate_frame": "camera",
        },
        hands={
            "left": HandPoseEstimate(
                side="left",
                source="mediapipe",
                coordinate_frame="image_normalized_relative_z",
                landmarks=[Landmark(name="wrist", x=0.5, y=0.5, z=-0.02, confidence=0.8)],
                confidence=0.8,
            )
        },
        provenance={"pipeline_version": "0.1.0"},
    )
    return record.to_dict()
