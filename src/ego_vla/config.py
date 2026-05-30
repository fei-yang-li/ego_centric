from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    name: str = "egocentric_vla_debug"
    episode_id: str | None = None
    language_instruction: str | None = None
    operator_id: str | None = None


@dataclass(slots=True)
class VideoConfig:
    sample_rate_hz: float = 5.0
    max_frames: int | None = None
    image_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass(slots=True)
class CameraConfig:
    intrinsics: dict[str, float] | None = None
    coordinate_frame: str = "camera"


@dataclass(slots=True)
class PoseConfig:
    backend: str = "mediapipe"
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_asset_path: str | None = None
    auto_download_model: bool = True


@dataclass(slots=True)
class BackendConfig:
    backend: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    depth: BackendConfig = field(default_factory=BackendConfig)
    ego_motion: BackendConfig = field(default_factory=BackendConfig)
    objects: BackendConfig = field(default_factory=BackendConfig)
    actions: BackendConfig = field(default_factory=BackendConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ProcessingConfig":
        return cls(
            dataset=_load_dataclass(DatasetConfig, raw.get("dataset", {})),
            video=_load_dataclass(VideoConfig, raw.get("video", {})),
            camera=_load_dataclass(CameraConfig, raw.get("camera", {})),
            pose=_load_dataclass(PoseConfig, raw.get("pose", {})),
            depth=_load_backend(raw.get("depth", {})),
            ego_motion=_load_backend(raw.get("ego_motion", {})),
            objects=_load_backend(raw.get("objects", {})),
            actions=_load_backend(raw.get("actions", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    if not isinstance(values, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(values).__name__}")
    allowed = cls.__dataclass_fields__.keys()
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**values)


def _load_backend(values: dict[str, Any]) -> BackendConfig:
    if not isinstance(values, dict):
        raise TypeError(f"Expected mapping for backend config, got {type(values).__name__}")
    backend = values.get("backend", "none")
    extra = {key: value for key, value in values.items() if key != "backend"}
    return BackendConfig(backend=backend, extra=extra)


def load_config(path: str | Path | None) -> ProcessingConfig:
    if path is None:
        return ProcessingConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ProcessingConfig.from_mapping(raw)


def write_default_config(path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(ProcessingConfig().to_dict(), handle, indent=2)
        handle.write("\n")
