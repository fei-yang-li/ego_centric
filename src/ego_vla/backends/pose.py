"""Pose (frame-level) backends."""

from __future__ import annotations

from ego_vla.config import PoseConfig
from ego_vla.stages import POSE_BACKENDS


@POSE_BACKENDS.register("none")
def _build_none(pose: PoseConfig):
    from ego_vla.estimators import NoopEstimator

    return NoopEstimator()


@POSE_BACKENDS.register("mediapipe")
def _build_mediapipe(pose: PoseConfig):
    from ego_vla.estimators import MediaPipeHolisticEstimator

    return MediaPipeHolisticEstimator(
        model_complexity=pose.model_complexity,
        min_detection_confidence=pose.min_detection_confidence,
        min_tracking_confidence=pose.min_tracking_confidence,
        model_asset_path=pose.model_asset_path,
        auto_download_model=pose.auto_download_model,
    )
