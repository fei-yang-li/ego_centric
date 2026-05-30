from __future__ import annotations

from typing import Protocol

from ego_vla.config import ProcessingConfig
from ego_vla.schemas import FrameEstimate, HandPoseEstimate, Landmark, PoseEstimate


POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


class FrameEstimator(Protocol):
    backend_name: str

    def estimate(self, frame_bgr: object, frame_index: int, timestamp_sec: float) -> FrameEstimate:
        ...

    def close(self) -> None:
        ...


class NoopEstimator:
    backend_name = "none"

    def estimate(self, frame_bgr: object, frame_index: int, timestamp_sec: float) -> FrameEstimate:
        return FrameEstimate(
            warnings=[
                "pose backend is disabled; record contains schema placeholders only",
            ]
        )

    def close(self) -> None:
        return None


class MediaPipeHolisticEstimator:
    backend_name = "mediapipe"

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe pose estimation requires optional dependencies. "
                "Install with `pip install -e \".[vision]\"` or set pose.backend to `none`."
            ) from exc
        self._cv2 = cv2
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame_bgr: object, frame_index: int, timestamp_sec: float) -> FrameEstimate:
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        results = self._holistic.process(frame_rgb)

        body_pose_3d = None
        if results.pose_world_landmarks is not None:
            body_pose_3d = PoseEstimate(
                source=self.backend_name,
                coordinate_frame="mediapipe_world_meters_approx",
                landmarks=_landmarks_from_mediapipe(
                    results.pose_world_landmarks.landmark,
                    POSE_LANDMARK_NAMES,
                    include_visibility=True,
                ),
                confidence=_mean_visibility(results.pose_world_landmarks.landmark),
                notes=[
                    "MediaPipe world landmarks are useful for relative pose, but should be calibrated before treating them as metric ground truth.",
                ],
            )

        body_pose_2d = None
        if results.pose_landmarks is not None:
            body_pose_2d = PoseEstimate(
                source=self.backend_name,
                coordinate_frame="image_normalized",
                landmarks=_landmarks_from_mediapipe(
                    results.pose_landmarks.landmark,
                    POSE_LANDMARK_NAMES,
                    include_visibility=True,
                ),
                confidence=_mean_visibility(results.pose_landmarks.landmark),
            )

        hands: dict[str, HandPoseEstimate] = {}
        if results.left_hand_landmarks is not None:
            hands["left"] = _hand_estimate("left", results.left_hand_landmarks.landmark)
        if results.right_hand_landmarks is not None:
            hands["right"] = _hand_estimate("right", results.right_hand_landmarks.landmark)

        warnings = []
        if body_pose_3d is None:
            warnings.append("body_pose_3d not detected for this frame")
        if not hands:
            warnings.append("hand landmarks not detected for this frame")

        return FrameEstimate(
            body_pose_3d=body_pose_3d,
            body_pose_2d=body_pose_2d,
            hands=hands,
            warnings=warnings,
        )

    def close(self) -> None:
        self._holistic.close()


def build_frame_estimator(config: ProcessingConfig) -> FrameEstimator:
    backend = config.pose.backend.lower()
    if backend == "none":
        return NoopEstimator()
    if backend == "mediapipe":
        return MediaPipeHolisticEstimator(
            model_complexity=config.pose.model_complexity,
            min_detection_confidence=config.pose.min_detection_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
        )
    raise ValueError(f"Unsupported pose backend: {config.pose.backend}")


def _hand_estimate(side: str, landmarks: object) -> HandPoseEstimate:
    return HandPoseEstimate(
        side=side,
        source="mediapipe",
        coordinate_frame="image_normalized_relative_z",
        landmarks=_landmarks_from_mediapipe(landmarks, HAND_LANDMARK_NAMES, include_visibility=False),
        notes=[
            "Hand z is relative to wrist and image scale; use a calibrated hand model for metric 3D.",
        ],
    )


def _landmarks_from_mediapipe(
    landmarks: object,
    names: list[str],
    include_visibility: bool,
) -> list[Landmark]:
    output: list[Landmark] = []
    for index, point in enumerate(landmarks):
        name = names[index] if index < len(names) else f"landmark_{index}"
        visibility = getattr(point, "visibility", None) if include_visibility else None
        output.append(
            Landmark(
                name=name,
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                visibility=float(visibility) if visibility is not None else None,
            )
        )
    return output


def _mean_visibility(landmarks: object) -> float | None:
    values = [
        float(getattr(point, "visibility"))
        for point in landmarks
        if getattr(point, "visibility", None) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)
