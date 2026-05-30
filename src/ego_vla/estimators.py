from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.request import urlretrieve

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

DEFAULT_HOLISTIC_TASK_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
DEFAULT_MODEL_CACHE_PATH = Path.home() / ".cache" / "ego_vla" / "models" / "holistic_landmarker.task"


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
        model_asset_path: str | None = None,
        auto_download_model: bool = True,
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
        self._mp = mp
        self._mode = "tasks"
        self._holistic = None
        self._landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            self._mode = "solutions"
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        try:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "Installed mediapipe package does not provide either `solutions` or `tasks`. "
                "Install the latest mediapipe or set pose.backend to `none`."
            ) from exc

        model_path = _resolve_model_asset_path(model_asset_path, auto_download_model)
        try:
            options = vision.HolisticLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_landmarks_confidence=min_detection_confidence,
                min_hand_landmarks_confidence=min_detection_confidence,
                output_segmentation_mask=False,
            )
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        except OSError as exc:
            if "libEGL" in str(exc):
                raise RuntimeError(
                    "MediaPipe Tasks requires the Linux system library `libEGL.so.1`. "
                    "Install it with `sudo apt-get install -y libegl1`, or run with `--pose-backend none`."
                ) from exc
            raise

    def estimate(self, frame_bgr: object, frame_index: int, timestamp_sec: float) -> FrameEstimate:
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        if self._mode == "solutions":
            results = self._holistic.process(frame_rgb)
            return self._estimate_from_landmarks(
                pose_world_landmarks=_extract_landmarks(results.pose_world_landmarks),
                pose_landmarks=_extract_landmarks(results.pose_landmarks),
                left_hand_landmarks=_extract_landmarks(results.left_hand_landmarks),
                right_hand_landmarks=_extract_landmarks(results.right_hand_landmarks),
            )

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = max(0, int(round(timestamp_sec * 1000)))
        results = self._landmarker.detect_for_video(image, timestamp_ms)
        return self._estimate_from_landmarks(
            pose_world_landmarks=_extract_landmarks(results.pose_world_landmarks),
            pose_landmarks=_extract_landmarks(results.pose_landmarks),
            left_hand_landmarks=_extract_landmarks(results.left_hand_landmarks),
            right_hand_landmarks=_extract_landmarks(results.right_hand_landmarks),
        )

    def _estimate_from_landmarks(
        self,
        pose_world_landmarks: object | None,
        pose_landmarks: object | None,
        left_hand_landmarks: object | None,
        right_hand_landmarks: object | None,
    ) -> FrameEstimate:
        body_pose_3d = None
        if pose_world_landmarks:
            body_pose_3d = PoseEstimate(
                source=self.backend_name,
                coordinate_frame="mediapipe_world_meters_approx",
                landmarks=_landmarks_from_mediapipe(
                    pose_world_landmarks,
                    POSE_LANDMARK_NAMES,
                    include_visibility=True,
                ),
                confidence=_mean_visibility(pose_world_landmarks),
                notes=[
                    "MediaPipe world landmarks are useful for relative pose, but should be calibrated before treating them as metric ground truth.",
                ],
            )

        body_pose_2d = None
        if pose_landmarks:
            body_pose_2d = PoseEstimate(
                source=self.backend_name,
                coordinate_frame="image_normalized",
                landmarks=_landmarks_from_mediapipe(
                    pose_landmarks,
                    POSE_LANDMARK_NAMES,
                    include_visibility=True,
                ),
                confidence=_mean_visibility(pose_landmarks),
            )

        hands: dict[str, HandPoseEstimate] = {}
        if left_hand_landmarks:
            hands["left"] = _hand_estimate("left", left_hand_landmarks)
        if right_hand_landmarks:
            hands["right"] = _hand_estimate("right", right_hand_landmarks)

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
        if self._holistic is not None:
            self._holistic.close()
        if self._landmarker is not None:
            self._landmarker.close()

def build_frame_estimator(config: ProcessingConfig) -> FrameEstimator:
    backend = config.pose.backend.lower()
    if backend == "none":
        return NoopEstimator()
    if backend == "mediapipe":
        return MediaPipeHolisticEstimator(
            model_complexity=config.pose.model_complexity,
            min_detection_confidence=config.pose.min_detection_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
            model_asset_path=config.pose.model_asset_path,
            auto_download_model=config.pose.auto_download_model,
        )
    raise ValueError(f"Unsupported pose backend: {config.pose.backend}")



def _resolve_model_asset_path(model_asset_path: str | None, auto_download_model: bool) -> Path:
    if model_asset_path:
        path = Path(model_asset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe model asset not found: {path}")
        return path
    if not auto_download_model:
        raise RuntimeError(
            "MediaPipe Tasks requires pose.model_asset_path when auto_download_model is false."
        )
    DEFAULT_MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_MODEL_CACHE_PATH.exists():
        urlretrieve(DEFAULT_HOLISTIC_TASK_URL, DEFAULT_MODEL_CACHE_PATH)
    return DEFAULT_MODEL_CACHE_PATH


def _extract_landmarks(container: object) -> object | None:
    if container is None:
        return None
    if hasattr(container, "landmark"):
        return container.landmark
    if isinstance(container, list):
        if not container:
            return None
        first = container[0]
        if hasattr(first, "x"):
            return container
        if isinstance(first, list) and first:
            return first
    return container

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
