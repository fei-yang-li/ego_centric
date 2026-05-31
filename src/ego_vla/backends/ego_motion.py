"""Ego-motion / camera-trajectory (record-level) backends.

Backends:

- ``none``       -- annotate ``ego_motion`` as not computed.
- ``opencv_vo``  -- monocular feature-based visual odometry (ORB + essential
  matrix + ``recoverPose``). Runs on CPU; the trajectory is *up to scale*
  (each step's translation is a unit vector) unless a per-step scale is given.
- ``dpvo`` / ``droid_slam`` / ``orbslam3`` -- adapters for the well-known SLAM/VO
  systems. These need a GPU and/or a compiled package, so the factories raise an
  actionable error pointing at their installation.

``opencv_vo`` uses the intrinsics resolved by the calibration stage
(``ctx.calibration``); if none are available it falls back to a 70 deg-FOV
pinhole prior and records a warning.
"""

from __future__ import annotations

from typing import Any

from ego_vla.backends._common import (
    compose_pose,
    intrinsic_matrix,
    load_cv2,
    pinhole_intrinsics,
    read_frame_bgr,
    rotation_matrix_to_quaternion,
)
from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import EgoMotionEstimate, VLAFrameRecord
from ego_vla.stages import EGO_MOTION_BACKENDS, StageContext

_NOT_COMPUTED_NOTE = (
    "Camera trajectory is not computed by the default pipeline; add SLAM or "
    "visual odometry for metric VLA state."
)


class NoEgoMotion:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        for record in records:
            record.ego_motion.backend = "none"
            record.ego_motion.status = "not_computed"
            if _NOT_COMPUTED_NOTE not in record.ego_motion.notes:
                record.ego_motion.notes.append(_NOT_COMPUTED_NOTE)

    def close(self) -> None:
        return None


class OpenCVVisualOdometry:
    backend_name = "opencv_vo"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        extra = dict(section.extra or {})
        self._config = config
        self._n_features = int(extra.get("n_features", 2000))
        self._min_matches = int(extra.get("min_matches", 12))
        self._ransac_threshold = float(extra.get("ransac_threshold", 1.0))
        self._step_scale = float(extra.get("step_scale", 1.0))
        self._fov_deg = float(extra.get("fallback_horizontal_fov_deg", 70.0))

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        if not records:
            return
        import numpy as np

        cv2 = load_cv2()
        intrinsics, intrinsics_note = self._resolve_intrinsics(ctx)
        k_matrix = intrinsic_matrix(intrinsics)

        orb = cv2.ORB_create(nfeatures=self._n_features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        prev_gray = None
        prev_kp = None
        prev_des = None
        frames_with_pose = 0

        for index, record in enumerate(records):
            frame = read_frame_bgr(ctx.resolve_frame_path(record))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            notes = [
                "Up-to-scale monocular VO (ORB + essential matrix); translation "
                "magnitude is not metric.",
            ]
            if intrinsics_note:
                notes.append(intrinsics_note)
            status = "ok"
            confidence: float | None = None

            if index == 0:
                status = "ok"
                confidence = 1.0
            else:
                rel = self._estimate_relative(
                    cv2, np, matcher, k_matrix, prev_kp, prev_des, keypoints, descriptors
                )
                if rel is None:
                    status = "partial"
                    notes.append("insufficient matches; carried previous pose")
                else:
                    r_rel, t_rel, inlier_ratio = rel
                    rotation, translation = compose_pose(
                        rotation, translation, r_rel, t_rel * self._step_scale
                    )
                    confidence = float(inlier_ratio)
                    frames_with_pose += 1

            record.ego_motion = EgoMotionEstimate(
                backend=self.backend_name,
                status=status,
                coordinate_frame="world_up_to_scale",
                translation_xyz=[float(v) for v in translation],
                quaternion_xyzw=rotation_matrix_to_quaternion(rotation),
                confidence=confidence,
                notes=notes,
            )

            prev_gray = gray
            prev_kp = keypoints
            prev_des = descriptors

        ctx.extra.setdefault("ego_motion", {}).update(
            {
                "backend": self.backend_name,
                "frames": len(records),
                "frames_with_pose": frames_with_pose,
                "scale": "up_to_scale",
                "intrinsics": intrinsics,
            }
        )

    def _estimate_relative(
        self,
        cv2: Any,
        np: Any,
        matcher: Any,
        k_matrix: Any,
        prev_kp: Any,
        prev_des: Any,
        kp: Any,
        des: Any,
    ):
        if prev_des is None or des is None or len(prev_kp) < self._min_matches or len(kp) < self._min_matches:
            return None
        matches = matcher.match(prev_des, des)
        if len(matches) < self._min_matches:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        prev_pts = np.float64([prev_kp[m.queryIdx].pt for m in matches])
        cur_pts = np.float64([kp[m.trainIdx].pt for m in matches])

        essential, mask = cv2.findEssentialMat(
            prev_pts,
            cur_pts,
            k_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self._ransac_threshold,
        )
        if essential is None or essential.shape != (3, 3):
            return None
        inliers, r_rel, t_rel, pose_mask = cv2.recoverPose(
            essential, prev_pts, cur_pts, k_matrix
        )
        if inliers < self._min_matches:
            return None
        inlier_ratio = float(inliers) / float(len(matches))
        return r_rel, t_rel.reshape(3), inlier_ratio

    def _resolve_intrinsics(self, ctx: StageContext) -> tuple[dict[str, float], str | None]:
        calibration = ctx.calibration
        if calibration and calibration.intrinsics:
            return calibration.intrinsics, None
        width = ctx.video_metadata.width
        height = ctx.video_metadata.height
        if not width or not height:
            raise RuntimeError("opencv_vo needs intrinsics or a known video resolution.")
        intrinsics = pinhole_intrinsics(width, height, self._fov_deg)
        return intrinsics, (
            f"No calibration available; used a {self._fov_deg} deg-FOV pinhole prior "
            "for intrinsics."
        )

    def close(self) -> None:
        return None


def _external_backend(name: str, guidance: str):
    def _factory(section: BackendConfig, config: ProcessingConfig):
        raise RuntimeError(
            f"The {name!r} ego-motion backend requires external setup. {guidance} "
            "For a CPU-only, up-to-scale trajectory use 'opencv_vo'."
        )

    return _factory


@EGO_MOTION_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoEgoMotion:
    return NoEgoMotion(section, config)


@EGO_MOTION_BACKENDS.register("opencv_vo")
def _build_opencv_vo(
    section: BackendConfig, config: ProcessingConfig
) -> OpenCVVisualOdometry:
    return OpenCVVisualOdometry(section, config)


EGO_MOTION_BACKENDS.register(
    "dpvo",
    _external_backend(
        "dpvo",
        "Install DPVO (https://github.com/princeton-vl/DPVO) and a CUDA GPU.",
    ),
)
EGO_MOTION_BACKENDS.register(
    "droid_slam",
    _external_backend(
        "droid_slam",
        "Install DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM); needs a CUDA GPU with ample VRAM.",
    ),
)
EGO_MOTION_BACKENDS.register(
    "orbslam3",
    _external_backend(
        "orbslam3",
        "Build ORB-SLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3) with Python bindings.",
    ),
)
