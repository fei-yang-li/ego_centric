"""Camera calibration (pre-processing) backends.

Calibration is the metric anchor for the depth and ego-motion stages. The
built-in ``none`` backend does not estimate intrinsics; it simply forwards any
intrinsics provided in ``camera.intrinsics`` and records that metric scale is
unavailable otherwise.

Future backends (e.g. ``opencv_checkerboard``, ``colmap_selfcal``) register
here and return a populated :class:`CameraCalibration`.
"""

from __future__ import annotations

from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import CameraCalibration
from ego_vla.stages import CALIBRATION_BACKENDS, StageContext


class StaticCalibration:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> CameraCalibration:
        camera = self._config.camera
        intrinsics = camera.intrinsics
        notes: list[str] = []
        if intrinsics:
            status = "provided"
        else:
            status = "not_computed"
            notes.append(
                "No calibration backend; camera intrinsics are unknown. Provide "
                "camera.intrinsics or add a calibration backend for metric depth/SLAM."
            )
        return CameraCalibration(
            backend=self.backend_name,
            status=status,
            coordinate_frame=camera.coordinate_frame,
            intrinsics=intrinsics,
            image_width=ctx.video_metadata.width or None,
            image_height=ctx.video_metadata.height or None,
            metric_scale=True if intrinsics else None,
            notes=notes,
        )


@CALIBRATION_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> StaticCalibration:
    return StaticCalibration(section, config)
