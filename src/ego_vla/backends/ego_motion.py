"""Ego-motion / camera-trajectory (record-level) backends.

A real SLAM/VO backend consumes the full frame sequence (and optionally depth +
calibration for metric scale) and writes a per-frame camera pose plus a
trajectory in metadata. The built-in ``none`` backend annotates each record's
``ego_motion`` signal as not computed.
"""

from __future__ import annotations

from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import VLAFrameRecord
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


@EGO_MOTION_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoEgoMotion:
    return NoEgoMotion(section, config)
