"""Object detection + tracking (record-level) backends.

A real backend detects objects per frame (closed-set or open-vocabulary) and
associates ``track_id`` across frames, optionally writing segmentation masks.
The built-in ``none`` backend leaves the per-frame ``objects`` list empty.
"""

from __future__ import annotations

from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import VLAFrameRecord
from ego_vla.stages import OBJECT_BACKENDS, StageContext


class NoObjects:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        # Objects default to an empty list on every record; nothing to add.
        return None

    def close(self) -> None:
        return None


@OBJECT_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoObjects:
    return NoObjects(section, config)
