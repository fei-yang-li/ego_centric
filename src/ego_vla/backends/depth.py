"""Depth (record-level) backends.

Depth is conceptually per-frame, but it runs as a ``RecordProcessor`` so a real
backend can batch frames through a GPU model and write depth maps next to the
frames. The built-in ``none`` backend only annotates each record's ``depth``
signal as not computed.
"""

from __future__ import annotations

from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import VLAFrameRecord
from ego_vla.stages import DEPTH_BACKENDS, StageContext

_NOT_COMPUTED_NOTE = "Depth is not computed by the default pipeline."


class NoDepth:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        for record in records:
            record.depth.backend = "none"
            record.depth.status = "not_computed"
            if _NOT_COMPUTED_NOTE not in record.depth.notes:
                record.depth.notes.append(_NOT_COMPUTED_NOTE)

    def close(self) -> None:
        return None


@DEPTH_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoDepth:
    return NoDepth(section, config)
