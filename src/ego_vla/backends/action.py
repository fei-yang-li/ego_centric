"""Action (clip-level) backends: temporal segmentation + captioning/labeling.

The action stage is the only one that produces clip-level records. It is split
into two pluggable sub-steps so segmentation and captioning evolve
independently:

- a ``Segmenter`` proposes clip boundaries (see ``backends/segmentation.py``);
- a ``Captioner`` labels/describes each clip (see ``backends/captioning.py``).

The ``segment_caption`` clip processor wires a chosen segmenter and captioner
together. Configure it via ``actions.extra``::

    "actions": {
      "backend": "segment_caption",
      "extra": {"segmenter": "fixed_window", "captioner": "template",
                "window_sec": 2.0}
    }
"""

from __future__ import annotations

from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import SegmentRecord, VLAFrameRecord
from ego_vla.stages import ACTION_BACKENDS, StageContext


class NoAction:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        return []

    def close(self) -> None:
        return None


class SegmentCaptionProcessor:
    backend_name = "segment_caption"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        from ego_vla.stages import CAPTIONERS, SEGMENTERS

        extra = dict(section.extra or {})
        self._segmenter_name = str(extra.get("segmenter", "fixed_window"))
        self._captioner_name = str(extra.get("captioner", "none"))
        self._segmenter = SEGMENTERS.create(self._segmenter_name, extra, config)
        self._captioner = CAPTIONERS.create(self._captioner_name, extra, config)
        self._instruction = config.dataset.language_instruction

    def run(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        segments = self._segmenter.segment(records, ctx)
        for segment in segments:
            segment.source.setdefault("segmenter", self._segmenter_name)
            if self._instruction and not segment.instruction:
                segment.instruction = self._instruction
            self._captioner.caption(segment, records, ctx)
        return segments

    def close(self) -> None:
        return None


@ACTION_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoAction:
    return NoAction(section, config)


@ACTION_BACKENDS.register("segment_caption")
def _build_segment_caption(
    section: BackendConfig, config: ProcessingConfig
) -> SegmentCaptionProcessor:
    return SegmentCaptionProcessor(section, config)
