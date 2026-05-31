"""Captioning / labeling backends (the "describe" half of the action stage).

A captioner fills a segment's caption/instruction/label in place. The built-in
backends are dependency-free:

- ``none``     -- leave the segment without a caption.
- ``template`` -- emit a deterministic, non-semantic placeholder caption derived
  from per-frame signals (hands/body/objects). This exists to exercise the
  end-to-end flow; it is NOT a real action label.

Real captioners register here, e.g. a ``gemini`` backend (Gemini API) or a
``qwen_vl`` backend (self-hosted Qwen2.5-VL, optionally SFT-distilled from
Gemini silver labels). Those should populate ``label``/``instruction``/
``caption``/``confidence`` and record model + version in ``source``.
"""

from __future__ import annotations

from ego_vla.config import ProcessingConfig
from ego_vla.schemas import SegmentRecord, VLAFrameRecord
from ego_vla.stages import CAPTIONERS, StageContext


class NoneCaptioner:
    name = "none"

    def __init__(self, extra: dict, config: ProcessingConfig) -> None:
        self._extra = extra

    def caption(
        self, segment: SegmentRecord, records: list[VLAFrameRecord], ctx: StageContext
    ) -> None:
        segment.source.setdefault("captioner", self.name)
        note = "caption not generated (captioner=none)"
        if note not in segment.notes:
            segment.notes.append(note)


class TemplateCaptioner:
    name = "template"

    def __init__(self, extra: dict, config: ProcessingConfig) -> None:
        self._extra = extra

    def caption(
        self, segment: SegmentRecord, records: list[VLAFrameRecord], ctx: StageContext
    ) -> None:
        member_ids = set(segment.frame_ids)
        members = [r for r in records if r.frame_id in member_ids]
        total = len(members)
        hand_frames = sum(1 for r in members if r.hands)
        body_frames = sum(1 for r in members if r.body_pose_2d or r.body_pose_3d)
        object_labels = sorted(
            {obj.label for r in members for obj in (r.objects or [])}
        )

        parts = [
            f"egocentric clip ({segment.start_sec:.2f}-{segment.end_sec:.2f}s, "
            f"{total} frames)"
        ]
        if hand_frames:
            parts.append(f"hands visible in {hand_frames}/{total} frames")
        if body_frames:
            parts.append(f"body pose in {body_frames}/{total} frames")
        if object_labels:
            parts.append("objects: " + ", ".join(object_labels))
        caption = "; ".join(parts)

        segment.caption = caption
        if not segment.instruction:
            segment.instruction = caption
        segment.objects = object_labels
        segment.status = "heuristic"
        segment.source.setdefault("captioner", self.name)
        note = (
            "template caption is a non-semantic placeholder; use a VLM backend "
            "(gemini/qwen_vl) for real action labels"
        )
        if note not in segment.notes:
            segment.notes.append(note)


@CAPTIONERS.register("none")
def _build_none(extra: dict, config: ProcessingConfig) -> NoneCaptioner:
    return NoneCaptioner(extra, config)


@CAPTIONERS.register("template")
def _build_template(extra: dict, config: ProcessingConfig) -> TemplateCaptioner:
    return TemplateCaptioner(extra, config)
