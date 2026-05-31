"""Temporal segmentation backends (the "slicing" half of the action stage).

A segmenter turns the frame-record sequence into clip boundaries without
assigning semantic labels. The built-in backends are dependency-free:

- ``none``         -- produce no segments.
- ``fixed_window`` -- uniform windows of ``window_sec`` with optional overlap.

Future segmenters (motion/optical-flow boundaries, hand-contact boundaries,
learned temporal action segmentation, or VLM-proposed boundaries) register here.
"""

from __future__ import annotations

from ego_vla.config import ProcessingConfig
from ego_vla.schemas import SegmentRecord, VLAFrameRecord
from ego_vla.stages import SEGMENTERS, StageContext

_EPS = 1e-9


def _segment_from_members(
    episode_id: str,
    ordinal: int,
    members: list[VLAFrameRecord],
    segmenter_name: str,
    params: dict[str, object],
) -> SegmentRecord:
    members = sorted(members, key=lambda r: r.timestamp_sec)
    frame_ids = [m.frame_id for m in members]
    key_frame_ids = [frame_ids[0]]
    if len(frame_ids) > 2:
        key_frame_ids.append(frame_ids[len(frame_ids) // 2])
    if len(frame_ids) > 1:
        key_frame_ids.append(frame_ids[-1])
    return SegmentRecord(
        segment_id=f"{episode_id}/seg_{ordinal:04d}",
        episode_id=episode_id,
        start_sec=round(members[0].timestamp_sec, 6),
        end_sec=round(members[-1].timestamp_sec, 6),
        start_frame=members[0].frame_index,
        end_frame=members[-1].frame_index,
        frame_ids=frame_ids,
        key_frame_ids=key_frame_ids,
        source={"segmenter": segmenter_name, "params": params},
    )


class NoneSegmenter:
    name = "none"

    def __init__(self, extra: dict, config: ProcessingConfig) -> None:
        self._extra = extra

    def segment(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        return []


class FixedWindowSegmenter:
    name = "fixed_window"

    def __init__(self, extra: dict, config: ProcessingConfig) -> None:
        self.window_sec = float(extra.get("window_sec", 2.0))
        self.stride_sec = float(extra.get("stride_sec", self.window_sec))
        self.min_frames = int(extra.get("min_frames", 1))
        if self.window_sec <= 0:
            raise ValueError("fixed_window segmenter requires window_sec > 0")
        if self.stride_sec <= 0:
            raise ValueError("fixed_window segmenter requires stride_sec > 0")

    def segment(
        self, records: list[VLAFrameRecord], ctx: StageContext
    ) -> list[SegmentRecord]:
        if not records:
            return []
        recs = sorted(records, key=lambda r: r.timestamp_sec)
        episode_id = recs[0].episode_id
        t_start = recs[0].timestamp_sec
        t_last = recs[-1].timestamp_sec

        params = {"window_sec": self.window_sec, "stride_sec": self.stride_sec}
        segments: list[SegmentRecord] = []
        ordinal = 0
        window_start = t_start
        while window_start <= t_last + _EPS:
            window_end = window_start + self.window_sec
            members = [
                r
                for r in recs
                if window_start - _EPS <= r.timestamp_sec < window_end - _EPS
            ]
            if len(members) >= self.min_frames:
                segments.append(
                    _segment_from_members(
                        episode_id, ordinal, members, self.name, params
                    )
                )
                ordinal += 1
            window_start += self.stride_sec
        return segments


@SEGMENTERS.register("none")
def _build_none(extra: dict, config: ProcessingConfig) -> NoneSegmenter:
    return NoneSegmenter(extra, config)


@SEGMENTERS.register("fixed_window")
def _build_fixed_window(extra: dict, config: ProcessingConfig) -> FixedWindowSegmenter:
    return FixedWindowSegmenter(extra, config)
