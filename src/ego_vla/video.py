from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ego_vla.schemas import VideoMetadata


@dataclass(slots=True)
class FrameSample:
    ordinal: int
    frame_index: int
    timestamp_sec: float
    image_bgr: object


def compute_sampled_frame_indices(
    total_frames: int,
    fps: float,
    sample_rate_hz: float,
    max_frames: int | None = None,
) -> list[int]:
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if sample_rate_hz <= 0:
        indices = list(range(total_frames))
    else:
        step = max(1, round(fps / sample_rate_hz))
        indices = list(range(0, total_frames, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


class VideoSampler:
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required. Install with `pip install -e .`.") from exc
        self._cv2 = cv2

    def probe(self) -> VideoMetadata:
        cap = self._open()
        try:
            width = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(self._cv2.CAP_PROP_FPS)) or 0.0
            frame_count_raw = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
            frame_count = frame_count_raw if frame_count_raw > 0 else None
            duration_sec = None
            if fps > 0 and frame_count is not None:
                duration_sec = frame_count / fps
            return VideoMetadata(
                source_path=str(self.video_path),
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_sec=duration_sec,
            )
        finally:
            cap.release()

    def samples(self, sample_rate_hz: float, max_frames: int | None = None) -> Iterator[FrameSample]:
        cap = self._open()
        try:
            fps = float(cap.get(self._cv2.CAP_PROP_FPS)) or 30.0
            frame_count = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
            target_indices = None
            if frame_count > 0:
                target_indices = set(
                    compute_sampled_frame_indices(frame_count, fps, sample_rate_hz, max_frames)
                )
            next_timestamp = 0.0
            period = 1.0 / sample_rate_hz if sample_rate_hz > 0 else 0.0
            yielded = 0
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_sec = frame_index / fps
                should_yield = False
                if target_indices is not None:
                    should_yield = frame_index in target_indices
                elif sample_rate_hz <= 0 or timestamp_sec + 1e-9 >= next_timestamp:
                    should_yield = True
                    next_timestamp += period
                if should_yield:
                    yield FrameSample(
                        ordinal=yielded,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image_bgr=frame,
                    )
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        break
                frame_index += 1
        finally:
            cap.release()

    def write_frame(
        self,
        image_bgr: object,
        path: str | Path,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params: list[int] = []
        if image_format.lower() in {"jpg", "jpeg"}:
            params = [int(self._cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        ok = self._cv2.imwrite(str(path), image_bgr, params)
        if not ok:
            raise RuntimeError(f"Failed to write frame: {path}")

    def _open(self) -> object:
        cap = self._cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        return cap
