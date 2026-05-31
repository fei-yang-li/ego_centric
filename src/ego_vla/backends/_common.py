"""Dependency-light helpers shared by the real backends.

Only ``numpy`` (a core dependency) is imported at module load. ``cv2`` and any
heavy optional libraries are imported lazily so importing ``ego_vla.backends``
stays cheap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

BBox = Sequence[float]


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - cv2 is a core dependency
        raise RuntimeError("OpenCV is required. Install with `pip install -e .`.") from exc
    return cv2


def read_frame_bgr(path: str | Path) -> Any:
    cv2 = load_cv2()
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not read frame image: {path}")
    return frame


def read_frame_rgb(path: str | Path) -> Any:
    cv2 = load_cv2()
    return cv2.cvtColor(read_frame_bgr(path), cv2.COLOR_BGR2RGB)


def resize_max_side(image: Any, max_side: int | None) -> Any:
    if not max_side:
        return image
    cv2 = load_cv2()
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def pinhole_intrinsics(width: int, height: int, horizontal_fov_deg: float) -> dict[str, float]:
    """Approximate pinhole intrinsics from a horizontal field-of-view guess."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    fov = math.radians(horizontal_fov_deg)
    fx = (width / 2.0) / math.tan(fov / 2.0)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}


def intrinsic_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_to_quaternion(matrix: Any) -> list[float]:
    """Convert a 3x3 rotation matrix to a unit quaternion ``[x, y, z, w]``."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm
    return [float(v) for v in q]


def compose_pose(
    r_world_prev: np.ndarray,
    t_world_prev: np.ndarray,
    r_rel: np.ndarray,
    t_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a relative motion onto a world pose (right-multiply convention)."""
    r_world = r_world_prev @ r_rel
    t_world = r_world_prev @ t_rel + t_world_prev
    return r_world, t_world


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """A minimal greedy IoU tracker for detect-only backends.

    Real tracking-by-detection systems (ByteTrack, BoT-SORT, OC-SORT) add motion
    models and re-identification; this is a lightweight, dependency-free
    associator used when the detector does not emit track ids itself.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self._next_id = 0
        self._tracks: dict[str, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[str]:
        assignments: list[str | None] = [None] * len(detections)
        used: set[str] = set()

        for det_idx, det in enumerate(detections):
            best_id: str | None = None
            best_iou = self.iou_threshold
            for track_id, track in self._tracks.items():
                if track_id in used or track["label"] != det["label"]:
                    continue
                overlap = iou(track["bbox"], det["bbox_xyxy"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is not None:
                used.add(best_id)
                assignments[det_idx] = best_id
                self._tracks[best_id]["bbox"] = det["bbox_xyxy"]
                self._tracks[best_id]["age"] = 0

        for det_idx, det in enumerate(detections):
            if assignments[det_idx] is None:
                track_id = f"t{self._next_id}"
                self._next_id += 1
                self._tracks[track_id] = {"bbox": det["bbox_xyxy"], "label": det["label"], "age": 0}
                used.add(track_id)
                assignments[det_idx] = track_id

        for track_id in list(self._tracks):
            if track_id not in used:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]

        return [track_id for track_id in assignments if track_id is not None]


def encode_depth_png16(depth: Any, metric: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode a float depth map to a 16-bit array plus decoding metadata.

    - metric: stored as millimetres (``uint16``), decode with ``value * scale``.
    - relative: per-frame min/max normalised to ``[0, 65535]`` (not comparable
      across frames; store ``min``/``max`` to undo).
    """
    depth_arr = np.asarray(depth, dtype=np.float64)
    if metric:
        millimetres = np.clip(depth_arr * 1000.0, 0, 65535).astype(np.uint16)
        return millimetres, {"unit": "millimeter", "scale": 0.001, "normalized": False}

    finite = np.isfinite(depth_arr)
    if not finite.any():
        return np.zeros(depth_arr.shape, dtype=np.uint16), {
            "unit": "normalized",
            "min": 0.0,
            "max": 0.0,
            "normalized": True,
        }
    d_min = float(depth_arr[finite].min())
    d_max = float(depth_arr[finite].max())
    if d_max - d_min < 1e-12:
        encoded = np.zeros(depth_arr.shape, dtype=np.uint16)
    else:
        normalized = np.clip((depth_arr - d_min) / (d_max - d_min), 0.0, 1.0)
        encoded = (normalized * 65535.0).astype(np.uint16)
    return encoded, {"unit": "normalized", "min": d_min, "max": d_max, "normalized": True}


def missing_dependency(backend: str, packages: str, extra: str | None = None) -> RuntimeError:
    hint = f"pip install {packages}"
    if extra:
        hint = f"pip install -e \".[{extra}]\"  (or: {hint})"
    return RuntimeError(
        f"The {backend!r} backend requires extra dependencies that are not installed. {hint}."
    )
