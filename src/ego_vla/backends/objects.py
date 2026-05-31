"""Object detection + tracking (record-level) backends.

Backends:

- ``none``           -- leave the per-frame ``objects`` list empty.
- ``yolo``           -- Ultralytics YOLO with built-in ByteTrack tracking
  (closed-set; ``track_id`` comes from the tracker).
- ``grounding_dino`` -- open-vocabulary detection (Grounding DINO via
  ``transformers``) associated across frames with a lightweight IoU tracker.

Detection is expressed as a ``detector(frame_bgr, frame_index) -> list[dict]``
callable with keys ``label``, ``bbox_xyxy``, ``confidence`` (and optional
``track_id``). ``DetectionTrackProcessor`` turns detections into per-frame
``ObjectEstimate`` records, so the orchestration is testable without any model.

Segmentation masks (e.g. SAM2 propagation) are a natural follow-on backend that
would populate ``ObjectEstimate.mask_path``.
"""

from __future__ import annotations

from typing import Any, Callable

from ego_vla.backends._common import IoUTracker, missing_dependency, read_frame_bgr
from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import ObjectEstimate, VLAFrameRecord
from ego_vla.stages import OBJECT_BACKENDS, StageContext

Detector = Callable[[Any, int], list[dict[str, Any]]]


class NoObjects:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        return None

    def close(self) -> None:
        return None


class DetectionTrackProcessor:
    """Generic detect-then-track processor.

    ``tracker`` may be ``None`` (the detector supplies ``track_id``) or an object
    exposing ``update(detections) -> list[str]`` (e.g. :class:`IoUTracker`).
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Any | None,
        backend_name: str,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self.backend_name = backend_name

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        frames_with_objects = 0
        track_ids: set[str] = set()
        for record in records:
            frame = read_frame_bgr(ctx.resolve_frame_path(record))
            detections = self._detector(frame, record.frame_index)
            if self._tracker is not None:
                assigned = self._tracker.update(detections)
            else:
                assigned = [det.get("track_id") for det in detections]

            objects: list[ObjectEstimate] = []
            for det, track_id in zip(detections, assigned):
                track_str = None if track_id is None else str(track_id)
                if track_str is not None:
                    track_ids.add(track_str)
                objects.append(
                    ObjectEstimate(
                        label=str(det["label"]),
                        bbox_xyxy=[float(v) for v in det["bbox_xyxy"]],
                        confidence=(
                            float(det["confidence"]) if det.get("confidence") is not None else None
                        ),
                        track_id=track_str,
                        mask_path=det.get("mask_path"),
                    )
                )
            record.objects = objects
            if objects:
                frames_with_objects += 1

        ctx.extra.setdefault("objects", {}).update(
            {
                "backend": self.backend_name,
                "frames_with_objects": frames_with_objects,
                "unique_tracks": len(track_ids),
            }
        )

    def close(self) -> None:
        return None


def _build_yolo_detector(extra: dict[str, Any]) -> Detector:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise missing_dependency("yolo", "ultralytics", extra="objects") from exc

    model = YOLO(str(extra.get("model", "yolov8n.pt")))
    conf = float(extra.get("conf", 0.25))
    classes = extra.get("classes")
    tracker_cfg = str(extra.get("tracker", "bytetrack.yaml"))
    device = extra.get("device")

    def detect(frame_bgr: Any, frame_index: int) -> list[dict[str, Any]]:
        results = model.track(
            frame_bgr,
            persist=True,
            conf=conf,
            classes=classes,
            tracker=tracker_cfg,
            device=device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = result.names
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
                track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
                detections.append(
                    {
                        "label": label,
                        "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                        "confidence": float(box.conf[0]),
                        "track_id": track_id,
                    }
                )
        return detections

    return detect


def _build_grounding_dino_detector(extra: dict[str, Any]) -> Detector:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise missing_dependency(
            "grounding_dino", "torch transformers pillow", extra="objects-openvocab"
        ) from exc

    model_name = str(extra.get("model", "IDEA-Research/grounding-dino-tiny"))
    prompt = str(extra.get("prompt", "person. hand. object."))
    box_threshold = float(extra.get("box_threshold", 0.3))
    text_threshold = float(extra.get("text_threshold", 0.25))
    device = str(extra.get("device", "cpu"))

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)

    def detect(frame_bgr: Any, frame_index: int) -> list[dict[str, Any]]:  # pragma: no cover - requires model
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        detections: list[dict[str, Any]] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append(
                {
                    "label": str(label),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "confidence": float(score),
                }
            )
        return detections

    return detect


@OBJECT_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoObjects:
    return NoObjects(section, config)


@OBJECT_BACKENDS.register("yolo")
def _build_yolo(section: BackendConfig, config: ProcessingConfig) -> DetectionTrackProcessor:
    extra = dict(section.extra or {})
    # YOLO's tracker assigns ids itself, so no separate tracker is used.
    return DetectionTrackProcessor(_build_yolo_detector(extra), tracker=None, backend_name="yolo")


@OBJECT_BACKENDS.register("grounding_dino")
def _build_grounding_dino(
    section: BackendConfig, config: ProcessingConfig
) -> DetectionTrackProcessor:
    extra = dict(section.extra or {})
    tracker = IoUTracker(
        iou_threshold=float(extra.get("iou_threshold", 0.3)),
        max_age=int(extra.get("max_age", 5)),
    )
    return DetectionTrackProcessor(
        _build_grounding_dino_detector(extra), tracker=tracker, backend_name="grounding_dino"
    )
