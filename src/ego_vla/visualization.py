from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from ego_vla.export import iter_jsonl


BODY_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

HAND_CONNECTIONS = [
    ("wrist", "thumb_cmc"),
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("wrist", "index_finger_mcp"),
    ("index_finger_mcp", "index_finger_pip"),
    ("index_finger_pip", "index_finger_dip"),
    ("index_finger_dip", "index_finger_tip"),
    ("wrist", "middle_finger_mcp"),
    ("middle_finger_mcp", "middle_finger_pip"),
    ("middle_finger_pip", "middle_finger_dip"),
    ("middle_finger_dip", "middle_finger_tip"),
    ("wrist", "ring_finger_mcp"),
    ("ring_finger_mcp", "ring_finger_pip"),
    ("ring_finger_pip", "ring_finger_dip"),
    ("ring_finger_dip", "ring_finger_tip"),
    ("wrist", "pinky_mcp"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
]


def write_dataset_report(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    max_records: int = 25,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    records = _load_records(dataset_dir)
    output_path = Path(output_path) if output_path else dataset_dir / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_records(records)
    metadata = _load_metadata(dataset_dir)
    html_text = _build_report_html(dataset_dir, output_path, records, summary, metadata, max_records)
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "report_path": str(output_path),
        "record_count": summary["record_count"],
        "body_pose_2d_frames": summary["body_pose_2d_frames"],
        "body_pose_3d_frames": summary["body_pose_3d_frames"],
        "hand_frames": summary["hand_frames"],
    }


def render_pose_overlay_video(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for pose rendering. Install with `pip install -e .`.") from exc

    dataset_dir = Path(dataset_dir)
    records = _load_records(dataset_dir)
    if max_frames is not None:
        records = records[:max_frames]
    if not records:
        raise ValueError(f"No records found in {dataset_dir / 'records.jsonl'}")

    output_path = Path(output_path) if output_path else dataset_dir / "pose_overlay.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_fps = fps or _infer_fps(records)

    first_frame = _read_record_frame(cv2, dataset_dir, records[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frames_rendered = 0
    pose_frames = 0
    hand_frames = 0
    try:
        for record in records:
            frame = _read_record_frame(cv2, dataset_dir, record)
            frame = _resize_if_needed(cv2, frame, width, height)
            drew_pose = _draw_record_pose(cv2, frame, record, min_confidence=min_confidence)
            pose_frames += int(drew_pose["body"])
            hand_frames += int(drew_pose["hands"])
            _draw_frame_label(cv2, frame, record)
            writer.write(frame)
            frames_rendered += 1
    finally:
        writer.release()

    return {
        "video_path": str(output_path),
        "frames_rendered": frames_rendered,
        "fps": render_fps,
        "body_pose_frames": pose_frames,
        "hand_pose_frames": hand_frames,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pose_backend_counts = Counter(
        record.get("provenance", {}).get("pose_backend", "unknown") for record in records
    )
    warning_counts = Counter(
        warning for record in records for warning in record.get("warnings", [])
    )
    hand_sides = Counter(
        side for record in records for side in record.get("hands", {}).keys()
    )
    return {
        "record_count": len(records),
        "body_pose_2d_frames": sum(1 for record in records if record.get("body_pose_2d")),
        "body_pose_3d_frames": sum(1 for record in records if record.get("body_pose_3d")),
        "hand_frames": sum(1 for record in records if record.get("hands")),
        "object_frames": sum(1 for record in records if record.get("objects")),
        "depth_frames": sum(
            1 for record in records if record.get("depth", {}).get("status") == "ok"
        ),
        "pose_backend_counts": dict(pose_backend_counts),
        "warning_counts": dict(warning_counts),
        "hand_side_counts": dict(hand_sides),
    }


def _load_records(dataset_dir: Path) -> list[dict[str, Any]]:
    records_path = dataset_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")
    return list(iter_jsonl(records_path))


def _load_metadata(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _build_report_html(
    dataset_dir: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    max_records: int,
) -> str:
    record_rows = "\n".join(
        _record_row(dataset_dir, output_path, record) for record in records[:max_records]
    )
    summary_items = "\n".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(json.dumps(value, ensure_ascii=False))}</li>"
        for key, value in summary.items()
    )
    metadata_block = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Egocentric VLA Dataset Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 0.5rem; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    img {{ max-width: 220px; max-height: 140px; }}
    pre {{ white-space: pre-wrap; background: #f7f9fb; padding: 0.75rem; overflow: auto; }}
    .warning {{ color: #9f580a; }}
    .small {{ font-size: 0.9rem; color: #52606d; }}
  </style>
</head>
<body>
  <h1>Egocentric VLA Dataset Report</h1>
  <p class="small">Dataset directory: {html.escape(str(dataset_dir))}</p>
  <h2>Summary</h2>
  <ul>{summary_items}</ul>
  <h2>Metadata</h2>
  <details open><summary>metadata.json</summary><pre>{metadata_block}</pre></details>
  <h2>Sample records</h2>
  <table>
    <thead>
      <tr>
        <th>Frame</th>
        <th>Preview</th>
        <th>Pose / hands</th>
        <th>Warnings</th>
        <th>Structured JSON</th>
      </tr>
    </thead>
    <tbody>
      {record_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _record_row(dataset_dir: Path, output_path: Path, record: dict[str, Any]) -> str:
    rgb_path = record.get("observation", {}).get("rgb", {}).get("path")
    preview = ""
    if rgb_path:
        frame_path = dataset_dir / rgb_path
        if frame_path.exists():
            rel_path = Path(_safe_relpath(frame_path, output_path.parent))
            preview = f'<img src="{html.escape(rel_path.as_posix())}" alt="{html.escape(record.get("frame_id", "frame"))}">'

    body_2d_count = len(record.get("body_pose_2d", {}).get("landmarks", []))
    body_3d_count = len(record.get("body_pose_3d", {}).get("landmarks", []))
    hand_counts = {
        side: len(hand.get("landmarks", []))
        for side, hand in record.get("hands", {}).items()
    }
    warnings = record.get("warnings", [])
    warnings_html = "<br>".join(html.escape(warning) for warning in warnings)
    if warnings_html:
        warnings_html = f'<span class="warning">{warnings_html}</span>'
    payload = html.escape(json.dumps(record, indent=2, ensure_ascii=False))
    return f"""<tr>
  <td>{html.escape(record.get("frame_id", ""))}<br><span class="small">{record.get("timestamp_sec", "")} sec</span></td>
  <td>{preview}</td>
  <td>body_2d={body_2d_count}<br>body_3d={body_3d_count}<br>hands={html.escape(json.dumps(hand_counts, ensure_ascii=False))}</td>
  <td>{warnings_html}</td>
  <td><details><summary>JSON</summary><pre>{payload}</pre></details></td>
</tr>"""


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path)


def _infer_fps(records: list[dict[str, Any]]) -> float:
    timestamps = [float(record.get("timestamp_sec", 0.0)) for record in records]
    deltas = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later - earlier > 1e-6
    ]
    if not deltas:
        return 10.0
    return max(1.0, min(60.0, 1.0 / median(deltas)))


def _read_record_frame(cv2: Any, dataset_dir: Path, record: dict[str, Any]) -> Any:
    rgb_path = record.get("observation", {}).get("rgb", {}).get("path")
    if not rgb_path:
        raise ValueError(f"Record has no observation.rgb.path: {record.get('frame_id')}")
    frame_path = dataset_dir / rgb_path
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Could not read frame image: {frame_path}")
    return frame


def _resize_if_needed(cv2: Any, frame: Any, width: int, height: int) -> Any:
    frame_height, frame_width = frame.shape[:2]
    if frame_width == width and frame_height == height:
        return frame
    return cv2.resize(frame, (width, height))


def _draw_record_pose(
    cv2: Any,
    frame: Any,
    record: dict[str, Any],
    min_confidence: float,
) -> dict[str, bool]:
    drew_body = False
    drew_hands = False
    body_pose = record.get("body_pose_2d")
    if body_pose:
        drew_body = _draw_landmark_set(
            cv2,
            frame,
            body_pose.get("landmarks", []),
            BODY_CONNECTIONS,
            point_color=(0, 255, 255),
            line_color=(0, 180, 255),
            min_confidence=min_confidence,
        )
    for side, hand in record.get("hands", {}).items():
        color = (0, 255, 0) if side == "left" else (255, 0, 255)
        drew = _draw_landmark_set(
            cv2,
            frame,
            hand.get("landmarks", []),
            HAND_CONNECTIONS,
            point_color=color,
            line_color=color,
            min_confidence=min_confidence,
        )
        drew_hands = drew_hands or drew
    return {"body": drew_body, "hands": drew_hands}


def _draw_landmark_set(
    cv2: Any,
    frame: Any,
    landmarks: list[dict[str, Any]],
    connections: list[tuple[str, str]],
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    min_confidence: float,
) -> bool:
    height, width = frame.shape[:2]
    points = {
        landmark.get("name"): _landmark_to_pixel(landmark, width, height, min_confidence)
        for landmark in landmarks
    }
    points = {name: point for name, point in points.items() if name and point}
    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], line_color, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(frame, point, 3, point_color, -1, cv2.LINE_AA)
    return bool(points)


def _landmark_to_pixel(
    landmark: dict[str, Any],
    width: int,
    height: int,
    min_confidence: float,
) -> tuple[int, int] | None:
    visibility = landmark.get("visibility")
    confidence = landmark.get("confidence")
    if visibility is not None and float(visibility) < min_confidence:
        return None
    if confidence is not None and float(confidence) < min_confidence:
        return None
    x = float(landmark.get("x", 0.0))
    y = float(landmark.get("y", 0.0))
    if not (-0.25 <= x <= 1.25 and -0.25 <= y <= 1.25):
        return None
    px = int(round(max(0.0, min(1.0, x)) * (width - 1)))
    py = int(round(max(0.0, min(1.0, y)) * (height - 1)))
    return px, py


def _draw_frame_label(cv2: Any, frame: Any, record: dict[str, Any]) -> None:
    label = f"{record.get('frame_id', '')}  t={record.get('timestamp_sec', '')}s"
    cv2.putText(
        frame,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
