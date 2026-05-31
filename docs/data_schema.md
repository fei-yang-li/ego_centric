# VLA frame record schema

The pipeline writes one JSON object per sampled frame to `records.jsonl`. When a
clip (action) stage is enabled, it additionally writes one clip per line to
`segments.jsonl` (see [Segment records](#segment-records)).

## Top-level fields

| Field | Type | Purpose |
| --- | --- | --- |
| `dataset_name` | string | Dataset split or project name. |
| `episode_id` | string | Sequence identifier, defaulting to the video stem. |
| `frame_id` | string | Stable id in `{episode_id}/{ordinal}` form. |
| `frame_index` | integer | Original video frame index. |
| `timestamp_sec` | float | Timestamp in the source video. |
| `observation.rgb` | object | Relative RGB frame path, size, and encoding. |
| `camera` | object | Intrinsics and coordinate-frame metadata. |
| `body_pose_3d` | object/null | Body landmarks in estimator-specific 3D coordinates. |
| `body_pose_2d` | object/null | Normalized image-space body landmarks. |
| `hands` | object | Left/right hand landmark estimates. |
| `objects` | array | Optional object detections and tracks. |
| `depth` | object | Optional dense or sparse depth signal. |
| `ego_motion` | object | Optional camera pose or visual odometry. |
| `action` | object | Optional action label, controls, or language instruction. |
| `provenance` | object | Backends and pipeline version used to create the record. |
| `warnings` | array | Per-frame missing-data notes. |

## Coordinate frames

The schema stores `coordinate_frame` on every spatial field. The default
MediaPipe backend emits:

- `image_normalized`: `x` and `y` are normalized to image width/height.
- `image_normalized_relative_z`: hand `z` is relative to wrist and image scale.
- `mediapipe_world_meters_approx`: body landmarks from MediaPipe world output.

These estimates are useful for debugging and bootstrapping. For high-quality VLA
training data, add calibration and a metric reconstruction backend:

- camera intrinsics and distortion coefficients;
- visual odometry or SLAM for `ego_motion`;
- monocular or multi-view depth for `depth`;
- object detection, segmentation, and tracking for `objects`;
- action labels, controls, or natural-language instructions for `action`.

## Example

```json
{
  "dataset_name": "egocentric_vla_debug",
  "episode_id": "debug_video",
  "frame_id": "debug_video/000000",
  "frame_index": 0,
  "timestamp_sec": 0.0,
  "observation": {
    "rgb": {
      "path": "frames/000000.jpg",
      "width": 1920,
      "height": 1080,
      "encoding": "jpg"
    },
    "source_video": "debug_video.mp4"
  },
  "camera": {
    "coordinate_frame": "camera"
  },
  "hands": {
    "left": {
      "side": "left",
      "source": "mediapipe",
      "coordinate_frame": "image_normalized_relative_z",
      "landmarks": []
    }
  },
  "provenance": {
    "pipeline_version": "0.1.0",
    "pose_backend": "mediapipe"
  }
}
```

## Segment records

When a clip (action) stage produces segments, each is written as one JSON
object per line in `segments.jsonl`. A segment is a contiguous slice of frames
with an optional action label, language instruction, and free-text caption.

| Field | Type | Purpose |
| --- | --- | --- |
| `segment_id` | string | Stable id in `{episode_id}/seg_{ordinal}` form. |
| `episode_id` | string | Episode the segment belongs to. |
| `start_sec` / `end_sec` | float | Clip time span in the source video. |
| `start_frame` / `end_frame` | integer | Original video frame indices. |
| `frame_ids` | array | `frame_id`s of the member frame records. |
| `key_frame_ids` | array | Representative frames (e.g. first/middle/last). |
| `label` | string/null | Discrete action label (closed-set), if any. |
| `instruction` | string/null | Natural-language instruction for the clip. |
| `caption` | string/null | Free-text description of the clip. |
| `objects` | array | Object labels involved in the clip. |
| `confidence` | float/null | Backend confidence. |
| `status` | string | `ok`, `heuristic`, etc. |
| `source` | object | Segmenter/captioner names, params, model + version. |
| `notes` | array | Caveats (e.g. placeholder caption). |

```json
{
  "segment_id": "debug_video/seg_0000",
  "episode_id": "debug_video",
  "start_sec": 0.0,
  "end_sec": 1.8,
  "start_frame": 0,
  "end_frame": 54,
  "frame_ids": ["debug_video/000000", "debug_video/000001"],
  "key_frame_ids": ["debug_video/000000", "debug_video/000001"],
  "instruction": "pick up the mug from the table",
  "caption": "the camera wearer reaches for and lifts a mug",
  "objects": ["mug"],
  "status": "ok",
  "source": {"segmenter": "fixed_window", "captioner": "qwen_vl", "model": "Qwen2.5-VL-7B"}
}
```
