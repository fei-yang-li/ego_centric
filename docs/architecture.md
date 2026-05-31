# Pipeline architecture

The dataset builder runs as ordered **stages** so estimators that operate at
different time scales compose cleanly without changing the per-frame record
contract. Every stage is a **registered backend** selected from config, so new
capabilities are added by registering a factory — not by editing the pipeline.

```
video.mp4
  │
  ▼
[1] PreProcessor        per video        → CameraCalibration
  │                                        (camera intrinsics / metric anchor)
  ▼
[2] FrameEstimator      per sampled frame → body/hand pose, (future: depth, detect)
  │                                        writes frames/000000.jpg, builds records
  ▼
[3] RecordProcessor[]   over all records  → depth, objects+tracking, ego-motion/SLAM
  │                     (in place)          (sequence-level association allowed here)
  ▼
[4] ClipProcessor       over all records  → SegmentRecord[]  (segments.jsonl)
                        (returns clips)      temporal action segmentation + captioning
```

## Stage types

| Stage | Protocol | Scope | Built-in backends | Future backends |
| --- | --- | --- | --- | --- |
| Calibration | `PreProcessor` | per video | `none` (forward `camera.intrinsics`) | OpenCV checkerboard, COLMAP self-cal |
| Pose | `FrameEstimator` | per frame | `none`, `mediapipe` | other body/hand models |
| Depth | `RecordProcessor` | per frame (batchable) | `none` | Depth Anything V2, Metric3D, UniDepth |
| Objects | `RecordProcessor` | frame detect + cross-frame track | `none` | YOLO / Grounding DINO + ByteTrack, SAM2 |
| Ego-motion | `RecordProcessor` | sequence | `none` | DPVO, DROID-SLAM, ORB-SLAM3 |
| Actions | `ClipProcessor` | clip | `none`, `segment_caption` | learned segmenters, Gemini / Qwen2.5-VL captioners |

All protocols live in [`stages.py`](../src/ego_vla/stages.py); each owns a
`Registry` from [`registry.py`](../src/ego_vla/registry.py).

## The action stage (segmentation + captioning)

The action stage is the only one that produces **clip-level** records. It is
split into two pluggable sub-steps so slicing and labeling evolve independently:

- a **`Segmenter`** proposes clip boundaries (no labels);
- a **`Captioner`** describes/labels each clip in place.

The built-in `segment_caption` clip processor wires a chosen segmenter +
captioner together via `actions.extra`:

```json
"actions": {
  "backend": "segment_caption",
  "extra": {
    "segmenter": "fixed_window",
    "captioner": "template",
    "window_sec": 2.0,
    "stride_sec": 2.0
  }
}
```

Built-in, dependency-free pieces (meant as a baseline / wiring demo):

- segmenters: `none`, `fixed_window`
- captioners: `none`, `template` (non-semantic placeholder caption)

Real captioners plug in here:

- **`gemini`** — call the Gemini API on each clip's key frames; best zero-shot
  quality, billed per clip, data leaves your environment.
- **`qwen_vl`** — self-hosted Qwen2.5-VL (optionally LoRA-SFT distilled from
  Gemini "silver" labels); cheaper at scale, private, controllable taxonomy.

Both should populate `label` / `instruction` / `caption` / `confidence` and
record the model name + version in `source`.

## Output

- `records.jsonl` — one `VLAFrameRecord` per sampled frame (unchanged contract).
- `segments.jsonl` — one `SegmentRecord` per clip (only written when a clip
  processor produces segments).
- `metadata.json` — now also includes a `stages` map, the resolved
  `calibration`, and `output.segment_count` / `output.segments_path`.

When all optional stages are `none` (the default), the output is identical to
the pose-only pipeline.

## Adding a backend

1. Pick the stage and its registry in `stages.py` (e.g. `DEPTH_BACKENDS`).
2. Create a module under `src/ego_vla/backends/` that registers a factory:

```python
from ego_vla.stages import DEPTH_BACKENDS

@DEPTH_BACKENDS.register("depth_anything_v2")
def _build(section, config):
    return DepthAnythingV2Processor(section, config)  # lazy-import heavy deps
```

3. Import the module from `backends/__init__.py` so it registers on load.
4. Select it from config (`"depth": {"backend": "depth_anything_v2", "extra": {...}}`).
5. Run `ego-vla list-backends` to confirm it is registered.

Guidelines:

- **Lazy-import** heavy/optional third-party libraries inside the factory or the
  backend's methods, so importing `ego_vla.backends` stays cheap and the package
  installs/imports without GPU or model dependencies.
- A `RecordProcessor` mutates `records` in place; a `ClipProcessor` returns new
  `SegmentRecord` objects.
- Record provenance and confidence; prefer adding `notes`/`status` over silently
  dropping data, matching the project's "auditable, not pretend-metric" stance.
