# Debug video workflow

When a first-person video is available, use this checklist to produce and audit a
debug dataset.

## 1. Prepare config

```bash
ego-vla init-config configs/local.json
```

Edit `configs/local.json`:

- set `video.sample_rate_hz` low for fast iteration, for example `2.0`;
- set `video.max_frames` when debugging a short clip;
- set `dataset.language_instruction` if the whole video has a task prompt;
- set `camera.intrinsics` if calibration is known.

## 2. Run extraction

```bash
ego-vla process input.mp4 --output outputs/input_debug --config configs/local.json
```

For MediaPipe Tasks, either rely on automatic model download or pass a local model:

```bash
ego-vla process input.mp4 --output outputs/input_debug --pose-model /path/to/holistic_landmarker.task
```

For a dependency-light smoke test:

```bash
ego-vla process input.mp4 --output outputs/input_no_pose --pose-backend none --max-frames 10
```

## 3. Inspect output

```bash
ego-vla inspect outputs/input_debug
ego-vla visualize-data outputs/input_debug --output outputs/input_debug/report.html
ego-vla render-pose outputs/input_debug --output outputs/input_debug/pose_overlay.mp4
```

Open `report.html`, `pose_overlay.mp4`, and `records.jsonl`, then check:

- timestamps are monotonic;
- frame paths exist under `frames/`;
- warnings explain missing hands/body pose;
- pose coordinates include a clear `coordinate_frame`;
- the overlay video draws landmarks where detection succeeded;
- provenance records each backend.

## 4. Improve data quality

First-person handheld videos often need additional modules before the data is
ready for serious VLA training. These plug in as pipeline stages — see
[architecture.md](architecture.md) for the stage model and how to add a backend,
and run `ego-vla list-backends` to see what is registered:

- add object detection and tracking for manipulated objects;
- add depth and camera trajectory for spatial reasoning;
- add action segmentation and labels;
- add camera calibration for metric pose and depth;
- add manual review for frames where hands are occluded or motion-blurred.
