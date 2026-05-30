# Egocentric VLA Dataset Builder

This project turns first-person handheld video into structured records that can
be used to train Vision-Language-Action (VLA) models.

The initial implementation focuses on a reproducible data pipeline:

- sample frames from an egocentric video;
- estimate body and hand pose with an optional MediaPipe backend;
- preserve timestamps, frame provenance, and camera metadata;
- export JSONL records with slots for RGB, 3D pose, ego-motion, depth, object
  tracks, action labels, and language instructions;
- keep estimator interfaces modular so depth, SLAM, object tracking, and action
  recognition can be added without changing the dataset contract.

## Why these fields matter for VLA

VLA training usually needs more than raw frames. This project records:

1. **Observation**: RGB frame path, resolution, timestamp, source video.
2. **Embodiment state**: hands, body pose, and confidence values.
3. **Scene state**: object detections/tracks and optional segmentation masks.
4. **Spatial context**: camera intrinsics, estimated camera pose, depth.
5. **Action supervision**: discrete action labels, continuous controls, or
   human-readable instructions when available.
6. **Provenance**: estimator backend, config, and per-record status so generated
   data is auditable.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vision]"

ego-vla init-config configs/local.json
ego-vla process /path/to/first_person_video.mp4 --output outputs/debug_run --config configs/local.json
```

If you only want to test the pipeline without installing MediaPipe, set
`pose.backend` to `"none"` in the config or run:

```bash
ego-vla process /path/to/video.mp4 --output outputs/no_pose --pose-backend none
```

With current MediaPipe releases, the pipeline uses the Tasks API and will
automatically download the official holistic landmarker `.task` model into
`~/.cache/ego_vla/models/` unless `pose.model_asset_path` is set. On Linux,
MediaPipe may also require `libEGL.so.1`; install it with `sudo apt-get install -y libegl1`
if model initialization fails with a `libEGL` error.

## Output layout

```text
outputs/debug_run/
  frames/
    000000.jpg
    000001.jpg
  records.jsonl
  metadata.json
```

Each line in `records.jsonl` is one VLA-ready frame record. The schema is
documented in [docs/data_schema.md](docs/data_schema.md).

## Visualization

After processing a video, generate a standalone HTML report to inspect the
structured records and frame previews:

```bash
ego-vla visualize-data outputs/debug_run --output outputs/debug_run/report.html
```

To audit pose quality visually, render an MP4 with 2D body pose and hand
landmarks drawn over the exported sampled frames:

```bash
ego-vla render-pose outputs/debug_run --output outputs/debug_run/pose_overlay.mp4
```

The overlay uses `body_pose_2d` and hand landmarks. It does not project
MediaPipe world landmarks back to pixels unless a 2D estimate is present,
because metric 3D projection requires camera calibration.

## Commands

```bash
ego-vla init-config configs/local.json
ego-vla process video.mp4 --output outputs/run --config configs/default.json
ego-vla inspect outputs/run
ego-vla visualize-data outputs/run --output outputs/run/report.html
ego-vla render-pose outputs/run --output outputs/run/pose_overlay.mp4
ego-vla schema
```

## Development

```bash
python -m unittest discover
python -m compileall src tests
```

## Notes on first-person handheld video

First-person handheld data is noisy: hands occlude objects, camera motion is
large, and the camera wearer is usually not fully visible. The project therefore
stores confidence and coordinate-frame metadata instead of pretending every
estimate is metric and complete. For production data, add calibration, SLAM or
visual odometry, depth estimation, and manual/automatic action labels.
