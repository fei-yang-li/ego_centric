from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ego_vla import __version__
from ego_vla.config import ProcessingConfig
from ego_vla.export import JsonlRecordWriter, write_jsonl, write_metadata
from ego_vla.schemas import ActionEstimate, VLAFrameRecord
from ego_vla.stages import (
    StageContext,
    build_clip_processor,
    build_frame_estimator,
    build_pre_processor,
    build_record_processors,
)
from ego_vla.video import VideoSampler


class EgoVlaPipeline:
    """Multi-stage egocentric-video -> VLA dataset pipeline.

    Stages run in order: calibration (pre) -> per-frame estimation (pose, ...)
    -> record processors (depth, objects, ego-motion) -> clip processor
    (action segmentation + captioning). Every stage is a registered backend; by
    default all optional stages are ``none`` and the output matches the
    pose-only pipeline.
    """

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()

    def process(self, video_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        records_path = output_dir / "records.jsonl"
        segments_path = output_dir / "segments.jsonl"
        metadata_path = output_dir / "metadata.json"

        sampler = VideoSampler(video_path)
        video_metadata = sampler.probe()
        episode_id = self.config.dataset.episode_id or video_path.stem
        image_format = self.config.video.image_format.lower().lstrip(".")

        ctx = StageContext(
            config=self.config,
            video_path=video_path,
            video_metadata=video_metadata,
            output_dir=output_dir,
            frames_dir=frames_dir,
        )

        # Stage 1: calibration (per video).
        pre_processor = build_pre_processor(self.config)
        calibration = pre_processor.run(ctx)
        ctx.calibration = calibration
        camera_field = self._camera_field(calibration)

        estimator = build_frame_estimator(self.config)
        record_processors = build_record_processors(self.config)
        clip_processor = build_clip_processor(self.config)

        records: list[VLAFrameRecord] = []
        segments: list[Any] = []
        try:
            # Stage 2: per-frame estimation. Frames are written immediately;
            # records are collected so later stages can see the full sequence.
            for sample in sampler.samples(
                sample_rate_hz=self.config.video.sample_rate_hz,
                max_frames=self.config.video.max_frames,
            ):
                frame_filename = f"{sample.ordinal:06d}.{image_format}"
                frame_path = frames_dir / frame_filename
                sampler.write_frame(
                    sample.image_bgr,
                    frame_path,
                    image_format=image_format,
                    jpeg_quality=self.config.video.jpeg_quality,
                )

                estimate = estimator.estimate(
                    sample.image_bgr,
                    frame_index=sample.frame_index,
                    timestamp_sec=sample.timestamp_sec,
                )
                records.append(
                    self._build_record(
                        episode_id=episode_id,
                        sample=sample,
                        estimate=estimate,
                        video_metadata=video_metadata,
                        video_path=video_path,
                        image_format=image_format,
                        frame_filename=frame_filename,
                        camera_field=camera_field,
                        pose_backend=estimator.backend_name,
                    )
                )

            # Stage 3: record processors mutate the sequence in place.
            for processor in record_processors:
                processor.run(records, ctx)

            # Stage 4: clip processor produces segment records.
            segments = clip_processor.run(records, ctx)
        finally:
            estimator.close()
            for processor in record_processors:
                _safe_close(processor)
            _safe_close(clip_processor)

        with JsonlRecordWriter(records_path) as writer:
            for record in records:
                writer.write(record)
            record_count = writer.count

        segment_count = 0
        segments_output: str | None = None
        if segments:
            segment_count = write_jsonl(segments_path, segments)
            segments_output = str(segments_path)

        metadata = {
            "dataset": asdict(self.config.dataset),
            "config": self.config.to_dict(),
            "video": asdict(video_metadata),
            "stages": {
                "calibration": pre_processor.backend_name,
                "pose": estimator.backend_name,
                "depth": self.config.depth.backend,
                "objects": self.config.objects.backend,
                "ego_motion": self.config.ego_motion.backend,
                "actions": clip_processor.backend_name,
            },
            "calibration": calibration.to_dict(),
            "output": {
                "records_path": str(records_path),
                "frames_dir": str(frames_dir),
                "record_count": record_count,
                "segments_path": segments_output,
                "segment_count": segment_count,
            },
            "schema_version": "vla_frame_record/v1",
            "pipeline_version": __version__,
        }
        if ctx.extra:
            metadata["stage_outputs"] = ctx.extra
        write_metadata(metadata_path, metadata)
        return metadata

    def _camera_field(self, calibration: Any) -> dict[str, Any]:
        intrinsics = calibration.intrinsics or self.config.camera.intrinsics
        coordinate_frame = (
            calibration.coordinate_frame or self.config.camera.coordinate_frame
        )
        return {"intrinsics": intrinsics, "coordinate_frame": coordinate_frame}

    def _build_record(
        self,
        *,
        episode_id: str,
        sample: Any,
        estimate: Any,
        video_metadata: Any,
        video_path: Path,
        image_format: str,
        frame_filename: str,
        camera_field: dict[str, Any],
        pose_backend: str,
    ) -> VLAFrameRecord:
        action = estimate.action
        if self.config.dataset.language_instruction and action.language_instruction is None:
            action = ActionEstimate(
                label=action.label,
                controls=action.controls,
                language_instruction=self.config.dataset.language_instruction,
                backend=action.backend,
                status=action.status,
                confidence=action.confidence,
            )

        return VLAFrameRecord(
            dataset_name=self.config.dataset.name,
            episode_id=episode_id,
            frame_id=f"{episode_id}/{sample.ordinal:06d}",
            frame_index=sample.frame_index,
            timestamp_sec=round(sample.timestamp_sec, 6),
            observation={
                "rgb": {
                    "path": str(Path("frames") / frame_filename),
                    "width": video_metadata.width,
                    "height": video_metadata.height,
                    "encoding": image_format,
                },
                "source_video": str(video_path),
            },
            camera=dict(camera_field),
            body_pose_3d=estimate.body_pose_3d,
            body_pose_2d=estimate.body_pose_2d,
            hands=estimate.hands,
            objects=estimate.objects,
            depth=estimate.depth,
            ego_motion=estimate.ego_motion,
            action=action,
            provenance={
                "pipeline_version": __version__,
                "pose_backend": pose_backend,
                "calibration_backend": self.config.calibration.backend,
                "depth_backend": self.config.depth.backend,
                "ego_motion_backend": self.config.ego_motion.backend,
                "object_backend": self.config.objects.backend,
                "action_backend": self.config.actions.backend,
            },
            warnings=estimate.warnings,
        )


def _safe_close(stage: Any) -> None:
    close = getattr(stage, "close", None)
    if callable(close):
        close()
