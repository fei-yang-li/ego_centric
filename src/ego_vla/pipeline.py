from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ego_vla import __version__
from ego_vla.config import ProcessingConfig
from ego_vla.estimators import build_frame_estimator
from ego_vla.export import JsonlRecordWriter, write_metadata
from ego_vla.schemas import ActionEstimate, DenseSignal, EgoMotionEstimate, VLAFrameRecord
from ego_vla.video import VideoSampler


class EgoVlaPipeline:
    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()

    def process(self, video_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        records_path = output_dir / "records.jsonl"
        metadata_path = output_dir / "metadata.json"

        sampler = VideoSampler(video_path)
        video_metadata = sampler.probe()
        episode_id = self.config.dataset.episode_id or video_path.stem
        image_format = self.config.video.image_format.lower().lstrip(".")

        estimator = build_frame_estimator(self.config)
        processed_records = 0
        try:
            with JsonlRecordWriter(records_path) as writer:
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

                    record = VLAFrameRecord(
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
                        camera={
                            "intrinsics": self.config.camera.intrinsics,
                            "coordinate_frame": self.config.camera.coordinate_frame,
                        },
                        body_pose_3d=estimate.body_pose_3d,
                        body_pose_2d=estimate.body_pose_2d,
                        hands=estimate.hands,
                        objects=estimate.objects,
                        depth=_apply_backend_status(
                            estimate.depth,
                            backend=self.config.depth.backend,
                            default_note="Depth is not computed by the default pipeline.",
                        ),
                        ego_motion=_apply_ego_motion_status(
                            estimate.ego_motion,
                            backend=self.config.ego_motion.backend,
                        ),
                        action=action,
                        provenance={
                            "pipeline_version": __version__,
                            "pose_backend": estimator.backend_name,
                            "depth_backend": self.config.depth.backend,
                            "ego_motion_backend": self.config.ego_motion.backend,
                            "object_backend": self.config.objects.backend,
                            "action_backend": self.config.actions.backend,
                        },
                        warnings=estimate.warnings,
                    )
                    writer.write(record)
                    processed_records = writer.count
        finally:
            estimator.close()

        metadata = {
            "dataset": asdict(self.config.dataset),
            "config": self.config.to_dict(),
            "video": asdict(video_metadata),
            "output": {
                "records_path": str(records_path),
                "frames_dir": str(frames_dir),
                "record_count": processed_records,
            },
            "schema_version": "vla_frame_record/v1",
            "pipeline_version": __version__,
        }
        write_metadata(metadata_path, metadata)
        return metadata


def _apply_backend_status(signal: DenseSignal, backend: str, default_note: str) -> DenseSignal:
    if backend == "none":
        signal.backend = "none"
        signal.status = "not_computed"
        if default_note not in signal.notes:
            signal.notes.append(default_note)
    return signal


def _apply_ego_motion_status(signal: EgoMotionEstimate, backend: str) -> EgoMotionEstimate:
    if backend == "none":
        signal.backend = "none"
        signal.status = "not_computed"
        note = "Camera trajectory is not computed by the default pipeline; add SLAM or visual odometry for metric VLA state."
        if note not in signal.notes:
            signal.notes.append(note)
    return signal
