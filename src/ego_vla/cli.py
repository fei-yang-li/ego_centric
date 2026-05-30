from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego_vla.config import load_config, write_default_config
from ego_vla.export import count_jsonl_records, iter_jsonl
from ego_vla.pipeline import EgoVlaPipeline
from ego_vla.schemas import json_schema_example
from ego_vla.visualization import render_pose_overlay_video, write_dataset_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ego-vla",
        description="Build VLA training records from first-person handheld video.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="Write a default JSON config.")
    init_parser.add_argument("path", type=Path)

    process_parser = subparsers.add_parser("process", help="Process one egocentric video.")
    process_parser.add_argument("video", type=Path)
    process_parser.add_argument("--output", "-o", type=Path, required=True)
    process_parser.add_argument("--config", "-c", type=Path, default=None)
    process_parser.add_argument("--sample-rate", type=float, default=None)
    process_parser.add_argument("--max-frames", type=int, default=None)
    process_parser.add_argument("--pose-backend", choices=["mediapipe", "none"], default=None)
    process_parser.add_argument("--pose-model", type=Path, default=None, help="Path to a MediaPipe .task model asset.")
    process_parser.add_argument("--language-instruction", type=str, default=None)

    inspect_parser = subparsers.add_parser("inspect", help="Summarize an output dataset directory.")
    inspect_parser.add_argument("dataset_dir", type=Path)

    report_parser = subparsers.add_parser(
        "visualize-data",
        help="Write an HTML report for a processed dataset.",
    )
    report_parser.add_argument("dataset_dir", type=Path)
    report_parser.add_argument("--output", "-o", type=Path, default=None)
    report_parser.add_argument("--max-records", type=int, default=25)

    render_parser = subparsers.add_parser(
        "render-pose",
        help="Render body and hand landmarks over exported frames.",
    )
    render_parser.add_argument("dataset_dir", type=Path)
    render_parser.add_argument("--output", "-o", type=Path, default=None)
    render_parser.add_argument("--fps", type=float, default=None)
    render_parser.add_argument("--max-frames", type=int, default=None)
    render_parser.add_argument("--min-confidence", type=float, default=0.0)

    subparsers.add_parser("schema", help="Print an example VLA frame record.")

    args = parser.parse_args(argv)

    if args.command == "init-config":
        write_default_config(args.path)
        print(f"Wrote config to {args.path}")
        return 0

    if args.command == "process":
        config = load_config(args.config)
        if args.sample_rate is not None:
            config.video.sample_rate_hz = args.sample_rate
        if args.max_frames is not None:
            config.video.max_frames = args.max_frames
        if args.pose_backend is not None:
            config.pose.backend = args.pose_backend
        if args.pose_model is not None:
            config.pose.model_asset_path = str(args.pose_model)
        if args.language_instruction is not None:
            config.dataset.language_instruction = args.language_instruction
        metadata = EgoVlaPipeline(config).process(args.video, args.output)
        print(json.dumps(metadata["output"], indent=2))
        return 0

    if args.command == "inspect":
        records_path = args.dataset_dir / "records.jsonl"
        count = count_jsonl_records(records_path)
        first = next(iter(iter_jsonl(records_path)), None) if count else None
        summary = {
            "dataset_dir": str(args.dataset_dir),
            "record_count": count,
            "first_frame_id": first.get("frame_id") if first else None,
            "first_timestamp_sec": first.get("timestamp_sec") if first else None,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "visualize-data":
        summary = write_dataset_report(
            args.dataset_dir,
            output_path=args.output,
            max_records=args.max_records,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "render-pose":
        summary = render_pose_overlay_video(
            args.dataset_dir,
            output_path=args.output,
            fps=args.fps,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "schema":
        print(json.dumps(json_schema_example(), indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
