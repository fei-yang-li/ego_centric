from __future__ import annotations

import argparse
from pathlib import Path

from ego_vla.config import load_config
from ego_vla.pipeline import EgoVlaPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one egocentric debug video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/example"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = EgoVlaPipeline(config).process(args.video, args.output)
    print(f"Wrote {metadata['output']['record_count']} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
