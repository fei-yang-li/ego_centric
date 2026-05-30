from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ego_vla.schemas import VLAFrameRecord


class JsonlRecordWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, record: VLAFrameRecord) -> None:
        json.dump(record.to_dict(), self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self.count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlRecordWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def count_jsonl_records(path: str | Path) -> int:
    record_path = Path(path)
    if not record_path.exists():
        return 0
    with record_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
