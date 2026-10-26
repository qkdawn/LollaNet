from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml


AXES = ("specificity", "independence", "non_template")
ALLOWED_SPLITS = {"train", "development"}


def load_charter(path: Path | None = None) -> dict[str, Any]:
    source = path or Path(__file__).with_name("charter.yaml")
    charter = yaml.safe_load(source.read_text(encoding="utf-8"))
    if tuple(charter["labels"]["axes"]) != AXES:
        raise ValueError("charter axes differ from the frozen three-axis contract")
    return charter


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_discovery_rows(rows: list[dict[str, Any]], expected: int | None = None) -> None:
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} discovery rows, found {len(rows)}")
    splits = {str(row["split"]) for row in rows}
    if not splits <= ALLOWED_SPLITS:
        raise ValueError(f"sealed or unknown split encountered: {sorted(splits - ALLOWED_SPLITS)}")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record IDs")
