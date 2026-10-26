from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from .common import atomic_write_json, normalized_local_checkpoint, read_json, sha256_file, stable_hash
from .freeze import REQUIRED_PATTERNS


def resign_checkpoint(charter_path: Path, candidate_id: str) -> dict[str, Any]:
    from fnmatch import fnmatch

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    candidates = {row["id"]: row for row in charter["candidates"]}
    if candidate_id not in candidates:
        raise ValueError(f"candidate is not preregistered: {candidate_id}")
    candidate = candidates[candidate_id]
    checkpoint = normalized_local_checkpoint(charter_path, candidate.get("local_checkpoint"))
    if checkpoint is None or not checkpoint.is_dir():
        raise ValueError("candidate checkpoint is missing")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    content = dict(manifest)
    previous_hash = content.pop("checkpoint_content_sha256", None)
    if previous_hash != stable_hash(content):
        raise ValueError("existing checkpoint manifest content hash mismatch")
    if (
        manifest.get("candidate_id") != candidate_id
        or manifest.get("model_id") != candidate["model_id"]
        or manifest.get("revision") != candidate["revision"]
        or Path(str(manifest.get("checkpoint", ""))).resolve() != checkpoint.resolve()
    ):
        raise ValueError("existing checkpoint manifest has a different candidate identity")

    files = manifest.get("files", [])
    root = checkpoint.resolve()
    covered: set[str] = set()
    for row in files:
        relative = str(row["path"])
        path = (root / relative).resolve()
        if root not in path.parents or relative in covered:
            raise ValueError("checkpoint manifest contains a path escape or duplicate")
        covered.add(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"checkpoint file differs from its frozen manifest: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file() and any(fnmatch(path.name, pattern) for pattern in REQUIRED_PATTERNS)
    }
    if covered != actual:
        raise ValueError("checkpoint required-file coverage differs from its frozen manifest")
    if len(files) != int(manifest["file_count"]) or sum(int(row["size"]) for row in files) != int(manifest["total_bytes"]):
        raise ValueError("checkpoint manifest count or total size is inconsistent")

    resigned = dict(manifest)
    resigned.pop("checkpoint_content_sha256", None)
    resigned["charter_sha256"] = sha256_file(charter_path)
    resigned["resigned_at"] = datetime.now(timezone.utc).isoformat()
    resigned["resigned_from_checkpoint_content_sha256"] = previous_hash
    resigned["checkpoint_content_sha256"] = stable_hash(resigned)
    atomic_write_json(manifest_path, resigned)
    return resigned


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebind a verified local checkpoint to a revised charter.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("qualification_charter.yaml"))
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    print(json.dumps(resign_checkpoint(args.charter, args.candidate), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

