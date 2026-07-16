#!/usr/bin/env python3
"""Verify the frozen source-grounded retrospective task-pack artifacts.

Passing verifies only that the local charter, task packet, and scoring rubric
still match their declared byte sizes and SHA-256 hashes. It does not establish
source validity, independent execution or review, or model effectiveness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "private"
    / "evaluations"
    / "model-test-v2-2026-07-16"
    / "59-public-case-retrospective-integrity-manifest.json"
)


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"manifest": str(manifest_path), "artifacts_checked": 0, "passed": False, "errors": [str(exc)]}

    if not isinstance(manifest, dict):
        return {"manifest": str(manifest_path), "artifacts_checked": 0, "passed": False, "errors": ["manifest must be an object"]}
    if manifest.get("type") != "source_grounded_retrospective_integrity_manifest":
        errors.append("unexpected manifest type")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest must contain a non-empty artifacts list")
        artifacts = []

    base = manifest_path.parent
    seen_paths: set[str] = set()
    checked = 0
    for index, artifact in enumerate(artifacts, start=1):
        label = f"artifact {index}"
        if not isinstance(artifact, dict):
            errors.append(f"{label}: must be an object")
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{label}: missing path")
            continue
        if raw_path in seen_paths:
            errors.append(f"{label}: duplicate path {raw_path}")
            continue
        seen_paths.add(raw_path)
        path = (base / raw_path).resolve()
        if base not in path.parents or not path.is_file():
            errors.append(f"{label}: invalid or missing artifact path {raw_path}")
            continue
        expected_hash = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"{label}: missing or invalid SHA-256")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            errors.append(f"{label}: SHA-256 mismatch for {raw_path}")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            errors.append(f"{label}: missing or invalid byte count")
        elif path.stat().st_size != expected_bytes:
            errors.append(f"{label}: byte-count mismatch for {raw_path}")
        checked += 1

    return {
        "manifest": str(manifest_path),
        "artifacts_checked": checked,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen source-grounded retrospective task-pack artifacts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Integrity manifest to verify.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    args = parser.parse_args()
    report = verify_manifest(args.manifest)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"PASS: {report['artifacts_checked']} frozen retrospective task-pack artifacts match the manifest.")
    else:
        print("FAIL: retrospective task-pack integrity check")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
