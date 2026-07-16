#!/usr/bin/env python3
"""Verify frozen independent source-content review packet hashes before handoff.

This checks file identity only. A successful result does not prove reviewer
independence, source sufficiency, model effectiveness, or a completed review.
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
    / "55-independent-source-content-review-integrity-manifest.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"manifest path must be relative: {raw_path}")
    resolved_base = base.resolve()
    resolved_path = (base / path).resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise ValueError(f"manifest path escapes its directory: {raw_path}")
    return resolved_path


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    waves_checked = 0
    artifacts_checked = 0

    if not manifest_path.is_file():
        return {
            "manifest": str(manifest_path),
            "waves_checked": waves_checked,
            "artifacts_checked": artifacts_checked,
            "passed": False,
            "errors": [f"manifest not found: {manifest_path}"],
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "manifest": str(manifest_path),
            "waves_checked": waves_checked,
            "artifacts_checked": artifacts_checked,
            "passed": False,
            "errors": [f"cannot read manifest: {exc}"],
        }

    if manifest.get("type") != "independent_source_review_integrity_manifest":
        errors.append("unexpected manifest type")
    waves = manifest.get("waves")
    if not isinstance(waves, list) or not waves:
        errors.append("manifest must contain a non-empty waves list")
        waves = []

    base = manifest_path.parent
    artifact_specs = (
        ("source_packet_path", "source_packet_sha256", "packet_bytes"),
        ("review_record_template_path", "review_record_template_sha256", "template_bytes"),
    )
    for wave in waves:
        if not isinstance(wave, dict):
            errors.append("wave entry must be an object")
            continue
        name = str(wave.get("wave", "<unnamed wave>"))
        waves_checked += 1
        for path_key, hash_key, byte_key in artifact_specs:
            raw_path = wave.get(path_key)
            expected_hash = wave.get(hash_key)
            expected_bytes = wave.get(byte_key)
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"{name}: missing {path_key}")
                continue
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                errors.append(f"{name}: invalid {hash_key}")
                continue
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                errors.append(f"{name}: invalid {byte_key}")
                continue
            try:
                artifact = safe_relative_path(base, raw_path)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
                continue
            if not artifact.is_file():
                errors.append(f"{name}: missing {path_key}: {raw_path}")
                continue
            artifacts_checked += 1
            if sha256(artifact) != expected_hash:
                errors.append(f"{name}: {path_key} SHA-256 mismatch")
            if artifact.stat().st_size != expected_bytes:
                errors.append(f"{name}: {path_key} byte-count mismatch")

    return {
        "manifest": str(manifest_path),
        "waves_checked": waves_checked,
        "artifacts_checked": artifacts_checked,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify frozen independent source-content review packet hashes."
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="Integrity manifest JSON path."
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable verification report.")
    args = parser.parse_args()
    report = verify_manifest(args.manifest)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(
            f"PASS: {report['waves_checked']} waves and {report['artifacts_checked']} "
            "frozen source-review artifacts match the manifest."
        )
    else:
        print("FAIL: source-review integrity verification")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
