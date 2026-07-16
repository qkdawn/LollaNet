#!/usr/bin/env python3
"""Check that the prepared independent source-review handoff covers every model once.

This is a coverage/integrity-preparation check. Passing does not prove an
independent reviewer has run a review, that sources support the full claims,
or that a model is effective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16"


def load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return None
    return value


def check_campaign(base: Path) -> dict[str, Any]:
    base = base.resolve()
    errors: list[str] = []
    paths = {
        "coverage": base / "12-all-model-coverage.csv",
        "source_matrix": base / "27-all-model-core-claim-source-locator-gap.json",
        "handoff": base / "54-independent-source-content-review-handoff-manifest.json",
        "integrity": base / "55-independent-source-content-review-integrity-manifest.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if errors:
        return {"base": str(base), "models_checked": 0, "waves_checked": 0, "passed": False, "errors": errors}

    with paths["coverage"].open(encoding="utf-8-sig", newline="") as handle:
        coverage = list(csv.DictReader(handle))
    source_matrix = load_json(paths["source_matrix"], "source matrix", errors)
    handoff = load_json(paths["handoff"], "handoff manifest", errors)
    integrity = load_json(paths["integrity"], "integrity manifest", errors)
    if not source_matrix or not handoff or not integrity:
        return {"base": str(base), "models_checked": len(coverage), "waves_checked": 0, "passed": False, "errors": errors}

    source_rows = source_matrix.get("rows")
    handoff_waves = handoff.get("waves")
    integrity_waves = integrity.get("waves")
    if not isinstance(source_rows, list):
        errors.append("source matrix rows must be a list")
        source_rows = []
    if not isinstance(handoff_waves, list) or not handoff_waves:
        errors.append("handoff waves must be a non-empty list")
        handoff_waves = []
    if not isinstance(integrity_waves, list) or not integrity_waves:
        errors.append("integrity waves must be a non-empty list")
        integrity_waves = []

    coverage_ids = [row.get("model_id", "") for row in coverage]
    source_ids = [row.get("model_id", "") for row in source_rows if isinstance(row, dict)]
    if not coverage_ids or any(not item for item in coverage_ids):
        errors.append("coverage rows must have model_id")
    if len(coverage_ids) != len(set(coverage_ids)):
        errors.append("coverage contains duplicate model_id")
    if set(coverage_ids) != set(source_ids):
        errors.append("coverage and source-matrix model ID sets differ")
    for row in coverage:
        gate = row.get("quality_gate", "")
        if "same_reviewer_direct_source_locator_captured" not in gate:
            errors.append(f"{row.get('model_id', '<unknown>')}: missing direct-source-locator state")
        if "independent_content_review_pending" not in gate:
            errors.append(f"{row.get('model_id', '<unknown>')}: missing independent-content-review pending state")
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        if row.get("source_claim_status") != "exact_locator_captured_pending_independent_content_review":
            errors.append(f"{row.get('model_id', '<unknown>')}: unexpected source-claim status")

    handoff_by_wave = {str(w.get("wave")): w for w in handoff_waves if isinstance(w, dict)}
    integrity_by_wave = {str(w.get("wave")): w for w in integrity_waves if isinstance(w, dict)}
    if set(handoff_by_wave) != set(integrity_by_wave):
        errors.append("handoff and integrity wave sets differ")
    packet_ids: list[str] = []
    coverage_wave_by_id = {row.get("model_id", ""): row.get("wave", "") for row in coverage}
    for wave_name, wave in handoff_by_wave.items():
        model_ids = wave.get("model_ids")
        if not isinstance(model_ids, list) or not model_ids:
            errors.append(f"{wave_name}: missing model_ids")
            continue
        if len(model_ids) != wave.get("models"):
            errors.append(f"{wave_name}: declared model count differs from model_ids")
        packet_ids.extend(model_ids)
        for model_id in model_ids:
            if coverage_wave_by_id.get(model_id) != wave_name:
                errors.append(f"{wave_name}: model wave mismatch: {model_id}")
        integrity_wave = integrity_by_wave.get(wave_name)
        if not integrity_wave:
            continue
        if integrity_wave.get("models") != wave.get("models"):
            errors.append(f"{wave_name}: handoff/integrity model count mismatch")
        for path_key, hash_key, byte_key, heading in (
            ("source_packet_path", "source_packet_sha256", "packet_bytes", "### "),
            ("review_record_template_path", "review_record_template_sha256", "template_bytes", "### "),
        ):
            raw_path = integrity_wave.get(path_key)
            if not isinstance(raw_path, str):
                errors.append(f"{wave_name}: missing {path_key}")
                continue
            artifact = (base / raw_path).resolve()
            if base not in artifact.parents or not artifact.is_file():
                errors.append(f"{wave_name}: invalid or missing {path_key}")
                continue
            content = artifact.read_text(encoding="utf-8")
            if content.count(heading) != wave.get("models"):
                errors.append(f"{wave_name}: {path_key} card count does not match declared models")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != integrity_wave.get(hash_key):
                errors.append(f"{wave_name}: {path_key} SHA-256 mismatch")
            if artifact.stat().st_size != integrity_wave.get(byte_key):
                errors.append(f"{wave_name}: {path_key} byte-count mismatch")

    if len(packet_ids) != len(set(packet_ids)):
        errors.append("handoff packet assignments contain duplicate model_id")
    if set(packet_ids) != set(coverage_ids):
        errors.append("handoff packet assignments do not cover exactly the coverage model IDs")

    return {
        "base": str(base),
        "models_checked": len(coverage_ids),
        "waves_checked": len(handoff_by_wave),
        "status_counts": Counter(row.get("source_claim_status") for row in source_rows),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check source-review handoff coverage and frozen artifacts.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Private evaluation campaign directory.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    args = parser.parse_args()
    report = check_campaign(args.base)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    elif report["passed"]:
        print(f"PASS: {report['models_checked']} models across {report['waves_checked']} source-review waves are covered and frozen artifacts match.")
    else:
        print("FAIL: source-review coverage check")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
