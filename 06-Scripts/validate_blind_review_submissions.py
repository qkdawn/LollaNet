#!/usr/bin/env python3
"""Validate private independent blind-review submissions.

A passing report verifies only the frozen-record format, wave coverage, and the
blind-packet hash declared by a submission. It does not prove that a reviewer
was actually independent or blind, that a judgement is correct, or that any
model is effective. The review coordinator must perform those checks before
unblinding, aggregation, or governance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16"
PREFERENCES = {"A", "B", "tie", "invalid", "abstain"}
COMPARISON_ASSESSMENTS = {"A", "B", "tie", "neither", "unclear", "not_assessed"}
UNSUPPORTED_CLAIM_ASSESSMENTS = {"A", "B", "both", "neither", "unclear", "not_assessed"}
INDEPENDENCE_FIELDS = (
    "participated_in_lollanet_card_authoring",
    "participated_in_task_or_answer_generation",
    "participated_in_prior_blind_review_or_scoring",
    "saw_randomization_mapping_before_scoring",
    "saw_full_ablation_mapping_before_scoring",
    "used_private_user_data",
)


def read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return None
    return value


def non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_iso_date(value: Any) -> bool:
    if not non_empty_string(value):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def campaign_metadata(base: Path, errors: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return expected unit IDs and frozen packet hashes keyed by wave."""
    handoff = read_json(base / "16-independent-review-handoff-manifest.json", "handoff manifest", errors)
    integrity = read_json(base / "28-independent-review-integrity-manifest.json", "integrity manifest", errors)
    if handoff is None or integrity is None:
        return {}, {}

    handoff_waves = handoff.get("waves")
    integrity_waves = integrity.get("waves")
    if not isinstance(handoff_waves, list) or not isinstance(integrity_waves, list):
        errors.append("handoff and integrity manifests must contain waves lists")
        return {}, {}

    expected: dict[str, list[str]] = {}
    for entry in handoff_waves:
        if not isinstance(entry, dict) or not non_empty_string(entry.get("wave")) or not isinstance(entry.get("unit_ids"), list):
            errors.append("handoff contains invalid wave definition")
            continue
        wave = entry["wave"]
        unit_ids = entry["unit_ids"]
        if not all(non_empty_string(unit_id) for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
            errors.append(f"{wave}: invalid unit_ids")
            continue
        if wave in expected:
            errors.append(f"handoff contains duplicate wave {wave}")
            continue
        expected[wave] = unit_ids

    packet_hashes: dict[str, str] = {}
    for entry in integrity_waves:
        if not isinstance(entry, dict) or not non_empty_string(entry.get("wave")) or not non_empty_string(entry.get("blind_packet_sha256")):
            errors.append("integrity manifest contains invalid wave definition")
            continue
        wave = entry["wave"]
        packet_hash = entry["blind_packet_sha256"]
        if len(packet_hash) != 64:
            errors.append(f"{wave}: invalid blind_packet_sha256")
            continue
        if wave in packet_hashes:
            errors.append(f"integrity manifest contains duplicate wave {wave}")
            continue
        packet_hashes[wave] = packet_hash

    if set(expected) != set(packet_hashes):
        errors.append("handoff and integrity manifest wave sets differ")
    return expected, packet_hashes


def validate_submission(
    path: Path,
    expected: dict[str, list[str]],
    packet_hashes: dict[str, str],
    seen_submission_ids: set[str],
    seen_reviewer_wave: set[tuple[str, str]],
    submissions_dir: Path,
) -> tuple[dict[str, str] | None, list[str]]:
    errors: list[str] = []
    submission = read_json(path, str(path), errors)
    if submission is None:
        return None, errors
    prefix = path.name

    if submission.get("type") != "independent_blind_review_submission":
        errors.append(f"{prefix}: unexpected type")
    if submission.get("campaign") != "model-test-v2-2026-07-16":
        errors.append(f"{prefix}: unexpected campaign")
    if submission.get("status") != "frozen_submitted":
        errors.append(f"{prefix}: status must be frozen_submitted")

    submission_id = submission.get("submission_id")
    if not non_empty_string(submission_id):
        errors.append(f"{prefix}: missing submission_id")
    elif submission_id in seen_submission_ids:
        errors.append(f"{prefix}: duplicate submission_id {submission_id}")
    else:
        seen_submission_ids.add(submission_id)

    wave = submission.get("wave")
    if not non_empty_string(wave) or wave not in expected:
        errors.append(f"{prefix}: unknown wave {wave!r}")
        wave = ""

    reviewer = submission.get("reviewer")
    reviewer_id = ""
    if not isinstance(reviewer, dict):
        errors.append(f"{prefix}: reviewer must be an object")
    else:
        reviewer_id = reviewer.get("reviewer_id", "")
        if not non_empty_string(reviewer_id):
            errors.append(f"{prefix}: missing reviewer.reviewer_id")
            reviewer_id = ""
        if not parse_iso_date(reviewer.get("review_date")):
            errors.append(f"{prefix}: reviewer.review_date must be YYYY-MM-DD")
        if not non_empty_string(reviewer.get("agent_or_tool")):
            errors.append(f"{prefix}: missing reviewer.agent_or_tool")
        declaration = reviewer.get("independence_declaration")
        if not isinstance(declaration, dict):
            errors.append(f"{prefix}: missing independence declaration")
        else:
            for field in INDEPENDENCE_FIELDS:
                if declaration.get(field) is not False:
                    errors.append(f"{prefix}: independence declaration must set {field} to false")
            if not non_empty_string(declaration.get("conflict_statement")):
                errors.append(f"{prefix}: missing conflict_statement")

    if wave:
        try:
            relative_parent = path.parent.resolve().relative_to(submissions_dir.resolve())
        except ValueError:
            errors.append(f"{prefix}: file must be inside submissions directory")
        else:
            if relative_parent != Path(wave):
                errors.append(f"{prefix}: submission directory must match wave {wave}")
        expected_hash = packet_hashes.get(wave)
        if submission.get("blind_packet_sha256") != expected_hash:
            errors.append(f"{prefix}: blind_packet_sha256 does not match frozen packet for {wave}")

    if wave and reviewer_id:
        reviewer_wave = (wave, reviewer_id)
        if reviewer_wave in seen_reviewer_wave:
            errors.append(f"{prefix}: duplicate reviewer_id {reviewer_id} for {wave}")
        else:
            seen_reviewer_wave.add(reviewer_wave)

    records = submission.get("records")
    if not isinstance(records, list):
        errors.append(f"{prefix}: records must be a list")
        records = []
    record_ids: list[str] = []
    for index, record in enumerate(records, start=1):
        record_prefix = f"{prefix}: records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{record_prefix} must be an object")
            continue
        unit_id = record.get("unit_id")
        if not non_empty_string(unit_id):
            errors.append(f"{record_prefix}: missing unit_id")
        else:
            record_ids.append(unit_id)
        if record.get("preference") not in PREFERENCES:
            errors.append(f"{record_prefix}: preference must be one of {sorted(PREFERENCES)}")
        for field in ("correctness", "boundary_handling", "action_increment"):
            if record.get(field) not in COMPARISON_ASSESSMENTS:
                errors.append(f"{record_prefix}: {field} must be one of {sorted(COMPARISON_ASSESSMENTS)}")
        if record.get("unsupported_claims") not in UNSUPPORTED_CLAIM_ASSESSMENTS:
            errors.append(f"{record_prefix}: unsupported_claims must be one of {sorted(UNSUPPORTED_CLAIM_ASSESSMENTS)}")
        if not non_empty_string(record.get("rationale")):
            errors.append(f"{record_prefix}: missing rationale")

    if len(record_ids) != len(set(record_ids)):
        errors.append(f"{prefix}: duplicate unit_id in records")
    if wave:
        expected_ids = expected[wave]
        if set(record_ids) != set(expected_ids) or len(record_ids) != len(expected_ids):
            errors.append(f"{prefix}: records must cover every unit_id exactly once for {wave}")

    wave_summary = submission.get("wave_summary")
    if not isinstance(wave_summary, dict):
        errors.append(f"{prefix}: wave_summary must be an object")
    else:
        if not non_empty_string(wave_summary.get("unresolved_disagreements")):
            errors.append(f"{prefix}: missing wave_summary.unresolved_disagreements")
        if not non_empty_string(wave_summary.get("limitations")):
            errors.append(f"{prefix}: missing wave_summary.limitations")

    if errors:
        return None, errors
    return {"wave": wave, "reviewer_id": reviewer_id, "submission": str(path)}, []


def validate_campaign(
    base: Path,
    submissions_dir: Path | None = None,
    require_two_per_wave: bool = False,
) -> dict[str, Any]:
    base = base.resolve()
    submissions_dir = (submissions_dir or base / "blind-review-submissions").resolve()
    errors: list[str] = []
    expected, packet_hashes = campaign_metadata(base, errors)

    if submissions_dir.exists() and not submissions_dir.is_dir():
        errors.append(f"submissions path is not a directory: {submissions_dir}")
        submission_paths: list[Path] = []
    elif submissions_dir.exists():
        submission_paths = sorted(path for path in submissions_dir.rglob("*.json") if path.is_file())
    else:
        submission_paths = []

    valid: list[dict[str, str]] = []
    seen_submission_ids: set[str] = set()
    seen_reviewer_wave: set[tuple[str, str]] = set()
    for path in submission_paths:
        item, item_errors = validate_submission(
            path, expected, packet_hashes, seen_submission_ids, seen_reviewer_wave, submissions_dir
        )
        errors.extend(item_errors)
        if item is not None:
            valid.append(item)

    reviewers_by_wave: dict[str, set[str]] = {wave: set() for wave in expected}
    for item in valid:
        reviewers_by_wave[item["wave"]].add(item["reviewer_id"])
    two_submission_waves = sorted(wave for wave, reviewers in reviewers_by_wave.items() if len(reviewers) >= 2)
    if require_two_per_wave:
        incomplete = sorted(wave for wave, reviewers in reviewers_by_wave.items() if len(reviewers) < 2)
        if incomplete:
            errors.append("fewer than two format-valid declared-independent submissions for: " + ", ".join(incomplete))

    return {
        "base": str(base),
        "submissions_dir": str(submissions_dir),
        "waves_expected": len(expected),
        "submissions_found": len(submission_paths),
        "format_valid_submissions": len(valid),
        "declared_independent_reviewer_counts": {wave: len(reviewers_by_wave[wave]) for wave in sorted(reviewers_by_wave)},
        "two_submission_format_gate_waves": two_submission_waves,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen independent blind-review submissions; this is not a proof of actual independence or model efficacy."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Campaign evaluation directory.")
    parser.add_argument("--submissions-dir", type=Path, help="Override blind-review submissions directory.")
    parser.add_argument("--require-two-per-wave", action="store_true", help="Fail unless every wave has two distinct format-valid declared-independent submissions.")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable validation report.")
    args = parser.parse_args()
    report = validate_campaign(args.base, args.submissions_dir, args.require_two_per_wave)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(
            f"PASS: {report['format_valid_submissions']} format-valid frozen blind-review submission(s); "
            f"{len(report['two_submission_format_gate_waves'])}/{report['waves_expected']} waves meet the two-submission format gate."
        )
        print("Coordinator verification, controlled unblinding, and no automatic model-state change remain required.")
    else:
        print("FAIL: blind-review submission validation")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
