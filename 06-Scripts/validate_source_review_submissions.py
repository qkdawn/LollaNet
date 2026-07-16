#!/usr/bin/env python3
"""Validate private independent source-content review submissions.

The validator checks frozen-submission structure, coverage, and matching frozen
source-packet hashes. A passing result does not establish a reviewer's actual
independence, source sufficiency, or any model's effectiveness; coordinator
review remains mandatory before any governance decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16"
DISPOSITIONS = {
    "supports_narrow_claim",
    "supports_only_narrower_claim",
    "unsupported_or_mismapped",
    "source_inaccessible",
    "conflicting_evidence",
    "unclear",
}
CONFIDENCES = {"low", "medium", "high"}
NEXT_STEPS = {
    "maintain_review_record",
    "continue_research",
    "consider_independent_task_replication",
    "consider_governance_research",
}
INDEPENDENCE_FIELDS = (
    "participated_in_lollanet_card_authoring",
    "participated_in_source_location_or_prior_audit",
    "participated_in_task_or_answer_generation",
    "participated_in_governance_for_this_wave",
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
    handoff = read_json(base / "54-independent-source-content-review-handoff-manifest.json", "handoff manifest", errors)
    integrity = read_json(base / "55-independent-source-content-review-integrity-manifest.json", "integrity manifest", errors)
    if handoff is None or integrity is None:
        return {}, {}
    handoff_waves = handoff.get("waves")
    integrity_waves = integrity.get("waves")
    if not isinstance(handoff_waves, list) or not isinstance(integrity_waves, list):
        errors.append("handoff and integrity manifests must contain waves lists")
        return {}, {}
    expected: dict[str, list[str]] = {}
    for wave in handoff_waves:
        if not isinstance(wave, dict) or not non_empty_string(wave.get("wave")) or not isinstance(wave.get("model_ids"), list):
            errors.append("handoff contains invalid wave definition")
            continue
        name = wave["wave"]
        model_ids = wave["model_ids"]
        if not all(non_empty_string(model_id) for model_id in model_ids) or len(model_ids) != len(set(model_ids)):
            errors.append(f"{name}: invalid model_ids")
            continue
        if name in expected:
            errors.append(f"handoff contains duplicate wave {name}")
            continue
        expected[name] = model_ids
    packet_hashes: dict[str, str] = {}
    for wave in integrity_waves:
        if not isinstance(wave, dict) or not non_empty_string(wave.get("wave")) or not non_empty_string(wave.get("source_packet_sha256")):
            errors.append("integrity manifest contains invalid wave definition")
            continue
        name = wave["wave"]
        if name in packet_hashes:
            errors.append(f"integrity manifest contains duplicate wave {name}")
            continue
        packet_hashes[name] = wave["source_packet_sha256"]
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
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    submission = read_json(path, str(path), errors)
    if submission is None:
        return None, errors
    prefix = path.name
    if submission.get("type") != "independent_source_content_review_submission":
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
            relative_parts = path.relative_to(submissions_dir).parts
        except ValueError:
            relative_parts = ()
        if len(relative_parts) < 2 or relative_parts[0] != wave:
            errors.append(f"{prefix}: submission path must be under source-review-submissions/{wave}/")
    if wave and reviewer_id:
        reviewer_wave = (reviewer_id, wave)
        if reviewer_wave in seen_reviewer_wave:
            errors.append(f"{prefix}: duplicate reviewer_id for {wave}; submit revisions for coordinator reconciliation, not double-counting")
        else:
            seen_reviewer_wave.add(reviewer_wave)
    if wave and submission.get("source_packet_sha256") != packet_hashes.get(wave):
        errors.append(f"{prefix}: source_packet_sha256 does not match frozen packet for {wave}")

    records = submission.get("records")
    if not isinstance(records, list) or not records:
        errors.append(f"{prefix}: records must be a non-empty list")
        records = []
    record_ids: list[str] = []
    for index, record in enumerate(records, start=1):
        label = f"{prefix}: record {index}"
        if not isinstance(record, dict):
            errors.append(f"{label} must be an object")
            continue
        model_id = record.get("model_id")
        if not non_empty_string(model_id):
            errors.append(f"{label}: missing model_id")
        else:
            record_ids.append(model_id)
        disposition = record.get("disposition")
        if disposition not in DISPOSITIONS:
            errors.append(f"{label}: invalid disposition")
        source_refs = record.get("source_refs")
        if not isinstance(source_refs, list):
            errors.append(f"{label}: source_refs must be a list")
            source_refs = []
        if disposition != "source_inaccessible" and not source_refs:
            errors.append(f"{label}: accessible disposition requires at least one source ref")
        for ref_index, source_ref in enumerate(source_refs, start=1):
            if not isinstance(source_ref, dict) or not all(
                non_empty_string(source_ref.get(field)) for field in ("stable_url", "version_or_access_date", "locator")
            ):
                errors.append(f"{label}: source ref {ref_index} requires stable_url, version_or_access_date, and locator")
        for field in ("support_or_limit", "counterevidence_or_boundary", "access_notes"):
            if not non_empty_string(record.get(field)):
                errors.append(f"{label}: missing {field}")
        if disposition == "source_inaccessible" and not non_empty_string(record.get("access_notes")):
            errors.append(f"{label}: source_inaccessible requires access_notes")
        if record.get("confidence") not in CONFIDENCES:
            errors.append(f"{label}: invalid confidence")
    if wave and set(record_ids) != set(expected[wave]):
        errors.append(f"{prefix}: record model IDs do not exactly cover {wave}")
    if len(record_ids) != len(set(record_ids)):
        errors.append(f"{prefix}: duplicate model_id in records")

    summary = submission.get("wave_summary")
    if not isinstance(summary, dict):
        errors.append(f"{prefix}: wave_summary must be an object")
    else:
        if summary.get("recommended_next_step") not in NEXT_STEPS:
            errors.append(f"{prefix}: invalid recommended_next_step")
        if not non_empty_string(summary.get("unresolved_disagreements")):
            errors.append(f"{prefix}: missing unresolved_disagreements")
    return ({"path": str(path), "wave": wave, "reviewer_id": reviewer_id} if not errors else None), errors


def validate_campaign(base: Path, submissions_dir: Path | None = None, require_two_per_wave: bool = False) -> dict[str, Any]:
    base = base.resolve()
    errors: list[str] = []
    expected, packet_hashes = campaign_metadata(base, errors)
    if not expected or not packet_hashes:
        return {"base": str(base), "passed": False, "errors": errors, "valid_submission_count": 0, "wave_summary": {}}
    submissions_dir = (submissions_dir or (base / "source-review-submissions")).resolve()
    if base not in submissions_dir.parents and submissions_dir != base:
        errors.append("submissions directory must stay under campaign base")
        return {"base": str(base), "passed": False, "errors": errors, "valid_submission_count": 0, "wave_summary": {}}
    files = sorted(submissions_dir.rglob("*.json")) if submissions_dir.is_dir() else []
    seen_submission_ids: set[str] = set()
    seen_reviewer_wave: set[tuple[str, str]] = set()
    valid: list[dict[str, Any]] = []
    for path in files:
        result, submission_errors = validate_submission(
            path, expected, packet_hashes, seen_submission_ids, seen_reviewer_wave, submissions_dir
        )
        errors.extend(submission_errors)
        if result:
            valid.append(result)
    by_wave: dict[str, list[str]] = {wave: [] for wave in expected}
    for result in valid:
        by_wave[result["wave"]].append(result["reviewer_id"])
    wave_summary = {
        wave: {
            "declared_independent_valid_submission_count": len(reviewer_ids),
            "reviewer_ids": sorted(reviewer_ids),
            "two_submission_format_gate_met": len(reviewer_ids) >= 2,
            "coordinator_identity_and_substance_verification_required": True,
        }
        for wave, reviewer_ids in by_wave.items()
    }
    missing_two = [wave for wave, summary in wave_summary.items() if not summary["two_submission_format_gate_met"]]
    if require_two_per_wave and missing_two:
        errors.append("fewer than two format-valid declared-independent submissions for: " + ", ".join(missing_two))
    return {
        "base": str(base),
        "submissions_dir": str(submissions_dir),
        "submission_files_checked": len(files),
        "valid_submission_count": len(valid),
        "wave_summary": wave_summary,
        "completion_format_gate_met": not missing_two,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate private independent source-review submissions without changing model governance.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Private evaluation campaign directory.")
    parser.add_argument("--submissions-dir", type=Path, help="Optional source-review submissions directory beneath campaign base.")
    parser.add_argument("--require-two-per-wave", action="store_true", help="Fail unless every wave has two format-valid declared-independent submissions.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    args = parser.parse_args()
    report = validate_campaign(args.base, args.submissions_dir, args.require_two_per_wave)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        ready_waves = sum(1 for summary in report["wave_summary"].values() if summary["two_submission_format_gate_met"])
        print(
            f"PASS: {report['valid_submission_count']} format-valid frozen submission(s); "
            f"{ready_waves}/{len(report['wave_summary'])} waves meet the two-submission format gate. "
            "Coordinator verification is still required."
        )
    else:
        print("FAIL: independent source-review submission validation")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
