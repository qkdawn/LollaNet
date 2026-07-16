#!/usr/bin/env python3
"""Check declared unit coverage and obvious mapping leakage in frozen blind packets.

This static checker verifies that the packet text still exposes exactly the
pre-registered blind-review units and two answer variants per unit. It also
looks for direct label-to-condition assignments (for example ``A = Full``),
not for ordinary methodological wording such as "mapping withheld". Passing
this check does not prove that prompts are valid, reviewers are blind, or a
model is effective.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16"
UNIT_HEADING = re.compile(r"^##\s+([A-Za-z0-9-]+)(?:\s*/|\s+—)", re.MULTILINE)
ANSWER_A = re.compile(r"(?m)^(?:###\s*Answer A\b|\*\*A[：:]\*\*)")
ANSWER_B = re.compile(r"(?m)^(?:###\s*Answer B\b|\*\*B[：:]\*\*)")
DIRECT_ASSIGNMENT = re.compile(
    r"(?im)(?:\b(?:answer|label|condition)\s*)?[AB]\s*(?:=|:|：|is|为)\s*(?:\*\*)?\s*[`\"']?\s*(?:full|ablation|baseline)\b"
)
MODEL_ID_OR_CARD_PATH = re.compile(r"(?i)(?:\bmm-[a-z0-9-]+\b|01-Models[/\\])")


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


def safe_relative_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"path must be relative: {raw_path}")
    resolved_base = base.resolve()
    resolved_path = (base / path).resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise ValueError(f"path escapes campaign directory: {raw_path}")
    return resolved_path


def packet_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(UNIT_HEADING.finditer(text))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[match.start() : end]))
    return blocks


def check_campaign(base: Path) -> dict[str, Any]:
    base = base.resolve()
    errors: list[str] = []
    handoff = read_json(base / "16-independent-review-handoff-manifest.json", "handoff manifest", errors)
    integrity = read_json(base / "28-independent-review-integrity-manifest.json", "integrity manifest", errors)
    if handoff is None or integrity is None:
        return {"base": str(base), "passed": False, "errors": errors}

    handoff_waves = handoff.get("waves")
    integrity_waves = integrity.get("waves")
    if not isinstance(handoff_waves, list) or not isinstance(integrity_waves, list):
        errors.append("handoff and integrity manifests must contain waves lists")
        return {"base": str(base), "passed": False, "errors": errors}
    integrity_by_wave: dict[str, dict[str, Any]] = {}
    for entry in integrity_waves:
        if not isinstance(entry, dict) or not non_empty_string(entry.get("wave")):
            errors.append("integrity manifest contains invalid wave definition")
            continue
        wave = entry["wave"]
        if wave in integrity_by_wave:
            errors.append(f"integrity manifest contains duplicate wave {wave}")
        integrity_by_wave[wave] = entry

    results: list[dict[str, Any]] = []
    all_units = 0
    for entry in handoff_waves:
        if not isinstance(entry, dict):
            errors.append("handoff contains non-object wave definition")
            continue
        wave = entry.get("wave")
        expected_ids = entry.get("unit_ids")
        raw_packet = entry.get("blind_packet")
        if not non_empty_string(wave) or not isinstance(expected_ids, list) or not all(non_empty_string(item) for item in expected_ids):
            errors.append("handoff contains invalid wave/unit definition")
            continue
        if len(expected_ids) != len(set(expected_ids)):
            errors.append(f"{wave}: handoff has duplicate unit_ids")
            continue
        if not non_empty_string(raw_packet):
            errors.append(f"{wave}: missing blind_packet")
            continue
        integrity_entry = integrity_by_wave.get(wave)
        if integrity_entry is None:
            errors.append(f"{wave}: missing from integrity manifest")
            continue
        if integrity_entry.get("blind_packet_path") != raw_packet:
            errors.append(f"{wave}: handoff blind_packet differs from integrity blind_packet_path")
        try:
            packet_path = safe_relative_path(base, raw_packet)
        except ValueError as exc:
            errors.append(f"{wave}: {exc}")
            continue
        if not packet_path.is_file():
            errors.append(f"{wave}: blind packet missing: {raw_packet}")
            continue
        text = packet_path.read_text(encoding="utf-8")
        blocks = packet_blocks(text)
        found_ids = [unit_id for unit_id, _ in blocks]
        duplicate_ids = sorted(unit_id for unit_id, count in Counter(found_ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"{wave}: duplicate packet unit_ids: {', '.join(duplicate_ids)}")
        if set(found_ids) != set(expected_ids) or len(found_ids) != len(expected_ids):
            errors.append(f"{wave}: packet unit_ids do not exactly match handoff unit_ids")
        missing_answer_markers: list[str] = []
        for unit_id, block in blocks:
            if len(ANSWER_A.findall(block)) != 1 or len(ANSWER_B.findall(block)) != 1:
                missing_answer_markers.append(unit_id)
        if missing_answer_markers:
            errors.append(f"{wave}: each unit must contain exactly one A and B answer marker; invalid: {', '.join(missing_answer_markers)}")
        assignment_hits = [match.group(0) for match in DIRECT_ASSIGNMENT.finditer(text)]
        card_identity_hits = [match.group(0) for match in MODEL_ID_OR_CARD_PATH.finditer(text)]
        if assignment_hits:
            errors.append(f"{wave}: direct A/B-to-condition assignment leak: {assignment_hits[0]!r}")
        if card_identity_hits:
            errors.append(f"{wave}: model ID or card-path leak: {card_identity_hits[0]!r}")
        results.append(
            {
                "wave": wave,
                "expected_units": len(expected_ids),
                "packet_units": len(found_ids),
                "direct_assignment_leaks": len(assignment_hits),
                "card_identity_leaks": len(card_identity_hits),
            }
        )
        all_units += len(expected_ids)

    handoff_names = {entry.get("wave") for entry in handoff_waves if isinstance(entry, dict)}
    if handoff_names != set(integrity_by_wave):
        errors.append("handoff and integrity manifest wave sets differ")
    return {"base": str(base), "waves_checked": len(results), "units_checked": all_units, "results": results, "passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check blind packet unit coverage and obvious mapping leakage.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Campaign evaluation directory.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    args = parser.parse_args()
    report = check_campaign(args.base)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"PASS: {report['waves_checked']} blind packets expose exactly {report['units_checked']} declared units with paired A/B answers and no direct identity/assignment leak.")
    else:
        print("FAIL: blind-packet structure check")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
