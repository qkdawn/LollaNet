#!/usr/bin/env python3
"""Check withheld blind-review randomization maps against declared wave units.

The checker runs only in the private campaign workspace. It verifies that every
pre-registered unit has one A/B mapping entry, that entries contain opposite
conditions, and that both label orientations occur. It emits only aggregate
orientation counts. It does not release per-unit mappings, score answers, or
change any model state.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16"


def read_json(path: Path, label: str, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None


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


def expected_conditions(wave: str) -> set[str]:
    if wave == "wave-04-contested-models":
        return {"contest", "alternative"}
    return {"full", "ablation"}


def check_campaign(base: Path) -> dict[str, Any]:
    base = base.resolve()
    errors: list[str] = []
    handoff = read_json(base / "16-independent-review-handoff-manifest.json", "handoff manifest", errors)
    integrity = read_json(base / "28-independent-review-integrity-manifest.json", "integrity manifest", errors)
    if not isinstance(handoff, dict) or not isinstance(integrity, dict):
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
    total_units = 0
    for entry in handoff_waves:
        if not isinstance(entry, dict):
            errors.append("handoff contains non-object wave definition")
            continue
        wave = entry.get("wave")
        unit_ids = entry.get("unit_ids")
        raw_mapping = entry.get("mapping_withheld_path")
        if not non_empty_string(wave) or not isinstance(unit_ids, list) or not all(non_empty_string(unit_id) for unit_id in unit_ids):
            errors.append("handoff contains invalid wave/unit definition")
            continue
        if len(unit_ids) != len(set(unit_ids)):
            errors.append(f"{wave}: handoff has duplicate unit_ids")
            continue
        if not non_empty_string(raw_mapping):
            errors.append(f"{wave}: missing mapping_withheld_path")
            continue
        integrity_entry = integrity_by_wave.get(wave)
        if integrity_entry is None:
            errors.append(f"{wave}: missing from integrity manifest")
            continue
        if integrity_entry.get("mapping_withheld_path") != raw_mapping:
            errors.append(f"{wave}: handoff mapping path differs from integrity manifest")
        try:
            mapping_path = safe_relative_path(base, raw_mapping)
        except ValueError as exc:
            errors.append(f"{wave}: {exc}")
            continue
        mapping = read_json(mapping_path, f"{wave} mapping", errors)
        if not isinstance(mapping, dict):
            continue
        if set(mapping) != set(unit_ids) or len(mapping) != len(unit_ids):
            errors.append(f"{wave}: mapping keys do not exactly match handoff unit_ids")
        pair = expected_conditions(wave)
        orientation: Counter[str] = Counter()
        for unit_id in unit_ids:
            value = mapping.get(unit_id)
            if not isinstance(value, dict) or set(value) != {"A", "B"}:
                errors.append(f"{wave}: {unit_id} mapping must contain exactly A and B")
                continue
            a, b = value["A"], value["B"]
            if not non_empty_string(a) or not non_empty_string(b):
                errors.append(f"{wave}: {unit_id} mapping labels must be non-empty strings")
                continue
            normalized = {a.casefold(), b.casefold()}
            if normalized != pair:
                errors.append(f"{wave}: {unit_id} must map opposite {sorted(pair)} conditions")
                continue
            if a.casefold() == b.casefold():
                errors.append(f"{wave}: {unit_id} maps A and B to the same condition")
                continue
            orientation[f"A={a.casefold()}"] += 1
        if len(unit_ids) > 1:
            required_orientations = {f"A={condition}" for condition in pair}
            if not required_orientations.issubset(set(orientation)):
                errors.append(f"{wave}: label orientation is one-sided; both A-condition orientations are required")
        results.append(
            {
                "wave": wave,
                "units": len(unit_ids),
                "condition_pair": sorted(pair),
                "a_label_orientation_counts": dict(sorted(orientation.items())),
            }
        )
        total_units += len(unit_ids)

    handoff_names = {entry.get("wave") for entry in handoff_waves if isinstance(entry, dict)}
    if handoff_names != set(integrity_by_wave):
        errors.append("handoff and integrity manifest wave sets differ")
    return {"base": str(base), "waves_checked": len(results), "units_checked": total_units, "results": results, "passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check private blind randomization maps against pre-registered unit coverage.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Campaign evaluation directory.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable aggregate report.")
    args = parser.parse_args()
    report = check_campaign(args.base)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"PASS: {report['waves_checked']} withheld maps cover exactly {report['units_checked']} units with opposite, non-one-sided A/B conditions.")
    else:
        print("FAIL: blind-mapping consistency check")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
