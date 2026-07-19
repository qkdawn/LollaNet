"""Prepare a stratified, leakage-blinded sample for LLM teacher labeling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

try:
    from .download_free_pilot import external_output
except ImportError:
    from download_free_pilot import external_output


FORBIDDEN_PACKET_KEYS = {
    "cik", "issuer_name", "ticker_at_collection", "route_labels", "exploratory_outcome",
    "outcome", "entry_date", "exit_date", "raw_price_return",
}


def _packet(record: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(name for name, value in (record["financial_features"] | record["market_state"]).items() if value is None)
    return {
        "episode_id": record["episode_id"],
        "task": "Select and order mental models for a point-in-time value-investment analysis of a US non-financial issuer.",
        "filing_form": record["filing"]["form"],
        "sector_code": record["market_state"].get("sector_code"),
        "financial_features": record["financial_features"],
        "market_state": record["market_state"],
        "missing_fields": missing,
        "source_quality": {
            "financials": "SEC EDGAR Company Facts filtered to the filing availability boundary",
            "market": "exploratory Yahoo adjusted prices using an unverified current-ticker historical mapping",
            "delisting_return_coverage": False,
        },
    }


def _stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    year = record["availability"]["public_available_at"][:4]
    sector = str(record["market_state"].get("sector_code") or "unknown")
    # Existing labels are used only to balance the sample and are never copied into teacher packets.
    route_pattern = "+".join(sorted(record["route_labels"].get("triggered_rules", []))) or "base_only"
    return year, sector, route_pattern


def prepare(input_path: Path, output: Path, sample_size: int = 1000, batch_size: int = 25, seed: int = 20260719) -> dict[str, Any]:
    input_path, output = external_output(input_path), external_output(output)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    eligible = [record for record in records if 2017 <= int(record["availability"]["public_available_at"][:4]) <= 2023]
    if sample_size > len(eligible):
        raise ValueError("sample size exceeds eligible records")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        groups[_stratum(record)].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    ordered_groups = sorted(groups)
    while len(selected) < sample_size:
        progressed = False
        for key in ordered_groups:
            if groups[key] and len(selected) < sample_size:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    packets = [_packet(record) for record in selected]
    for packet in packets:
        serialized = json.dumps(packet)
        if any(f'"{key}"' in serialized for key in FORBIDDEN_PACKET_KEYS):
            raise ValueError(f"teacher packet leaked a forbidden key: {packet['episode_id']}")
    output.mkdir(parents=True, exist_ok=True)
    batches = output / "batches"
    batches.mkdir(exist_ok=True)
    for index in range(0, len(packets), batch_size):
        destination = batches / f"batch_{index // batch_size:03d}.json"
        destination.write_text(json.dumps(packets[index:index + batch_size], indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "teacher_sample_version": "llm-teacher-sample-v1",
        "sample_size": len(packets),
        "batch_size": batch_size,
        "batch_count": (len(packets) + batch_size - 1) // batch_size,
        "seed": seed,
        "eligible_years": [2017, 2023],
        "stratification": ["availability_year", "SIC two-digit sector", "hidden rule-pattern for sample balance only"],
        "teacher_visible_fields": sorted(packets[0]),
        "forbidden_teacher_inputs": sorted(FORBIDDEN_PACKET_KEYS),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, args.sample_size, args.batch_size)
    print(f"Prepared {manifest['sample_size']} blinded teacher packets in {manifest['batch_count']} batches")


if __name__ == "__main__":
    main()
