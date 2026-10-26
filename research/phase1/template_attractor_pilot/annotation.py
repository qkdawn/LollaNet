from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any


RUBRIC = """# Natural-Trajectory Blind Annotation Rubric

Rate the answer itself. Do not infer which model, condition, or intervention produced it.

## Formulaic expression (0-4, primary label)

- 0: Direct, object-specific organization; almost no replaceable assistant scaffolding.
- 1: Mostly specific; one minor generic transition or conventional phrase.
- 2: Mixed; useful specific content is organized partly through reusable scaffolding.
- 3: Strong template structure, generic headings, or broadly replaceable passages dominate.
- 4: The answer could largely be transferred to another prompt by replacing a few nouns.

Formal language, lists, or a common answer are not automatically formulaic. Judge whether the
organization and wording respond to this object, not whether a banned phrase appears.

## Specificity (0-4)

- 0: Almost entirely generic.
- 1: Mentions the object but gives little object-specific reasoning.
- 2: Some concrete reasoning, mixed with generic material.
- 3: Mostly tied to the prompt's variables and constraints.
- 4: Highly specific; the reasoning would not transfer unchanged to another prompt.

## Usefulness and validity (0-4 each)

Usefulness asks whether the answer advances the task. Validity asks whether its factual and logical
claims are supportable from the prompt and general knowledge. Do not reward novelty at their expense.

Mark `exclude=1` only for an empty answer, unreadable corruption, or a response too truncated to rate.
Annotators must work independently and must not compare scores before submission.
"""


FORBIDDEN_PACKET_FIELDS = {
    "case_id", "split", "domain", "sample_index", "seed", "diagnostic_rule_metrics",
    "prompt_tokens", "response_tokens", "generation_finished", "record_id",
}


def _alias(packet_id: str, record_id: str) -> str:
    return "item-" + sha256(f"{packet_id}:{record_id}".encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_template(path: Path, aliases: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "formulaic_0_4", "specificity_0_4", "usefulness_0_4", "validity_0_4", "exclude_0_1", "notes"])
        writer.writerows([[alias, "", "", "", "", "", ""] for alias in aliases])


def build_annotation_delivery(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_ids = [str(row["record_id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("annotation source contains duplicate record IDs")
    mappings: dict[str, dict[str, str]] = {}
    packet_artifacts: dict[str, dict[str, str]] = {}
    orders: list[list[str]] = []
    for packet_index, packet_id in enumerate(("annotator_a", "annotator_b")):
        order = list(record_ids)
        random.Random(2026072511 + packet_index).shuffle(order)
        orders.append(order)
        by_id = {str(row["record_id"]): row for row in records}
        packet_rows = []
        mapping: dict[str, str] = {}
        for source_id in order:
            alias = _alias(packet_id, source_id)
            mapping[alias] = source_id
            packet_rows.append({"item_id": alias, "prompt": by_id[source_id]["prompt"], "response": by_id[source_id]["response"]})
        packet_path = output_dir / f"{packet_id}_packet.jsonl"
        template_path = output_dir / f"{packet_id}_scores.csv"
        _write_jsonl(packet_path, packet_rows)
        _write_template(template_path, [str(row["item_id"]) for row in packet_rows])
        mappings[packet_id] = mapping
        packet_artifacts[packet_id] = {"packet": str(packet_path.resolve()), "scores": str(template_path.resolve())}
    if orders[0] == orders[1]:
        raise RuntimeError("independent annotation packets unexpectedly use identical order")
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    rubric_path = output_dir / "rubric.md"
    rubric_path.write_text(RUBRIC, encoding="utf-8")
    audit = audit_annotation_delivery(output_dir, len(records))
    manifest = {
        "type": "template_attractor_blind_annotation_delivery_v1",
        "status": "ready_for_two_independent_annotators",
        "record_count": len(records),
        "packets": packet_artifacts,
        "rubric": str(rubric_path.resolve()),
        "private_mapping": str(mapping_path.resolve()),
        "audit": audit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def audit_annotation_delivery(output_dir: Path, expected_count: int) -> dict[str, Any]:
    observed_aliases = []
    for packet_id in ("annotator_a", "annotator_b"):
        packet_path = output_dir / f"{packet_id}_packet.jsonl"
        rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected_count or len({row["item_id"] for row in rows}) != expected_count:
            raise ValueError(f"{packet_id} packet has invalid coverage")
        if any(FORBIDDEN_PACKET_FIELDS.intersection(row) for row in rows):
            raise ValueError(f"{packet_id} packet leaks forbidden metadata")
        if any(set(row) != {"item_id", "prompt", "response"} for row in rows):
            raise ValueError(f"{packet_id} packet has unexpected fields")
        observed_aliases.append([str(row["item_id"]) for row in rows])
    if observed_aliases[0] == observed_aliases[1]:
        raise ValueError("annotation packets do not have independent ordering and aliases")
    return {
        "status": "passed",
        "forbidden_metadata_absent": True,
        "independent_aliases_and_order": True,
        "packet_counts": [len(value) for value in observed_aliases],
    }

