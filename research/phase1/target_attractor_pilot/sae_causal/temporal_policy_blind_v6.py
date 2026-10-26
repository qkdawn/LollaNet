from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from .intervention_blind_v2 import RUBRIC


EXPECTED_RECORDS = 76


def _alias(prompt: str, response: str) -> str:
    visible = json.dumps([prompt, response], ensure_ascii=False, separators=(",", ":"))
    digest = sha256(f"temporal-policy-blind-v6:{visible}".encode()).hexdigest()
    return f"blind-{digest[:16]}"


def build_packet(source: Path, output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["record_id"]) for row in rows]
    if len(rows) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
        raise ValueError("temporal-policy smoke must contain 76 unique records")
    items: dict[str, dict[str, str]] = {}
    mapping: dict[str, list[str]] = {}
    visible_by_alias: dict[str, tuple[str, str]] = {}
    for row in rows:
        prompt, response = str(row["prompt"]), str(row["response"])
        if not response.strip():
            raise ValueError("temporal-policy smoke contains an empty response")
        item_id = _alias(prompt, response)
        visible = (prompt, response)
        if item_id in visible_by_alias and visible_by_alias[item_id] != visible:
            raise ValueError("blind alias collision")
        visible_by_alias[item_id] = visible
        items.setdefault(item_id, {"item_id": item_id, "prompt": prompt, "response": response})
        mapping.setdefault(item_id, []).append(str(row["record_id"]))
    packet = list(items.values())
    random.Random(2026080607).shuffle(packet)
    ordered_mapping = {row["item_id"]: mapping[row["item_id"]] for row in packet}
    output.mkdir(parents=True, exist_ok=True)
    (output / "packet.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    (output / "private_mapping.json").write_text(
        json.dumps(ordered_mapping, indent=2), encoding="utf-8"
    )
    (output / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id", "specificity_0_4", "independence_0_4", "non_template_0_4",
            "usefulness_0_4", "validity_0_4", "exclude_0_1", "notes",
        ])
        writer.writerows([[row["item_id"], "", "", "", "", "", "", ""] for row in packet])
    return {
        "status": "awaiting_blind_scores",
        "records": len(rows),
        "unique_texts": len(packet),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_packet(args.source, args.output))


if __name__ == "__main__":
    main()
