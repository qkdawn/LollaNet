from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any


RUBRIC = """---
type: private_research_artifact
status: discovery_only
project: lollanet
date: 2026-07-26
---

# Intervention smoke blind-scoring rubric

Judge only the prompt and response. Do not infer model, feature, sign, strength, or condition.

Score in half-point increments from 0 to 4:

- specificity: 0 generic; 4 reasoning tightly tied to the prompt's actual variables.
- independence: 0 accepts framing; 4 reconstructs the problem and identifies a decisive test.
- non_template: 0 dominated by reusable assistant scaffolding; 4 object-specific organization and language.
- usefulness: 0 does not advance the task; 4 directly and substantially advances it.
- validity: 0 logically/factually unsupported; 4 well-supported and coherent.

Do not reward stock phrases or lists by themselves. Use exclude=1 only for empty, corrupt, or
unrateably truncated output. These scores are model-assisted and discovery-only.
"""


def _alias(record_id: str) -> str:
    return "causal-" + sha256(f"sae-causal-smoke-v2:{record_id}".encode()).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_packet(source: Path, output: Path) -> dict[str, Any]:
    rows = _read_jsonl(source)
    if len(rows) != 52 or len({row["record_id"] for row in rows}) != 52:
        raise ValueError("engineering smoke must contain 52 unique records")
    order = list(rows)
    random.Random(2026072709).shuffle(order)
    mapping = {_alias(str(row["record_id"])): str(row["record_id"]) for row in order}
    packet = [{
        "item_id": _alias(str(row["record_id"])),
        "prompt": row["prompt"],
        "response": row["response"],
    } for row in order]
    output.mkdir(parents=True, exist_ok=True)
    (output / "packet.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet), encoding="utf-8",
    )
    (output / "private_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (output / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id", "specificity_0_4", "independence_0_4", "non_template_0_4",
            "usefulness_0_4", "validity_0_4", "exclude_0_1", "notes",
        ])
        writer.writerows([[row["item_id"], "", "", "", "", "", "", ""] for row in packet])
    return {"status": "awaiting_blind_scores", "records": len(packet)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_packet(args.source, args.output))


if __name__ == "__main__":
    main()
