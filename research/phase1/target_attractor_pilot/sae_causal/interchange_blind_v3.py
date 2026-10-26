from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from .intervention_blind_v2 import RUBRIC


def _alias(record_id: str) -> str:
    return "interchange-" + sha256(
        f"sae-interchange-v3b:{record_id}".encode()
    ).hexdigest()[:16]


def build_packet(source: Path, output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 36 or len({row["record_id"] for row in rows}) != 36:
        raise ValueError("interchange smoke must contain 36 unique records")
    order = list(rows)
    random.Random(2026073017).shuffle(order)
    mapping = {_alias(str(row["record_id"])): str(row["record_id"]) for row in order}
    packet = [
        {
            "item_id": _alias(str(row["record_id"])),
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in order
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "packet.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    (output / "private_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    (output / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id",
            "specificity_0_4",
            "independence_0_4",
            "non_template_0_4",
            "usefulness_0_4",
            "validity_0_4",
            "exclude_0_1",
            "notes",
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
