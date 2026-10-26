from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Scores were assigned to unique texts in first-appearance order in the blinded packet.
SCORES = {
    1: (2.0, 2.0, 0.0, 2.0, 3.0),
    2: (2.5, 2.0, 0.0, 2.5, 2.5),
    3: (2.5, 2.0, 0.0, 2.5, 3.0),
    4: (2.0, 2.0, 0.0, 2.0, 3.0),
    5: (2.5, 2.0, 0.0, 2.5, 3.0),
    6: (2.5, 2.5, 0.0, 2.5, 3.0),
    7: (2.0, 1.5, 0.0, 2.0, 2.5),
    8: (2.5, 2.0, 0.0, 2.5, 3.0),
    9: (2.5, 2.0, 0.0, 2.5, 2.5),
    10: (2.0, 1.5, 0.0, 2.0, 2.0),
    11: (2.5, 2.0, 0.0, 2.5, 3.0),
    12: (2.0, 1.5, 0.0, 2.0, 2.5),
    13: (2.5, 1.5, 0.0, 2.5, 2.5),
    14: (2.0, 1.5, 0.0, 2.0, 2.5),
    15: (2.5, 2.0, 0.0, 2.5, 2.5),
    16: (2.5, 2.5, 0.0, 2.5, 3.0),
    17: (2.5, 2.0, 0.0, 2.5, 3.0),
    18: (2.5, 2.0, 0.5, 2.5, 3.0),
    19: (2.5, 2.0, 0.0, 2.5, 2.5),
    20: (2.5, 2.0, 0.0, 2.5, 3.0),
    21: (2.0, 1.5, 0.0, 2.0, 2.5),
    22: (2.5, 2.0, 0.0, 2.5, 2.5),
    23: (2.5, 2.0, 0.0, 2.5, 3.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = [
        json.loads(line)
        for line in args.packet.read_text(encoding="utf-8").splitlines()
        if line
    ]
    unique: dict[str, int] = {}
    rows = []
    for item in packet:
        index = unique.setdefault(str(item["response"]), len(unique) + 1)
        score = SCORES[index]
        rows.append([
            item["item_id"],
            *score,
            0,
            "codex_model_assisted_discovery; unique_text_%02d" % index,
        ])
    if len(unique) != len(SCORES) or len(rows) != 36:
        raise ValueError("blind packet differs from the scored 23-text/36-record packet")
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
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
        writer.writerows(rows)
    print({"records": len(rows), "unique_texts": len(unique), "status": "complete"})


if __name__ == "__main__":
    main()
