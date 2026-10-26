from __future__ import annotations

import csv
import json

from research.phase1.target_attractor_pilot.sae_causal.analyze_interchange_v3 import (
    _load_scores,
)
from research.phase1.target_attractor_pilot.sae_causal.subspace_interchange_blind_v4 import (
    build_packet,
)


def test_blind_packet_deduplicates_visible_text_without_leaking_metadata(tmp_path) -> None:
    source = tmp_path / "subspace_trajectories.jsonl"
    rows = []
    for index in range(60):
        text_index = index // 2
        rows.append({
            "record_id": f"source-feature-method-direction-{index:02d}",
            "prompt": f"prompt {text_index}",
            "response": f"response {text_index}",
            "condition": "subspace_interchange",
            "method": "sae_random",
            "axis": "specificity",
            "source_class": "high",
            "target_class": "low",
        })
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "blind"
    result = build_packet(source, output)
    packet = [
        json.loads(line)
        for line in (output / "packet.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mapping = json.loads((output / "private_mapping.json").read_text(encoding="utf-8"))

    assert result == {
        "status": "awaiting_blind_scores",
        "records": 60,
        "unique_texts": 30,
    }
    assert len(packet) == len(mapping) == 30
    assert len({row["item_id"] for row in packet}) == 30
    assert {row["item_id"] for row in packet} == set(mapping)
    assert all(set(row) == {"item_id", "prompt", "response"} for row in packet)
    assert all(len(record_ids) == 2 for record_ids in mapping.values())
    assert sorted(record_id for ids in mapping.values() for record_id in ids) == sorted(
        row["record_id"] for row in rows
    )
    assert all(
        hidden not in row["item_id"]
        for row in packet
        for hidden in (
            "condition", "method", "axis", "direction", "source", "feature",
            "subspace", "interchange",
        )
    )

    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id", "specificity_0_4", "independence_0_4",
            "non_template_0_4", "usefulness_0_4", "validity_0_4",
            "exclude_0_1", "notes",
        ])
        writer.writerows(
            [[row["item_id"], 3, 2, 1, 3, 3, 0, ""] for row in packet]
        )
    expanded = _load_scores(output / "scores.csv", output / "private_mapping.json")
    assert len(expanded) == 60
    assert expanded[rows[0]["record_id"]] == expanded[rows[1]["record_id"]]
