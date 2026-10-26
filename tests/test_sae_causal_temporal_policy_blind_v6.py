from __future__ import annotations

import json

from research.phase1.target_attractor_pilot.sae_causal.temporal_policy_blind_v6 import (
    build_packet,
)


def test_temporal_blind_packet_deduplicates_visible_text(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [
        {"record_id": f"r{i}", "prompt": "p", "response": "same" if i < 2 else f"x{i}"}
        for i in range(76)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = build_packet(source, tmp_path / "blind")
    assert result["records"] == 76
    assert result["unique_texts"] == 75
    mapping = json.loads((tmp_path / "blind" / "private_mapping.json").read_text())
    assert sorted(len(ids) for ids in mapping.values())[-1] == 2
