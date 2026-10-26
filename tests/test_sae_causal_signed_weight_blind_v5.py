from __future__ import annotations

import json

from research.phase1.target_attractor_pilot.sae_causal.signed_weight_blind_v5 import (
    build_packet,
)


def test_signed_weight_blind_packet_is_bijective_and_reuses_visible_aliases(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [{
        "record_id": f"hidden-condition-{index}",
        "prompt": f"prompt {index // 2}",
        "response": f"response {index // 2}",
    } for index in range(40)]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "blind"
    result = build_packet(source, output)
    packet = [
        json.loads(line) for line in (output / "packet.jsonl").read_text().splitlines()
    ]
    mapping = json.loads((output / "private_mapping.json").read_text())
    assert result == {"status": "awaiting_blind_scores", "records": 40, "unique_texts": 20}
    assert all(set(row) == {"item_id", "prompt", "response"} for row in packet)
    assert all(row["item_id"].startswith("blind-") for row in packet)
    assert len({record for ids in mapping.values() for record in ids}) == 40
