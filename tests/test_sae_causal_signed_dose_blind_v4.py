from __future__ import annotations

import json

from research.phase1.target_attractor_pilot.sae_causal.signed_dose_blind_v4 import build_packet


def test_signed_dose_blind_packet_reuses_visible_text_aliases(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [{
        "record_id": f"axis-method-direction-alpha-source-{index}",
        "prompt": f"prompt {index // 2}", "response": f"response {index // 2}",
    } for index in range(300)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "blind"
    result = build_packet(source, output)
    packet = [json.loads(line) for line in (output / "packet.jsonl").read_text(encoding="utf-8").splitlines()]
    mapping = json.loads((output / "private_mapping.json").read_text(encoding="utf-8"))
    assert result["records"] == 300
    assert result["unique_texts"] == len(packet) == 150
    assert all(set(row) == {"item_id", "prompt", "response"} for row in packet)
    assert all(len(ids) == 2 for ids in mapping.values())
    assert len({record for ids in mapping.values() for record in ids}) == 300
    assert all(row["item_id"].startswith("blind-") for row in packet)
