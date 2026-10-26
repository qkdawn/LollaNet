from __future__ import annotations

import csv
import json

import pytest

from research.phase1.template_attractor_pilot.annotation import build_annotation_delivery
from research.phase1.template_attractor_pilot.annotation_app import ScoreStore, validate_score_payload


def _delivery(tmp_path):
    records = [
        {"record_id": f"r-{i}", "prompt": f"prompt {i}", "response": f"response {i}"}
        for i in range(4)
    ]
    return build_annotation_delivery(records, tmp_path)


def test_score_payload_requires_half_point_scale() -> None:
    valid = {
        "item_id": "item-1", "formulaic_0_4": 2.5, "specificity_0_4": 3,
        "usefulness_0_4": 3.5, "validity_0_4": 4, "exclude_0_1": 0, "notes": "ok",
    }
    assert validate_score_payload(valid)["formulaic_0_4"] == "2.5"
    with pytest.raises(ValueError, match="half-point"):
        validate_score_payload({**valid, "formulaic_0_4": 2.2})
    with pytest.raises(ValueError, match="0 or 1"):
        validate_score_payload({**valid, "exclude_0_1": 2})


def test_store_saves_atomically_and_resumes_progress(tmp_path) -> None:
    manifest = _delivery(tmp_path)
    packet = tmp_path / "annotator_a_packet.jsonl"
    scores = tmp_path / "annotator_a_scores.csv"
    rubric = tmp_path / "rubric.md"
    store = ScoreStore(packet, scores, rubric)
    first_id = store.item_ids[0]
    result = store.save({
        "item_id": first_id, "formulaic_0_4": 2.5, "specificity_0_4": 3,
        "usefulness_0_4": 3, "validity_0_4": 3.5, "exclude_0_1": 0, "notes": "saved",
    })
    assert result["completed_count"] == 1
    resumed = ScoreStore(packet, scores, rubric).state()
    assert resumed["completed_count"] == 1
    assert resumed["items"][0]["scores"]["notes"] == "saved"
    with scores.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in rows] == store.item_ids
    assert manifest["audit"]["forbidden_metadata_absent"]


def test_store_rejects_item_from_another_packet(tmp_path) -> None:
    _delivery(tmp_path)
    store = ScoreStore(
        tmp_path / "annotator_a_packet.jsonl", tmp_path / "annotator_a_scores.csv", tmp_path / "rubric.md",
    )
    with pytest.raises(ValueError, match="outside"):
        store.save({
            "item_id": "item-from-b", "formulaic_0_4": 2, "specificity_0_4": 2,
            "usefulness_0_4": 2, "validity_0_4": 2, "exclude_0_1": 0,
        })

