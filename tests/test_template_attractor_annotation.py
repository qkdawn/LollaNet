from __future__ import annotations

import csv
import json

from research.phase1.template_attractor_pilot.annotation import audit_annotation_delivery, build_annotation_delivery
from research.phase1.template_attractor_pilot.natural_trajectories import (
    _generation_seed, _ordered_present_records, _record_id,
    apply_existing_release_gate, load_natural_cases,
)


def test_blind_annotation_delivery_hides_provenance(tmp_path) -> None:
    records = [
        {
            "record_id": f"record-{index}", "case_id": f"case-{index // 2}",
            "split": "train", "seed": index, "prompt": f"prompt {index}",
            "response": f"response {index}", "diagnostic_rule_metrics": {"score": index},
        }
        for index in range(6)
    ]
    manifest = build_annotation_delivery(records, tmp_path)
    assert manifest["status"] == "ready_for_two_independent_annotators"
    assert audit_annotation_delivery(tmp_path, 6)["status"] == "passed"
    packet_a = [json.loads(line) for line in (tmp_path / "annotator_a_packet.jsonl").read_text(encoding="utf-8").splitlines()]
    packet_b = [json.loads(line) for line in (tmp_path / "annotator_b_packet.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == {"item_id", "prompt", "response"} for row in packet_a + packet_b)
    assert [row["item_id"] for row in packet_a] != [row["item_id"] for row in packet_b]


def test_score_templates_cover_each_blind_item(tmp_path) -> None:
    records = [{"record_id": f"r-{i}", "prompt": "p", "response": "r"} for i in range(4)]
    build_annotation_delivery(records, tmp_path)
    for packet_id in ("annotator_a", "annotator_b"):
        with (tmp_path / f"{packet_id}_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert all(row["formulaic_0_4"] == "" for row in rows)


def test_natural_trajectory_ids_and_seeds_are_deterministic() -> None:
    assert _record_id("ta-001", 1) == "ta-001__sample_01"
    assert _generation_seed(0, 0) != _generation_seed(0, 1)
    assert _generation_seed(0, 0) != _generation_seed(1, 0)


def test_resume_checkpoint_keeps_existing_records_in_expected_order() -> None:
    expected = ["case-a__sample_00", "case-a__sample_01", "case-b__sample_00"]
    records = {
        "case-b__sample_00": {"record_id": "case-b__sample_00"},
        "case-a__sample_00": {"record_id": "case-a__sample_00"},
    }
    assert [row["record_id"] for row in _ordered_present_records(expected, records)] == [
        "case-a__sample_00", "case-b__sample_00",
    ]


def test_existing_release_gate_blocks_truncated_run(tmp_path) -> None:
    (tmp_path / "trajectory_audit.json").write_text(
        json.dumps({"response_tokens": {"truncation_rate": 0.75}}), encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"status": "awaiting_independent_blind_annotations", "artifacts": {}}), encoding="utf-8",
    )
    result = apply_existing_release_gate(tmp_path)
    assert result["status"] == "blocked_excessive_truncation"
    assert not result["annotation_release_gate"]["passed"]
    assert result["artifacts"]["annotation_manifest"]["legacy_packet_files_must_not_be_scored"]


def test_external_natural_casebank_is_validated(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    rows = [
        {"case_id": f"c-{index}", "split": split, "domain": "test", "prompt": f"question {index}"}
        for index, split in enumerate(["train"] * 4 + ["development"] * 4 + ["evaluation"] * 4)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cases = load_natural_cases(path)
    assert len(cases) == 12
    assert cases[0].anchors == ()
