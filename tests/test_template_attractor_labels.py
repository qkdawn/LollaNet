from __future__ import annotations

import csv
import json

from research.phase1.template_attractor_pilot.annotation import build_annotation_delivery
from research.phase1.template_attractor_pilot.labels import (
    prepare_consensus_labels, prepare_model_assisted_labels,
)


def _records() -> list[dict[str, object]]:
    return [
        {"record_id": f"r-{i}", "case_id": f"c-{i // 2}", "split": "train", "prompt": f"p{i}", "response": f"r{i}"}
        for i in range(6)
    ]


def _write_scores(path, formulaic: list[float]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row, score in zip(rows, formulaic):
        row.update({
            "formulaic_0_4": str(score), "specificity_0_4": "2",
            "usefulness_0_4": "3", "validity_0_4": "3", "exclude_0_1": "0",
        })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_incomplete_scores_do_not_unblind_or_fit(tmp_path) -> None:
    records = _records()
    annotation = tmp_path / "annotation"
    build_annotation_delivery(records, annotation)
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    result = prepare_consensus_labels(annotation, trajectories, tmp_path / "labels")
    assert result["status"] == "awaiting_complete_independent_annotations"
    assert result["missing_score_counts"] == {"annotator_a": 6, "annotator_b": 6}


def test_large_disagreement_creates_blind_adjudication_packet(tmp_path) -> None:
    records = _records()
    annotation = tmp_path / "annotation"
    build_annotation_delivery(records, annotation)
    _write_scores(annotation / "annotator_a_scores.csv", [0, 1, 2, 3, 4, 1])
    _write_scores(annotation / "annotator_b_scores.csv", [4, 1, 2, 3, 4, 1])
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    result = prepare_consensus_labels(annotation, trajectories, tmp_path / "labels")
    assert result["status"] == "awaiting_independent_adjudication"
    assert result["agreement"]["adjudication_count"] >= 1
    packet = (tmp_path / "labels" / "adjudicator_packet.jsonl").read_text(encoding="utf-8")
    assert "case_id" not in packet


def test_model_assisted_labels_are_discovery_only(tmp_path) -> None:
    records = _records()
    annotation = tmp_path / "annotation"
    build_annotation_delivery(records, annotation)
    scores = annotation / "annotator_a_scores.csv"
    _write_scores(scores, [0, 1, 3, 4, 1, 3])
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    result = prepare_model_assisted_labels(
        annotation, trajectories, scores, tmp_path / "labels", "codex-5.6",
    )
    assert result["status"] == "blocked_insufficient_class_coverage"
    assert result["annotation_provenance"]["annotator_type"] == "language_model"
    assert not result["annotation_provenance"]["independent_human_annotation"]
    assert "Discovery-only" in result["evidence_boundary"]
