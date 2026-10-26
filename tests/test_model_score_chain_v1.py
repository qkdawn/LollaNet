from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.phase1.mechanism_questions_v1.finalize_teacher_labels_v1 import finalize_teacher_labels
from research.phase1.mechanism_questions_v1.model_score_chain_v1 import (
    MODEL_SCORE_MANIFEST_FILENAME,
    build_model_score_packets,
    finalize_model_scores,
    make_model_score_label,
    model_score_ledger_entry,
    verify_model_score_manifest,
)
from research.phase1.mechanism_questions_v1.question_set_v1 import read_jsonl, write_jsonl
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import make_teacher_identity
from tests.test_llm_teacher_v1 import _write_chain


def _prepared(tmp_path: Path) -> tuple[Path, Path, dict]:
    paths = _write_chain(tmp_path / "dataset", review_status="accepted")
    finalize_teacher_labels(
        episodes_path=paths["episodes"],
        initial_labels_path=paths["initial"],
        review_labels_path=paths["review"],
        initial_audit_ledger_path=paths["initial_ledger"],
        review_audit_ledger_path=paths["review_ledger"],
        output_dir=paths["output"],
    )
    dataset_root = tmp_path / "dataset"
    output = paths["output"] / "model_scores"
    prepared = build_model_score_packets(
        source_manifest_path=paths["output"] / "teacher_chain_manifest.json",
        dataset_root=dataset_root,
        output_dir=output,
        max_score_gap=1.0,
    )
    return dataset_root, output, prepared


def _finalize(dataset_root: Path, output: Path, prepared: dict) -> dict:
    packet = prepared["packets"][0]
    scores_a = {model_id: 2.0 for model_id in packet["model_ids"]}
    scores_b = {model_id: 2.0 for model_id in packet["model_ids"]}
    close_model, disputed_model = packet["model_ids"][:2]
    scores_a[close_model], scores_b[close_model] = 3.0, 2.0
    scores_a[disputed_model], scores_b[disputed_model] = 4.0, 1.0
    identity_a = make_teacher_identity(provider="provider-a", model="score-model-a", deployment="run-a", version="v1")
    identity_b = make_teacher_identity(provider="provider-b", model="score-model-b", deployment="run-b", version="v1")
    labels = [
        make_model_score_label(
            packet=packet,
            slot="a",
            teacher_identity=identity_a,
            scores=scores_a,
            confidence=0.9,
            rationale="All models were scored independently against the fixed rubric.",
            raw_response={"scores": scores_a},
            api_response_id="response-a",
            run_id="run-a",
        ),
        make_model_score_label(
            packet=packet,
            slot="b",
            teacher_identity=identity_b,
            scores=scores_b,
            confidence=0.8,
            rationale="All models were scored independently against the fixed rubric.",
            raw_response={"scores": scores_b},
            api_response_id="response-b",
            run_id="run-b",
        ),
    ]
    labels_path = output / "teacher_model_scores.jsonl"
    ledger_path = output / "model_score_audit_ledger.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(ledger_path, [model_score_ledger_entry(label) for label in labels])
    return finalize_model_scores(
        source_manifest_path=output.parent / "teacher_chain_manifest.json",
        dataset_root=dataset_root,
        packets_path=prepared["packet_path"],
        labels_path=labels_path,
        audit_ledger_path=ledger_path,
        policy_path=prepared["policy_path"],
        prompt_contract_path=prepared["prompt_contract_path"],
        output_dir=output,
    )


def test_dense_model_score_chain_averages_close_scores_and_masks_wide_gap(tmp_path: Path) -> None:
    dataset_root, output, prepared = _prepared(tmp_path)
    manifest = _finalize(dataset_root, output, prepared)
    aggregate = read_jsonl(output / "aggregated_model_scores.jsonl")[0]
    close_model, disputed_model = prepared["model_ids"][:2]
    assert aggregate["activation_targets"][close_model] == pytest.approx(0.625)
    assert aggregate["activation_loss_mask"][close_model] is True
    assert aggregate["activation_loss_weight"][close_model] == pytest.approx(0.4)
    assert aggregate["activation_targets"][disputed_model] is None
    assert aggregate["activation_loss_mask"][disputed_model] is False
    assert aggregate["activation_loss_weight"][disputed_model] == 0.0
    assert aggregate["supervised_model_count"] == 19
    assert manifest["supervised_dimensions"] == 19
    verify_model_score_manifest(
        output / MODEL_SCORE_MANIFEST_FILENAME,
        dataset_root=dataset_root,
        expected_aggregate_path=output / "aggregated_model_scores.jsonl",
    )


def test_dense_model_score_manifest_rejects_aggregate_tamper(tmp_path: Path) -> None:
    dataset_root, output, prepared = _prepared(tmp_path)
    _finalize(dataset_root, output, prepared)
    aggregate_path = output / "aggregated_model_scores.jsonl"
    aggregate_path.write_text(aggregate_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(Exception, match="integrity mismatch"):
        verify_model_score_manifest(
            output / MODEL_SCORE_MANIFEST_FILENAME,
            dataset_root=dataset_root,
            expected_aggregate_path=aggregate_path,
        )


def test_dense_score_label_requires_all_twenty_models(tmp_path: Path) -> None:
    _, _, prepared = _prepared(tmp_path)
    packet = prepared["packets"][0]
    scores = {model_id: 2.0 for model_id in packet["model_ids"][:-1]}
    identity = make_teacher_identity(provider="provider-a", model="score-model-a", deployment="run-a", version="v1")
    with pytest.raises(Exception, match="cover exactly every"):
        make_model_score_label(
            packet=packet,
            slot="a",
            teacher_identity=identity,
            scores=scores,
            confidence=0.9,
            rationale="Incomplete on purpose.",
            raw_response={},
            api_response_id="response-a",
            run_id="run-a",
        )
