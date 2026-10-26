from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.phase1.mechanism_questions_v1.adjudication_chain_v1 import (
    ADJUDICATION_MANIFEST_FILENAME,
    adjudication_ledger_entry,
    build_adjudication_packets,
    finalize_adjudication,
    make_adjudicator_label,
    verify_adjudication_manifest,
)
from research.phase1.mechanism_questions_v1.finalize_teacher_labels_v1 import finalize_teacher_labels
from research.phase1.mechanism_questions_v1.question_set_v1 import read_jsonl, write_jsonl
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import make_teacher_identity
from tests.test_llm_teacher_v1 import _write_chain


def _prepared(tmp_path: Path) -> tuple[Path, Path, dict]:
    paths = _write_chain(tmp_path / "dataset", review_status="review_required")
    finalize_teacher_labels(
        episodes_path=paths["episodes"], initial_labels_path=paths["initial"], review_labels_path=paths["review"],
        initial_audit_ledger_path=paths["initial_ledger"], review_audit_ledger_path=paths["review_ledger"], output_dir=paths["output"],
    )
    dataset_root = tmp_path / "dataset"
    parent = paths["output"] / "teacher_chain_manifest.json"
    child = paths["output"] / "adjudication"
    prepared = build_adjudication_packets(parent_manifest_path=parent, dataset_root=dataset_root, output_dir=child)
    return dataset_root, child, prepared


def _finalize_synthetic_child(dataset_root: Path, child: Path, prepared: dict) -> dict:
    packet = prepared["packets"][0]
    identity_a = make_teacher_identity(provider="provider-a", model="model-a", deployment="run-a", version="v1")
    identity_b = make_teacher_identity(provider="provider-b", model="model-b", deployment="run-b", version="v1")
    labels = [
        make_adjudicator_label(packet=packet, slot="a", teacher_identity=identity_a, confidence=0.9, abstain=True, primary_model=None, auxiliary_models=[], rationale="No defensible route.", raw_response={"slot": "a", "route": "abstain"}, api_response_id="response-a", run_id="run-a"),
        make_adjudicator_label(packet=packet, slot="b", teacher_identity=identity_b, confidence=0.8, abstain=True, primary_model=None, auxiliary_models=[], rationale="No defensible route.", raw_response={"slot": "b", "route": "abstain"}, api_response_id="response-b", run_id="run-b"),
    ]
    labels_path = child / "adjudicator_labels.jsonl"
    ledger_path = child / "adjudication_audit_ledger.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(ledger_path, [adjudication_ledger_entry(label) for label in labels])
    return finalize_adjudication(
        parent_manifest_path=child.parent / "teacher_chain_manifest.json",
        dataset_root=dataset_root,
        packets_path=prepared["packet_path"], labels_path=labels_path, audit_ledger_path=ledger_path,
        policy_path=prepared["policy_path"], prompt_contract_path=prepared["prompt_contract_path"], output_dir=child,
    )


def test_dual_consensus_child_chain_is_deterministic(tmp_path: Path) -> None:
    dataset_root, child, prepared = _prepared(tmp_path)
    packet = prepared["packets"][0]
    manifest = _finalize_synthetic_child(dataset_root, child, prepared)
    assert manifest["consensus_episode_ids"] == [packet["episode_id"]]
    assert len(read_jsonl(child / "accepted_adjudication.jsonl")) == 1
    verify_adjudication_manifest(child / ADJUDICATION_MANIFEST_FILENAME, dataset_root=dataset_root, expected_accepted_path=child / "accepted_adjudication.jsonl")


def test_child_manifest_rejects_parent_byte_tamper(tmp_path: Path) -> None:
    dataset_root, child, prepared = _prepared(tmp_path)
    _finalize_synthetic_child(dataset_root, child, prepared)
    parent = dataset_root / "labels" / "train_development" / "teacher_chain_manifest.json"
    original = parent.read_bytes()
    parent.write_bytes(original + b" ")
    with pytest.raises(Exception, match="parent manifest"):
        verify_adjudication_manifest(child / ADJUDICATION_MANIFEST_FILENAME, dataset_root=dataset_root, expected_accepted_path=child / "accepted_adjudication.jsonl")
