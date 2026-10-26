from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.phase1.mechanism_questions_v1 import relevance_blind_model_score_cli_v1
from research.phase1.mechanism_questions_v1.audit_relevance_blind_model_scores_v1 import (
    build_relevance_blind_vs_hinted_audit,
)
from research.phase1.mechanism_questions_v1.relevance_blind_model_score_chain_v1 import (
    BLIND_SCORE_MANIFEST_FILENAME,
    build_relevance_blind_model_score_packets,
    finalize_relevance_blind_model_scores,
    make_relevance_blind_model_score_label,
    relevance_blind_model_score_ledger_entry,
    validate_relevance_blind_packet,
    verify_relevance_blind_model_score_manifest,
)
from research.phase1.mechanism_questions_v1.question_set_v1 import read_jsonl, write_jsonl
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import make_teacher_identity
from tests.test_model_score_chain_v1 import _finalize as finalize_hinted
from tests.test_model_score_chain_v1 import _prepared as prepare_hinted


def _blind_labels(packet: dict, *, scores_a: dict[str, float], scores_b: dict[str, float]) -> list[dict]:
    identity_a = make_teacher_identity(provider="blind-a", model="blind-model-a", deployment="run-a", version="v1")
    identity_b = make_teacher_identity(provider="blind-b", model="blind-model-b", deployment="run-b", version="v1")
    return [
        make_relevance_blind_model_score_label(
            packet=packet, slot="a", teacher_identity=identity_a, scores=scores_a,
            confidence=0.9, rationale="Scored every model using only the blind packet.",
            raw_response={"scores": scores_a}, api_response_id="blind-response-a", run_id="blind-run-a",
        ),
        make_relevance_blind_model_score_label(
            packet=packet, slot="b", teacher_identity=identity_b, scores=scores_b,
            confidence=0.8, rationale="Scored every model independently using only the blind packet.",
            raw_response={"scores": scores_b}, api_response_id="blind-response-b", run_id="blind-run-b",
        ),
    ]


def _prepare_blind(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    dataset_root, hinted_output, _ = prepare_hinted(tmp_path)
    source_manifest = hinted_output.parent / "teacher_chain_manifest.json"
    blind_output = hinted_output.parent / "relevance_blind_model_scores"
    prepared = build_relevance_blind_model_score_packets(
        source_manifest_path=source_manifest, dataset_root=dataset_root,
        output_dir=blind_output, max_score_gap=1.0,
    )
    return dataset_root, source_manifest, blind_output, prepared


def _finalize_blind(dataset_root: Path, source_manifest: Path, blind_output: Path, prepared: dict, scores_a: dict[str, float], scores_b: dict[str, float]) -> dict:
    labels = _blind_labels(prepared["packets"][0], scores_a=scores_a, scores_b=scores_b)
    labels_path = blind_output / "teacher_relevance_blind_model_scores.jsonl"
    ledger_path = blind_output / "relevance_blind_model_score_audit_ledger.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(ledger_path, [relevance_blind_model_score_ledger_entry(label) for label in labels])
    return finalize_relevance_blind_model_scores(
        source_manifest_path=source_manifest, dataset_root=dataset_root,
        packets_path=prepared["packet_path"], labels_path=labels_path,
        audit_ledger_path=ledger_path, policy_path=prepared["policy_path"],
        prompt_contract_path=prepared["prompt_contract_path"], output_dir=blind_output,
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_relevance_blind_packet_contains_only_raw_facts_question_and_twenty_cards(tmp_path: Path) -> None:
    _, _, _, prepared = _prepare_blind(tmp_path)
    packet = prepared["packets"][0]
    validate_relevance_blind_packet(packet)
    assert set(packet["raw_facts"]) == {"facts", "factual_uncertainties", "structured_evidence"}
    assert isinstance(packet["question"], str) and packet["question"]
    assert len(packet["models"]) == len(packet["model_ids"]) == 20
    keys = _all_keys(packet)
    assert not {
        "task_relevance", "task_relevance_profile", "task_focus_microconcept_id",
        "microconcept_states", "fact_microconcept_states", "candidate_subgraph",
        "candidate_graph", "candidate_model_ids", "model_selection_contracts",
        "goal", "constraints", "case_id", "source_refs",
    } & keys
    assert packet["visibility_attestation"] == {
        "condition": "relevance_blind", "task_relevance_hidden": True,
        "microconcept_states_hidden": True, "candidate_graph_hidden": True,
        "selection_contracts_hidden": True,
    }


def test_relevance_blind_v1_fixes_gap_threshold_at_one(tmp_path: Path) -> None:
    dataset_root, hinted_output, _ = prepare_hinted(tmp_path)
    with pytest.raises(Exception, match="fixes max_score_gap at 1.0"):
        build_relevance_blind_model_score_packets(
            source_manifest_path=hinted_output.parent / "teacher_chain_manifest.json",
            dataset_root=dataset_root, output_dir=hinted_output.parent / "invalid-blind",
            max_score_gap=1.5,
        )


def test_relevance_blind_packet_validator_rejects_recursive_hint_leak(tmp_path: Path) -> None:
    _, _, _, prepared = _prepare_blind(tmp_path)
    leaked = deepcopy(prepared["packets"][0])
    leaked["raw_facts"]["facts"][0]["task_relevance"] = 3
    with pytest.raises(Exception, match="leaked prohibited fields"):
        validate_relevance_blind_packet(leaked)


def test_relevance_blind_dual_teacher_averages_gap_lte_one_and_masks_gap_gt_one(tmp_path: Path) -> None:
    dataset_root, source_manifest, blind_output, prepared = _prepare_blind(tmp_path)
    model_ids = prepared["model_ids"]
    scores_a = {model_id: 2.0 for model_id in model_ids}
    scores_b = {model_id: 2.0 for model_id in model_ids}
    close_model, disputed_model = model_ids[:2]
    scores_a[close_model], scores_b[close_model] = 3.0, 2.0
    scores_a[disputed_model], scores_b[disputed_model] = 4.0, 1.0
    manifest = _finalize_blind(dataset_root, source_manifest, blind_output, prepared, scores_a, scores_b)
    aggregate = read_jsonl(blind_output / "aggregated_relevance_blind_model_scores.jsonl")[0]
    assert aggregate["activation_targets"][close_model] == pytest.approx(0.625)
    assert aggregate["activation_loss_mask"][close_model] is True
    assert aggregate["activation_targets"][disputed_model] is None
    assert aggregate["activation_loss_mask"][disputed_model] is False
    assert aggregate["supervised_model_count"] == 19
    assert manifest["supervised_dimensions"] == 19
    verify_relevance_blind_model_score_manifest(
        blind_output / BLIND_SCORE_MANIFEST_FILENAME,
        dataset_root=dataset_root,
        expected_aggregate_path=blind_output / "aggregated_relevance_blind_model_scores.jsonl",
    )


def test_relevance_blind_cli_assembles_two_full_independent_teacher_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, blind_output, prepared = _prepare_blind(tmp_path)
    packet = prepared["packets"][0]
    scores = {model_id: 2 for model_id in packet["model_ids"]}
    raw_a = blind_output / "raw-a.jsonl"
    raw_b = blind_output / "raw-b.jsonl"
    write_jsonl(raw_a, [{"episode_id": packet["episode_id"], "scores": scores, "confidence": 0.8, "rationale": "Teacher A scored all models."}])
    write_jsonl(raw_b, [{"episode_id": packet["episode_id"], "scores": scores, "confidence": 0.7, "rationale": "Teacher B scored all models independently."}])
    labels_path = blind_output / "assembled-labels.jsonl"
    ledger_path = blind_output / "assembled-ledger.jsonl"
    monkeypatch.setattr("sys.argv", [
        "relevance_blind_model_score_cli_v1", "assemble",
        "--packets", str(prepared["packet_path"]), "--raw-a", str(raw_a), "--raw-b", str(raw_b),
        "--output-labels", str(labels_path), "--output-ledger", str(ledger_path),
        "--provider-a", "provider-a", "--model-a", "model-a", "--deployment-a", "dep-a", "--version-a", "v1", "--run-id-a", "run-a",
        "--provider-b", "provider-b", "--model-b", "model-b", "--deployment-b", "dep-b", "--version-b", "v1", "--run-id-b", "run-b",
    ])
    assert relevance_blind_model_score_cli_v1.main() == 0
    labels = read_jsonl(labels_path)
    assert len(labels) == 2
    assert {row["teacher_slot"] for row in labels} == {"a", "b"}
    assert all(set(row["model_scores"]) == set(packet["model_ids"]) for row in labels)


def test_relevance_blind_finalizer_rejects_non_independent_teachers(tmp_path: Path) -> None:
    dataset_root, source_manifest, blind_output, prepared = _prepare_blind(tmp_path)
    packet = prepared["packets"][0]
    scores = {model_id: 2.0 for model_id in packet["model_ids"]}
    identity = make_teacher_identity(provider="same", model="same", deployment="one", version="v1")
    labels = [
        make_relevance_blind_model_score_label(
            packet=packet, slot=slot, teacher_identity=identity, scores=scores,
            confidence=0.8, rationale="Independent-looking response.", raw_response={"scores": scores},
            api_response_id=f"response-{slot}", run_id=f"run-{slot}",
        )
        for slot in ("a", "b")
    ]
    labels_path = blind_output / "teacher_relevance_blind_model_scores.jsonl"
    ledger_path = blind_output / "relevance_blind_model_score_audit_ledger.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(ledger_path, [relevance_blind_model_score_ledger_entry(label) for label in labels])
    with pytest.raises(Exception, match="distinct provider/model"):
        finalize_relevance_blind_model_scores(
            source_manifest_path=source_manifest, dataset_root=dataset_root,
            packets_path=prepared["packet_path"], labels_path=labels_path,
            audit_ledger_path=ledger_path, policy_path=prepared["policy_path"],
            prompt_contract_path=prepared["prompt_contract_path"], output_dir=blind_output,
        )


def test_audit_compares_blind_and_hinted_scores_by_model_and_episode(tmp_path: Path) -> None:
    dataset_root, hinted_output, hinted_prepared = prepare_hinted(tmp_path)
    finalize_hinted(dataset_root, hinted_output, hinted_prepared)
    source_manifest = hinted_output.parent / "teacher_chain_manifest.json"
    blind_output = hinted_output.parent / "relevance_blind_model_scores"
    blind_prepared = build_relevance_blind_model_score_packets(
        source_manifest_path=source_manifest, dataset_root=dataset_root,
        output_dir=blind_output, max_score_gap=1.0,
    )
    model_ids = blind_prepared["model_ids"]
    scores_a = {model_id: 2.0 for model_id in model_ids}
    scores_b = {model_id: 2.0 for model_id in model_ids}
    changed_model, previously_disputed_model = model_ids[:2]
    scores_a[changed_model], scores_b[changed_model] = 4.0, 3.0
    scores_a[previously_disputed_model], scores_b[previously_disputed_model] = 2.0, 2.0
    _finalize_blind(dataset_root, source_manifest, blind_output, blind_prepared, scores_a, scores_b)

    result = build_relevance_blind_vs_hinted_audit(
        dataset_root=dataset_root,
        blind_manifest_path=blind_output / BLIND_SCORE_MANIFEST_FILENAME,
        hinted_manifest_path=hinted_output / "model_score_manifest.json",
        output_dir=hinted_output.parent / "relevance_blind_audit",
    )
    assert len(result["dimensions"]) == 20
    assert len(result["by_model"]) == 20
    assert len(result["by_episode"]) == 1
    changed = next(row for row in result["dimensions"] if row["model_id"] == changed_model)
    assert changed["mean_score_delta_blind_minus_hinted"] == pytest.approx(1.0)
    unmasked = next(row for row in result["dimensions"] if row["model_id"] == previously_disputed_model)
    assert unmasked["blind_supervised"] is True
    assert unmasked["hinted_supervised"] is False
    assert unmasked["mask_changed"] is True
    assert result["summary"]["mask_change_count"] == 1
    assert json.loads(result["paths"]["summary"].read_text(encoding="utf-8"))["dimension_count"] == 20
