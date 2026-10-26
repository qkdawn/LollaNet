from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from research.phase1.mechanism_questions_v1.finalize_teacher_labels_v1 import finalize_teacher_labels
from research.phase1.mechanism_questions_v1.llm_teacher_v1 import _ledger_entry, initial_label_from_payload, review_label_from_payload
from research.phase1.mechanism_questions_v1.question_set_v1 import build_episodes, write_jsonl
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import MANIFEST_FILENAME
from research.phase1.microconcept_router_v1.data import (
    IGNORE_INDEX,
    RouterDataError,
    _encode_example,
    load_route_data,
    serializable_predictions,
)
from research.phase1.microconcept_router_v1.model import MicroconceptRouter, router_loss
from tests.test_mechanism_questions_v1 import fact_pack


def _payload() -> dict:
    return {
        "confidence": 0.8,
        "abstain": False,
        "primary_model": "mm-p1-bottleneck",
        "auxiliary_models": ["mm-p1-optimization-constraints"],
        "auxiliary_model_roles": [{"model_id": "mm-p1-optimization-constraints", "role": "diagnostic"}],
        "near_miss_models": ["mm-p1-rate-control"],
        "near_miss_exclusion_items": [{"model_id": "mm-p1-rate-control", "reason": "Rate control is not established."}],
        "cited_microconcept_ids": ["constraint_pressure"],
        "cited_evidence_refs": ["fact-1"],
        "route_rationale": "Constraint pressure supports the selected route.",
        "challenge_assessment": "No active challenge invalidates the route.",
        "addressed_challenge_microconcept_ids": [],
        "next_verification": "Relieve the constraint and measure throughput.",
        "abstain_reason": "",
    }


def _audit(char: str, response_id: str) -> dict:
    return {
        "api_response_id": response_id,
        "request_fingerprint_sha256": char * 64,
        "raw_response_fingerprint_sha256": ("1" if char == "a" else "2") * 64,
    }


def _build_trusted_dataset(root: Path, review_statuses: tuple[str, str, str] = ("accepted", "accepted", "accepted")) -> Path:
    base = build_episodes([fact_pack()])[0]
    episodes = []
    for index, split in enumerate(("train", "train", "development"), start=1):
        episode = deepcopy(base)
        episode["episode_id"] = f"hermetic::{split}::{index}"
        episode["split"] = split
        episodes.append(episode)

    labels_dir = root / "labels" / "train_development"
    episodes_path = root / "episodes" / "train_development.jsonl"
    labels_dir.mkdir(parents=True)
    episodes_path.parent.mkdir(parents=True)
    initial_labels, review_labels, initial_ledger, review_ledger = [], [], [], []
    for index, episode in enumerate(episodes):
        initial_audit = _audit("a", f"initial-{index}")
        review_audit = _audit("b", f"review-{index}")
        initial = initial_label_from_payload(
            episode, _payload(), provider="openai", model="teacher-a", deployment="primary", teacher_version="2026-07-01", audit=initial_audit,
        )
        review = review_label_from_payload(
            episode,
            initial,
            {"review_status": review_statuses[index], "confidence": 0.75, "audit_rationale": "The exact route is legal or remains explicitly gated."},
            provider="anthropic",
            model="teacher-b",
            deployment="audit",
            teacher_version="2026-07-02",
            audit=review_audit,
        )
        initial_labels.append(initial)
        review_labels.append(review)
        initial_ledger.append(_ledger_entry(initial, mode="initial", audit=initial_audit))
        review_ledger.append(_ledger_entry(review, mode="review", audit=review_audit, initial=initial))

    initial_path = labels_dir / "initial.jsonl"
    review_path = labels_dir / "review.jsonl"
    initial_ledger_path = labels_dir / "initial-audit.jsonl"
    review_ledger_path = labels_dir / "review-audit.jsonl"
    write_jsonl(episodes_path, episodes)
    write_jsonl(initial_path, initial_labels)
    write_jsonl(review_path, review_labels)
    write_jsonl(initial_ledger_path, initial_ledger)
    write_jsonl(review_ledger_path, review_ledger)
    finalize_teacher_labels(
        episodes_path=episodes_path,
        initial_labels_path=initial_path,
        review_labels_path=review_path,
        initial_audit_ledger_path=initial_ledger_path,
        review_audit_ledger_path=review_ledger_path,
        output_dir=labels_dir,
    )
    return root


@pytest.fixture(scope="module")
def trusted_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_trusted_dataset(tmp_path_factory.mktemp("trusted-router-dataset"))


@pytest.fixture(scope="module")
def bundle(trusted_dataset: Path):
    return load_route_data(
        trusted_dataset,
        minimum_primary_support=1,
        minimum_auxiliary_support=1,
    )


def _batch(rows):
    return {
        "status": torch.tensor(np.stack([row.status for row in rows]), dtype=torch.long),
        "direction": torch.tensor(np.stack([row.direction for row in rows]), dtype=torch.long),
        "numeric": torch.tensor(np.stack([row.numeric for row in rows]), dtype=torch.float32),
        "abstain": torch.tensor([row.abstain for row in rows], dtype=torch.float32),
        "primary": torch.tensor([row.primary_target for row in rows], dtype=torch.long),
        "auxiliary": torch.tensor(np.stack([row.auxiliary_targets for row in rows]), dtype=torch.float32),
    }


def test_router_rejects_bare_accepted_review_without_manifest(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels" / "train_development"
    label_dir.mkdir(parents=True)
    (label_dir / "accepted_review.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RouterDataError, match="bare accepted_review.jsonl is forbidden"):
        load_route_data(tmp_path, minimum_primary_support=1, minimum_auxiliary_support=1)


@pytest.mark.parametrize("relative_path", [
    Path("labels/train_development/accepted_review.jsonl"),
    Path("labels/train_development/review-audit.jsonl"),
    Path("episodes/train_development.jsonl"),
])
def test_router_rejects_any_manifest_bound_file_hash_change(trusted_dataset: Path, tmp_path: Path, relative_path: Path) -> None:
    copied = tmp_path / "dataset"
    shutil.copytree(trusted_dataset, copied)
    target = copied / relative_path
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RouterDataError, match="manifest (hash|byte count) mismatch"):
        load_route_data(copied, minimum_primary_support=1, minimum_auxiliary_support=1)


def test_router_rejects_manifest_metadata_tampering(trusted_dataset: Path, tmp_path: Path) -> None:
    copied = tmp_path / "dataset"
    shutil.copytree(trusted_dataset, copied)
    manifest_path = copied / "labels" / "train_development" / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted_episode_ids"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RouterDataError, match="accepted_episode_ids"):
        load_route_data(copied, minimum_primary_support=1, minimum_auxiliary_support=1)


def test_verified_load_uses_train_and_development_only(bundle) -> None:
    assert bundle.audit["requested_splits"] == ["train", "development"]
    assert bundle.audit["counts"] == {"train": 2, "development": 1, "sealed_test": 0}
    assert bundle.audit["input_label_consistency_enforced"] is True
    assert bundle.sealed_test == ()
    assert "train_development" in bundle.audit["teacher_chain_manifests"]
    assert set(bundle.audit["primary_supported_models"]) == set(bundle.support.primary_models)
    assert set(bundle.audit["auxiliary_supported_models"]) == set(bundle.support.auxiliary_models)


def test_review_required_rows_remain_in_chain_but_are_excluded_from_supervision(tmp_path: Path) -> None:
    dataset = _build_trusted_dataset(
        tmp_path / "partial-acceptance",
        review_statuses=("accepted", "review_required", "accepted"),
    )
    bundle = load_route_data(dataset, minimum_primary_support=1, minimum_auxiliary_support=1)
    assert len(bundle.train) == 1
    assert len(bundle.development) == 1
    assert bundle.audit["teacher_chain_episode_counts"] == {"train": 2, "development": 1, "sealed_test": 0}
    assert bundle.audit["accepted_supervision_counts"] == {"train": 1, "development": 1, "sealed_test": 0}
    assert bundle.audit["review_required_excluded_counts"] == {"train": 1, "development": 0, "sealed_test": 0}
    assert bundle.audit["review_required_excluded_total"] == 1


def test_model_masks_all_illegal_microconcept_edges(bundle) -> None:
    torch.manual_seed(7)
    router = MicroconceptRouter(bundle.graph, bundle.support, len(bundle.direction_vocab)).eval()
    output = router(_batch(bundle.train[:2]))
    illegal = ~router.edge_mask.unsqueeze(0).expand_as(output["attention"])
    assert torch.equal(output["attention"][illegal], torch.zeros_like(output["attention"][illegal]))
    assert output["primary"].shape == (2, len(bundle.support.primary_models))
    assert output["auxiliary"].shape == (2, len(bundle.support.auxiliary_models))


def test_development_unknown_direction_maps_to_unk(bundle) -> None:
    episode = {
        "episode_id": "synthetic-unknown-direction",
        "microconcept_states": [
            {
                "microconcept_id": concept_id,
                "status": "unknown",
                "intensity": 0,
                "direction": "direction_only_in_development",
                "temporal_persistence": 0,
                "task_relevance": 0,
            }
            for concept_id in bundle.graph.concept_ids
        ],
    }
    label = {"abstain": True, "primary_model": None, "auxiliary_models": []}
    encoded = _encode_example(episode, label, "development", bundle.graph, bundle.support, bundle.direction_vocab)
    assert set(encoded.direction.tolist()) == {bundle.direction_vocab.index("<UNK>")}


def test_predictions_enforce_abstain_and_auxiliary_limit(bundle) -> None:
    rows = bundle.development[:1] + bundle.train[:1]
    output = {
        "primary": np.array([[0.9] * len(bundle.support.primary_models), [0.9] * len(bundle.support.primary_models)]),
        "auxiliary": np.ones((2, len(bundle.support.auxiliary_models))),
        "abstain": np.array([0.9, 0.1]),
    }
    predictions = serializable_predictions(rows, output, bundle.support, auxiliary_threshold=0.5, abstain_threshold=0.5)
    assert predictions[0]["abstain"] is True
    assert predictions[0]["primary_model"] is None
    assert predictions[0]["auxiliary_models"] == []
    assert predictions[1]["abstain"] is False
    assert predictions[1]["primary_model"] in bundle.support.primary_models
    assert len(predictions[1]["auxiliary_models"]) <= 2


def test_malformed_microconcept_state_fails_fast(bundle) -> None:
    episode = {
        "episode_id": "synthetic-invalid-state",
        "microconcept_states": [
            {
                "microconcept_id": concept_id,
                "status": "unknown",
                "intensity": 0,
                "direction": "unknown",
                "temporal_persistence": 0,
                "task_relevance": 0,
            }
            for concept_id in bundle.graph.concept_ids[:-1]
        ],
    }
    label = {"abstain": True, "primary_model": None, "auxiliary_models": []}
    with pytest.raises(RouterDataError, match="fixed microconcepts"):
        _encode_example(episode, label, "train", bundle.graph, bundle.support, bundle.direction_vocab)


def test_primary_mask_without_supported_examples_has_finite_loss(bundle) -> None:
    torch.manual_seed(9)
    router = MicroconceptRouter(bundle.graph, bundle.support, len(bundle.direction_vocab))
    batch = _batch(bundle.train[:2])
    batch["primary"][:] = IGNORE_INDEX
    output = router(batch)
    loss, metrics = router_loss(output, batch)
    assert torch.isfinite(loss)
    assert metrics["primary"] == 0.0
