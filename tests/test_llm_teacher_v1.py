from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.phase1.mechanism_questions_v1.finalize_teacher_labels_v1 import finalize_teacher_labels
from research.phase1.mechanism_questions_v1.llm_teacher_v1 import (
    INITIAL_PROMPT_VERSION,
    REVIEW_PROMPT_VERSION,
    _ledger_entry,
    initial_label_from_payload,
    review_label_from_payload,
)
from research.phase1.mechanism_questions_v1.question_set_v1 import ValidationError, build_episodes, write_jsonl
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import (
    MANIFEST_FILENAME,
    canonical_sha256,
    require_exact_review_route,
    route_signature,
)
from tests.test_mechanism_questions_v1 import fact_pack


def _payload() -> dict:
    return {
        "confidence": 0.78,
        "abstain": False,
        "primary_model": "mm-p1-bottleneck",
        "auxiliary_models": ["mm-p1-optimization-constraints"],
        "auxiliary_model_roles": [{"model_id": "mm-p1-optimization-constraints", "role": "diagnostic"}],
        "near_miss_models": ["mm-p1-rate-control"],
        "near_miss_exclusion_items": [{"model_id": "mm-p1-rate-control", "reason": "The observed state does not establish a rate-control cause."}],
        "cited_microconcept_ids": ["constraint_pressure"],
        "cited_evidence_refs": ["fact-1"],
        "route_rationale": "The supported constraint state establishes a legal primary connection.",
        "challenge_assessment": "No active challenge condition invalidates the selected route.",
        "addressed_challenge_microconcept_ids": [],
        "next_verification": "Measure completion after relieving the cited constraint.",
        "abstain_reason": "",
    }


def _audit(seed: str, response_id: str) -> dict:
    return {
        "api_response_id": response_id,
        "request_fingerprint_sha256": seed * 64,
        "raw_response_fingerprint_sha256": seed.upper().encode().hex()[:64].ljust(64, "0"),
    }


def _chain(review_status: str = "accepted") -> tuple[dict, dict, dict, dict, dict]:
    episode = build_episodes([fact_pack()])[0]
    initial_audit = _audit("a", "initial-response")
    review_audit = _audit("b", "review-response")
    initial = initial_label_from_payload(
        episode,
        _payload(),
        provider=" OpenAI ",
        model=" Initial-Model ",
        deployment=" Primary ",
        teacher_version=" Initial-V1 ",
        audit=initial_audit,
    )
    review = review_label_from_payload(
        episode,
        initial,
        {"review_status": review_status, "confidence": 0.71, "audit_rationale": "The exact route satisfies the contract."},
        provider="Anthropic",
        model="Audit-Model",
        deployment="Audit",
        teacher_version="Audit-V1",
        audit=review_audit,
    )
    return episode, initial, review, _ledger_entry(initial, mode="initial", audit=initial_audit), _ledger_entry(review, mode="review", audit=review_audit, initial=initial)


def _write_chain(root: Path, *, review_status: str = "accepted") -> dict[str, Path | dict]:
    episode, initial, review, initial_ledger, review_ledger = _chain(review_status)
    episodes_path = root / "episodes" / "train_development.jsonl"
    labels_dir = root / "labels" / "train_development"
    labels_dir.mkdir(parents=True)
    episodes_path.parent.mkdir(parents=True)
    paths = {
        "episodes": episodes_path,
        "initial": labels_dir / "initial.jsonl",
        "review": labels_dir / "review.jsonl",
        "initial_ledger": labels_dir / "initial-audit.jsonl",
        "review_ledger": labels_dir / "review-audit.jsonl",
        "output": labels_dir,
    }
    write_jsonl(paths["episodes"], [episode])
    write_jsonl(paths["initial"], [initial])
    write_jsonl(paths["review"], [review])
    write_jsonl(paths["initial_ledger"], [initial_ledger])
    write_jsonl(paths["review_ledger"], [review_ledger])
    return {**paths, "episode": episode, "initial_label": initial, "review_label": review}


def _finalize(paths: dict[str, Path | dict]) -> dict:
    return finalize_teacher_labels(
        episodes_path=paths["episodes"],
        initial_labels_path=paths["initial"],
        review_labels_path=paths["review"],
        initial_audit_ledger_path=paths["initial_ledger"],
        review_audit_ledger_path=paths["review_ledger"],
        output_dir=paths["output"],
    )


def test_external_initial_and_independent_audit_provenance() -> None:
    _, initial, review, initial_ledger, review_ledger = _chain()
    assert initial["teacher_identity"] == {
        "schema_version": "teacher-identity-v1",
        "provider": "openai",
        "model": "initial-model",
        "deployment": "primary",
        "version": "initial-v1",
    }
    assert review["teacher_identity"]["provider"] == "anthropic"
    assert review["initial_label_sha256"] == canonical_sha256(initial)
    assert review_ledger["initial_label_sha256"] == canonical_sha256(initial)
    assert initial_ledger["label_sha256"] == canonical_sha256(initial)


def test_audit_model_cannot_alias_initial_structured_identity() -> None:
    episode = build_episodes([fact_pack()])[0]
    initial = initial_label_from_payload(
        episode, _payload(), provider="OpenAI", model="Same-Model", deployment="one", teacher_version="v1", audit=_audit("a", "i"),
    )
    with pytest.raises(ValidationError, match="different provider/model"):
        review_label_from_payload(
            episode,
            initial,
            {"review_status": "accepted", "confidence": 0.71, "audit_rationale": "Pass."},
            provider=" openai ",
            model=" same-model ",
            deployment="two",
            teacher_version="v2",
            audit=_audit("b", "r"),
        )


@pytest.mark.parametrize("review_status", ["accepted", "review_required"])
def test_review_status_never_allows_route_rewrite(review_status: str) -> None:
    _, initial, review, _, _ = _chain(review_status)
    review["auxiliary_models"] = []
    review["auxiliary_model_roles"] = {}
    with pytest.raises(ValidationError, match="exact initial abstain/primary/auxiliary route"):
        require_exact_review_route(initial, review)


def test_route_signature_includes_abstain_primary_and_auxiliary() -> None:
    route = {"abstain": False, "primary_model": "p", "auxiliary_models": ["b", "a"]}
    assert route_signature(route) == (False, "p", ("a", "b"))
    changed = deepcopy(route)
    changed["abstain"] = True
    assert route_signature(changed) != route_signature(route)


def test_finalizer_consumes_ledgers_and_emits_hash_bound_manifest(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    manifest = _finalize(paths)
    manifest_path = paths["output"] / MANIFEST_FILENAME
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest
    assert set(manifest["files"]) == {
        "episodes", "initial_labels", "review_labels", "initial_audit_ledger",
        "review_audit_ledger", "all_labels", "accepted_reviews",
    }
    assert manifest["prompt_versions"] == {"initial": [INITIAL_PROMPT_VERSION], "review": [REVIEW_PROMPT_VERSION]}
    assert manifest["accepted_episode_ids"] == [paths["episode"]["episode_id"]]
    assert manifest["teacher_identities"]["initial"][0]["model"] == "initial-model"
    assert manifest["teacher_identities"]["review"][0]["model"] == "audit-model"


@pytest.mark.parametrize("review_status", ["accepted", "review_required"])
def test_finalizer_rejects_route_rewrite_for_every_review_status(tmp_path: Path, review_status: str) -> None:
    paths = _write_chain(tmp_path, review_status=review_status)
    review = json.loads(paths["review"].read_text(encoding="utf-8").strip())
    review["auxiliary_models"] = []
    review["auxiliary_model_roles"] = {}
    ledger = json.loads(paths["review_ledger"].read_text(encoding="utf-8").strip())
    ledger["label_sha256"] = canonical_sha256(review)
    write_jsonl(paths["review"], [review])
    write_jsonl(paths["review_ledger"], [ledger])
    with pytest.raises(ValidationError, match="exact initial"):
        _finalize(paths)


def test_finalizer_rejects_tampered_audit_ledger(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    ledger = json.loads(paths["initial_ledger"].read_text(encoding="utf-8").strip())
    ledger["label_sha256"] = "0" * 64
    write_jsonl(paths["initial_ledger"], [ledger])
    with pytest.raises(ValidationError, match="ledger label hash"):
        _finalize(paths)


def test_finalizer_rejects_review_bound_to_previous_initial_bytes(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    initial = json.loads(paths["initial"].read_text(encoding="utf-8").strip())
    initial["confidence"] = 0.77
    write_jsonl(paths["initial"], [initial])
    with pytest.raises(ValidationError, match="bound to initial label"):
        _finalize(paths)


def test_finalizer_rechecks_structured_teacher_independence(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    initial = paths["initial_label"]
    review = deepcopy(paths["review_label"])
    review["teacher_identity"] = deepcopy(initial["teacher_identity"])
    review["teacher_model"] = initial["teacher_model"]
    review["teacher_version"] = initial["teacher_version"]
    review["provenance"]["teacher_identity"] = deepcopy(initial["teacher_identity"])
    review_ledger = json.loads(paths["review_ledger"].read_text(encoding="utf-8").strip())
    review_ledger["teacher_identity"] = deepcopy(initial["teacher_identity"])
    review_ledger["teacher_model"] = initial["teacher_model"]
    review_ledger["teacher_version"] = initial["teacher_version"]
    review_ledger["label_sha256"] = canonical_sha256(review)
    write_jsonl(paths["review"], [review])
    write_jsonl(paths["review_ledger"], [review_ledger])
    with pytest.raises(ValidationError, match="different provider/model"):
        _finalize(paths)
