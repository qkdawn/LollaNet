from __future__ import annotations

from copy import deepcopy

import pytest

from research.phase1.mechanism_questions_v1.audit_question_conditioning_v1 import (
    audit_question_conditioning,
    validate_question_conditioning_audit,
)
from research.phase1.mechanism_questions_v1.public_corpus_v1 import _variants
from research.phase1.mechanism_questions_v1.question_set_v1 import (
    PROFILE_BY_QUESTION_KIND,
    ValidationError,
    _validate_counterfactual,
    load_semantic_assets,
    question_conditioned_microconcept_states,
    task_relevance_basis,
)


def _states(active: dict[str, dict]) -> list[dict]:
    rows = []
    for concept_id in sorted(load_semantic_assets()["concept_ids"]):
        item = active.get(concept_id)
        if item is None:
            rows.append({
                "microconcept_id": concept_id,
                "status": "unknown",
                "intensity": 0,
                "direction": "unknown",
                "temporal_persistence": 0,
                "task_relevance": 0,
                "evidence_refs": [],
            })
        else:
            rows.append({"microconcept_id": concept_id, **item})
    return rows


def _variant(kind: str, focus: str) -> dict:
    return {
        "question_kind": kind,
        "task_relevance_profile": PROFILE_BY_QUESTION_KIND[kind],
        "task_focus_microconcept_id": focus,
        "counterfactual_delta": None,
    }


def test_same_question_kind_uses_fact_specific_relevance_not_one_fixed_template() -> None:
    low_pressure = _states({
        "constraint_pressure": {
            "status": "candidate", "intensity": 1, "direction": "moderate",
            "temporal_persistence": 1, "task_relevance": 1, "evidence_refs": ["fact-1"],
        },
        "evidence_quality": {
            "status": "supported", "intensity": 3, "direction": "reviewed",
            "temporal_persistence": 2, "task_relevance": 3, "evidence_refs": ["fact-1"],
        },
    })
    high_pressure = _states({
        "constraint_pressure": {
            "status": "supported", "intensity": 3, "direction": "binding",
            "temporal_persistence": 3, "task_relevance": 3, "evidence_refs": ["fact-2"],
        },
        "evidence_quality": {
            "status": "mixed", "intensity": 1, "direction": "partial",
            "temporal_persistence": 1, "task_relevance": 1, "evidence_refs": ["fact-2"],
        },
    })
    variant = _variant("mechanism_diagnosis", "constraint_pressure")
    low_basis = task_relevance_basis(variant, low_pressure, load_semantic_assets()["concept_ids"])
    high_basis = task_relevance_basis(variant, high_pressure, load_semantic_assets()["concept_ids"])

    assert low_basis["profile"] == high_basis["profile"]
    assert low_basis["vector"] != high_basis["vector"]
    changed = {
        concept_id for concept_id in low_basis["vector"]
        if low_basis["vector"][concept_id] != high_basis["vector"][concept_id]
    }
    assert {"constraint_pressure", "evidence_quality"}.issubset(changed)

    rows = [
        {
            "episode_id": "case-low::mechanism_diagnosis", "case_id": "case-low",
            "question_kind": "mechanism_diagnosis", "split": "development",
            "microconcept_states": question_conditioned_microconcept_states(low_pressure, variant),
            "counterfactual_delta": None,
        },
        {
            "episode_id": "case-high::mechanism_diagnosis", "case_id": "case-high",
            "question_kind": "mechanism_diagnosis", "split": "development",
            "microconcept_states": question_conditioned_microconcept_states(high_pressure, variant),
            "counterfactual_delta": None,
        },
    ]
    audit = audit_question_conditioning(rows)
    metrics = audit["question_kind_task_relevance"]["mechanism_diagnosis"]
    assert metrics["unique_vector_count"] == 2
    assert metrics["same_kind_differing_pair_rate"] == 1.0
    assert audit["checks"]["same_kind_fact_specific_task_relevance"] is True


def test_generated_counterfactual_applies_all_declared_non_task_axes_and_audit_detects_tampering() -> None:
    facts = [{
        "fact_id": "fact-1", "statement": "A public recall records no injury entries yet.",
        "evidence_refs": ["src-1"], "available_at": "2024-01-01T00:00:00Z",
        "injury_entries": 0, "incident_entries": 2, "affected_units": 5000,
    }]
    fact_states = _states({
        "outcome_asymmetry": {
            "status": "supported", "intensity": 2, "direction": "downside_skewed",
            "temporal_persistence": 2, "task_relevance": 2, "evidence_refs": ["fact-1"],
        },
        "evidence_quality": {
            "status": "mixed", "intensity": 2, "direction": "partial",
            "temporal_persistence": 2, "task_relevance": 2, "evidence_refs": ["fact-1"],
        },
    })
    generated = _variants(0, "product_organization", facts, fact_states)
    variant = next(item for item in generated if item["question_kind"] == "conditional_or_counterfactual")
    changes = variant["counterfactual_delta"]["microconcept_state_changes"]
    assert {change["axis"] for change in changes} == {
        "status", "intensity", "direction", "temporal_persistence",
    }

    conditioned = question_conditioned_microconcept_states(fact_states, variant)
    outcome = next(state for state in conditioned if state["microconcept_id"] == "outcome_asymmetry")
    assert outcome["status"] == "candidate"
    assert outcome["intensity"] == 3
    assert outcome["direction"] == "counterfactual_harm_signal_increasing"
    assert outcome["temporal_persistence"] == 3

    episode = {
        "episode_id": "case-1::conditional_or_counterfactual", "case_id": "case-1",
        "question_kind": "conditional_or_counterfactual", "split": "development",
        "microconcept_states": conditioned,
        "counterfactual_delta": variant["counterfactual_delta"],
    }
    audit = audit_question_conditioning([episode])
    assert audit["counterfactuals"]["declared_axes"] == {
        "direction": 1, "intensity": 1, "status": 1, "temporal_persistence": 1,
    }
    assert audit["checks"]["all_declared_counterfactual_state_changes_applied"] is True
    validate_question_conditioning_audit(audit)

    tampered = deepcopy(episode)
    next(state for state in tampered["microconcept_states"] if state["microconcept_id"] == "outcome_asymmetry")["direction"] = "unchanged"
    failed = audit_question_conditioning([tampered])
    assert failed["checks"]["all_declared_counterfactual_state_changes_applied"] is False
    with pytest.raises(ValidationError, match="state_changes_applied"):
        validate_question_conditioning_audit(failed)


def test_task_only_counterfactual_delta_is_rejected() -> None:
    variant = {
        "counterfactual_delta": {
            "changes": [{"fact_id": "fact-1", "field": "value", "before": 1, "after": 2}],
            "microconcept_state_changes": [{
                "microconcept_id": "constraint_pressure", "axis": "task_relevance", "before": 2, "after": 3,
            }],
        }
    }
    with pytest.raises(ValidationError, match="non-task"):
        _validate_counterfactual(
            variant,
            {"fact-1": {"fact_id": "fact-1", "value": 1}},
            load_semantic_assets()["concept_ids"],
        )
