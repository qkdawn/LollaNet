from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.phase1.controllability_audit_v1.contracts import (
    CharterValidationError,
    preflight_open_data,
    validate_charter,
)


def valid_charter() -> dict:
    return {
        "study_id": "test",
        "paper": {"attractor_claim_allowed": False},
        "constructs": {
            "target_axes": [
                {"name": "specificity", "separate_label": True},
                {"name": "independence", "separate_label": True},
                {"name": "non_template", "separate_label": True},
            ],
            "composite_quality_score": False,
            "positive_control": {"name": "sentiment", "included_in_target_axes": False},
        },
        "conditions": [
            {"name": name, "required": True}
            for name in (
                "no_intervention", "prompting", "caa_diffmean", "reft", "sae_subspace",
                "matched_random_subspace",
            )
        ],
        "splits": {
            "grouping_key": "case_id", "case_disjoint": True, "selection_split": "train",
            "tuning_split": "development",
            "evaluation": {
                "sealed": True, "accessible_during_development": False,
                "release_mode": "once_after_freeze",
            },
        },
        "controls": {
            "matched_random_subspace": {
                "match": ["layer", "rank", "update_norm", "token_window", "injection_policy"],
                "minimum_draws": 3,
            },
            "positive_control_required_before_target_claims": True,
        },
        "reporting": {
            "target_effects_separate_by_axis": True, "cross_axis_effect_matrix": True,
            "quality_preservation_separate": True, "composite_overall_quality_forbidden": True,
        },
        "statistics": {
            "independent_unit": "case_id", "within_case_repetitions": ["seed", "response", "token"],
            "report_effect_sizes": True, "report_confidence_intervals": True,
            "comparison_family": {"primary": "axes_by_method", "multiplicity_strategy": "holm"},
        },
        "annotation": {
            "model_assisted_labels_are_human_validation": False, "human_validation_required": True,
            "independent_blind_annotators": 2, "report_agreement": True,
        },
        "evidence_boundary": {
            "prior_versions_are_discovery_only": True, "formal_evaluation_accessed": False,
            "formal_evaluation_results_available": False,
        },
        "open_data": {},
    }


def test_valid_charter_passes() -> None:
    validate_charter(valid_charter())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["constructs"].update(composite_quality_score=True), "composite_quality_score"),
        (lambda c: c["constructs"].pop("positive_control"), "positive control"),
        (lambda c: c["splits"]["evaluation"].update(sealed=False), "evaluation must remain sealed"),
        (lambda c: c["statistics"].update(independent_unit="response"), "case_id"),
        (lambda c: c["conditions"].pop(3), "missing required intervention"),
        (lambda c: c["controls"]["matched_random_subspace"].update(match=["rank"]), "not fully matched"),
    ],
)
def test_invalid_charters_fail_closed(mutate, message: str) -> None:
    charter = copy.deepcopy(valid_charter())
    mutate(charter)
    with pytest.raises(CharterValidationError, match=message):
        validate_charter(charter)


def test_open_data_preflight_counts_only_train_and_development(tmp_path: Path) -> None:
    source = tmp_path / "open.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"case_id": case_id, "split": split})
            for case_id, split in (("a", "train"), ("a", "train"), ("b", "development"))
        ),
        encoding="utf-8",
    )
    charter = valid_charter()
    charter["open_data"] = {
        "responses": {"path": str(source), "expected_records": 3, "allowed_splits": ["train", "development"]}
    }
    report = preflight_open_data(charter)
    assert report["sources"]["responses"]["records"] == 3
    assert report["sources"]["responses"]["cases"] == 2
    assert report["sources"]["responses"]["status"] == "ready"
    assert len(report["sources"]["responses"]["sha256"]) == 64


def test_open_data_preflight_rejects_evaluation_rows(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(json.dumps({"case_id": "sealed", "split": "evaluation"}), encoding="utf-8")
    charter = valid_charter()
    charter["open_data"] = {"responses": {"path": str(source)}}
    with pytest.raises(CharterValidationError, match="forbidden split"):
        preflight_open_data(charter)


def test_open_data_preflight_rejects_case_leakage_across_splits(tmp_path: Path) -> None:
    source = tmp_path / "leaked.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"case_id": "same-case", "split": split})
            for split in ("train", "development")
        ),
        encoding="utf-8",
    )
    charter = valid_charter()
    charter["open_data"] = {"responses": {"path": str(source)}}
    with pytest.raises(CharterValidationError, match="multiple splits"):
        preflight_open_data(charter)


def test_open_data_preflight_rejects_source_mapping_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"case_id": "a", "split": "train"}), encoding="utf-8")
    second.write_text(json.dumps({"case_id": "a", "split": "development"}), encoding="utf-8")
    charter = valid_charter()
    charter["open_data"] = {"first": {"path": str(first)}, "second": {"path": str(second)}}
    with pytest.raises(CharterValidationError, match="mapping differs"):
        preflight_open_data(charter)
