from __future__ import annotations

import pytest

from research.phase1.lollanet_cma.common import stable_hash
from research.phase1.lollanet_cma.sealed_eval.conditions import derive_routed_rows
from research.phase1.lollanet_cma.sealed_eval.metrics import (
    binary_brier,
    expected_calibration_error,
    faithfulness_matrix,
    faithfulness_diagonal_bootstrap,
    gold_probability,
    paired_stratified_bootstrap,
    summarize_rows,
    yes_probability,
)


def _row(record_id: str, family: str, correct: bool, condition: str = "base") -> dict:
    row = {
        "condition_id": condition,
        "record_id": record_id,
        "model_family": family,
        "query_type": f"query-{family}",
        "rung": 1,
        "correct": correct,
        "parse_ok": True,
        "gold_answer_margin": 1.0 if correct else -1.0,
        "brier_score": 0.1 if correct else 0.9,
        "predicted_answer_probability": 0.8 if correct else 0.7,
        "output_tokens": 10,
        "latency_seconds": 1.0,
        "peak_gpu_memory_bytes": 100,
    }
    row["record_content_sha256"] = stable_hash(row)
    return row


def test_binary_probability_and_calibration_metrics_are_well_defined():
    assert yes_probability(0.0, 0.0) == 0.5
    assert gold_probability(2.0, 0.0, "yes") > 0.5
    assert gold_probability(2.0, 0.0, "no") < 0.5
    assert binary_brier(0.8, "yes") == pytest.approx(0.04)
    assert expected_calibration_error([0.8, 0.7], [True, False], bins=10) == pytest.approx(0.45)


def test_routed_conditions_are_derived_from_hash_verified_family_runs():
    expected_ids = ["a", "b"]
    controller = {"a": {"model_family": "a"}, "b": {"model_family": "b"}}
    runs = {
        "family_direction__a": [
            _row("a", "a", True, "family_direction__a"),
            _row("b", "b", False, "family_direction__a"),
        ],
        "family_direction__b": [
            _row("a", "a", False, "family_direction__b"),
            _row("b", "b", True, "family_direction__b"),
        ],
    }
    derived = derive_routed_rows(
        "oracle_correct", expected_ids, controller, runs, {"a": "b", "b": "a"},
        {"a": "b", "b": "a"},
    )
    assert [row["correct"] for row in derived] == [True, True]
    assert [row["source_condition_id"] for row in derived] == [
        "family_direction__a", "family_direction__b",
    ]
    assert all(row["record_content_sha256"] == stable_hash({
        key: value for key, value in row.items() if key != "record_content_sha256"
    }) for row in derived)


def test_condition_summary_reports_macro_and_calibration_metrics():
    rows = [_row("a-1", "a", True), _row("a-2", "a", True), _row("b-1", "b", False)]
    summary = summarize_rows(rows, ["a", "b"], ece_bins=10)
    assert summary["overall_accuracy"] == pytest.approx(2 / 3)
    assert summary["macro_family_accuracy"] == 0.5
    assert summary["query_type_accuracy"] == {"query-a": 1.0, "query-b": 0.0}
    assert summary["mean_brier_score"] == pytest.approx((0.1 + 0.1 + 0.9) / 3)


def test_paired_bootstrap_is_stratified_deterministic_and_paired():
    left = [
        _row("a-1", "a", True), _row("a-2", "a", True),
        _row("b-1", "b", True), _row("b-2", "b", False),
    ]
    right = [
        _row("a-1", "a", False), _row("a-2", "a", True),
        _row("b-1", "b", False), _row("b-2", "b", False),
    ]
    first = paired_stratified_bootstrap(left, right, ["a", "b"], 1000, 7, 0.95)
    second = paired_stratified_bootstrap(left, right, ["a", "b"], 1000, 7, 0.95)
    assert first == second
    assert first["left_minus_right"] == 0.5
    assert first["lower"] <= first["left_minus_right"] <= first["upper"]


def test_faithfulness_matrix_uses_all_direction_family_combinations():
    base = [_row("a", "a", False), _row("b", "b", False)]
    directions = {
        "a": [_row("a", "a", True), _row("b", "b", False)],
        "b": [_row("a", "a", False), _row("b", "b", True)],
    }
    result = faithfulness_matrix(base, directions, ["a", "b"])
    assert result["matrix"] == {"a": {"a": 1.0, "b": 0.0}, "b": {"a": 0.0, "b": 1.0}}
    assert result["diagonal_advantage"] == {"a": 1.0, "b": 1.0}
    assert result["positive_diagonal_advantage_count"] == 2


def test_faithfulness_bootstrap_is_paired_stratified_and_deterministic():
    base = [
        _row("a-1", "a", False), _row("a-2", "a", False),
        _row("b-1", "b", False), _row("b-2", "b", False),
    ]
    directions = {
        "a": [
            _row("a-1", "a", True), _row("a-2", "a", True),
            _row("b-1", "b", False), _row("b-2", "b", False),
        ],
        "b": [
            _row("a-1", "a", False), _row("a-2", "a", False),
            _row("b-1", "b", True), _row("b-2", "b", True),
        ],
    }
    first = faithfulness_diagonal_bootstrap(base, directions, ["a", "b"], 100, 5, 0.95)
    second = faithfulness_diagonal_bootstrap(base, directions, ["a", "b"], 100, 5, 0.95)
    assert first == second
    assert first["significant_positive_count"] == 2
