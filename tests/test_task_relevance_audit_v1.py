from __future__ import annotations

import numpy as np

from research.phase1.microconcept_router_v1.soft_data import DenseRouterExample
from research.phase1.microconcept_router_v1.task_relevance_audit import (
    _constant_task_relevance,
    _question_type,
    _same_fact_delta_metrics,
    _sibling_cycle_task_relevance,
    _variant_rows,
)


def _row(question_type: str, relevance: tuple[float, float]) -> DenseRouterExample:
    return DenseRouterExample(
        episode_id=f"case-1::{question_type}",
        split="development",
        status=np.asarray([1, 2], dtype=np.int64),
        direction=np.asarray([2, 3], dtype=np.int64),
        numeric=np.asarray([[0.3, 0.6, relevance[0]], [0.7, 0.2, relevance[1]]], dtype=np.float32),
        activation_targets=np.asarray([0.25, 0.75], dtype=np.float32),
        activation_mask=np.asarray([1.0, 1.0], dtype=np.float32),
        activation_weights=np.asarray([1.0, 0.5], dtype=np.float32),
    )


def test_constant_task_relevance_does_not_mutate_other_state_or_targets() -> None:
    row = _row("mechanism_diagnosis", (0.0, 1.0))
    changed = _constant_task_relevance((row,), 0.5)[0]
    assert np.array_equal(changed.status, row.status)
    assert np.array_equal(changed.direction, row.direction)
    assert np.array_equal(changed.numeric[:, :2], row.numeric[:, :2])
    assert np.all(changed.numeric[:, 2] == 0.5)
    assert np.array_equal(changed.activation_targets, row.activation_targets)
    assert _question_type(changed) == "mechanism_diagnosis"


def test_sibling_cycle_uses_only_another_legal_question_profile() -> None:
    rows = (
        _row("conditional_or_counterfactual", (1.0, 0.0)),
        _row("decision_tradeoff", (0.0, 1.0)),
        _row("evidence_discrimination", (0.3, 0.7)),
        _row("mechanism_diagnosis", (0.7, 0.3)),
    )
    changed = _sibling_cycle_task_relevance(rows)
    original_profiles = {tuple(row.numeric[:, 2].tolist()) for row in rows}
    for original, modified in zip(rows, changed, strict=True):
        assert tuple(modified.numeric[:, 2].tolist()) in original_profiles
        assert not np.array_equal(modified.numeric[:, 2], original.numeric[:, 2])
        assert np.array_equal(modified.numeric[:, :2], original.numeric[:, :2])
        assert np.array_equal(modified.status, original.status)
        assert np.array_equal(modified.direction, original.direction)


def test_matched_no_task_variant_removes_only_episode_specific_task_channel() -> None:
    row = _row("decision_tradeoff", (0.3, 1.0))
    reference = {
        "status": np.asarray([4, 4], dtype=np.int64),
        "direction": np.asarray([5, 5], dtype=np.int64),
        "numeric_non_task": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    no_task = _variant_rows((row,), "no_task", reference)[0]
    assert np.array_equal(no_task.status, row.status)
    assert np.array_equal(no_task.direction, row.direction)
    assert np.array_equal(no_task.numeric[:, :2], row.numeric[:, :2])
    assert np.all(no_task.numeric[:, 2] == 0.0)

    task_only = _variant_rows((row,), "task_only", reference)[0]
    assert np.array_equal(task_only.status, reference["status"])
    assert np.array_equal(task_only.direction, reference["direction"])
    assert np.array_equal(task_only.numeric[:, :2], reference["numeric_non_task"])
    assert np.array_equal(task_only.numeric[:, 2], row.numeric[:, 2])


def test_same_fact_delta_metrics_checks_direction_not_just_output_change() -> None:
    left = _row("decision_tradeoff", (1.0, 0.0))
    right = _row("evidence_discrimination", (0.0, 1.0))
    right = DenseRouterExample(
        episode_id=right.episode_id,
        split=right.split,
        status=right.status,
        direction=right.direction,
        numeric=right.numeric,
        activation_targets=np.asarray([0.75, 0.25], dtype=np.float32),
        activation_mask=right.activation_mask,
        activation_weights=right.activation_weights,
    )
    perfect_prediction = np.stack([left.activation_targets, right.activation_targets])
    metrics = _same_fact_delta_metrics(perfect_prediction, (left, right))
    assert metrics["fact_pairs"] == 1
    assert metrics["weighted_delta_mae"] == 0.0
    assert np.isclose(metrics["delta_pearson_correlation"], 1.0)
    assert metrics["sign_agreement_when_absolute_target_delta_at_least_0_25"] == 1.0
