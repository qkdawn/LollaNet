from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from research.phase1.lollanet_cma.sealed_eval.diagnostics import (
    binary_auroc,
    evaluate_linear_probe,
    mean_token_level_kl,
    select_probe_l2,
    token_level_kl,
    unrelated_control_result,
)


def test_token_kl_uses_aligned_full_vocabulary_logits():
    logits = np.array([1.0, 0.0, -1.0])
    assert token_level_kl(logits, logits) == pytest.approx(0.0)
    assert token_level_kl(np.array([3.0, 0.0]), np.array([0.0, 3.0])) > 0
    assert mean_token_level_kl([0.1, 0.3]) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="aligned"):
        token_level_kl(np.array([1.0, 2.0]), np.array([1.0]))


def test_probe_l2_selection_is_development_only_and_deterministic():
    train_x = np.array([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    train_y = np.array([0, 0, 1, 1])
    development_x = np.array([[-1.5, 0.0], [1.5, 0.0]])
    development_y = np.array([0, 1])
    first, summaries = select_probe_l2(
        train_x, train_y, development_x, development_y, [0.1, 1.0, 10.0], 1e-6,
    )
    second, _ = select_probe_l2(
        train_x, train_y, development_x, development_y, [0.1, 1.0, 10.0], 1e-6,
    )
    assert first["l2"] == second["l2"]
    assert len(summaries) == 3
    result = evaluate_linear_probe(first, development_x, development_y)
    assert result["accuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert binary_auroc(np.array([0.0, 0.0, 1.0]), np.array([0, 1, 1])) == pytest.approx(0.75)


def test_unrelated_control_gate_uses_the_worst_family_direction():
    passed = unrelated_control_result(0.50, {"a": 0.49, "b": 0.48}, 0.02)
    assert passed["passed"] is True
    assert passed["worst_condition"] == "b"
    failed = unrelated_control_result(0.50, {"a": 0.479, "b": 0.55}, 0.02)
    assert failed["passed"] is False


def test_formal_diagnostic_contracts_are_resolved_without_claiming_results():
    charter = yaml.safe_load(Path(
        "research/phase1/lollanet_cma/sealed_eval/sealed_eval_charter.yaml",
    ).read_text(encoding="utf-8"))
    assert charter["token_level_kl"]["cache_rule"] == (
        "separate_condition_consistent_kv_caches_in_sequential_batch_one_forwards"
    )
    assert charter["hidden_state_probe"]["train_split"] == "vector_train"
    assert charter["hidden_state_probe"]["regularization_selection_split"] == "development"
    assert charter["hidden_state_probe"]["generated_text_or_gold_answer_tokens_forbidden"] is True
    assert all(value["status"] == "resolved" for value in charter["readiness_blockers"].values())
