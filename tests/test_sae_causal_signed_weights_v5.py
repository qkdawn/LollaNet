from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.fit_signed_weights_v5 import (
    fit_signed_weights,
    split_train_pairs,
    weighted_activation_update,
)


def test_fit_signed_weights_recovers_relative_signs_without_development() -> None:
    rng = np.random.default_rng(7)
    differences = rng.normal(size=(40, 3))
    target = 2.0 * differences[:, 0] - 0.7 * differences[:, 1] + rng.normal(0, 0.05, 40)
    cases = np.asarray([f"case-{index // 2:02d}" for index in range(40)])
    selected, fit = fit_signed_weights(
        differences, target, cases, bootstraps=40, seed=11
    )
    columns = selected.tolist()
    weights = dict(zip(columns, fit["selected_normalized_signed_weights"]))
    assert 0 in columns and 1 in columns
    assert weights[0] > 0
    assert weights[1] < 0
    assert fit["oof_spearman"] > 0.9
    assert fit["passes_train_only_gate"] is True


def test_weighted_activation_update_preserves_signed_relative_weights() -> None:
    source_indices = np.asarray([[10, 20], [10, -1]])
    source_values = np.asarray([[2.0, 4.0], [3.0, 0.0]])
    target_indices = np.asarray([[10, 20], [10, -1]])
    target_values = np.asarray([[1.0, 1.0], [1.0, 0.0]])
    decoder = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    update = weighted_activation_update(
        source_indices,
        source_values,
        target_indices,
        target_values,
        [10, 20],
        decoder,
        np.asarray([1.0, -0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(update, [[1.0, -1.5], [2.0, 0.0]])


def test_train_holdout_split_is_case_grouped_and_deterministic() -> None:
    pairs = [
        {"pair_id": f"p-{index}", "case_id": f"c-{index // 2}"}
        for index in range(20)
    ]
    fit, holdout = split_train_pairs(pairs, "specificity")
    repeated_fit, repeated_holdout = split_train_pairs(pairs, "specificity")
    assert fit == repeated_fit
    assert holdout == repeated_holdout
    assert {row["case_id"] for row in fit}.isdisjoint(
        {row["case_id"] for row in holdout}
    )


def test_import_does_not_load_pyarrow() -> None:
    module = (
        "research.phase1.target_attractor_pilot.sae_causal."
        "analyze_signed_weight_holdout_v5"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module}; assert 'pyarrow' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
