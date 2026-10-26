from __future__ import annotations

import numpy as np
import pytest

from research.phase1.template_attractor_pilot.natural_probe_audit import (
    clustered_bootstrap, grouped_discovery_cross_validation,
)


def test_clustered_bootstrap_preserves_case_clusters_and_is_deterministic() -> None:
    y = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    activation = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    text = np.asarray([0.4, 0.5, 0.6, 0.7, 0.45, 0.55])
    case_ids = ["a", "a", "b", "b", "c", "c"]
    first = clustered_bootstrap(y, activation, text, case_ids, iterations=200, seed=7)
    second = clustered_bootstrap(y, activation, text, case_ids, iterations=200, seed=7)
    assert first == second
    assert first["requested_iterations"] == 200
    assert first["valid_iterations"] > 0
    assert first["activation_minus_text_auroc"]["median"] >= 0.0


def test_clustered_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        clustered_bootstrap(
            np.asarray([0, 1]), np.asarray([0.1]), np.asarray([0.2, 0.8]), ["a", "b"],
        )


def test_grouped_cross_validation_keeps_case_pairs_together() -> None:
    case_ids = [f"case-{index // 2}" for index in range(20)]
    y = np.asarray([0, 0, 1, 1] * 5, dtype=np.int64)
    activation = np.column_stack([y * 4 - 2, y * 2 - 1]).astype(np.float32)
    texts = ["same prefix" for _ in y]
    formats = np.zeros((len(y), 2), dtype=np.float64)
    result = grouped_discovery_cross_validation(
        activation, texts, formats, y, case_ids,
        activation_c=0.1, text_c=0.1, format_c=0.1,
        folds=5, repeats=2, seed=3,
    )
    assert result["valid_repeats"] == 2
    assert result["activation_auroc"]["median"] == 1.0
    assert result["activation_minus_text_auroc"]["median"] > 0.0
