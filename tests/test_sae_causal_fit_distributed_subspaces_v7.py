from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.fit_distributed_subspaces_v7 import (
    augment_bidirectional,
    grouped_oof_ridge,
)


def test_grouped_oof_ridge_recovers_distributed_signal() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(12), 2)
    x = rng.normal(size=(24, 16))
    y = x[:, 2] - 0.7 * x[:, 9] + rng.normal(scale=0.05, size=24)
    confounds = rng.normal(size=(24, 3))
    result = grouped_oof_ridge(x, y, confounds, groups, alpha=1.0, folds=4)
    assert result["spearman"] > 0.6


def test_augment_bidirectional_balances_pair_orientation() -> None:
    x, y, confounds, groups = augment_bidirectional(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([1.0, 2.0]),
        np.asarray([[1.0], [2.0]]),
        np.asarray(["a", "b"]),
    )
    np.testing.assert_allclose(x[2:], -x[:2])
    assert y.tolist() == [1.0, 1.0, -1.0, -1.0]
    assert groups.tolist() == ["a", "b", "a", "b"]
