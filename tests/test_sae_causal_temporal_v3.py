from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.analyze_temporal_v3 import (
    _feature_trajectory,
    _window_statistics,
)


def test_feature_trajectory_respects_topk_membership() -> None:
    indices = np.asarray([[[3, 4], [4, 5]]])
    values = np.asarray([[[2.0, 1.0], [7.0, 8.0]]])
    np.testing.assert_allclose(_feature_trajectory(indices, values, 4), [[1.0, 7.0]])


def test_window_statistics_uses_oriented_high_minus_low_pairs() -> None:
    trajectory = np.asarray([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    result = _window_statistics(
        trajectory,
        {"low": 0, "high": 1},
        [{"low_record_id": "low", "high_record_id": "high"}],
        4,
    )
    assert result["mean_delta"] == 2.0
    assert result["sign_consistency"] == 1.0
