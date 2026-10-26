from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.adaptive_prescreen_v2 import (
    _select_prescreen_schedule,
)


def test_prescreen_schedule_keeps_train_direction_and_matched_control() -> None:
    data = {
        "candidates": {
            axis: [{"feature": index + 10, "train_rho": rho}]
            for index, (axis, rho) in enumerate(
                zip(("specificity", "independence", "non_template"), (0.2, -0.3, 0.1))
            )
        },
        "matched_random_controls": {"10": [100], "11": [101], "12": [102]},
    }
    schedule = _select_prescreen_schedule(data, features_per_axis=1, controls_per_feature=1)

    assert len(schedule) == 6
    assert sum(not row["random_control"] for row in schedule) == 3
    assert sum(row["random_control"] for row in schedule) == 3
    assert [row["target_improving_feature_sign"] for row in schedule[::2]] == [1, -1, 1]
    assert all(row["candidate_train_rho"] != 0 for row in schedule)
