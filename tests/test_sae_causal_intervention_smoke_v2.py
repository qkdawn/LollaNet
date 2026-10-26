from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.intervention_smoke_v2 import _select_schedule


def test_smoke_schedule_pairs_candidate_with_matched_control() -> None:
    data = {
        "candidates": {
            axis: [{"feature": index + 10}]
            for index, axis in enumerate(("specificity", "independence", "non_template"))
        },
        "matched_random_controls": {"10": [100], "11": [101], "12": [102]},
    }
    schedule = _select_schedule(data, features_per_axis=1, controls_per_feature=1)
    assert len(schedule) == 6
    assert sum(not row["random_control"] for row in schedule) == 3
    assert sum(row["random_control"] for row in schedule) == 3
