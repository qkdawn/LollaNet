from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.paired_discovery_v3 import (
    build_matched_pairs,
)


def _label(record_id: str, specificity: float, independence: float, non_template: float) -> dict:
    return {
        "record_id": record_id,
        "case_id": "case-1",
        "split": "train",
        "domain": "research",
        "specificity": specificity,
        "independence": independence,
        "non_template": non_template,
        "usefulness": 3.0,
        "validity": 3.0,
    }


def test_matching_changes_one_axis_and_does_not_reuse_responses() -> None:
    labels = [
        _label("low-a", 1.0, 2.0, 1.0),
        _label("high-a", 3.0, 2.0, 1.0),
        _label("low-b", 1.0, 2.5, 1.0),
        _label("high-b", 2.0, 2.5, 1.0),
    ]
    trajectories = [
        {"record_id": row["record_id"], "response_tokens": 100 + index}
        for index, row in enumerate(labels)
    ]

    pairs = build_matched_pairs(trajectories, labels)["specificity"]

    assert len(pairs) == 2
    used = [record_id for pair in pairs for record_id in (pair.low_record_id, pair.high_record_id)]
    assert len(used) == len(set(used))
    assert all(pair.maximum_other_axis_delta <= 0.5 for pair in pairs)


def test_matching_rejects_cross_axis_and_quality_changes() -> None:
    low = _label("low", 1.0, 1.0, 1.0)
    cross_axis = _label("cross", 3.0, 3.0, 1.0)
    poor_quality = _label("quality", 3.0, 1.0, 1.0)
    poor_quality["validity"] = 1.0
    labels = [low, cross_axis, poor_quality]
    trajectories = [{"record_id": row["record_id"], "response_tokens": 100} for row in labels]

    assert build_matched_pairs(trajectories, labels)["specificity"] == []
