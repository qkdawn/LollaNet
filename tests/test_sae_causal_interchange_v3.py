from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.interchange_smoke_v3 import (
    match_frequency_control,
    match_frequency_control_for_pairs,
)


def test_frequency_control_excludes_candidate_pool_and_matches_count() -> None:
    indices = np.asarray([[[1, 2], [1, 3]], [[4, 2], [4, 3]]])
    control, target_count, control_count = match_frequency_control(
        indices, feature=1, excluded={1, 2, 3}
    )
    assert control == 4
    assert target_count == control_count == 2


def test_pair_matched_control_must_change_on_every_pair() -> None:
    indices = np.asarray([
        [[1, 2], [1, 3]],
        [[1, 4], [1, 3]],
        [[1, 2], [1, 5]],
        [[1, 4], [1, 5]],
    ])
    values = np.ones_like(indices, dtype=np.float32)
    pairs = [
        {"low_record_id": "a", "high_record_id": "b"},
        {"low_record_id": "c", "high_record_id": "d"},
    ]
    control, _, _ = match_frequency_control_for_pairs(
        indices,
        values,
        feature=1,
        excluded={1, 2, 3, 5},
        record_index={"a": 0, "b": 1, "c": 2, "d": 3},
        pairs=pairs,
    )
    assert control == 4
