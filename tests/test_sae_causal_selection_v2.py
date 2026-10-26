from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.candidate_selection_v2 import rank_candidates_train_only
from research.phase1.target_attractor_pilot.sae_causal.contracts import AXES
from research.phase1.target_attractor_pilot.sae_causal.sae import SparseCodes


def test_development_labels_cannot_change_selected_features() -> None:
    count = 30
    indices = np.column_stack([
        np.ones(count, dtype=np.int32),
        np.full(count, 2, dtype=np.int32),
    ])
    values = np.column_stack([
        np.arange(count, dtype=np.float32),
        np.arange(count, dtype=np.float32)[::-1],
    ])
    codes = SparseCodes(indices, values, [str(i) for i in range(count)], 8)
    split = np.asarray(["train"] * 20 + ["development"] * 10)
    train_signal = np.arange(count, dtype=float)
    first = SimpleNamespace(
        record_ids=codes.record_ids,
        split=split,
        residual={axis: train_signal.copy() for axis in AXES},
    )
    second = SimpleNamespace(
        record_ids=codes.record_ids,
        split=split,
        residual={axis: train_signal.copy() for axis in AXES},
    )
    for axis in AXES:
        second.residual[axis][20:] = second.residual[axis][20:][::-1]
    selected_first = rank_candidates_train_only(codes, first, 1, 5)
    selected_second = rank_candidates_train_only(codes, second, 1, 5)
    assert {
        axis: rows[0].feature for axis, rows in selected_first.items()
    } == {
        axis: rows[0].feature for axis, rows in selected_second.items()
    }
