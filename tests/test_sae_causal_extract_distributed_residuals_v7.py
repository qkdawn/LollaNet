from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.extract_distributed_residuals_v7 import (
    reconstruct_train_rows,
    summarize_windows,
)


def test_reconstruct_train_rows_excludes_development() -> None:
    packet = [
        {"item_id": "a", "prompt": "p1", "response": "r1"},
        {"item_id": "b", "prompt": "p2", "response": "r2"},
    ]
    mapping = {"a": "r1", "b": "r2"}
    labels = [
        {"record_id": "r1", "case_id": "c1", "split": "train"},
        {"record_id": "r2", "case_id": "c2", "split": "development"},
    ]
    rows = reconstruct_train_rows(packet, mapping, labels)
    assert [row["record_id"] for row in rows] == ["r1"]


def test_summarize_windows_masks_unavailable_late_tokens() -> None:
    hidden = np.arange(10, dtype=np.float32).reshape(5, 2)
    values, mask = summarize_windows(hidden)
    np.testing.assert_allclose(values[0], hidden[:4].mean(axis=0))
    np.testing.assert_allclose(values[1], hidden[4:5].mean(axis=0))
    assert mask.tolist() == [True, True, False, False, False]
