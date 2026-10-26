from __future__ import annotations

import numpy as np
import torch

from research.phase1.target_attractor_pilot.sae_causal.temporal_policy_holdout_v6 import (
    _anchor,
    build_plan,
    temporal_hook,
)


def test_temporal_policy_plan_has_76_unique_train_holdout_records() -> None:
    pairs = [
        {"pair_id": "p1", "axis": "non_template"},
        {"pair_id": "p2", "axis": "non_template"},
    ]
    plan = build_plan(pairs)
    assert len(plan) == 76
    assert len({row["record_id"] for row in plan}) == 76
    assert {row["policy"] for row in plan} == {
        "none", "prefix_once", "persistent", "closed_loop"
    }


def test_persistent_hook_skips_prefill_and_updates_cached_steps() -> None:
    hook, state = temporal_hook(np.asarray([1.0, 0.0]), "persistent", 2.0, 4)
    prefill = torch.zeros((1, 3, 2))
    assert hook(None, None, prefill) is prefill
    first = hook(None, None, torch.zeros((1, 1, 2)))
    second = hook(None, None, torch.zeros((1, 1, 2)))
    torch.testing.assert_close(first[0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(second[0, 0], torch.tensor([1.0, 0.0]))
    assert state["generation_steps_seen"] == 2
    assert state["intervention_steps"] == 2
    assert state["applied_update_energy"] == 2.0


def test_closed_loop_only_corrects_toward_anchor_with_cap() -> None:
    hook, state = temporal_hook(
        np.asarray([1.0, 0.0]), "closed_loop", 2.0, 4,
        source_anchor=3.0, target_anchor=1.0, alpha=1.0,
    )
    corrected = hook(None, None, torch.tensor([[[0.0, 4.0]]]))
    torch.testing.assert_close(corrected[0, 0], torch.tensor([1.0, 4.0]))
    already_high = hook(None, None, torch.tensor([[[3.5, 4.0]]]))
    torch.testing.assert_close(already_high[0, 0], torch.tensor([3.0, 4.0]))
    assert state["intervention_steps"] == 2


def test_anchor_uses_late_half_residual_projection() -> None:
    residual = np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
    assert _anchor(residual, np.asarray([1.0, 0.0])) == 6.0
