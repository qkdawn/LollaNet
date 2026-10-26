from __future__ import annotations

import numpy as np
import torch

from research.phase1.target_attractor_pilot.sae_causal.distributed_intervention_v7 import (
    build_plan,
    window_steering_hook,
)


def test_distributed_plan_has_56_unique_records() -> None:
    pairs = {
        "specificity": [{"pair_id": "s1"}, {"pair_id": "s2"}],
        "independence": [{"pair_id": "i1"}, {"pair_id": "i2"}],
    }
    plan = build_plan(pairs)
    assert len(plan) == 56
    assert len({row["record_id"] for row in plan}) == 56


def test_window_hook_only_updates_selected_generation_steps() -> None:
    hook, state = window_steering_hook(np.asarray([1.0, 0.0]), 2.0, 1, 3)
    outputs = [hook(None, None, torch.zeros((1, 1, 2))) for _ in range(4)]
    torch.testing.assert_close(outputs[0], torch.zeros((1, 1, 2)))
    torch.testing.assert_close(outputs[1][0, 0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(outputs[2][0, 0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(outputs[3], torch.zeros((1, 1, 2)))
    assert state["intervention_steps"] == 2
