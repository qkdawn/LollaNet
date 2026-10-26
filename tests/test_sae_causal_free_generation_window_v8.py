from __future__ import annotations

import numpy as np
import torch

from research.phase1.target_attractor_pilot.sae_causal.free_generation_window_v8 import (
    build_plan,
    free_generation_hook,
)


def test_free_plan_has_52_unique_records() -> None:
    pairs = {"specificity": [{"pair_id": "s1"}, {"pair_id": "s2"}], "independence": [{"pair_id": "i1"}, {"pair_id": "i2"}]}
    plan = build_plan(pairs)
    assert len(plan) == 52
    assert len({row["record_id"] for row in plan}) == 52


def test_free_hook_can_change_prefill_last_token() -> None:
    hook, state = free_generation_hook(np.asarray([1.0, 0.0]), 2.0, 0, 1)
    output = hook(None, None, torch.zeros((1, 4, 2)))
    torch.testing.assert_close(output[0, -1], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(output[0, :-1], torch.zeros((3, 2)))
    assert state["intervention_steps"] == 1
