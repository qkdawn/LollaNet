from __future__ import annotations

import torch

from research.phase1.target_attractor_pilot.sae_causal.causal_das_v9 import (
    deterministic_case_split,
    fit_grouped_ridge_head,
    interchange_update,
    orthonormal_basis,
    pooled_window,
    temporal_head_representation,
)


def test_interchange_update_only_swaps_the_learned_subspace() -> None:
    basis = orthonormal_basis(torch.tensor([[1.0], [0.0], [0.0]]))
    target = torch.tensor([[1.0, 2.0, 3.0]])
    source = torch.tensor([[5.0, 7.0, 9.0]])
    update = interchange_update(target, source, basis)
    torch.testing.assert_close(update, torch.tensor([[4.0, 0.0, 0.0]]))


def test_pooled_window_handles_short_responses() -> None:
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.testing.assert_close(pooled_window(hidden, (4, 8)), hidden[4])


def test_deterministic_case_split_is_disjoint_and_stable() -> None:
    cases = [f"c{i}" for i in range(10)]
    first = deterministic_case_split(cases, "specificity")
    second = deterministic_case_split(list(reversed(cases)), "specificity")
    assert first == second
    assert not first[0].intersection(first[1])
    assert first[0] | first[1] == set(cases)


def test_grouped_ridge_head_recovers_two_dimensional_contrast() -> None:
    generator = torch.Generator().manual_seed(4)
    cache = {}
    pairs = []
    for index in range(12):
        low = torch.randn((8, 6), generator=generator)
        high = low.clone()
        high[:, 1] += 2.0
        high[:, 4] -= 1.0
        low_id, high_id = f"l{index}", f"h{index}"
        cache[low_id] = {"final": low}
        cache[high_id] = {"final": high}
        pairs.append({
            "low_record_id": low_id, "high_record_id": high_id,
            "case_id": f"c{index}", "response_token_delta": 0,
        })
    head, fit = fit_grouped_ridge_head(cache, pairs, (0, 8))
    assert fit["selected"]["oof_auroc"] > 0.9
    assert float(head[1]) > 0
    assert float(head[4]) < 0


def test_temporal_head_representation_concatenates_fixed_windows() -> None:
    hidden = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    representation = temporal_head_representation(hidden)
    assert representation.shape == (20,)
    torch.testing.assert_close(representation[:4], hidden[:4].mean(dim=0))
    torch.testing.assert_close(representation[-4:], torch.zeros(4))
