from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.phase1.target_attractor_pilot.sae_causal.candidates import matched_random_features, rank_candidates
from research.phase1.target_attractor_pilot.sae_causal.causal import evaluate_causal_gate, steering_hook
from research.phase1.target_attractor_pilot.sae_causal.confounds import residualize_axes, visible_markers
from research.phase1.target_attractor_pilot.sae_causal.contracts import AXES, assert_discovery_rows, load_charter
from research.phase1.target_attractor_pilot.sae_causal.sae import SparseCodes
from research.phase1.target_attractor_pilot.sae_causal.sae import QwenScopeTopKSAE
from research.phase1.target_attractor_pilot.sae_causal.resources import WindowsCommitMemory
from research.phase1.target_attractor_pilot.sae_causal.stages import (
    StageLocked, choose_representation_result, fit_sae_targeted_steering, require_causal_features,
)


def test_charter_is_discovery_only_and_factorized() -> None:
    charter = load_charter()
    assert tuple(charter["labels"]["axes"]) == AXES
    assert charter["evidence_boundary"]["forbidden_splits"] == ["evaluation"]


def test_discovery_guard_rejects_evaluation() -> None:
    with pytest.raises(ValueError, match="sealed"):
        assert_discovery_rows([{"record_id": "x", "split": "evaluation"}])


def test_visible_markers_are_mechanical_confounds() -> None:
    markers = visible_markers("# H\n\n- a\n1. b")
    assert markers["heading_count"] == 1
    assert markers["bullet_count"] == 1
    assert markers["numbered_count"] == 1


def test_residualization_fits_train_and_preserves_order() -> None:
    trajectories, labels = [], []
    for index in range(12):
        split = "train" if index < 8 else "development"
        trajectories.append({"record_id": str(index), "response": "x" * (index + 2), "response_tokens": index + 1,
                             "generation_finished": True})
        labels.append({"record_id": str(index), "case_id": f"c{index % 3}", "domain": "d", "split": split,
                       "specificity": index % 5, "independence": (index * 2) % 5, "non_template": (index * 3) % 5})
    result = residualize_axes(trajectories, labels)
    assert result.record_ids == [str(i) for i in range(12)]
    assert all(result.residual[axis].shape == (12,) for axis in AXES)
    assert all(np.isfinite(result.pipelines[axis].coefficients).all() for axis in AXES)


def test_sparse_candidate_ranking_and_matched_controls() -> None:
    count, top_k = 30, 2
    indices = np.tile(np.array([[1, 3]], dtype=np.int32), (count, 1))
    values = np.column_stack([np.arange(count), np.ones(count)]).astype(np.float32)
    codes = SparseCodes(indices, values, [str(i) for i in range(count)], 8)
    split = np.asarray(["train"] * 20 + ["development"] * 10)
    signal = np.arange(count, dtype=float)
    residuals = SimpleNamespace(record_ids=codes.record_ids, split=split,
                                residual={axis: signal for axis in AXES})
    ranked = rank_candidates(codes, residuals, candidates_per_axis=1, minimum_active_train=5)
    assert all(rows[0].feature == 1 for rows in ranked.values())
    assert matched_random_features(codes, 1, 1)[0] != 1


def test_qwen_scope_encoder_matches_official_formula_without_relu_or_decoder_bias() -> None:
    import torch

    encoder = torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 1.0]])
    decoder = torch.zeros(3, 2)
    encoder_bias = torch.tensor([0.0, 0.0, -5.0])
    decoder_bias = torch.tensor([100.0, 100.0])
    sae = QwenScopeTopKSAE(encoder, decoder, encoder_bias, decoder_bias, top_k=2)
    indices, values = sae.encode(np.array([[1.0, 1.0]], dtype=np.float32))
    assert indices.tolist() == [[0, 1]]
    assert values.tolist() == [[1.0, -2.0]]


def test_causal_gate_requires_specific_monotonic_effect_and_random_control() -> None:
    rows = []
    for feature, random_control, scale in ((7, False, 1.0), (9, True, 0.05)):
        for signed_strength in (-2, -1, 1, 2):
            rows.append({"candidate_feature": 7, "feature": feature, "random_control": random_control,
                         "signed_strength": signed_strength, "delta_specificity": signed_strength * scale,
                         "delta_independence": 0.05, "delta_non_template": 0.05,
                         "delta_usefulness": 0.0, "delta_validity": 0.0})
    gate = evaluate_causal_gate(rows, "specificity", 7)
    assert gate.passed


def test_steering_hook_changes_prompt_end_then_cached_steps() -> None:
    import torch

    hook = steering_hook(torch.tensor([1.0, 2.0]), coefficient=2.0, token_start=0, token_stop=2)
    prompt = hook(None, None, torch.zeros(1, 3, 2))
    first_cached = hook(None, None, torch.zeros(1, 1, 2))
    second_cached = hook(None, None, torch.zeros(1, 1, 2))
    assert torch.equal(prompt[0, -1], torch.tensor([2.0, 4.0]))
    assert torch.equal(prompt[0, 0], torch.zeros(2))
    assert torch.equal(first_cached[0, 0], torch.tensor([2.0, 4.0]))
    assert torch.equal(second_cached, torch.zeros(1, 1, 2))


def test_later_stages_are_locked_and_sae_ts_solver_works(tmp_path: Path) -> None:
    with pytest.raises(StageLocked):
        require_causal_features(tmp_path / "missing.json")
    solution = fit_sae_targeted_steering(np.eye(2), np.array([1.0, -1.0]), ridge=0.0)
    assert np.allclose(solution, [1.0, -1.0])
    assert choose_representation_result(False, True) == "distributed_causal_subspace"


def test_commit_memory_contract_reports_headroom() -> None:
    status = WindowsCommitMemory(62.0, 64.0, 2.0)
    assert status.headroom_gb == status.limit_gb - status.committed_gb
