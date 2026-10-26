from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from research.phase1.closed_loop_control_v1.charter import validate_pilot_charter
from research.phase1.closed_loop_control_v1.control import (
    ClosedLoopConfig,
    ClosedLoopGenerationHook,
    EnsembleLinearObserver,
    LowRankFeedbackPolicy,
    Perturbation,
    fit_group_bootstrap_observer,
    recovery_metrics,
)


def observer(hidden: int = 2, uncertainty_offset: float = 0.0) -> EnsembleLinearObserver:
    # Features are [rolling hidden, prompt-end hidden]. All members read hidden[0].
    weights = np.zeros((3, 3, hidden * 2), dtype=np.float32)
    weights[:, 0, 0] = 1.0
    weights[0, 0, 0] += uncertainty_offset
    weights[2, 0, 0] -= uncertainty_offset
    return EnsembleLinearObserver(
        weights, np.zeros((3, 3), dtype=np.float32),
        np.zeros(hidden * 2, dtype=np.float32), np.ones(hidden * 2, dtype=np.float32),
        window_size=1,
    )


def policy(hidden: int = 2) -> LowRankFeedbackPolicy:
    basis = np.zeros((hidden, 1), dtype=np.float32)
    basis[0, 0] = 1.0
    error_gain = np.zeros((1, 3), dtype=np.float32)
    error_gain[0, 0] = 1.0
    return LowRankFeedbackPolicy(basis, error_gain, np.zeros_like(error_gain))


def config(**overrides) -> ClosedLoopConfig:
    values = {
        "axis": "specificity", "target": 1.0, "deadband": 0.1,
        "maximum_uncertainty": 0.2, "control_interval": 1,
        "start_step": 0, "stop_step": 10, "maximum_step_norm": 0.5,
        "total_energy_budget": 1.0, "gate_mode": "closed_loop",
    }
    values.update(overrides)
    return ClosedLoopConfig(**values)


def prefill(hook: ClosedLoopGenerationHook) -> None:
    result = hook(None, None, torch.tensor([[[0.0, 0.0], [0.25, 0.0]]]))
    torch.testing.assert_close(result, torch.tensor([[[0.0, 0.0], [0.25, 0.0]]]))


def test_closed_loop_corrects_only_when_below_target() -> None:
    hook = ClosedLoopGenerationHook(observer(), policy(), config())
    prefill(hook)
    low = hook(None, None, torch.tensor([[[0.0, 0.0]]]))
    high = hook(None, None, torch.tensor([[[1.2, 0.0]]]))
    torch.testing.assert_close(low[0, 0], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(high[0, 0], torch.tensor([1.2, 0.0]))
    assert [row["gate_open"] for row in hook.trace] == [True, False]


def test_uncertain_observer_abstains() -> None:
    hook = ClosedLoopGenerationHook(observer(uncertainty_offset=1.0), policy(), config(maximum_uncertainty=0.1))
    prefill(hook)
    unchanged = hook(None, None, torch.tensor([[[0.5, 0.0]]]))
    torch.testing.assert_close(unchanged[0, 0], torch.tensor([0.5, 0.0]))
    assert hook.trace[0]["gate_open"] is False


def test_perturbation_precedes_observation_and_control() -> None:
    perturbation = Perturbation(step=0, update=np.asarray([-1.0, 0.0]), family="reverse_target_axis")
    hook = ClosedLoopGenerationHook(observer(), policy(), config(), [perturbation])
    prefill(hook)
    changed = hook(None, None, torch.tensor([[[1.0, 0.0]]]))
    # The observer sees 0 after perturbation, so the controller adds its 0.5 cap.
    torch.testing.assert_close(changed[0, 0], torch.tensor([0.5, 0.0]))
    assert hook.trace[0]["scores"][0] == pytest.approx(0.0)
    assert hook.trace[0]["perturbation_family"] == "reverse_target_axis"


def test_energy_budget_is_never_exceeded() -> None:
    hook = ClosedLoopGenerationHook(
        observer(), policy(), config(maximum_step_norm=1.0, total_energy_budget=0.5)
    )
    prefill(hook)
    for _ in range(4):
        hook(None, None, torch.zeros((1, 1, 2)))
    assert hook.energy_used <= 0.5 + 1e-6
    assert hook.summary()["intervention_steps"] == 1


def test_withdrawal_disables_control_but_keeps_observing() -> None:
    hook = ClosedLoopGenerationHook(observer(), policy(), config(withdrawal_step=1))
    prefill(hook)
    first = hook(None, None, torch.zeros((1, 1, 2)))
    second = hook(None, None, torch.zeros((1, 1, 2)))
    assert first[0, 0, 0] > 0
    assert second[0, 0, 0] == 0
    assert hook.trace[1]["control_withdrawn"] is True


def test_recovery_metric_requires_consecutive_in_band_steps() -> None:
    trace = [
        {"step": 2, "scores": [0.0, 0.0, 0.0]},
        {"step": 3, "scores": [0.95, 0.0, 0.0]},
        {"step": 4, "scores": [0.7, 0.0, 0.0]},
        {"step": 5, "scores": [0.98, 0.0, 0.0]},
        {"step": 6, "scores": [1.02, 0.0, 0.0]},
    ]
    result = recovery_metrics(trace, "specificity", 2, target=1.0, tolerance=0.05)
    assert result["recovered"] is True
    assert result["recovery_step"] == 5
    assert result["recovery_latency_tokens"] == 3
    assert result["behavioral_recovery_required_separately"] is True


def test_group_bootstrap_observer_fits_case_groups() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(24, 4))
    targets = np.column_stack((features[:, 0], features[:, 1], features[:, 2]))
    groups = [f"case-{index // 4}" for index in range(24)]
    fitted = fit_group_bootstrap_observer(features, targets, groups, members=4, seed=3)
    estimate = fitted.estimate([features[0, :2]], features[0, 2:])
    assert estimate.scores.shape == (3,)
    assert estimate.uncertainty.shape == (3,)


def test_private_pilot_charter_passes_contract() -> None:
    path = Path(
        "private/projects/lollanet/evidence/controllability-audit/closed_loop_pilot_charter_v1.yaml"
    )
    validate_pilot_charter(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_charter_rejects_premature_attractor_claim() -> None:
    path = Path(
        "private/projects/lollanet/evidence/controllability-audit/closed_loop_pilot_charter_v1.yaml"
    )
    charter = yaml.safe_load(path.read_text(encoding="utf-8"))
    charter["claim_boundary"]["internal_attractor_claim_allowed"] = True
    with pytest.raises(ValueError, match="attractor claims"):
        validate_pilot_charter(charter)


def test_nonstandard_positive_control_axis_uses_same_hook() -> None:
    hidden = 2
    weights = np.zeros((3, 1, hidden * 2), dtype=np.float32)
    weights[:, 0, 0] = 1.0
    sentiment_observer = EnsembleLinearObserver(
        weights, np.zeros((3, 1), dtype=np.float32), np.zeros(hidden * 2),
        np.ones(hidden * 2), window_size=1, axes=("sentiment",),
    )
    sentiment_policy = LowRankFeedbackPolicy(
        np.asarray([[1.0], [0.0]]), np.ones((1, 1)), np.zeros((1, 1)),
        axes=("sentiment",),
    )
    sentiment_config = ClosedLoopConfig(
        axis="sentiment", target=1.0, deadband=0.1, maximum_uncertainty=0.1,
        control_interval=1, start_step=0, stop_step=2, maximum_step_norm=0.5,
        total_energy_budget=0.25,
    )
    hook = ClosedLoopGenerationHook(sentiment_observer, sentiment_policy, sentiment_config)
    prefill(hook)
    changed = hook(None, None, torch.zeros((1, 1, hidden)))
    torch.testing.assert_close(changed[0, 0], torch.tensor([0.5, 0.0]))
    result = recovery_metrics(hook.trace, "sentiment", 0, target=0.0, tolerance=0.01, consecutive_steps=1)
    assert result["recovered"] is True
