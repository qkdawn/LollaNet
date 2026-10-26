from __future__ import annotations

import numpy as np
import pytest

from research.phase1.target_attractor_pilot.core import (
    Schedule, intervention_at_step, matched_dynamics, norm_matched_random_direction, summarize_trace,
)
from research.phase1.target_attractor_pilot.data import direction_rows, smoke_cases


def test_direction_source_and_smoke_cases_never_access_evaluation() -> None:
    rows = direction_rows()
    assert len(rows) == 32
    assert {row["label"] for row in rows} == {0, 1}
    assert all(case.split == "development" for case in smoke_cases(4))
    assert len(smoke_cases(99)) == 4


def test_frozen_schedule_releases_and_perturbs_at_exact_steps() -> None:
    schedule = Schedule(target_pulse_tokens=2, release_tokens=3, perturbation_tokens=2)
    modes = [intervention_at_step("target_pulse_release_perturb", step, schedule) for step in range(9)]
    assert modes == ["target", "target", "none", "none", "none", "anti_target", "anti_target", "none", "none"]
    assert [intervention_at_step("target_pulse_release", step, schedule) for step in range(4)] == [
        "target", "target", "none", "none",
    ]
    with pytest.raises(ValueError, match="unknown condition"):
        intervention_at_step("typo", 0, schedule)


def test_random_control_is_norm_matched_and_orthogonal() -> None:
    target = np.asarray([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    control = norm_matched_random_direction(target, 7)
    assert np.linalg.norm(control) == pytest.approx(np.linalg.norm(target), rel=1e-6)
    assert float(control @ target) == pytest.approx(0.0, abs=1e-6)


def test_trace_summary_uses_pre_intervention_state_and_frozen_windows() -> None:
    trace = [{"step": step, "projection_before": float(step)} for step in range(10)]
    summary = summarize_trace(trace, Schedule(2, 3, 2))
    assert summary == {
        "pulse_mean": 0.5,
        "release_mean": 3.0,
        "perturbation_mean": 5.5,
        "recovery_mean": 8.0,
    }


def test_recovery_is_measured_against_matched_unperturbed_time_trend() -> None:
    records = [
        {"case_id": "a", "condition": "target_pulse_release", "trace_summary": {
            "perturbation_mean": 10.0, "recovery_mean": 6.0,
        }},
        {"case_id": "a", "condition": "target_pulse_release_perturb", "trace_summary": {
            "perturbation_mean": 8.0, "recovery_mean": 5.5,
        }},
    ]
    result = matched_dynamics(records)
    assert result["mean_perturbation_effect_vs_unperturbed"] == -2.0
    assert result["mean_residual_gap_after_recovery"] == -0.5
    assert result["mean_recovery_fraction"] == 0.75
