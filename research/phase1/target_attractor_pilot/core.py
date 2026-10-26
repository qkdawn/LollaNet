from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers


@dataclass(frozen=True)
class Schedule:
    target_pulse_tokens: int = 16
    release_tokens: int = 16
    perturbation_tokens: int = 8

    @property
    def perturbation_start(self) -> int:
        return self.target_pulse_tokens + self.release_tokens

    @property
    def perturbation_end(self) -> int:
        return self.perturbation_start + self.perturbation_tokens


def intervention_at_step(condition: str, step: int, schedule: Schedule) -> str:
    if condition == "continuous_target":
        return "target"
    if condition in {"target_pulse_release", "target_pulse_release_perturb"}:
        if step < schedule.target_pulse_tokens:
            return "target"
        if condition.endswith("perturb") and schedule.perturbation_start <= step < schedule.perturbation_end:
            return "anti_target"
    if condition == "random_pulse_release" and step < schedule.target_pulse_tokens:
        return "random"
    if condition != "baseline" and condition not in {
        "continuous_target", "target_pulse_release", "target_pulse_release_perturb", "random_pulse_release",
    }:
        raise ValueError(f"unknown condition: {condition}")
    return "none"


def norm_matched_random_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    target = np.asarray(direction, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    if not np.isfinite(target_norm) or target_norm == 0:
        raise ValueError("target direction is degenerate")
    unit_target = target / target_norm
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(target.shape)
    random = random - float(random @ unit_target) * unit_target
    random_norm = float(np.linalg.norm(random))
    if not np.isfinite(random_norm) or random_norm < 1e-12:
        raise ValueError("random control direction is degenerate")
    return (random / random_norm * target_norm).astype(np.float32)


@contextmanager
def scheduled_residual_intervention(
    model: Any,
    layer_index: int,
    target_direction: np.ndarray,
    random_direction: np.ndarray,
    strength: float,
    monitor_layer_index: int | None = None,
    monitor_direction: np.ndarray | None = None,
) -> Iterator[dict[str, Any]]:
    import torch

    target = torch.as_tensor(target_direction)
    random = torch.as_tensor(random_direction)
    monitor_layer_index = layer_index if monitor_layer_index is None else monitor_layer_index
    monitor_direction = target_direction if monitor_direction is None else monitor_direction
    monitor = torch.as_tensor(monitor_direction)
    state: dict[str, Any] = {
        "mode": "none", "step": -1, "calls": 0, "monitor_calls": 0,
        "trace": [], "finite": True,
    }

    def intervention_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        target_vector = target.to(device=hidden.device, dtype=hidden.dtype)
        random_vector = random.to(device=hidden.device, dtype=hidden.dtype)
        mode = str(state["mode"])
        if mode == "target":
            modified[:, -1, :] += target_vector * float(strength)
        elif mode == "anti_target":
            modified[:, -1, :] -= target_vector * float(strength)
        elif mode == "random":
            modified[:, -1, :] += random_vector * float(strength)
        after = modified[:, -1, :].float()
        state["finite"] = bool(state["finite"] and torch.isfinite(after).all().item())
        state["calls"] += 1
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    def monitor_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        value = hidden[:, -1, :].float()
        monitor_vector = monitor.to(device=hidden.device, dtype=torch.float32)
        state["finite"] = bool(state["finite"] and torch.isfinite(value).all().item())
        state["trace"].append({
            "step": int(state["step"]),
            "mode": str(state["mode"]),
            "projection_before": float((value * monitor_vector).sum(dim=-1).mean().item()),
        })
        state["monitor_calls"] += 1
        return output

    if monitor_layer_index <= layer_index:
        raise ValueError("monitor layer must be downstream of the intervention layer")
    intervention_handle = _layers(model)[layer_index].register_forward_hook(intervention_hook)
    monitor_handle = _layers(model)[monitor_layer_index].register_forward_hook(monitor_hook)
    try:
        yield state
    finally:
        intervention_handle.remove()
        monitor_handle.remove()


def window_mean(trace: list[dict[str, Any]], start: int, end: int) -> float | None:
    values = [float(row["projection_before"]) for row in trace if start <= int(row["step"]) < end]
    return float(np.mean(values)) if values else None


def summarize_trace(trace: list[dict[str, Any]], schedule: Schedule) -> dict[str, float | None]:
    return {
        "pulse_mean": window_mean(trace, 0, schedule.target_pulse_tokens),
        "release_mean": window_mean(trace, schedule.target_pulse_tokens, schedule.perturbation_start),
        "perturbation_mean": window_mean(trace, schedule.perturbation_start, schedule.perturbation_end),
        "recovery_mean": window_mean(trace, schedule.perturbation_end, schedule.perturbation_end + 16),
    }


def matched_dynamics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compare perturbation recovery with the case-matched unperturbed trajectory."""
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        by_case.setdefault(str(row["case_id"]), {})[str(row["condition"])] = row
    perturbation_effects: list[float] = []
    residual_gaps: list[float] = []
    recovery_fractions: list[float] = []
    for conditions in by_case.values():
        perturbed = conditions.get("target_pulse_release_perturb")
        reference = conditions.get("target_pulse_release")
        if perturbed is None or reference is None:
            continue
        perturbed_summary = perturbed["trace_summary"]
        reference_summary = reference["trace_summary"]
        values = (
            perturbed_summary.get("perturbation_mean"), reference_summary.get("perturbation_mean"),
            perturbed_summary.get("recovery_mean"), reference_summary.get("recovery_mean"),
        )
        if any(value is None for value in values):
            continue
        perturbation_effect = float(values[0] - values[1])
        residual_gap = float(values[2] - values[3])
        perturbation_effects.append(perturbation_effect)
        residual_gaps.append(residual_gap)
        if abs(perturbation_effect) > 1e-12:
            recovery_fractions.append(1.0 - abs(residual_gap) / abs(perturbation_effect))
    return {
        "mean_perturbation_effect_vs_unperturbed": (
            float(np.mean(perturbation_effects)) if perturbation_effects else None
        ),
        "mean_residual_gap_after_recovery": float(np.mean(residual_gaps)) if residual_gaps else None,
        "mean_recovery_fraction": float(np.mean(recovery_fractions)) if recovery_fractions else None,
    }
