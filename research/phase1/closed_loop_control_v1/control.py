from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


AXES = ("specificity", "independence", "non_template")
GATE_MODES = {"closed_loop", "always_on", "disabled"}


@dataclass(frozen=True)
class ObserverEstimate:
    scores: np.ndarray
    velocity: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class Perturbation:
    step: int
    update: np.ndarray
    family: str

    def __post_init__(self) -> None:
        update = np.asarray(self.update)
        if self.step < 0 or update.ndim != 1 or not np.isfinite(update).all():
            raise ValueError("perturbation requires a nonnegative step and finite vector")


@dataclass(frozen=True)
class ClosedLoopConfig:
    axis: str
    target: float
    deadband: float
    maximum_uncertainty: float
    control_interval: int
    start_step: int
    stop_step: int
    maximum_step_norm: float
    total_energy_budget: float
    gate_mode: str = "closed_loop"
    withdrawal_step: int | None = None

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("control axis must be nonempty")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"unknown gate mode: {self.gate_mode}")
        if self.deadband < 0 or self.maximum_uncertainty < 0:
            raise ValueError("deadband and uncertainty threshold must be nonnegative")
        if self.control_interval <= 0 or not 0 <= self.start_step < self.stop_step:
            raise ValueError("invalid control interval or step window")
        if self.maximum_step_norm <= 0 or self.total_energy_budget <= 0:
            raise ValueError("control norm and energy budget must be positive")
        if self.withdrawal_step is not None and self.withdrawal_step < self.start_step:
            raise ValueError("withdrawal cannot precede the control start")


class EnsembleLinearObserver:
    """A rolling linear observer with ensemble disagreement as uncertainty.

    Each member maps the concatenation of the rolling activation mean and the
    prompt-end activation to the three response-quality axes. Prompt conditioning
    prevents a single global target state from being imposed across unrelated cases.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        window_size: int,
        axes: Sequence[str] = AXES,
    ) -> None:
        weights = np.asarray(weights, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        feature_scale = np.asarray(feature_scale, dtype=np.float32)
        if weights.ndim != 3:
            raise ValueError("observer weights must have shape [members, axes, features]")
        if bias.shape != weights.shape[:2]:
            raise ValueError("observer bias shape mismatch")
        if feature_mean.shape != (weights.shape[2],) or feature_scale.shape != feature_mean.shape:
            raise ValueError("observer feature normalization shape mismatch")
        if len(tuple(axes)) != weights.shape[1] or window_size <= 0:
            raise ValueError("observer axes or window size mismatch")
        if np.any(feature_scale <= 0):
            raise ValueError("observer feature scales must be positive")
        self.weights = weights
        self.bias = bias
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.window_size = int(window_size)
        self.axes = tuple(axes)

    @property
    def hidden_width(self) -> int:
        if self.weights.shape[2] % 2:
            raise ValueError("prompt-conditioned observer feature width must be even")
        return self.weights.shape[2] // 2

    def estimate(
        self,
        history: Sequence[np.ndarray],
        prompt_context: np.ndarray,
        previous_scores: np.ndarray | None = None,
    ) -> ObserverEstimate:
        if not history:
            raise ValueError("observer requires at least one generated activation")
        values = np.stack([np.asarray(row, dtype=np.float32) for row in history[-self.window_size :]])
        prompt = np.asarray(prompt_context, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.hidden_width or prompt.shape != (self.hidden_width,):
            raise ValueError("observer activation width mismatch")
        feature = np.concatenate((values.mean(axis=0), prompt))
        normalized = (feature - self.feature_mean) / self.feature_scale
        member_scores = np.einsum("maf,f->ma", self.weights, normalized) + self.bias
        scores = member_scores.mean(axis=0)
        uncertainty = member_scores.std(axis=0)
        if previous_scores is None:
            velocity = np.zeros_like(scores)
        else:
            previous = np.asarray(previous_scores, dtype=np.float32)
            if previous.shape != scores.shape:
                raise ValueError("previous observer score shape mismatch")
            velocity = scores - previous
        return ObserverEstimate(scores=scores, velocity=velocity, uncertainty=uncertainty)


class LowRankFeedbackPolicy:
    """Prompt-conditioned state feedback mapped through a low-rank actuator."""

    def __init__(
        self,
        basis: np.ndarray,
        error_gain: np.ndarray,
        velocity_gain: np.ndarray,
        prompt_gain: np.ndarray | None = None,
        axes: Sequence[str] = AXES,
    ) -> None:
        basis = np.asarray(basis, dtype=np.float32)
        error_gain = np.asarray(error_gain, dtype=np.float32)
        velocity_gain = np.asarray(velocity_gain, dtype=np.float32)
        prompt_gain_array = None if prompt_gain is None else np.asarray(prompt_gain, dtype=np.float32)
        if basis.ndim != 2 or basis.shape[1] == 0:
            raise ValueError("actuator basis must have shape [hidden, rank]")
        if basis.shape[1] > basis.shape[0]:
            raise ValueError("actuator rank cannot exceed hidden width")
        axes_tuple = tuple(axes)
        if not axes_tuple or len(set(axes_tuple)) != len(axes_tuple):
            raise ValueError("actuator axes must be nonempty and unique")
        if error_gain.shape != (basis.shape[1], len(axes_tuple)):
            raise ValueError("error gain must map all axes into the low-rank coefficients")
        if velocity_gain.shape != error_gain.shape:
            raise ValueError("velocity gain shape mismatch")
        if prompt_gain_array is not None and prompt_gain_array.shape != (basis.shape[1], basis.shape[0]):
            raise ValueError("prompt gain must have shape [rank, hidden]")
        q, _ = np.linalg.qr(basis)
        self.basis = q[:, : basis.shape[1]].astype(np.float32)
        self.error_gain = error_gain
        self.velocity_gain = velocity_gain
        self.prompt_gain = prompt_gain_array
        self.axes = axes_tuple

    @property
    def hidden_width(self) -> int:
        return self.basis.shape[0]

    def update(
        self,
        error: np.ndarray,
        velocity: np.ndarray,
        prompt_context: np.ndarray,
    ) -> np.ndarray:
        error = np.asarray(error, dtype=np.float32)
        velocity = np.asarray(velocity, dtype=np.float32)
        prompt = np.asarray(prompt_context, dtype=np.float32)
        if error.shape != (len(self.axes),) or velocity.shape != error.shape:
            raise ValueError("feedback state must contain all response-quality axes")
        if prompt.shape != (self.hidden_width,):
            raise ValueError("feedback prompt-context width mismatch")
        coefficients = self.error_gain @ error - self.velocity_gain @ velocity
        if self.prompt_gain is not None:
            coefficients = coefficients + self.prompt_gain @ prompt
        return (self.basis @ coefficients).astype(np.float32)


class ClosedLoopGenerationHook:
    """One hook that applies scheduled perturbations and state-feedback control.

    Combining both operations fixes their order: perturbation, observation, gate,
    then control. The hook ignores prompt prefill and acts only on cached one-token
    generation calls.
    """

    def __init__(
        self,
        observer: EnsembleLinearObserver,
        policy: LowRankFeedbackPolicy,
        config: ClosedLoopConfig,
        perturbations: Sequence[Perturbation] = (),
    ) -> None:
        if observer.hidden_width != policy.hidden_width:
            raise ValueError("observer and actuator hidden widths differ")
        if observer.axes != policy.axes:
            raise ValueError("observer and actuator axes differ")
        if config.axis not in observer.axes:
            raise ValueError("configured axis is absent from observer and actuator")
        self.observer = observer
        self.policy = policy
        self.config = config
        self.axis_index = observer.axes.index(config.axis)
        self.perturbations = {row.step: row for row in perturbations}
        if len(self.perturbations) != len(tuple(perturbations)):
            raise ValueError("only one perturbation is allowed per generation step")
        for row in self.perturbations.values():
            if np.asarray(row.update).shape != (policy.hidden_width,):
                raise ValueError("perturbation hidden width mismatch")
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.prompt_context: np.ndarray | None = None
        self.history: list[np.ndarray] = []
        self.previous_scores: np.ndarray | None = None
        self.energy_used = 0.0
        self.trace: list[dict[str, Any]] = []

    def _control_is_available(self, step: int) -> bool:
        cfg = self.config
        if cfg.gate_mode == "disabled" or not cfg.start_step <= step < cfg.stop_step:
            return False
        if cfg.withdrawal_step is not None and step >= cfg.withdrawal_step:
            return False
        return (step - cfg.start_step) % cfg.control_interval == 0

    def _bounded_update(self, raw_update: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(raw_update))
        remaining_energy = max(0.0, self.config.total_energy_budget - self.energy_used)
        energy_tolerance = max(1e-9, self.config.total_energy_budget * 1e-6)
        if remaining_energy <= energy_tolerance:
            return np.zeros_like(raw_update)
        cap = min(self.config.maximum_step_norm, float(np.sqrt(remaining_energy)))
        if norm <= 1e-12 or cap <= 0:
            return np.zeros_like(raw_update)
        return raw_update * min(1.0, cap / norm)

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("closed-loop hook expects [1, tokens, hidden] activations")
        if hidden.shape[1] != 1:
            self.prompt_context = hidden[0, -1].detach().float().cpu().numpy()
            return output
        if self.prompt_context is None:
            raise RuntimeError("closed-loop hook saw generation before prompt prefill")

        step = self.step
        self.step += 1
        raw = hidden[0, 0].detach().float().cpu().numpy()
        perturbation = self.perturbations.get(step)
        perturbed = raw.copy()
        perturbation_norm = 0.0
        perturbation_family = None
        if perturbation is not None:
            delta = np.asarray(perturbation.update, dtype=np.float32)
            perturbed += delta
            perturbation_norm = float(np.linalg.norm(delta))
            perturbation_family = perturbation.family

        self.history.append(perturbed)
        estimate = self.observer.estimate(self.history, self.prompt_context, self.previous_scores)
        self.previous_scores = estimate.scores.copy()
        error = np.zeros(len(self.observer.axes), dtype=np.float32)
        error[self.axis_index] = float(self.config.target - estimate.scores[self.axis_index])
        eligible = self._control_is_available(step)
        if self.config.gate_mode == "closed_loop":
            eligible = bool(
                eligible
                and error[self.axis_index] > self.config.deadband
                and estimate.uncertainty[self.axis_index] <= self.config.maximum_uncertainty
            )
        raw_control = (
            self.policy.update(error, estimate.velocity, self.prompt_context)
            if eligible else np.zeros(self.policy.hidden_width, dtype=np.float32)
        )
        control = self._bounded_update(raw_control)
        control_norm = float(np.linalg.norm(control))
        self.energy_used += control_norm**2

        changed_vector = perturbed + control
        changed = hidden.clone()
        changed[0, 0] = torch.as_tensor(changed_vector, device=hidden.device, dtype=hidden.dtype)
        self.trace.append({
            "step": step,
            "axis_names": list(self.observer.axes),
            "scores": estimate.scores.tolist(),
            "velocity": estimate.velocity.tolist(),
            "uncertainty": estimate.uncertainty.tolist(),
            "axis_error": float(error[self.axis_index]),
            "gate_open": bool(eligible),
            "control_norm": control_norm,
            "cumulative_control_energy": float(self.energy_used),
            "perturbation_family": perturbation_family,
            "perturbation_norm": perturbation_norm,
            "control_withdrawn": bool(
                self.config.withdrawal_step is not None and step >= self.config.withdrawal_step
            ),
        })
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def summary(self) -> dict[str, Any]:
        applied = sum(row["control_norm"] > 1e-8 for row in self.trace)
        return {
            "generation_steps_seen": self.step,
            "intervention_steps": applied,
            "intervention_rate": (
                applied / len(self.trace)
                if self.trace else 0.0
            ),
            "cumulative_control_energy": float(self.energy_used),
            "perturbation_steps": [row["step"] for row in self.trace if row["perturbation_family"]],
            "trace": self.trace,
        }


def fit_group_bootstrap_observer(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    ridge: float = 10.0,
    members: int = 8,
    seed: int = 0,
    window_size: int = 4,
    axes: Sequence[str] = AXES,
) -> EnsembleLinearObserver:
    """Fit an ensemble by bootstrapping independent case groups, not responses."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    group_array = np.asarray(groups)
    axes_tuple = tuple(axes)
    if x.ndim != 2 or y.shape != (len(x), len(axes_tuple)) or group_array.shape != (len(x),):
        raise ValueError("observer fit arrays have incompatible shapes")
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 3 or ridge <= 0 or members < 2:
        raise ValueError("observer ensemble requires >=3 groups, positive ridge and >=2 members")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    rng = np.random.default_rng(seed)
    weight_rows, bias_rows = [], []
    for _ in range(members):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_array == group) for group in sampled_groups])
        design = np.column_stack((np.ones(len(indices)), normalized[indices]))
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y[indices])
        bias_rows.append(coefficients[0])
        weight_rows.append(coefficients[1:].T)
    return EnsembleLinearObserver(
        np.stack(weight_rows), np.stack(bias_rows), mean, scale, window_size, axes_tuple
    )


def recovery_metrics(
    trace: Sequence[dict[str, Any]],
    axis: str,
    perturbation_step: int,
    target: float,
    tolerance: float,
    consecutive_steps: int = 2,
) -> dict[str, Any]:
    if not axis or perturbation_step < 0 or tolerance < 0 or consecutive_steps <= 0:
        raise ValueError("invalid recovery-metric configuration")
    if trace and "axis_names" in trace[0]:
        axis_names = tuple(trace[0]["axis_names"])
        if axis not in axis_names:
            raise ValueError("recovery axis is absent from trace")
        axis_index = axis_names.index(axis)
    elif axis in AXES:
        axis_index = AXES.index(axis)
    else:
        raise ValueError("nonstandard recovery axes require axis_names in the trace")
    post = [row for row in trace if int(row["step"]) >= perturbation_step]
    recovered_at: int | None = None
    for index in range(0, len(post) - consecutive_steps + 1):
        window = post[index : index + consecutive_steps]
        if all(abs(float(row["scores"][axis_index]) - target) <= tolerance for row in window):
            recovered_at = int(window[0]["step"])
            break
    return {
        "axis": axis,
        "perturbation_step": perturbation_step,
        "recovered": recovered_at is not None,
        "recovery_step": recovered_at,
        "recovery_latency_tokens": None if recovered_at is None else recovered_at - perturbation_step,
        "post_perturbation_observations": len(post),
        "internal_state_only": True,
        "behavioral_recovery_required_separately": True,
    }
