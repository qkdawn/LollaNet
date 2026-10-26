from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class StageLocked(RuntimeError):
    pass


def require_causal_features(gate_manifest: Path) -> list[dict[str, Any]]:
    if not gate_manifest.is_file():
        raise StageLocked("causal feature gate has not been run")
    manifest = json.loads(gate_manifest.read_text(encoding="utf-8"))
    features = list(manifest.get("causal_features", []))
    if manifest.get("status") != "passed" or not features:
        raise StageLocked("no feature passed the frozen causal gate")
    return features


def integrated_gradients_attribution(gradients: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Feature-node attribution used to seed Sparse Feature Circuits."""
    if gradients.shape != activations.shape:
        raise ValueError("gradient and activation paths must have identical shape")
    return np.trapz(gradients, axis=0) * (activations[-1] - activations[0]) / max(len(gradients) - 1, 1)


def circuit_faithfulness(full_effect: float, retained_effect: float) -> float:
    return float(retained_effect / full_effect) if abs(full_effect) > 1e-12 else 0.0


def fit_sae_targeted_steering(effect_matrix: np.ndarray, target_effect: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge solution for SAE-TS: choose steering coefficients matching sparse target effects."""
    design = np.asarray(effect_matrix, dtype=np.float64)
    target = np.asarray(target_effect, dtype=np.float64)
    return np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ target)


def das_interchange_loss(predicted: Any, counterfactual_target: Any) -> Any:
    """Squared interchange-intervention loss; optimization lives in the model runner."""
    return ((predicted - counterfactual_target) ** 2).mean()


def choose_representation_result(sae_passed: bool, das_passed: bool) -> str:
    if das_passed and not sae_passed:
        return "distributed_causal_subspace"
    if sae_passed and das_passed:
        return "sparse_and_distributed_controls_both_supported"
    if sae_passed:
        return "sparse_causal_features"
    return "no_causal_representation_established"
