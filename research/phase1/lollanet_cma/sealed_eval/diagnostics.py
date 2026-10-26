from __future__ import annotations

from typing import Any

import numpy as np


def token_level_kl(steered_logits: np.ndarray, reference_logits: np.ndarray) -> float:
    steered = np.asarray(steered_logits, dtype=np.float64)
    reference = np.asarray(reference_logits, dtype=np.float64)
    if steered.ndim != 1 or steered.shape != reference.shape or steered.size < 2:
        raise ValueError("token KL requires aligned one-dimensional full-vocabulary logits")
    if not np.isfinite(steered).all() or not np.isfinite(reference).all():
        raise ValueError("token KL logits must be finite")
    steered_log_probs = steered - np.logaddexp.reduce(steered)
    reference_log_probs = reference - np.logaddexp.reduce(reference)
    probability = np.exp(steered_log_probs)
    value = float(np.sum(probability * (steered_log_probs - reference_log_probs)))
    return max(0.0, value)


def mean_token_level_kl(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("mean token KL requires non-empty finite non-negative values")
    return float(array.mean())


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or targets.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("AUROC requires aligned finite one-dimensional values")
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("AUROC requires both binary classes")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = targets == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def fit_l2_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    l2: float,
    standard_deviation_floor: float,
) -> dict[str, Any]:
    features = np.asarray(activations, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError("probe training requires aligned activation rows and labels")
    if not np.isfinite(features).all() or set(targets.tolist()) != {0, 1}:
        raise ValueError("probe training data must be finite and contain both labels")
    if l2 <= 0 or standard_deviation_floor <= 0:
        raise ValueError("probe regularization and scale floor must be positive")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), standard_deviation_floor)
    standardized = (features - mean) / scale
    signed = targets.astype(np.float64) * 2.0 - 1.0
    intercept = float(signed.mean())
    centered_target = signed - intercept
    kernel = standardized @ standardized.T
    dual = np.linalg.solve(
        kernel + float(l2) * np.eye(features.shape[0], dtype=np.float64),
        centered_target,
    )
    weights = standardized.T @ dual
    return {
        "weights": weights,
        "intercept": intercept,
        "feature_mean": mean,
        "feature_scale": scale,
        "l2": float(l2),
        "standard_deviation_floor": float(standard_deviation_floor),
    }


def evaluate_linear_probe(
    probe: dict[str, Any], activations: np.ndarray, labels: np.ndarray,
) -> dict[str, float]:
    features = np.asarray(activations, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(probe["weights"], dtype=np.float64)
    mean = np.asarray(probe["feature_mean"], dtype=np.float64)
    scale = np.asarray(probe["feature_scale"], dtype=np.float64)
    if features.ndim != 2 or targets.shape != (features.shape[0],) or features.shape[1] != weights.size:
        raise ValueError("probe evaluation data differ from the fitted feature contract")
    scores = ((features - mean) / scale) @ weights + float(probe["intercept"])
    predicted = (scores >= 0).astype(np.int64)
    signed = targets.astype(np.float64) * 2.0 - 1.0
    return {
        "accuracy": float(np.mean(predicted == targets)),
        "auroc": binary_auroc(scores, targets),
        "mean_gold_margin": float(np.mean(signed * scores)),
    }


def select_probe_l2(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    development_activations: np.ndarray,
    development_labels: np.ndarray,
    l2_grid: list[float],
    standard_deviation_floor: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if not l2_grid or len(l2_grid) != len(set(l2_grid)) or any(value <= 0 for value in l2_grid):
        raise ValueError("probe L2 grid must contain distinct positive values")
    candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    for l2 in l2_grid:
        probe = fit_l2_linear_probe(
            train_activations, train_labels, float(l2), standard_deviation_floor,
        )
        metrics = evaluate_linear_probe(probe, development_activations, development_labels)
        candidates.append((probe, {"l2": float(l2), **metrics}))
    chosen_probe, _ = max(
        candidates,
        key=lambda item: (
            item[1]["accuracy"], item[1]["auroc"], item[1]["mean_gold_margin"], item[1]["l2"],
        ),
    )
    return chosen_probe, [metrics for _, metrics in candidates]


def unrelated_control_result(
    base_accuracy: float,
    family_direction_accuracy: dict[str, float],
    maximum_allowed_loss: float,
) -> dict[str, Any]:
    numerical_tolerance = 1e-12
    values = {key: float(value) for key, value in family_direction_accuracy.items()}
    if not values or not 0 <= base_accuracy <= 1 or any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError("control accuracies must be non-empty probabilities")
    if not 0 <= maximum_allowed_loss <= 1:
        raise ValueError("maximum allowed control loss must be a probability")
    deltas = {key: value - float(base_accuracy) for key, value in sorted(values.items())}
    worst_condition = min(deltas, key=deltas.get)
    worst_delta = deltas[worst_condition]
    return {
        "base_accuracy": float(base_accuracy),
        "family_direction_accuracy": dict(sorted(values.items())),
        "accuracy_delta_vs_base": deltas,
        "worst_condition": worst_condition,
        "minimum_accuracy_delta_vs_base": worst_delta,
        "maximum_allowed_accuracy_loss": float(maximum_allowed_loss),
        "numerical_tolerance": numerical_tolerance,
        "passed": bool(worst_delta >= -float(maximum_allowed_loss) - numerical_tolerance),
    }
