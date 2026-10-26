from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def yes_probability(yes_score: float, no_score: float) -> float:
    difference = float(yes_score) - float(no_score)
    if difference >= 0:
        return float(1.0 / (1.0 + np.exp(-difference)))
    exp_difference = float(np.exp(difference))
    return float(exp_difference / (1.0 + exp_difference))


def gold_probability(yes_score: float, no_score: float, gold_answer: str) -> float:
    probability = yes_probability(yes_score, no_score)
    if gold_answer == "yes":
        return probability
    if gold_answer == "no":
        return 1.0 - probability
    raise ValueError(f"unsupported binary gold answer: {gold_answer}")


def binary_brier(yes_probability_value: float, gold_answer: str) -> float:
    if gold_answer not in {"yes", "no"}:
        raise ValueError(f"unsupported binary gold answer: {gold_answer}")
    target = 1.0 if gold_answer == "yes" else 0.0
    return float((float(yes_probability_value) - target) ** 2)


def expected_calibration_error(
    confidence: list[float],
    correctness: list[bool],
    bins: int,
) -> float:
    if not confidence or len(confidence) != len(correctness) or bins < 1:
        raise ValueError("ECE requires aligned non-empty values and at least one bin")
    confidence_values = np.asarray(confidence, dtype=np.float64)
    correct_values = np.asarray(correctness, dtype=np.float64)
    if not np.isfinite(confidence_values).all() or np.any(
        (confidence_values < 0) | (confidence_values > 1)
    ):
        raise ValueError("ECE confidence values must be finite probabilities")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence_values >= edges[index]) & (confidence_values <= edges[index + 1])
        else:
            mask = (confidence_values >= edges[index]) & (confidence_values < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(confidence_values[mask].mean()) - float(correct_values[mask].mean())
            )
    return float(result)


def _accuracy_by(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(bool(row["correct"]))
    return {group: float(np.mean(values)) for group, values in sorted(grouped.items())}


def summarize_rows(rows: list[dict[str, Any]], families: list[str], ece_bins: int) -> dict[str, Any]:
    record_ids = [str(row["record_id"]) for row in rows]
    if not rows or len(record_ids) != len(set(record_ids)):
        raise ValueError("condition summary requires non-empty unique records")
    family_accuracy = _accuracy_by(rows, "model_family")
    if set(family_accuracy) != set(families):
        raise ValueError("condition summary does not cover every frozen family")
    confidences = [float(row["predicted_answer_probability"]) for row in rows]
    kl_values = [float(row["token_level_kl"]) for row in rows if row.get("token_level_kl") is not None]
    return {
        "record_count": len(rows),
        "overall_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "family_accuracy": family_accuracy,
        "macro_family_accuracy": float(np.mean([family_accuracy[family] for family in families])),
        "query_type_accuracy": _accuracy_by(rows, "query_type"),
        "rung_accuracy": _accuracy_by(rows, "rung"),
        "parse_failure_rate": float(np.mean([not bool(row["parse_ok"]) for row in rows])),
        "mean_gold_answer_margin": float(np.mean([float(row["gold_answer_margin"]) for row in rows])),
        "mean_brier_score": float(np.mean([float(row["brier_score"]) for row in rows])),
        "expected_calibration_error": expected_calibration_error(
            confidences, [bool(row["correct"]) for row in rows], ece_bins,
        ),
        "mean_output_tokens": float(np.mean([int(row["output_tokens"]) for row in rows])),
        "median_latency_seconds": float(np.median([float(row["latency_seconds"]) for row in rows])),
        "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in rows),
        "mean_token_level_kl": float(np.mean(kl_values)) if kl_values else None,
    }


def paired_stratified_bootstrap(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    families: list[str],
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if samples < 1 or not 0 < confidence_level < 1:
        raise ValueError("bootstrap samples and confidence level are invalid")
    left = {str(row["record_id"]): row for row in left_rows}
    right = {str(row["record_id"]): row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows) or set(left) != set(right):
        raise ValueError("paired bootstrap requires identical unique record identities")
    family_ids: dict[str, list[str]] = {}
    for family in families:
        ids = sorted(record_id for record_id, row in left.items() if row["model_family"] == family)
        if not ids or any(right[record_id]["model_family"] != family for record_id in ids):
            raise ValueError(f"paired bootstrap family coverage differs: {family}")
        family_ids[family] = ids
    if set().union(*[set(ids) for ids in family_ids.values()]) != set(left):
        raise ValueError("paired bootstrap rows contain an unexpected family")

    def macro_difference(sampled: dict[str, list[str]]) -> float:
        differences = []
        for family in families:
            ids = sampled[family]
            differences.append(float(np.mean([
                float(bool(left[record_id]["correct"])) - float(bool(right[record_id]["correct"]))
                for record_id in ids
            ])))
        return float(np.mean(differences))

    point = macro_difference(family_ids)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = {
            family: [ids[position] for position in rng.integers(0, len(ids), size=len(ids))]
            for family, ids in family_ids.items()
        }
        draws[index] = macro_difference(sampled)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "metric": "macro_family_accuracy_difference",
        "left_minus_right": point,
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "resampling": "record_id_stratified_by_model_family",
    }


def faithfulness_matrix(
    base_rows: list[dict[str, Any]],
    direction_rows: dict[str, list[dict[str, Any]]],
    families: list[str],
) -> dict[str, Any]:
    base_accuracy = _accuracy_by(base_rows, "model_family")
    if set(base_accuracy) != set(families) or set(direction_rows) != set(families):
        raise ValueError("faithfulness matrix lacks a frozen family")
    matrix: dict[str, dict[str, float]] = {}
    diagonal_advantage: dict[str, float] = {}
    for direction_family in families:
        accuracy = _accuracy_by(direction_rows[direction_family], "model_family")
        if set(accuracy) != set(families):
            raise ValueError(f"faithfulness direction lacks family coverage: {direction_family}")
        effects = {
            task_family: float(accuracy[task_family] - base_accuracy[task_family])
            for task_family in families
        }
        matrix[direction_family] = effects
        off_diagonal = [effects[family] for family in families if family != direction_family]
        diagonal_advantage[direction_family] = float(
            effects[direction_family] - np.mean(off_diagonal)
        )
    return {
        "effect_definition": "direction_accuracy_minus_base_accuracy",
        "matrix": matrix,
        "diagonal_advantage": diagonal_advantage,
        "positive_diagonal_advantage_count": sum(value > 0 for value in diagonal_advantage.values()),
    }


def faithfulness_diagonal_bootstrap(
    base_rows: list[dict[str, Any]],
    direction_rows: dict[str, list[dict[str, Any]]],
    families: list[str],
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if samples < 1 or not 0 < confidence_level < 1:
        raise ValueError("faithfulness bootstrap settings are invalid")
    base = {str(row["record_id"]): row for row in base_rows}
    if len(base) != len(base_rows):
        raise ValueError("faithfulness base rows contain duplicate records")
    family_ids = {
        family: sorted(record_id for record_id, row in base.items() if row["model_family"] == family)
        for family in families
    }
    if any(not ids for ids in family_ids.values()) or set().union(*map(set, family_ids.values())) != set(base):
        raise ValueError("faithfulness base family coverage is incomplete")
    tail = (1.0 - confidence_level) / 2.0
    results: dict[str, Any] = {}
    for direction_index, direction_family in enumerate(families):
        observed = {str(row["record_id"]): row for row in direction_rows[direction_family]}
        if len(observed) != len(direction_rows[direction_family]) or set(observed) != set(base):
            raise ValueError(f"faithfulness direction coverage differs: {direction_family}")

        def advantage(sampled: dict[str, list[str]]) -> float:
            effects = {}
            for task_family, ids in sampled.items():
                effects[task_family] = float(np.mean([
                    float(bool(observed[record_id]["correct"]))
                    - float(bool(base[record_id]["correct"]))
                    for record_id in ids
                ]))
            return float(
                effects[direction_family]
                - np.mean([value for family, value in effects.items() if family != direction_family])
            )

        point = advantage(family_ids)
        rng = np.random.default_rng(int(seed) + direction_index)
        draws = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sampled = {
                family: [ids[position] for position in rng.integers(0, len(ids), size=len(ids))]
                for family, ids in family_ids.items()
            }
            draws[index] = advantage(sampled)
        lower = float(np.quantile(draws, tail))
        upper = float(np.quantile(draws, 1.0 - tail))
        results[direction_family] = {
            "diagonal_advantage": point,
            "lower": lower,
            "upper": upper,
            "confidence_level": float(confidence_level),
            "bootstrap_samples": int(samples),
            "seed": int(seed) + direction_index,
            "significant_positive": bool(lower > 0),
        }
    return {
        "families": results,
        "significant_positive_count": sum(
            value["significant_positive"] for value in results.values()
        ),
    }
