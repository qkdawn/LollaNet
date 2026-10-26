"""Audit whether the dense router is taking a task-relevance shortcut.

The audit deliberately avoids sealed-test data.  It combines interventions on an
existing checkpoint, simple shortcut baselines, and matched retraining ablations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .model import MicroconceptRouter, output_probabilities
from .soft_data import DenseRouteDataBundle, DenseRouterDataset, DenseRouterExample, load_dense_score_data
from .soft_run import _checkpoint_score, _seed, _train_epoch


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question_type(row: DenseRouterExample) -> str:
    parts = row.episode_id.rsplit("::", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Episode id lacks question type: {row.episode_id}")
    return parts[1]


def _fact_id(row: DenseRouterExample) -> str:
    return row.episode_id.rsplit("::", 1)[0]


def _loader(rows: Iterable[DenseRouterExample], batch_size: int) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(DenseRouterDataset(tuple(rows)), batch_size=batch_size, shuffle=False, num_workers=0)


def _prediction_matrix(
    model: MicroconceptRouter,
    rows: tuple[DenseRouterExample, ...],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _loader(rows, batch_size):
            output = model({key: value.to(device) for key, value in batch.items()})
            parts.append(output_probabilities(output)["all_activation"].cpu().numpy())
    return np.concatenate(parts, axis=0)


def _binary_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | int | None]:
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, np.logical_not(actual)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), actual).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1, "positives": int(actual.sum())}


def _metrics(
    predicted: np.ndarray,
    rows: tuple[DenseRouterExample, ...],
    activation_threshold: float,
) -> dict[str, Any]:
    target = np.stack([row.activation_targets for row in rows])
    mask = np.stack([row.activation_mask for row in rows]).astype(bool)
    weight = np.stack([row.activation_weights for row in rows]) * mask
    denominator = float(weight.sum())
    absolute_error = np.abs(predicted - target)
    binary = _binary_metrics(
        predicted[mask] >= activation_threshold,
        target[mask] >= activation_threshold,
    )
    return {
        "weighted_mae": float((absolute_error * weight).sum() / denominator) if denominator else None,
        "masked_mae": float(absolute_error[mask].mean()) if mask.any() else None,
        "high_activation": binary,
        "supervised_dimensions": int(mask.sum()),
    }


def _intervention_result(
    model: MicroconceptRouter,
    baseline_prediction: np.ndarray,
    rows: tuple[DenseRouterExample, ...],
    batch_size: int,
    device: torch.device,
    activation_threshold: float,
) -> dict[str, Any]:
    prediction = _prediction_matrix(model, rows, batch_size, device)
    change = np.abs(prediction - baseline_prediction)
    result = _metrics(prediction, rows, activation_threshold)
    result.update({
        "mean_absolute_prediction_change": float(change.mean()),
        "max_absolute_prediction_change": float(change.max()),
    })
    return result


def _constant_task_relevance(
    rows: tuple[DenseRouterExample, ...], value: float,
) -> tuple[DenseRouterExample, ...]:
    values = []
    for row in rows:
        numeric = row.numeric.copy()
        numeric[:, 2] = float(value)
        values.append(replace(row, numeric=numeric))
    return tuple(values)


def _mean_task_relevance(
    rows: tuple[DenseRouterExample, ...], train_rows: tuple[DenseRouterExample, ...],
) -> tuple[DenseRouterExample, ...]:
    mean = np.stack([row.numeric[:, 2] for row in train_rows]).mean(axis=0)
    values = []
    for row in rows:
        numeric = row.numeric.copy()
        numeric[:, 2] = mean
        values.append(replace(row, numeric=numeric))
    return tuple(values)


def _sibling_cycle_task_relevance(
    rows: tuple[DenseRouterExample, ...],
) -> tuple[DenseRouterExample, ...]:
    by_fact: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_fact[_fact_id(row)].append(index)
    changed = list(rows)
    for indices in by_fact.values():
        ordered = sorted(indices, key=lambda index: rows[index].episode_id)
        if len(ordered) < 2:
            continue
        for offset, index in enumerate(ordered):
            donor = rows[ordered[(offset + 1) % len(ordered)]]
            numeric = rows[index].numeric.copy()
            numeric[:, 2] = donor.numeric[:, 2]
            changed[index] = replace(rows[index], numeric=numeric)
    return tuple(changed)


def _permuted_rows(
    rows: tuple[DenseRouterExample, ...], permutation: np.ndarray, feature_group: str,
) -> tuple[DenseRouterExample, ...]:
    changed = []
    for index, donor_index in enumerate(permutation.tolist()):
        row = rows[index]
        donor = rows[donor_index]
        status = row.status.copy()
        direction = row.direction.copy()
        numeric = row.numeric.copy()
        if feature_group == "task_relevance":
            numeric[:, 2] = donor.numeric[:, 2]
        elif feature_group == "status":
            status = donor.status.copy()
        elif feature_group == "direction":
            direction = donor.direction.copy()
        elif feature_group == "intensity":
            numeric[:, 0] = donor.numeric[:, 0]
        elif feature_group == "temporal_persistence":
            numeric[:, 1] = donor.numeric[:, 1]
        elif feature_group == "non_task_state":
            status = donor.status.copy()
            direction = donor.direction.copy()
            numeric[:, :2] = donor.numeric[:, :2]
        else:
            raise ValueError(f"unknown feature group: {feature_group}")
        changed.append(replace(row, status=status, direction=direction, numeric=numeric))
    return tuple(changed)


def _permutation_audit(
    model: MicroconceptRouter,
    baseline_prediction: np.ndarray,
    baseline_metrics: dict[str, Any],
    rows: tuple[DenseRouterExample, ...],
    batch_size: int,
    device: torch.device,
    activation_threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for feature_group in (
        "task_relevance", "status", "direction", "intensity",
        "temporal_persistence", "non_task_state",
    ):
        mae_values = []
        prediction_changes = []
        f1_values = []
        for _ in range(repeats):
            permutation = rng.permutation(len(rows))
            changed = _permuted_rows(rows, permutation, feature_group)
            prediction = _prediction_matrix(model, changed, batch_size, device)
            metrics = _metrics(prediction, rows, activation_threshold)
            mae_values.append(float(metrics["weighted_mae"]))
            prediction_changes.append(float(np.abs(prediction - baseline_prediction).mean()))
            f1 = metrics["high_activation"]["f1"]
            if f1 is not None:
                f1_values.append(float(f1))
        result[feature_group] = {
            "repeats": repeats,
            "weighted_mae_mean": float(np.mean(mae_values)),
            "weighted_mae_std": float(np.std(mae_values)),
            "weighted_mae_degradation_mean": float(np.mean(mae_values) - float(baseline_metrics["weighted_mae"])),
            "mean_absolute_prediction_change_mean": float(np.mean(prediction_changes)),
            "high_activation_f1_mean": float(np.mean(f1_values)) if f1_values else None,
        }
    return result


def _mode_by_concept(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        Counter(values[:, concept].tolist()).most_common(1)[0][0]
        for concept in range(values.shape[1])
    ], dtype=values.dtype)


def _reference_state(train_rows: tuple[DenseRouterExample, ...]) -> dict[str, np.ndarray]:
    status = np.stack([row.status for row in train_rows])
    direction = np.stack([row.direction for row in train_rows])
    numeric = np.stack([row.numeric for row in train_rows])
    return {
        "status": _mode_by_concept(status),
        "direction": _mode_by_concept(direction),
        "numeric_non_task": numeric[:, :, :2].mean(axis=0).astype(np.float32),
    }


def _variant_rows(
    rows: tuple[DenseRouterExample, ...], variant: str, reference: dict[str, np.ndarray],
) -> tuple[DenseRouterExample, ...]:
    if variant not in {"full", "no_task", "task_only", "prior_only"}:
        raise ValueError(f"unknown matched-ablation variant: {variant}")
    changed = []
    for row in rows:
        status = row.status.copy()
        direction = row.direction.copy()
        numeric = row.numeric.copy()
        if variant in {"task_only", "prior_only"}:
            status = reference["status"].copy()
            direction = reference["direction"].copy()
            numeric[:, :2] = reference["numeric_non_task"]
        if variant in {"no_task", "prior_only"}:
            # The v1.1 architecture has an independent factual path, so zeroing
            # task relevance removes only question modulation.
            numeric[:, 2] = 0.0
        changed.append(replace(row, status=status, direction=direction, numeric=numeric))
    return tuple(changed)


def _new_model(bundle: DenseRouteDataBundle, config: dict[str, Any], device: torch.device) -> MicroconceptRouter:
    architecture = config["architecture"]
    return MicroconceptRouter(
        bundle.graph,
        bundle.support,
        len(bundle.direction_vocab),
        hidden_dim=int(architecture["node_hidden_dim"]),
        relation_embedding_dim=int(architecture["relation_embedding_dim"]),
        model_embedding_dim=int(architecture["model_embedding_dim"]),
        task_relevance_attention_strength=float(architecture.get("task_relevance_attention_strength", 0.15)),
        task_relevance_residual_strength=float(architecture.get("task_relevance_residual_strength", 0.25)),
    ).to(device)


def _matched_ablation_run(
    bundle: DenseRouteDataBundle,
    config: dict[str, Any],
    variant: str,
    seed: int,
    device: torch.device,
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    _seed(seed)
    variant_config = copy.deepcopy(config)
    if variant in {"no_task", "prior_only"}:
        variant_config["training"]["attention_alignment_loss_weight"] = 0.0
    training = variant_config["training"]
    threshold = float(variant_config["inference"]["activation_threshold"])
    train_rows = _variant_rows(bundle.train, variant, reference)
    development_rows = _variant_rows(bundle.development, variant, reference)
    model = _new_model(bundle, variant_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(training["epochs"]) + 1):
        _train_epoch(model, train_rows, optimizer, int(training["batch_size"]), device, training)
        predicted = _prediction_matrix(model, development_rows, int(training["batch_size"]), device)
        metrics = _metrics(predicted, development_rows, threshold)
        score = _checkpoint_score(metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("matched ablation did not produce a checkpoint")
    model.load_state_dict(best_state)
    predicted = _prediction_matrix(model, development_rows, int(training["batch_size"]), device)
    metrics = _metrics(predicted, development_rows, threshold)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "weighted_mae": metrics["weighted_mae"],
        "high_activation_f1": metrics["high_activation"]["f1"],
    }


def _matched_ablation_audit(
    bundle: DenseRouteDataBundle,
    config: dict[str, Any],
    seeds: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    reference = _reference_state(bundle.train)
    result: dict[str, Any] = {}
    for variant in ("full", "no_task", "task_only", "prior_only"):
        runs = [
            _matched_ablation_run(bundle, config, variant, seed, device, reference)
            for seed in seeds
        ]
        mae = np.asarray([run["weighted_mae"] for run in runs], dtype=float)
        f1 = np.asarray([run["high_activation_f1"] for run in runs], dtype=float)
        result[variant] = {
            "runs": runs,
            "weighted_mae_mean": float(mae.mean()),
            "weighted_mae_std": float(mae.std()),
            "high_activation_f1_mean": float(f1.mean()),
            "high_activation_f1_std": float(f1.std()),
        }
    prior_mae = result["prior_only"]["weighted_mae_mean"]
    full_mae = result["full"]["weighted_mae_mean"]
    total_gain = prior_mae - full_mae
    result["derived"] = {
        "full_gain_over_prior_mae": total_gain,
        "task_only_gain_over_prior_mae": prior_mae - result["task_only"]["weighted_mae_mean"],
        "no_task_gain_over_prior_mae": prior_mae - result["no_task"]["weighted_mae_mean"],
        "task_only_fraction_of_full_gain": (
            (prior_mae - result["task_only"]["weighted_mae_mean"]) / total_gain if total_gain > 0 else None
        ),
        "no_task_fraction_of_full_gain": (
            (prior_mae - result["no_task"]["weighted_mae_mean"]) / total_gain if total_gain > 0 else None
        ),
        "full_mae_advantage_over_task_only": result["task_only"]["weighted_mae_mean"] - full_mae,
        "full_mae_advantage_over_no_task": result["no_task"]["weighted_mae_mean"] - full_mae,
    }
    return result


def _weighted_model_prior(train_rows: tuple[DenseRouterExample, ...]) -> np.ndarray:
    model_count = train_rows[0].activation_targets.shape[0]
    numerator = np.zeros(model_count, dtype=np.float64)
    denominator = np.zeros(model_count, dtype=np.float64)
    for row in train_rows:
        weight = row.activation_mask * row.activation_weights
        numerator += row.activation_targets * weight
        denominator += weight
    return np.divide(numerator, denominator, out=np.full(model_count, 0.5), where=denominator > 0)


def _weighted_average_targets(rows: Iterable[DenseRouterExample], fallback: np.ndarray) -> np.ndarray:
    rows = tuple(rows)
    numerator = np.zeros_like(fallback, dtype=np.float64)
    denominator = np.zeros_like(fallback, dtype=np.float64)
    for row in rows:
        weight = row.activation_mask * row.activation_weights
        numerator += row.activation_targets * weight
        denominator += weight
    return np.divide(numerator, denominator, out=fallback.copy(), where=denominator > 0)


def _shortcut_baselines(
    bundle: DenseRouteDataBundle, activation_threshold: float,
) -> dict[str, Any]:
    prior = _weighted_model_prior(bundle.train)
    prior_prediction = np.repeat(prior[None, :], len(bundle.development), axis=0)
    by_question: dict[str, list[DenseRouterExample]] = defaultdict(list)
    for row in bundle.train:
        by_question[_question_type(row)].append(row)
    question_means = {
        name: _weighted_average_targets(rows, prior)
        for name, rows in by_question.items()
    }
    question_prediction = np.stack([
        question_means.get(_question_type(row), prior)
        for row in bundle.development
    ])

    def task_knn(k: int) -> np.ndarray:
        predictions = []
        for row in bundle.development:
            distances = np.asarray([
                np.abs(candidate.numeric[:, 2] - row.numeric[:, 2]).mean()
                for candidate in bundle.train
            ])
            nearest = np.argsort(distances, kind="stable")[:k]
            predictions.append(_weighted_average_targets((bundle.train[index] for index in nearest), prior))
        return np.stack(predictions)

    return {
        "model_prior": _metrics(prior_prediction, bundle.development, activation_threshold),
        "question_type_model_mean": _metrics(question_prediction, bundle.development, activation_threshold),
        "task_relevance_1nn": _metrics(task_knn(1), bundle.development, activation_threshold),
        "task_relevance_2nn": _metrics(task_knn(2), bundle.development, activation_threshold),
        "interpretation": (
            "These baselines are deliberately simple. If task-relevance nearest-neighbor matches or beats the router "
            "on fact-held-out development Episodes, the dataset permits a task-template shortcut."
        ),
    }


def _template_concentration(bundle: DenseRouteDataBundle) -> dict[str, Any]:
    all_rows = (*bundle.train, *bundle.development)
    by_fact: dict[str, list[DenseRouterExample]] = defaultdict(list)
    for row in all_rows:
        by_fact[_fact_id(row)].append(row)
    non_task_pair_count = 0
    identical_non_task_pair_count = 0
    for fact_rows in by_fact.values():
        for left_index, left in enumerate(fact_rows):
            for right in fact_rows[:left_index]:
                non_task_pair_count += 1
                identical_non_task_pair_count += bool(
                    np.array_equal(left.status, right.status)
                    and np.array_equal(left.direction, right.direction)
                    and np.array_equal(left.numeric[:, :2], right.numeric[:, :2])
                )
    by_question: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in all_rows:
        by_question[_question_type(row)].append(row.numeric[:, 2])
    within = []
    per_type: dict[str, Any] = {}
    centroids: dict[str, np.ndarray] = {}
    for question_type, values in sorted(by_question.items()):
        matrix = np.stack(values)
        distances = [
            float(np.abs(matrix[left] - matrix[right]).mean())
            for left in range(len(matrix))
            for right in range(left)
        ]
        within.extend(distances)
        centroids[question_type] = matrix.mean(axis=0)
        per_type[question_type] = {
            "episodes": len(matrix),
            "unique_vectors": len({tuple(row.tolist()) for row in matrix}),
            "mean_within_type_pairwise_mad": float(np.mean(distances)) if distances else None,
        }
    between = [
        float(np.abs(centroids[left] - centroids[right]).mean())
        for left_index, left in enumerate(sorted(centroids))
        for right in sorted(centroids)[:left_index]
    ]
    train_centroids: dict[str, np.ndarray] = {}
    for question_type in sorted({_question_type(row) for row in bundle.train}):
        train_centroids[question_type] = np.stack([
            row.numeric[:, 2] for row in bundle.train if _question_type(row) == question_type
        ]).mean(axis=0)
    correct = 0
    for row in bundle.development:
        predicted = min(
            train_centroids,
            key=lambda name: (float(np.abs(row.numeric[:, 2] - train_centroids[name]).mean()), name),
        )
        correct += predicted == _question_type(row)
    within_mean = float(np.mean(within)) if within else None
    between_mean = float(np.mean(between)) if between else None
    return {
        "per_question_type": per_type,
        "mean_within_question_type_pairwise_mad": within_mean,
        "mean_between_question_type_centroid_mad": between_mean,
        "within_to_between_ratio": within_mean / between_mean if within_mean is not None and between_mean else None,
        "development_question_type_accuracy_from_task_relevance_nearest_centroid": correct / len(bundle.development),
        "same_fact_question_pairs": non_task_pair_count,
        "same_fact_pairs_with_identical_status_direction_intensity_persistence": identical_non_task_pair_count,
        "same_fact_non_task_identity_rate": (
            identical_non_task_pair_count / non_task_pair_count if non_task_pair_count else None
        ),
    }


def _legal_relevance_target_relation(bundle: DenseRouteDataBundle) -> dict[str, Any]:
    edge_mask = bundle.graph.edge_relation >= 0
    relevance_values = []
    targets = []
    model_positions = []
    for row in (*bundle.train, *bundle.development):
        for position in range(len(bundle.graph.model_ids)):
            if not row.activation_mask[position]:
                continue
            relevance_values.append(float(row.numeric[:, 2][edge_mask[position]].mean()))
            targets.append(float(row.activation_targets[position]))
            model_positions.append(position)
    relevance = np.asarray(relevance_values)
    target = np.asarray(targets)
    model_position = np.asarray(model_positions)
    centered_relevance = []
    centered_target = []
    for position in np.unique(model_position):
        selected = model_position == position
        centered_relevance.extend((relevance[selected] - relevance[selected].mean()).tolist())
        centered_target.extend((target[selected] - target[selected].mean()).tolist())
    return {
        "supervised_dimensions": len(target),
        "pearson_correlation": float(np.corrcoef(relevance, target)[0, 1]),
        "model_centered_pearson_correlation": float(np.corrcoef(centered_relevance, centered_target)[0, 1]),
        "feature": "mean task_relevance over each model's legal microconcept edges",
    }



def _same_fact_delta_metrics(
    predicted: np.ndarray,
    rows: tuple[DenseRouterExample, ...],
) -> dict[str, Any]:
    by_fact: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_fact[_fact_id(row)].append(index)
    target_deltas = []
    prediction_deltas = []
    weights = []
    pair_cosines = []
    pair_count = 0
    for indices in by_fact.values():
        ordered = sorted(indices)
        for left_offset, left in enumerate(ordered):
            for right in ordered[:left_offset]:
                common = np.logical_and(rows[left].activation_mask > 0, rows[right].activation_mask > 0)
                if not common.any():
                    continue
                target_delta = rows[left].activation_targets[common] - rows[right].activation_targets[common]
                prediction_delta = predicted[left, common] - predicted[right, common]
                pair_weight = np.minimum(
                    rows[left].activation_weights[common], rows[right].activation_weights[common],
                )
                target_deltas.extend(target_delta.tolist())
                prediction_deltas.extend(prediction_delta.tolist())
                weights.extend(pair_weight.tolist())
                denominator = float(np.linalg.norm(target_delta) * np.linalg.norm(prediction_delta))
                if denominator > 0:
                    pair_cosines.append(float(np.dot(target_delta, prediction_delta) / denominator))
                pair_count += 1
    target = np.asarray(target_deltas, dtype=float)
    prediction = np.asarray(prediction_deltas, dtype=float)
    weight = np.asarray(weights, dtype=float)
    significant = np.abs(target) >= 0.25
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0
        else None
    )
    return {
        "fact_pairs": pair_count,
        "jointly_supervised_delta_dimensions": len(target),
        "weighted_delta_mae": float((np.abs(prediction - target) * weight).sum() / weight.sum()) if weight.sum() else None,
        "delta_pearson_correlation": correlation,
        "mean_pair_cosine_similarity": float(np.mean(pair_cosines)) if pair_cosines else None,
        "significant_target_delta_dimensions": int(significant.sum()),
        "sign_agreement_when_absolute_target_delta_at_least_0_25": (
            float((np.sign(prediction[significant]) == np.sign(target[significant])).mean())
            if significant.any() else None
        ),
        "prediction_to_target_delta_standard_deviation_ratio": (
            float(np.std(prediction) / np.std(target)) if np.std(target) > 0 else None
        ),
    }


def _supervision_selection_audit(bundle: DenseRouteDataBundle) -> dict[str, Any]:
    edge_mask = bundle.graph.edge_relation >= 0
    records = []
    for row in (*bundle.train, *bundle.development):
        for position, model_id in enumerate(bundle.graph.model_ids):
            relevance = row.numeric[:, 2][edge_mask[position]]
            records.append({
                "question_type": _question_type(row),
                "model_id": model_id,
                "edge_degree": int(edge_mask[position].sum()),
                "legal_relevance_mean": float(relevance.mean()),
                "candidate": bool((relevance > 0).any()),
                "supervised": bool(row.activation_mask[position]),
            })

    def grouped_rate(key: str) -> dict[str, Any]:
        grouped: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            grouped[str(record[key])].append(record["supervised"])
        return {
            name: {"dimensions": len(values), "supervised_rate": float(np.mean(values))}
            for name, values in sorted(grouped.items())
        }

    relevance = np.asarray([record["legal_relevance_mean"] for record in records], dtype=float)
    supervised = np.asarray([record["supervised"] for record in records], dtype=float)
    bins: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        value = record["legal_relevance_mean"]
        label = "low_[0,1/3]" if value <= 1 / 3 else "medium_(1/3,2/3]" if value <= 2 / 3 else "high_(2/3,1]"
        bins[label].append(record["supervised"])
    return {
        "dimensions": len(records),
        "point_biserial_correlation_legal_relevance_vs_supervised": (
            float(np.corrcoef(relevance, supervised)[0, 1])
            if np.std(relevance) > 0 and np.std(supervised) > 0 else None
        ),
        "by_question_type": grouped_rate("question_type"),
        "by_edge_degree": grouped_rate("edge_degree"),
        "by_candidate_membership": grouped_rate("candidate"),
        "by_legal_relevance_bin": {
            name: {"dimensions": len(values), "supervised_rate": float(np.mean(values))}
            for name, values in sorted(bins.items())
        },
        "interpretation": (
            "Large rate differences indicate that teacher-agreement masking is not missing at random; "
            "reported router metrics then emphasize easier-to-agree dimensions."
        ),
    }

def _risk_summary(report: dict[str, Any]) -> dict[str, Any]:
    flags = []
    architecture = report["architecture_observation"]
    if not architecture.get("factual_path_independent_of_task_relevance", False):
        flags.append("task_relevance_can_erase_or_contaminate_factual_path")
    concentration = report["task_relevance_template_concentration"]
    if concentration["development_question_type_accuracy_from_task_relevance_nearest_centroid"] >= 0.9:
        flags.append("task_relevance_nearly_identifies_question_template")
    shortcut = report["shortcut_baselines"]
    baseline_mae = report["checkpoint_interventions"]["baseline"]["weighted_mae"]
    if shortcut["task_relevance_1nn"]["weighted_mae"] <= baseline_mae:
        flags.append("task_relevance_1nn_matches_or_beats_router")
    derived = report["matched_retraining"]["derived"]
    fraction = derived["task_only_fraction_of_full_gain"]
    if fraction is not None and fraction >= 0.65:
        flags.append("task_only_recovers_most_of_full_mae_gain")
    non_task_advantage = derived["full_mae_advantage_over_task_only"]
    no_task_advantage = derived["full_mae_advantage_over_no_task"]
    ignores_non_task = non_task_advantage <= 0.002
    return {
        "classification": "high_shortcut_risk" if len(flags) >= 2 else "monitor",
        "flags": flags,
        "does_not_show_complete_non_task_ignorance": not ignores_non_task and no_task_advantage > 0.0,
        "judgment": (
            "The current dataset and architecture permit a strong task-relevance shortcut. "
            "The router still benefits from non-task microstate, so the evidence does not show complete neglect of factual state."
            if len(flags) >= 2 and not ignores_non_task
            else "No decisive shortcut conclusion; retain the audit for larger fact-held-out data."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root)
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    bundle = load_dense_score_data(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = _new_model(bundle, config, device)
    model.load_state_dict(checkpoint["state_dict"])
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["inference"]["activation_threshold"])
    baseline_prediction = _prediction_matrix(model, bundle.development, batch_size, device)
    baseline_metrics = _metrics(baseline_prediction, bundle.development, threshold)
    numeric_projection_weight = model.numeric_projection[0].weight.detach().cpu().numpy()
    numeric_projection_norms = {
        name: float(np.linalg.norm(numeric_projection_weight[:, position]))
        for position, name in enumerate(("intensity", "temporal_persistence"))
    }

    interventions = {
        "baseline": baseline_metrics,
        "zero_task_relevance_fair_ablation": _intervention_result(
            model, baseline_prediction, _constant_task_relevance(bundle.development, 0.0),
            batch_size, device, threshold,
        ),
        "constant_one_task_relevance": _intervention_result(
            model, baseline_prediction, _constant_task_relevance(bundle.development, 1.0),
            batch_size, device, threshold,
        ),
        "train_mean_task_relevance": _intervention_result(
            model, baseline_prediction, _mean_task_relevance(bundle.development, bundle.train),
            batch_size, device, threshold,
        ),
        "sibling_question_task_relevance_cycle": _intervention_result(
            model, baseline_prediction, _sibling_cycle_task_relevance(bundle.development),
            batch_size, device, threshold,
        ),
    }
    interventions["permutation"] = _permutation_audit(
        model,
        baseline_prediction,
        baseline_metrics,
        bundle.development,
        batch_size,
        device,
        threshold,
        int(args.permutation_repeats),
        int(args.seed),
    )

    seeds = tuple(int(args.seed) + offset for offset in range(int(args.retrain_seeds)))
    previous_threads = torch.get_num_threads()
    if args.cpu:
        torch.set_num_threads(1)
    try:
        matched = _matched_ablation_audit(bundle, config, seeds, device)
    finally:
        torch.set_num_threads(previous_threads)

    manifest_path = dataset_root / "labels" / "train_development" / "model_scores" / "model_score_manifest.json"
    config_path = run_dir / "run_config.json"
    report: dict[str, Any] = {
        "version": "task-relevance-dependency-audit-v1.1.0",
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "device": str(device),
        "sealed_test_read": False,
        "label_boundary": "Weak dual-teacher scores; this audit does not establish Gold validity",
        "source_hashes": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "model_score_manifest_sha256": _sha256(manifest_path),
            "run_config_sha256": _sha256(config_path) if config_path.is_file() else None,
        },
        "architecture_observation": {
            "factual_path_independent_of_task_relevance": True,
            "task_relevance_attention_strength": float(model.task_relevance_attention_strength.detach().cpu()),
            "task_relevance_residual_strength": float(model.task_relevance_residual_strength.detach().cpu()),
            "numeric_projection_input_column_l2_norm": numeric_projection_norms,
            "zero_ablation_is_fair_dependency_test": True,
            "reason": (
                "The factual node and base value path exclude task_relevance. Zeroing relevance removes only "
                "the attention/focused-residual modulation and preserves fact-conditioned context."
            ),
        },
        "checkpoint_interventions": interventions,
        "same_fact_question_delta": _same_fact_delta_metrics(baseline_prediction, bundle.development),
        "matched_retraining": matched,
        "shortcut_baselines": _shortcut_baselines(bundle, threshold),
        "task_relevance_template_concentration": _template_concentration(bundle),
        "legal_relevance_target_relation": _legal_relevance_target_relation(bundle),
        "supervision_selection": _supervision_selection_audit(bundle),
        "teacher_packet_circularity_observation": {
            "teacher_sees_question": True,
            "teacher_sees_microconcept_states_including_task_relevance": True,
            "teacher_sees_candidate_subgraph_derived_from_task_relevance": True,
            "teacher_sees_raw_facts": False,
            "judgment": (
                "This is not direct target leakage, but it is circular Weak supervision: the teacher target can be influenced "
                "by the same engineered task signal that the router later consumes."
            ),
        },
        "recommended_acceptance_gates": [
            "Evaluate on fact-held-out development data; never split sibling questions from one fact across train and development.",
            "The full router should beat task-relevance-only and task-relevance nearest-neighbor baselines by a predeclared MAE margin.",
            "Permutation of non-task factual state must cause a measurable degradation; otherwise factual microstate is being ignored.",
            "Within each question type, task_relevance vectors need fact-specific diversity rather than a nearly fixed template.",
            "Keep the factual path independent of task relevance and interpret zero relevance only as removal of question modulation.",
            "Add a teacher audit that scores from fact plus question while hiding task_relevance, then compare it with the current state-visible teacher chain.",
        ],
        "limitations": [
            "Only 16 train and 16 development Episodes are available, grouped into four facts per split.",
            "Development is used for checkpoint selection, so absolute metrics are exploratory and optimistic.",
            "Matched ablations compare information availability, not causal truth of the Weak teacher labels.",
            "High-activation F1 is prevalence-sensitive; weighted MAE is the primary comparison in this audit.",
        ],
    }
    report["risk_summary"] = _risk_summary(report)
    output_path = Path(args.output) if args.output else run_dir / "reports" / "task_relevance_audit.json"
    _atomic_json(output_path, report)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--retrain-seeds", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    output = run(args)
    print(f"Wrote task-relevance dependency audit: {output}")


if __name__ == "__main__":
    main()
