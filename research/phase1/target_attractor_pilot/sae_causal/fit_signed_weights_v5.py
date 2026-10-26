from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .analyze_temporal_v3 import _feature_trajectory
from .contracts import AXES


RIDGE_ALPHAS = np.logspace(-4, 3, 15)
MINIMUM_STABLE_FEATURES = 2
MINIMUM_SIGN_CONSISTENCY = 0.70
MINIMUM_OOF_SPEARMAN = 0.10
MINIMUM_OOF_SIGN_ACCURACY = 0.55
HOLDOUT_FRACTION = 0.20


def _symmetric_design(
    differences: np.ndarray, target_delta: np.ndarray, cases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.vstack((-0.5 * differences, 0.5 * differences)),
        np.concatenate((-0.5 * target_delta, 0.5 * target_delta)),
        np.concatenate((cases, cases)),
    )


def split_train_pairs(
    pairs: list[dict[str, Any]], axis: str, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = sorted({str(pair["case_id"]) for pair in pairs})
    holdout_count = max(2, int(np.ceil(len(cases) * holdout_fraction)))
    ordered = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f"signed-weights-v5|{axis}|{case}|20260804".encode("utf-8")
        ).hexdigest(),
    )
    holdout_cases = set(ordered[:holdout_count])
    fit = [pair for pair in pairs if str(pair["case_id"]) not in holdout_cases]
    holdout = [pair for pair in pairs if str(pair["case_id"]) in holdout_cases]
    if not fit or not holdout:
        raise ValueError("fit/holdout split must be non-empty")
    return fit, holdout


class _RidgeModel:
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
    ) -> None:
        self.mean = mean
        self.scale = scale
        self.coefficients = coefficients
        self.intercept = intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.coefficients + self.intercept

    @property
    def original_coefficients(self) -> np.ndarray:
        return self.coefficients / self.scale


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> _RidgeModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (x - mean) / scale
    centered_y = y - np.mean(y)
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(x.shape[1]),
        standardized.T @ centered_y,
    )
    return _RidgeModel(mean, scale, coefficients, float(np.mean(y)))


def _group_count(groups: np.ndarray, maximum: int) -> int:
    count = min(maximum, len(np.unique(groups)))
    if count < 2:
        raise ValueError("at least two groups are required")
    return count


def _group_splits(groups: np.ndarray, maximum: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    folds = _group_count(groups, maximum)
    _, group_indices, group_counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    order = np.argsort(group_counts)[::-1]
    fold_loads = np.zeros(folds, dtype=np.int64)
    group_to_fold = np.zeros(len(group_counts), dtype=np.int64)
    for group_index in order:
        fold = int(np.argmin(fold_loads))
        group_to_fold[group_index] = fold
        fold_loads[fold] += group_counts[group_index]
    sample_folds = group_to_fold[group_indices]
    return [
        (np.flatnonzero(sample_folds != fold), np.flatnonzero(sample_folds == fold))
        for fold in range(folds)
    ]


def _select_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    splits = _group_splits(groups, 4)
    mean_errors = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for train, test in splits:
            prediction = _fit_ridge(x[train], y[train], float(alpha)).predict(x[test])
            errors.append(float(np.mean((y[test] - prediction) ** 2)))
        mean_errors.append(float(np.mean(errors)))
    return float(RIDGE_ALPHAS[int(np.argmin(mean_errors))])


def _r2_score(y: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 0.0 if total <= 0 else 1.0 - residual / total


def fit_signed_weights(
    differences: np.ndarray,
    target_delta: np.ndarray,
    cases: np.ndarray,
    *,
    bootstraps: int = 200,
    seed: int = 20260727,
) -> tuple[np.ndarray, dict[str, Any]]:
    x, y, groups = _symmetric_design(differences, target_delta, cases)
    oof = np.zeros_like(y, dtype=np.float64)
    outer_alphas: list[float] = []
    for train, test in _group_splits(groups, 5):
        alpha = _select_alpha(x[train], y[train], groups[train])
        model = _fit_ridge(x[train], y[train], alpha)
        oof[test] = model.predict(x[test])
        outer_alphas.append(alpha)

    final_alpha = _select_alpha(x, y, groups)
    final_model = _fit_ridge(x, y, final_alpha)
    coefficients = final_model.original_coefficients

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    bootstrap_coefficients = np.zeros((bootstraps, x.shape[1]), dtype=np.float64)
    for bootstrap in range(bootstraps):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        bootstrap_model = _fit_ridge(x[rows], y[rows], final_alpha)
        bootstrap_coefficients[bootstrap] = bootstrap_model.original_coefficients

    median = np.median(bootstrap_coefficients, axis=0)
    sign_consistency = np.maximum(
        np.mean(bootstrap_coefficients > 0, axis=0),
        np.mean(bootstrap_coefficients < 0, axis=0),
    )
    selected = np.flatnonzero(sign_consistency >= MINIMUM_SIGN_CONSISTENCY)
    selected = selected[np.argsort(np.abs(median[selected]))[::-1]]
    raw_weights = median[selected]
    weights = (
        raw_weights / np.max(np.abs(raw_weights))
        if len(raw_weights) and np.max(np.abs(raw_weights)) > 0
        else raw_weights
    )
    rho = spearmanr(y, oof).statistic if np.unique(oof).size > 1 else 0.0
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    sign_accuracy = float(np.mean(np.sign(oof) == np.sign(y)))
    gate = bool(
        len(selected) >= MINIMUM_STABLE_FEATURES
        and rho >= MINIMUM_OOF_SPEARMAN
        and sign_accuracy >= MINIMUM_OOF_SIGN_ACCURACY
    )
    return selected.astype(np.int64), {
        "final_alpha": final_alpha,
        "outer_alphas": outer_alphas,
        "coefficients": coefficients.tolist(),
        "bootstrap_median_coefficients": median.tolist(),
        "bootstrap_sign_consistency": sign_consistency.tolist(),
        "selected_columns": selected.tolist(),
        "selected_normalized_signed_weights": weights.tolist(),
        "oof_spearman": rho,
        "oof_r2": float(_r2_score(y, oof)),
        "oof_sign_accuracy": sign_accuracy,
        "passes_train_only_gate": gate,
        "gate": {
            "minimum_stable_features": MINIMUM_STABLE_FEATURES,
            "minimum_sign_consistency": MINIMUM_SIGN_CONSISTENCY,
            "minimum_oof_spearman": MINIMUM_OOF_SPEARMAN,
            "minimum_oof_sign_accuracy": MINIMUM_OOF_SIGN_ACCURACY,
        },
    }


def weighted_activation_update(
    source_indices: np.ndarray,
    source_values: np.ndarray,
    target_indices: np.ndarray,
    target_values: np.ndarray,
    features: list[int],
    decoder: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if len(features) != len(weights) or decoder.shape[1] != len(features):
        raise ValueError("feature, decoder, and weight dimensions must match")
    source = np.stack([
        _feature_trajectory(source_indices[None], source_values[None], feature)[0]
        for feature in features
    ], axis=1)
    target = np.stack([
        _feature_trajectory(target_indices[None], target_values[None], feature)[0]
        for feature in features
    ], axis=1)
    return (((source - target) * weights[None, :]) @ decoder.T).astype(np.float32)


def main() -> None:
    from .fit_subspaces_v4 import (
        _activity_matched_controls,
        _load_temporal_codes,
        _pair_differences,
        _record_feature_matrix,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--temporal-analysis", type=Path, required=True)
    parser.add_argument("--temporal-codes", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=200)
    args = parser.parse_args()

    import torch

    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal_analysis.read_text(encoding="utf-8"))
    record_ids, indices, values = _load_temporal_codes(args.temporal_codes)
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    raw_sae = torch.load(args.sae, map_location="cpu", weights_only=True)
    decoder = raw_sae["W_dec"].float().numpy()
    if decoder.shape[1] != 65536:
        decoder = decoder.T
    if decoder.shape[1] != 65536:
        raise ValueError(f"unexpected decoder shape {decoder.shape}")
    del raw_sae
    gc.collect()

    all_candidates = {
        int(feature)
        for axis in AXES
        for feature in temporal["axes"][axis]["early_available_candidates"]
    }
    arrays: dict[str, np.ndarray] = {}
    axes: dict[str, Any] = {}
    for axis_index, axis in enumerate(AXES):
        candidates = [
            int(feature) for feature in temporal["axes"][axis]["early_available_candidates"]
        ]
        matrix = _record_feature_matrix(indices, values, candidates)
        all_train_pairs = [
            pair for pair in discovery["pairs"][axis] if pair["split"] == "train"
        ]
        train_pairs, holdout_pairs = split_train_pairs(all_train_pairs, axis)
        differences, target_delta, cases = _pair_differences(
            matrix, record_index, train_pairs
        )
        selected_columns, fit = fit_signed_weights(
            differences,
            target_delta,
            cases,
            bootstraps=args.bootstraps,
            seed=2026080400 + axis_index,
        )
        selected_features = [candidates[int(column)] for column in selected_columns]
        weights = np.asarray(fit["selected_normalized_signed_weights"], dtype=np.float32)
        holdout_record_indices = np.asarray([
            record_index[str(pair[key])]
            for pair in holdout_pairs
            for key in ("low_record_id", "high_record_id")
        ], dtype=np.int64)
        random_features = _activity_matched_controls(
            indices[holdout_record_indices], selected_features, all_candidates
        ) if selected_features else []
        arrays[f"{axis}__features"] = np.asarray(selected_features, dtype=np.int64)
        arrays[f"{axis}__weights"] = weights
        arrays[f"{axis}__decoder"] = decoder[:, selected_features].astype(np.float32)
        arrays[f"{axis}__random_features"] = np.asarray(random_features, dtype=np.int64)
        arrays[f"{axis}__random_decoder"] = decoder[:, random_features].astype(np.float32)
        axes[axis] = {
            "candidate_features": candidates,
            "selected_features": selected_features,
            "random_features": random_features,
            "normalized_signed_weights": weights.tolist(),
            "train_pairs": len(train_pairs),
            "train_cases": len(set(str(pair["case_id"]) for pair in train_pairs)),
            "fit_case_ids": sorted({str(pair["case_id"]) for pair in train_pairs}),
            "internal_holdout_pairs": len(holdout_pairs),
            "internal_holdout_cases": sorted({
                str(pair["case_id"]) for pair in holdout_pairs
            }),
            "internal_holdout_pair_ids": [
                str(pair["pair_id"]) for pair in holdout_pairs
            ],
            "fit": fit,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "signed_weight_operators.npz", **arrays)
    manifest = {
        "type": "sae_signed_relative_feature_weights_v5",
        "status": "train_only_fit_complete",
        "selection_split": "train",
        "train_partition": "deterministic case-grouped 80% fit / 20% internal holdout",
        "development_accessed_during_fit": False,
        "formal_evaluation_accessed": False,
        "axes": axes,
        "evidence_boundary": (
            "Grouped out-of-fold association and bootstrap sign stability are discovery gates, "
            "not causal evidence. Only blinded generation intervention can test behavior."
        ),
    }
    (args.output / "signed_weights_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        axis: {
            "selected_features": axes[axis]["selected_features"],
            "weights": axes[axis]["normalized_signed_weights"],
            "oof_spearman": axes[axis]["fit"]["oof_spearman"],
            "oof_sign_accuracy": axes[axis]["fit"]["oof_sign_accuracy"],
            "passes": axes[axis]["fit"]["passes_train_only_gate"],
        }
        for axis in AXES
    }, indent=2))


if __name__ == "__main__":
    main()
