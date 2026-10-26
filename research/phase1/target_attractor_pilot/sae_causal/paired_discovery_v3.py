from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from .contracts import AXES, assert_discovery_rows, read_jsonl


TARGET_MINIMUM_DELTA = {
    "specificity": 1.0,
    "independence": 0.5,
    "non_template": 1.0,
}


@dataclass(frozen=True)
class MatchedPair:
    pair_id: str
    axis: str
    case_id: str
    split: str
    low_record_id: str
    high_record_id: str
    target_delta: float
    maximum_other_axis_delta: float
    response_token_delta: int
    maximum_quality_delta: float


def _load_codes(path: Path) -> tuple[sparse.csr_matrix, list[str], int]:
    with np.load(path) as archive:
        indices = archive["indices"].copy()
        values = archive["values"].copy()
        record_ids = archive["record_ids"].tolist()
        width = int(archive["width"])
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    matrix = sparse.coo_matrix(
        (values.reshape(-1), (rows, indices.reshape(-1))),
        shape=(indices.shape[0], width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, record_ids, width


def build_matched_pairs(
    trajectories: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    maximum_other_axis_delta: float = 0.5,
    maximum_response_token_delta: int = 96,
    maximum_quality_delta: float = 1.0,
) -> dict[str, list[MatchedPair]]:
    trajectory_by_id = {str(row["record_id"]): row for row in trajectories}
    labels_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        labels_by_case[str(row["case_id"])].append(row)

    output: dict[str, list[MatchedPair]] = {}
    for axis in AXES:
        other_axes = tuple(value for value in AXES if value != axis)
        selected: list[MatchedPair] = []
        for case_id, case_rows in labels_by_case.items():
            candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
            for left_index, left in enumerate(case_rows):
                for right in case_rows[left_index + 1 :]:
                    low, high = (left, right) if float(left[axis]) <= float(right[axis]) else (right, left)
                    target_delta = float(high[axis]) - float(low[axis])
                    other_delta = max(abs(float(high[value]) - float(low[value])) for value in other_axes)
                    low_trajectory = trajectory_by_id[str(low["record_id"])]
                    high_trajectory = trajectory_by_id[str(high["record_id"])]
                    token_delta = abs(
                        int(high_trajectory["response_tokens"]) - int(low_trajectory["response_tokens"])
                    )
                    quality_delta = max(
                        abs(float(high[value]) - float(low[value]))
                        for value in ("usefulness", "validity")
                    )
                    if (
                        target_delta < TARGET_MINIMUM_DELTA[axis]
                        or other_delta > maximum_other_axis_delta
                        or token_delta > maximum_response_token_delta
                        or quality_delta > maximum_quality_delta
                    ):
                        continue
                    priority = (-target_delta, other_delta, float(token_delta), quality_delta)
                    candidates.append((priority, low, high))

            used: set[str] = set()
            pair_number = 0
            for _priority, low, high in sorted(candidates, key=lambda row: row[0]):
                low_id, high_id = str(low["record_id"]), str(high["record_id"])
                if low_id in used or high_id in used:
                    continue
                used.update((low_id, high_id))
                low_trajectory = trajectory_by_id[low_id]
                high_trajectory = trajectory_by_id[high_id]
                target_delta = float(high[axis]) - float(low[axis])
                other_delta = max(abs(float(high[value]) - float(low[value])) for value in other_axes)
                token_delta = abs(
                    int(high_trajectory["response_tokens"]) - int(low_trajectory["response_tokens"])
                )
                quality_delta = max(
                    abs(float(high[value]) - float(low[value]))
                    for value in ("usefulness", "validity")
                )
                selected.append(MatchedPair(
                    pair_id=f"{axis}__{case_id}__{pair_number:02d}",
                    axis=axis,
                    case_id=case_id,
                    split=str(low["split"]),
                    low_record_id=low_id,
                    high_record_id=high_id,
                    target_delta=target_delta,
                    maximum_other_axis_delta=other_delta,
                    response_token_delta=token_delta,
                    maximum_quality_delta=quality_delta,
                ))
                pair_number += 1
        output[axis] = selected
    return output


def _paired_matrix(
    matrix: sparse.csr_matrix, record_index: dict[str, int], pairs: list[MatchedPair]
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    low = np.asarray([record_index[pair.low_record_id] for pair in pairs])
    high = np.asarray([record_index[pair.high_record_id] for pair in pairs])
    differences = (matrix[high] - matrix[low]).tocsr()
    target = np.asarray([pair.target_delta for pair in pairs], dtype=np.float64)
    cases = np.asarray([pair.case_id for pair in pairs])
    return differences, target, cases


def _paired_feature_statistics(
    differences: sparse.csr_matrix, target: np.ndarray, minimum_active_pairs: int
) -> dict[int, dict[str, float]]:
    active = np.asarray((differences != 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(active >= minimum_active_pairs)
    output: dict[int, dict[str, float]] = {}
    for feature in eligible:
        values = differences[:, feature].toarray().ravel()
        nonzero = values != 0
        rho = (
            spearmanr(values, target).statistic
            if np.unique(values).size > 1 and np.unique(target).size > 1
            else 0.0
        )
        nonzero_values = values[nonzero]
        median = float(np.median(nonzero_values))
        direction = 1.0 if median >= 0 else -1.0
        consistency = float(np.mean(np.sign(nonzero_values) == direction))
        spread = float(np.std(nonzero_values, ddof=1)) if len(nonzero_values) > 1 else 0.0
        standardized_mean = float(np.mean(nonzero_values) / (spread + 1e-8))
        output[int(feature)] = {
            "active_pairs": int(active[feature]),
            "paired_spearman": 0.0 if not np.isfinite(rho) else float(rho),
            "median_activation_delta": median,
            "sign_consistency": consistency,
            "standardized_mean_delta": standardized_mean,
        }
    return output


def _stable_elastic_net(
    differences: sparse.csr_matrix,
    target: np.ndarray,
    cases: np.ndarray,
    *,
    bootstraps: int,
    minimum_active_pairs: int,
    alpha: float,
    l1_ratio: float,
    seed: int,
) -> dict[int, dict[str, float]]:
    active = np.asarray((differences != 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(active >= minimum_active_pairs)
    reduced = differences[:, eligible].tocsr()
    unique_cases = np.unique(cases)
    rng = np.random.default_rng(seed)
    coefficients = np.zeros((bootstraps, len(eligible)), dtype=np.float32)

    for bootstrap in range(bootstraps):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        sampled_rows = np.concatenate([np.flatnonzero(cases == case) for case in sampled_cases])
        delta = reduced[sampled_rows]
        outcome = target[sampled_rows]
        design = sparse.vstack((-0.5 * delta, 0.5 * delta), format="csr")
        centered_outcome = np.concatenate((-0.5 * outcome, 0.5 * outcome))
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(design)
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=5000,
            selection="random",
            random_state=seed + bootstrap,
        )
        model.fit(scaled, centered_outcome)
        coefficients[bootstrap] = model.coef_ / np.maximum(scaler.scale_, 1e-8)

    output: dict[int, dict[str, float]] = {}
    for column, feature in enumerate(eligible):
        values = coefficients[:, column]
        nonzero = np.abs(values) > 1e-8
        output[int(feature)] = {
            "selection_frequency": float(np.mean(nonzero)),
            "median_coefficient": float(np.median(values[nonzero])) if nonzero.any() else 0.0,
            "coefficient_sign_consistency": (
                float(max(np.mean(values[nonzero] > 0), np.mean(values[nonzero] < 0)))
                if nonzero.any() else 0.0
            ),
        }
    return output


def discover(
    trajectories: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    codes: sparse.csr_matrix,
    record_ids: list[str],
    *,
    bootstraps: int = 50,
    minimum_active_pairs: int = 8,
    alpha: float = 0.01,
    l1_ratio: float = 0.8,
) -> dict[str, Any]:
    pairs = build_matched_pairs(trajectories, labels)
    record_index = {record_id: index for index, record_id in enumerate(record_ids)}
    results: dict[str, Any] = {}
    for axis in AXES:
        train_pairs = [pair for pair in pairs[axis] if pair.split == "train"]
        development_pairs = [pair for pair in pairs[axis] if pair.split == "development"]
        train_x, train_y, train_cases = _paired_matrix(codes, record_index, train_pairs)
        development_x, development_y, _ = _paired_matrix(codes, record_index, development_pairs)
        paired_train = _paired_feature_statistics(train_x, train_y, minimum_active_pairs)
        stable = _stable_elastic_net(
            train_x,
            train_y,
            train_cases,
            bootstraps=bootstraps,
            minimum_active_pairs=minimum_active_pairs,
            alpha=alpha,
            l1_ratio=l1_ratio,
            seed=2026072900 + AXES.index(axis),
        )
        development = _paired_feature_statistics(development_x, development_y, 1)
        stable_top = sorted(
            stable,
            key=lambda feature: (
                stable[feature]["selection_frequency"],
                abs(stable[feature]["median_coefficient"]),
            ),
            reverse=True,
        )[:32]
        paired_top = sorted(
            paired_train,
            key=lambda feature: (
                paired_train[feature]["sign_consistency"],
                abs(paired_train[feature]["standardized_mean_delta"]),
                abs(paired_train[feature]["paired_spearman"]),
            ),
            reverse=True,
        )[:32]
        intersection = [feature for feature in stable_top if feature in set(paired_top)]
        union = list(dict.fromkeys((*intersection, *stable_top, *paired_top)))
        feature_rows = []
        for feature in union:
            feature_rows.append({
                "feature": feature,
                "paired_train": paired_train.get(feature),
                "stable_sparse_probe": stable.get(feature),
                "paired_development_report_only": development.get(feature),
            })
        results[axis] = {
            "train_pairs": len(train_pairs),
            "train_cases": len(set(pair.case_id for pair in train_pairs)),
            "development_pairs": len(development_pairs),
            "development_cases": len(set(pair.case_id for pair in development_pairs)),
            "candidate_sets": {
                "intersection": intersection,
                "stable_sparse_probe_top32": stable_top,
                "paired_effect_top32": paired_top,
                "union_for_temporal_attribution": union,
            },
            "features": feature_rows,
        }
    return {
        "type": "sae_paired_candidate_discovery_v3",
        "status": "ready_for_temporal_attribution",
        "selection_split": "train",
        "development_use": "report_only",
        "pair_design": "same-prompt matched observational; not strict generated counterfactual",
        "parameters": {
            "target_minimum_delta": TARGET_MINIMUM_DELTA,
            "maximum_other_axis_delta": 0.5,
            "maximum_response_token_delta": 96,
            "maximum_quality_delta": 1.0,
            "minimum_active_pairs": minimum_active_pairs,
            "bootstraps": bootstraps,
            "elastic_net_alpha": alpha,
            "elastic_net_l1_ratio": l1_ratio,
        },
        "pairs": {axis: [asdict(pair) for pair in rows] for axis, rows in pairs.items()},
        "axes": results,
        "evidence_boundary": (
            "Candidate discovery only. Observational matching, sparse-probe selection, and "
            "development association do not establish a causal feature."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=50)
    parser.add_argument("--minimum-active-pairs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--l1-ratio", type=float, default=0.8)
    args = parser.parse_args()

    trajectories = read_jsonl(args.trajectories)
    labels = read_jsonl(args.labels)
    assert_discovery_rows(trajectories, expected=600)
    assert_discovery_rows(labels, expected=600)
    codes, record_ids, _width = _load_codes(args.codes)
    if record_ids != [str(row["record_id"]) for row in trajectories]:
        raise ValueError("codes do not match frozen trajectory order")
    result = discover(
        trajectories,
        labels,
        codes,
        record_ids,
        bootstraps=args.bootstraps,
        minimum_active_pairs=args.minimum_active_pairs,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        axis: {
            "train_pairs": result["axes"][axis]["train_pairs"],
            "development_pairs": result["axes"][axis]["development_pairs"],
            "intersection": len(result["axes"][axis]["candidate_sets"]["intersection"]),
        }
        for axis in AXES
    }, indent=2))


if __name__ == "__main__":
    main()
