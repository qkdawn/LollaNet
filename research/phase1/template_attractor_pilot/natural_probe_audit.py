from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .natural_probe import (
    MINIMUM_AUROC_INCREMENT,
    _binary_label,
    _metrics,
    _fit_logistic,
    _fit_text,
    _prefixes,
    _read_jsonl,
    _sha256,
    _verify_alignment,
    _write_json,
    format_features,
)


BOOTSTRAP_SEED = 20260726
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_CV_REPEATS = 10
DEFAULT_CV_FOLDS = 5


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "lower_2_5_percentile": float(np.percentile(array, 2.5)),
        "upper_97_5_percentile": float(np.percentile(array, 97.5)),
    }


def clustered_bootstrap(
    y_true: np.ndarray,
    activation_probability: np.ndarray,
    text_probability: np.ndarray,
    case_ids: list[str],
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if not (len(y_true) == len(activation_probability) == len(text_probability) == len(case_ids)):
        raise ValueError("bootstrap inputs are not aligned")
    by_case: dict[str, list[int]] = defaultdict(list)
    for index, case_id in enumerate(case_ids):
        by_case[case_id].append(index)
    unique_cases = sorted(by_case)
    if len(unique_cases) < 2:
        raise ValueError("clustered bootstrap requires at least two cases")

    rng = np.random.default_rng(seed)
    activation_values: list[float] = []
    text_values: list[float] = []
    delta_values: list[float] = []
    skipped = 0
    for _ in range(iterations):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        indices = np.asarray(
            [index for case_id in sampled_cases for index in by_case[str(case_id)]],
            dtype=np.int64,
        )
        sampled_y = y_true[indices]
        if set(int(value) for value in sampled_y) != {0, 1}:
            skipped += 1
            continue
        activation_auroc = _metrics(sampled_y, activation_probability[indices])["auroc"]
        text_auroc = _metrics(sampled_y, text_probability[indices])["auroc"]
        activation_values.append(activation_auroc)
        text_values.append(text_auroc)
        delta_values.append(activation_auroc - text_auroc)
    if not delta_values:
        raise ValueError("no valid clustered bootstrap sample contained both classes")

    deltas = np.asarray(delta_values, dtype=np.float64)
    return {
        "requested_iterations": iterations,
        "valid_iterations": len(delta_values),
        "skipped_single_class_iterations": skipped,
        "seed": seed,
        "activation_auroc": _interval(activation_values),
        "text_auroc": _interval(text_values),
        "activation_minus_text_auroc": _interval(delta_values),
        "fraction_delta_above_zero": float(np.mean(deltas > 0.0)),
        "fraction_delta_at_least_prespecified_gate": float(
            np.mean(deltas >= MINIMUM_AUROC_INCREMENT)
        ),
        "interpretation_boundary": (
            "Case-clustered percentile intervals condition on the already selected models and reuse "
            "the development labels used for selection. They describe sensitivity, not confirmatory "
            "generalization uncertainty."
        ),
    }


def _leave_one_case_out(
    y_true: np.ndarray,
    activation_probability: np.ndarray,
    text_probability: np.ndarray,
    case_ids: list[str],
) -> dict[str, Any]:
    rows = []
    case_array = np.asarray(case_ids)
    for case_id in sorted(set(case_ids)):
        keep = case_array != case_id
        if set(int(value) for value in y_true[keep]) != {0, 1}:
            rows.append({"held_out_case_id": case_id, "status": "single_class_after_removal"})
            continue
        activation_auroc = _metrics(y_true[keep], activation_probability[keep])["auroc"]
        text_auroc = _metrics(y_true[keep], text_probability[keep])["auroc"]
        rows.append({
            "held_out_case_id": case_id,
            "status": "scored",
            "activation_auroc": activation_auroc,
            "text_auroc": text_auroc,
            "activation_minus_text_auroc": activation_auroc - text_auroc,
        })
    scored = [row for row in rows if row["status"] == "scored"]
    deltas = [float(row["activation_minus_text_auroc"]) for row in scored]
    return {
        "eligible_case_count": len(set(case_ids)),
        "scored_removals": len(scored),
        "single_class_removals": len(rows) - len(scored),
        "minimum_activation_minus_text_auroc": min(deltas),
        "maximum_activation_minus_text_auroc": max(deltas),
        "fraction_positive": float(np.mean(np.asarray(deltas) > 0.0)),
        "fraction_passing_prespecified_gate": float(
            np.mean(np.asarray(deltas) >= MINIMUM_AUROC_INCREMENT)
        ),
        "rows": rows,
    }


def grouped_discovery_cross_validation(
    activation_matrix: np.ndarray,
    texts: list[str],
    format_matrix: np.ndarray,
    y_true: np.ndarray,
    case_ids: list[str],
    activation_c: float,
    text_c: float,
    format_c: float,
    folds: int = DEFAULT_CV_FOLDS,
    repeats: int = DEFAULT_CV_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    if folds < 2 or repeats < 1:
        raise ValueError("grouped cross-validation requires at least two folds and one repeat")
    if len(set(case_ids)) < folds:
        raise ValueError("grouped cross-validation has fewer cases than folds")
    if not (len(activation_matrix) == len(texts) == len(format_matrix) == len(y_true) == len(case_ids)):
        raise ValueError("grouped cross-validation inputs are not aligned")

    groups = np.asarray(case_ids)
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed + repeat,
        )
        activation_probability = np.full(len(y_true), np.nan, dtype=np.float64)
        text_probability = np.full(len(y_true), np.nan, dtype=np.float64)
        format_probability = np.full(len(y_true), np.nan, dtype=np.float64)
        valid = True
        for train_index, test_index in splitter.split(activation_matrix, y_true, groups):
            if set(int(value) for value in y_true[train_index]) != {0, 1}:
                valid = False
                break
            activation_model = _fit_logistic(
                activation_matrix[train_index], y_true[train_index], activation_c,
            )
            text_model = _fit_text(
                [texts[int(index)] for index in train_index], y_true[train_index], text_c,
            )
            format_model = _fit_logistic(
                format_matrix[train_index], y_true[train_index], format_c,
            )
            activation_probability[test_index] = activation_model.predict_proba(
                activation_matrix[test_index]
            )[:, 1]
            text_probability[test_index] = text_model.predict_proba(
                [texts[int(index)] for index in test_index]
            )[:, 1]
            format_probability[test_index] = format_model.predict_proba(
                format_matrix[test_index]
            )[:, 1]
        if not valid or any(
            np.isnan(values).any()
            for values in (activation_probability, text_probability, format_probability)
        ):
            rows.append({"repeat": repeat, "status": "invalid_training_fold"})
            continue
        activation_metrics = _metrics(y_true, activation_probability)
        text_metrics = _metrics(y_true, text_probability)
        format_metrics = _metrics(y_true, format_probability)
        rows.append({
            "repeat": repeat,
            "status": "scored",
            "activation_auroc": activation_metrics["auroc"],
            "activation_auprc": activation_metrics["auprc"],
            "text_auroc": text_metrics["auroc"],
            "text_auprc": text_metrics["auprc"],
            "format_auroc": format_metrics["auroc"],
            "format_auprc": format_metrics["auprc"],
            "activation_minus_text_auroc": activation_metrics["auroc"] - text_metrics["auroc"],
        })
    scored = [row for row in rows if row["status"] == "scored"]
    if not scored:
        raise ValueError("no grouped cross-validation repeat was valid")
    deltas = [float(row["activation_minus_text_auroc"]) for row in scored]
    return {
        "folds": folds,
        "requested_repeats": repeats,
        "valid_repeats": len(scored),
        "seed": seed,
        "discovery_records": len(y_true),
        "discovery_cases": len(set(case_ids)),
        "activation_auroc": _interval([float(row["activation_auroc"]) for row in scored]),
        "text_auroc": _interval([float(row["text_auroc"]) for row in scored]),
        "format_auroc": _interval([float(row["format_auroc"]) for row in scored]),
        "activation_minus_text_auroc": _interval(deltas),
        "fraction_delta_above_zero": float(np.mean(np.asarray(deltas) > 0.0)),
        "fraction_delta_at_least_prespecified_gate": float(
            np.mean(np.asarray(deltas) >= MINIMUM_AUROC_INCREMENT)
        ),
        "rows": rows,
        "interpretation_boundary": (
            "Repeated grouped cross-validation uses only train and development cases and fixes the "
            "layer, window, and regularization values selected in the original development analysis. "
            "It is a discovery sensitivity analysis, not sealed evaluation."
        ),
    }


def _case_label_audit(labels: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    by_case: dict[str, list[int | None]] = defaultdict(list)
    for index in indices:
        row = labels[int(index)]
        by_case[str(row["case_id"])].append(_binary_label(float(row["formulaic"])))
    exact_formulaic_pairs = 0
    binary_comparable_pairs = 0
    binary_agreeing_pairs = 0
    negative_cases = 0
    positive_cases = 0
    ambiguous_cases = 0
    for case_id, binary_values in by_case.items():
        raw_values = [float(labels[int(index)]["formulaic"]) for index in indices if labels[int(index)]["case_id"] == case_id]
        if len(raw_values) == 2 and raw_values[0] == raw_values[1]:
            exact_formulaic_pairs += 1
        observed = [value for value in binary_values if value is not None]
        negative_cases += int(0 in observed)
        positive_cases += int(1 in observed)
        ambiguous_cases += int(any(value is None for value in binary_values))
        if len(binary_values) == 2 and all(value is not None for value in binary_values):
            binary_comparable_pairs += 1
            binary_agreeing_pairs += int(binary_values[0] == binary_values[1])
    return {
        "case_count": len(by_case),
        "cases_with_at_least_one_negative_response": negative_cases,
        "cases_with_at_least_one_positive_response": positive_cases,
        "cases_with_at_least_one_ambiguous_response": ambiguous_cases,
        "two_sample_cases_with_identical_ordinal_score": exact_formulaic_pairs,
        "two_sample_cases_comparable_on_binary_label": binary_comparable_pairs,
        "two_sample_cases_agreeing_on_binary_label": binary_agreeing_pairs,
        "binary_pair_agreement": (
            binary_agreeing_pairs / binary_comparable_pairs if binary_comparable_pairs else None
        ),
    }


def audit_probe(
    fit_manifest_path: Path,
    output_path: Path,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    if fit_manifest.get("status") not in {
        "discovery_fit_complete_no_evaluation", "frozen_before_evaluation",
        "stopped_no_internal_increment",
    }:
        raise ValueError("development audit requires a completed probe fit")
    models_path = Path(str(fit_manifest["models"]["path"]))
    if not models_path.is_file() or _sha256(models_path) != fit_manifest["models"]["sha256"]:
        raise ValueError("frozen probe models changed before development audit")
    for name in ("label_manifest", "trajectory_manifest", "labels"):
        source = fit_manifest["sources"][name]
        path = Path(str(source["path"]))
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen probe source changed before development audit: {name}")

    label_manifest = json.loads(
        Path(fit_manifest["sources"]["label_manifest"]["path"]).read_text(encoding="utf-8")
    )
    trajectory_manifest = json.loads(
        Path(fit_manifest["sources"]["trajectory_manifest"]["path"]).read_text(encoding="utf-8")
    )
    labels = _read_jsonl(Path(str(label_manifest["labels"])))
    trajectories = _read_jsonl(Path(str(trajectory_manifest["artifacts"]["trajectories"]["path"])))
    activation_index = json.loads(
        Path(str(trajectory_manifest["artifacts"]["activation_index"]["path"])).read_text(encoding="utf-8")
    )
    _verify_alignment(labels, trajectories, activation_index)

    development_indices = np.asarray([
        index for index, row in enumerate(labels)
        if row["split"] == "development" and not int(row["exclude"])
        and _binary_label(float(row["formulaic"])) is not None
    ], dtype=np.int64)
    development_y = np.asarray([
        int(_binary_label(float(labels[int(index)]["formulaic"])))
        for index in development_indices
    ], dtype=np.int64)
    if set(int(value) for value in development_y) != {0, 1}:
        raise ValueError("development audit requires both binary classes")

    models = joblib.load(models_path)
    activations_path = Path(str(trajectory_manifest["artifacts"]["activations"]["path"]))
    archive = np.load(activations_path)
    matrix = np.asarray(
        archive[f"{fit_manifest['selected_window']}__layer_{fit_manifest['selected_layer']}"],
        dtype=np.float32,
    )
    activation_probability = models["activation"].predict_proba(matrix[development_indices])[:, 1]
    prefix_rows = _read_jsonl(Path(str(trajectory_manifest["artifacts"]["prefixes"]["path"])))
    texts = _prefixes(prefix_rows, str(fit_manifest["selected_window"]))
    development_text = [texts[int(index)] for index in development_indices]
    text_probability = models["text"].predict_proba(development_text)[:, 1]
    all_format = format_features(texts)
    format_probability = models["format"].predict_proba(all_format[development_indices])[:, 1]
    case_ids = [str(labels[int(index)]["case_id"]) for index in development_indices]

    discovery_indices = np.asarray([
        index for index, row in enumerate(labels)
        if row["split"] in {"train", "development"} and not int(row["exclude"])
        and _binary_label(float(row["formulaic"])) is not None
    ], dtype=np.int64)
    discovery_y = np.asarray([
        int(_binary_label(float(labels[int(index)]["formulaic"])))
        for index in discovery_indices
    ], dtype=np.int64)
    discovery_case_ids = [str(labels[int(index)]["case_id"]) for index in discovery_indices]
    selected_window = str(fit_manifest["selected_window"])
    discovery_texts = [texts[int(index)] for index in discovery_indices]

    activation_metrics = _metrics(
        development_y, activation_probability, float(fit_manifest["activation"]["threshold"])
    )
    text_metrics = _metrics(
        development_y, text_probability, float(fit_manifest["text_baseline"]["threshold"])
    )
    format_metrics = _metrics(
        development_y, format_probability, float(fit_manifest["format_baseline"]["threshold"])
    )
    result = {
        "type": "template_attractor_development_stability_audit_v1",
        "status": "development_stability_audit_complete_no_evaluation",
        "evidence_tier": fit_manifest.get("evidence_tier"),
        "selected_window": fit_manifest["selected_window"],
        "selected_layer": fit_manifest["selected_layer"],
        "development_records": len(development_indices),
        "development_class_counts": {
            "negative": int(np.sum(development_y == 0)),
            "positive": int(np.sum(development_y == 1)),
        },
        "point_metrics": {
            "activation": activation_metrics,
            "text_baseline": text_metrics,
            "format_baseline": format_metrics,
            "activation_minus_text_auroc": activation_metrics["auroc"] - text_metrics["auroc"],
        },
        "case_label_audit": _case_label_audit(
            labels,
            np.asarray([index for index, row in enumerate(labels) if row["split"] == "development"]),
        ),
        "case_clustered_bootstrap": clustered_bootstrap(
            development_y, activation_probability, text_probability, case_ids,
            iterations=iterations,
        ),
        "leave_one_case_out": _leave_one_case_out(
            development_y, activation_probability, text_probability, case_ids,
        ),
        "grouped_discovery_cross_validation": grouped_discovery_cross_validation(
            matrix[discovery_indices],
            discovery_texts,
            all_format[discovery_indices],
            discovery_y,
            discovery_case_ids,
            activation_c=float(fit_manifest["activation"]["selected"]["c"]),
            text_c=float(fit_manifest["text_baseline"]["selection"]["selected"]["c"]),
            format_c=float(fit_manifest["format_baseline"]["selection"]["selected"]["c"]),
        ),
        "fit_manifest": {"path": str(fit_manifest_path.resolve()), "sha256": _sha256(fit_manifest_path)},
        "evaluation_records_accessed": 0,
        "evaluation_labels_used": False,
        "evaluation_release_allowed": False,
        "boundary": (
            "This audit uses the same development labels that selected the probe and therefore cannot "
            "replace independent annotation or sealed evaluation."
        ),
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit natural-probe development stability without evaluation.")
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    result = audit_probe(args.fit_manifest, args.output, iterations=args.bootstrap_iterations)
    print(result["status"])


if __name__ == "__main__":
    main()
