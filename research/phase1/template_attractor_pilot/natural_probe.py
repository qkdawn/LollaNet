from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


C_GRID = (0.01, 0.1, 1.0, 10.0)
MINIMUM_AUROC_INCREMENT = 0.02


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    if set(int(value) for value in y_true) != {0, 1}:
        raise ValueError("probe metrics require both binary classes")
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def _threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    return max(
        candidates,
        key=lambda value: (_metrics(y_true, probabilities, value)["balanced_accuracy"], -value),
    )


def _fit_logistic(x: Any, y: np.ndarray, c_value: float) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=3000, random_state=20260726, class_weight="balanced"),
    ).fit(x, y)


def _fit_text(texts: list[str], y: np.ndarray, c_value: float) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=10000),
        LogisticRegression(C=c_value, max_iter=3000, random_state=20260726, class_weight="balanced"),
    ).fit(texts, y)


def format_features(texts: list[str]) -> np.ndarray:
    import re

    return np.asarray([
        [
            len(text), text.count("\n"), text.count("：") + text.count(":"),
            len(re.findall(r"(?m)^\s*\d+[.、]", text)), text.count("**"),
            sum(text.count(marker) for marker in ("以下", "首先", "其次", "此外", "总之", "综上")),
        ]
        for text in texts
    ], dtype=np.float64)


def _binary_label(value: float) -> int | None:
    if value <= 1.5:
        return 0
    if value >= 2.5:
        return 1
    return None


def _eligible_indices(labels: list[dict[str, Any]], split: str) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    values = []
    for index, row in enumerate(labels):
        label = _binary_label(float(row["formulaic"]))
        if row["split"] == split and not int(row["exclude"]) and label is not None:
            indices.append(index)
            values.append(label)
    y = np.asarray(values, dtype=np.int64)
    if len(y) < 4 or set(int(value) for value in y) != {0, 1}:
        raise ValueError(f"{split} lacks binary class coverage")
    return np.asarray(indices, dtype=np.int64), y


def _prefixes(prefix_rows: list[dict[str, Any]], window: str) -> list[str]:
    return [str(row[window]) for row in prefix_rows]


def _best_c(
    fit_function: Any, train_x: Any, train_y: np.ndarray, development_x: Any,
    development_y: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    candidates = []
    models = {}
    for c_value in C_GRID:
        model = fit_function(train_x, train_y, c_value)
        probability = model.predict_proba(development_x)[:, 1]
        metrics = _metrics(development_y, probability)
        candidates.append({"c": c_value, **metrics})
        models[c_value] = model
    selected = max(candidates, key=lambda row: (row["auroc"], row["auprc"], row["balanced_accuracy"], -row["c"]))
    return models[float(selected["c"])], {"selected": selected, "candidates": candidates}


def _verify_alignment(
    labels: list[dict[str, Any]], trajectories: list[dict[str, Any]], activation_index: dict[str, Any],
) -> None:
    label_ids = [str(row["record_id"]) for row in labels]
    trajectory_ids = [str(row["record_id"]) for row in trajectories]
    if label_ids != trajectory_ids or label_ids != activation_index["record_ids"]:
        raise ValueError("labels, trajectories, and activations are not identically aligned")
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("natural probe inputs contain duplicate record IDs")


def fit_probe(
    label_manifest_path: Path,
    trajectory_manifest_path: Path,
    output_dir: Path,
    allow_model_assisted: bool = False,
) -> dict[str, Any]:
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    model_assisted = label_manifest.get("status") == "ready_for_discovery_probe_fit"
    labels_ready = label_manifest.get("status") == "ready_for_probe_fit" or (
        allow_model_assisted and model_assisted
    )
    if not labels_ready:
        result = {
            "type": "template_attractor_natural_probe_fit_v1",
            "status": "blocked_labels_not_ready",
            "observed_label_status": label_manifest.get("status"),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "fit_manifest.json", result)
        return result
    trajectory_manifest = json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
    if (
        trajectory_manifest.get("status") != "awaiting_independent_blind_annotations"
        or not trajectory_manifest.get("annotation_release_gate", {}).get("passed")
    ):
        raise ValueError("natural trajectory run did not pass its annotation release gate")

    labels_path = Path(str(label_manifest["labels"]))
    trajectories_path = Path(str(trajectory_manifest["artifacts"]["trajectories"]["path"]))
    activations_path = Path(str(trajectory_manifest["artifacts"]["activations"]["path"]))
    activation_index_path = Path(str(trajectory_manifest["artifacts"]["activation_index"]["path"]))
    prefixes_path = Path(str(trajectory_manifest["artifacts"]["prefixes"]["path"]))
    for name, path in (
        ("trajectories", trajectories_path), ("activations", activations_path),
        ("activation_index", activation_index_path), ("prefixes", prefixes_path),
    ):
        expected = trajectory_manifest["artifacts"][name]["sha256"]
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"natural probe source changed: {name}")

    labels = _read_jsonl(labels_path)
    trajectories = _read_jsonl(trajectories_path)
    prefix_rows = _read_jsonl(prefixes_path)
    activation_index = json.loads(activation_index_path.read_text(encoding="utf-8"))
    _verify_alignment(labels, trajectories, activation_index)
    if [row["record_id"] for row in prefix_rows] != activation_index["record_ids"]:
        raise ValueError("frozen text prefixes are not aligned with activations")
    train_indices, train_y = _eligible_indices(labels, "train")
    development_indices, development_y = _eligible_indices(labels, "development")

    archive = np.load(activations_path)
    activation_candidates = []
    activation_models = {}
    for window in activation_index["windows"]:
        for layer in activation_index["layers"]:
            key = f"{window}__layer_{layer}"
            matrix = np.asarray(archive[key], dtype=np.float32)
            model, selection = _best_c(
                _fit_logistic, matrix[train_indices], train_y,
                matrix[development_indices], development_y,
            )
            row = {"window": window, "layer": int(layer), **selection["selected"], "c_grid": selection["candidates"]}
            activation_candidates.append(row)
            activation_models[(window, int(layer), float(selection["selected"]["c"]))] = model
    selected_activation = max(
        activation_candidates,
        key=lambda row: (row["auroc"], row["auprc"], row["balanced_accuracy"], -row["layer"], -row["c"]),
    )
    selected_key = (
        str(selected_activation["window"]), int(selected_activation["layer"]), float(selected_activation["c"]),
    )
    activation_model = activation_models[selected_key]
    selected_matrix = np.asarray(
        archive[f"{selected_key[0]}__layer_{selected_key[1]}"], dtype=np.float32,
    )
    activation_dev_probability = activation_model.predict_proba(selected_matrix[development_indices])[:, 1]
    activation_threshold = _threshold(development_y, activation_dev_probability)
    activation_dev_metrics = _metrics(development_y, activation_dev_probability, activation_threshold)

    texts = _prefixes(prefix_rows, selected_key[0])
    train_text = [texts[index] for index in train_indices]
    development_text = [texts[index] for index in development_indices]
    text_model, text_selection = _best_c(
        _fit_text, train_text, train_y, development_text, development_y,
    )
    text_dev_probability = text_model.predict_proba(development_text)[:, 1]
    text_threshold = _threshold(development_y, text_dev_probability)
    text_dev_metrics = _metrics(development_y, text_dev_probability, text_threshold)

    all_format = format_features(texts)
    format_model, format_selection = _best_c(
        _fit_logistic, all_format[train_indices], train_y,
        all_format[development_indices], development_y,
    )
    format_dev_probability = format_model.predict_proba(all_format[development_indices])[:, 1]
    format_threshold = _threshold(development_y, format_dev_probability)
    format_dev_metrics = _metrics(development_y, format_dev_probability, format_threshold)

    incremental_gate = {
        "minimum_auroc_increment_over_text": MINIMUM_AUROC_INCREMENT,
        "observed_auroc_increment": activation_dev_metrics["auroc"] - text_dev_metrics["auroc"],
        "activation_auprc_not_below_text": activation_dev_metrics["auprc"] >= text_dev_metrics["auprc"],
    }
    incremental_gate["passed"] = bool(
        incremental_gate["observed_auroc_increment"] >= MINIMUM_AUROC_INCREMENT
        and incremental_gate["activation_auprc_not_below_text"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_path = output_dir / "frozen_models.joblib"
    joblib.dump({
        "activation": activation_model, "text": text_model, "format": format_model,
    }, models_path)
    manifest = {
        "type": "template_attractor_natural_probe_fit_v1",
        "status": (
            "discovery_fit_complete_no_evaluation"
            if model_assisted else
            "frozen_before_evaluation" if incremental_gate["passed"] else "stopped_no_internal_increment"
        ),
        "evidence_tier": "discovery_only_model_assisted" if model_assisted else "human_annotated_pilot",
        "selected_window": selected_key[0], "selected_layer": selected_key[1],
        "class_definition": label_manifest["class_definition"],
        "train_records": int(len(train_indices)), "development_records": int(len(development_indices)),
        "selection_rule": "maximize development AUROC, then AUPRC, balanced accuracy, shallower layer, smaller C",
        "activation": {
            "threshold": activation_threshold, "development_metrics": activation_dev_metrics,
            "selected": selected_activation, "all_candidates": activation_candidates,
        },
        "text_baseline": {
            "threshold": text_threshold, "development_metrics": text_dev_metrics,
            "selection": text_selection,
        },
        "format_baseline": {
            "threshold": format_threshold, "development_metrics": format_dev_metrics,
            "selection": format_selection,
        },
        "incremental_gate": incremental_gate,
        "sources": {
            "label_manifest": {"path": str(label_manifest_path.resolve()), "sha256": _sha256(label_manifest_path)},
            "trajectory_manifest": {"path": str(trajectory_manifest_path.resolve()), "sha256": _sha256(trajectory_manifest_path)},
            "labels": {"path": str(labels_path.resolve()), "sha256": _sha256(labels_path)},
        },
        "models": {"path": str(models_path.resolve()), "sha256": _sha256(models_path)},
        "code_sha256": _code_hash(),
        "evaluation_labels_used": False,
        "evaluation_release_allowed": not model_assisted,
    }
    _write_json(output_dir / "fit_manifest.json", manifest)
    return manifest


def evaluate_probe(fit_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("evaluation output already exists; natural probe evaluation is single-release")
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    if fit_manifest.get("status") != "frozen_before_evaluation":
        raise ValueError("evaluation release requires a passed and frozen development gate")
    if fit_manifest.get("code_sha256") != _code_hash():
        raise ValueError("probe code changed after development freeze")
    models_path = Path(str(fit_manifest["models"]["path"]))
    if _sha256(models_path) != fit_manifest["models"]["sha256"]:
        raise ValueError("frozen probe models changed")

    for name in ("label_manifest", "trajectory_manifest", "labels"):
        source = fit_manifest["sources"][name]
        path = Path(str(source["path"]))
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen probe source changed after development selection: {name}")
    label_manifest = json.loads(Path(fit_manifest["sources"]["label_manifest"]["path"]).read_text(encoding="utf-8"))
    trajectory_manifest = json.loads(Path(fit_manifest["sources"]["trajectory_manifest"]["path"]).read_text(encoding="utf-8"))
    labels = _read_jsonl(Path(str(label_manifest["labels"])))
    for name in ("trajectories", "activations", "activation_index", "prefixes"):
        artifact = trajectory_manifest["artifacts"][name]
        path = Path(str(artifact["path"]))
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen trajectory artifact changed before evaluation: {name}")
    trajectories = _read_jsonl(Path(str(trajectory_manifest["artifacts"]["trajectories"]["path"])))
    evaluation_indices, evaluation_y = _eligible_indices(labels, "evaluation")
    models = joblib.load(models_path)

    activations_path = Path(str(trajectory_manifest["artifacts"]["activations"]["path"]))
    archive = np.load(activations_path)
    matrix = np.asarray(
        archive[f"{fit_manifest['selected_window']}__layer_{fit_manifest['selected_layer']}"], dtype=np.float32,
    )
    activation_probability = models["activation"].predict_proba(matrix[evaluation_indices])[:, 1]

    prefix_rows = _read_jsonl(Path(str(trajectory_manifest["artifacts"]["prefixes"]["path"])))
    texts = _prefixes(prefix_rows, str(fit_manifest["selected_window"]))
    evaluation_text = [texts[index] for index in evaluation_indices]
    text_probability = models["text"].predict_proba(evaluation_text)[:, 1]
    all_format = format_features(texts)
    format_probability = models["format"].predict_proba(all_format[evaluation_indices])[:, 1]

    result = {
        "type": "template_attractor_natural_probe_evaluation_v1",
        "status": "evaluation_complete_descriptive_only",
        "evaluation_records": int(len(evaluation_indices)),
        "activation": _metrics(evaluation_y, activation_probability, float(fit_manifest["activation"]["threshold"])),
        "text_baseline": _metrics(evaluation_y, text_probability, float(fit_manifest["text_baseline"]["threshold"])),
        "format_baseline": _metrics(evaluation_y, format_probability, float(fit_manifest["format_baseline"]["threshold"])),
        "boundary": "The four evaluation case_ids support only a descriptive pilot comparison, not a confirmatory generalization claim.",
        "fit_manifest": {"path": str(fit_manifest_path.resolve()), "sha256": _sha256(fit_manifest_path)},
    }
    result["activation_minus_text_auroc"] = result["activation"]["auroc"] - result["text_baseline"]["auroc"]
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit or release the natural-trajectory activation probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--label-manifest", type=Path, required=True)
    fit_parser.add_argument("--trajectory-manifest", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--allow-model-assisted", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--fit-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_probe(
            args.label_manifest, args.trajectory_manifest, args.output,
            allow_model_assisted=args.allow_model_assisted,
        )
    else:
        result = evaluate_probe(args.fit_manifest, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
