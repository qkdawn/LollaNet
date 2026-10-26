from __future__ import annotations

import json

import numpy as np
import pytest

from research.phase1.template_attractor_pilot.natural_probe import (
    _binary_label, _metrics, _sha256, _threshold, evaluate_probe, fit_probe, format_features,
)


def test_binary_label_contract_is_frozen() -> None:
    assert _binary_label(0.0) == 0
    assert _binary_label(1.5) == 0
    assert _binary_label(2.0) is None
    assert _binary_label(2.5) == 1
    assert _binary_label(4.0) == 1


def test_threshold_and_metrics_use_development_probabilities() -> None:
    y = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = _threshold(y, probability)
    assert 0.4 < threshold <= 0.6
    assert _metrics(y, probability, threshold)["balanced_accuracy"] == 1.0


def test_format_features_are_fixed_width() -> None:
    values = format_features(["直接回答。", "以下几点：\n1. **第一**：内容"])
    assert values.shape == (2, 6)
    assert values[1, 1] > values[0, 1]


def test_fit_stops_before_sources_when_labels_are_not_ready(tmp_path) -> None:
    label_manifest = tmp_path / "labels.json"
    label_manifest.write_text(json.dumps({"status": "awaiting_complete_independent_annotations"}), encoding="utf-8")
    result = fit_probe(label_manifest, tmp_path / "missing-trajectory.json", tmp_path / "fit")
    assert result["status"] == "blocked_labels_not_ready"
    assert (tmp_path / "fit" / "fit_manifest.json").is_file()


def test_model_assisted_labels_require_explicit_opt_in(tmp_path) -> None:
    label_manifest = tmp_path / "labels.json"
    label_manifest.write_text(json.dumps({"status": "ready_for_discovery_probe_fit"}), encoding="utf-8")
    result = fit_probe(label_manifest, tmp_path / "missing-trajectory.json", tmp_path / "fit")
    assert result["status"] == "blocked_labels_not_ready"


def test_evaluation_rejects_unpassed_development_gate(tmp_path) -> None:
    manifest = tmp_path / "fit.json"
    manifest.write_text(json.dumps({"status": "stopped_no_internal_increment"}), encoding="utf-8")
    with pytest.raises(ValueError, match="passed and frozen"):
        evaluate_probe(manifest, tmp_path / "evaluation.json")


def test_separable_activation_runs_through_frozen_evaluation(tmp_path) -> None:
    labels = []
    trajectories = []
    prefixes = []
    activations = []
    splits = ["train"] * 8 + ["development"] * 4 + ["evaluation"] * 4
    binary = [0, 1] * 8
    for index, (split, label) in enumerate(zip(splits, binary)):
        record_id = f"r-{index:02d}"
        labels.append({
            "record_id": record_id, "case_id": f"c-{index:02d}", "split": split,
            "formulaic": 3.0 if label else 1.0, "exclude": 0,
        })
        trajectories.append({"record_id": record_id, "response": "same response"})
        prefixes.append({"record_id": record_id, "early_4": "相同文本"})
        activations.append([float(label) * 4 - 2, float(label) * 2 - 1])

    labels_path = tmp_path / "labels.jsonl"
    trajectories_path = tmp_path / "trajectories.jsonl"
    prefixes_path = tmp_path / "prefixes.jsonl"
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    trajectories_path.write_text("".join(json.dumps(row) + "\n" for row in trajectories), encoding="utf-8")
    prefixes_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prefixes), encoding="utf-8")
    activations_path = tmp_path / "activations.npz"
    np.savez_compressed(activations_path, early_4__layer_0=np.asarray(activations, dtype=np.float32))
    activation_index_path = tmp_path / "activation-index.json"
    activation_index_path.write_text(json.dumps({
        "record_ids": [row["record_id"] for row in labels], "windows": ["early_4"], "layers": [0],
    }), encoding="utf-8")
    label_manifest_path = tmp_path / "label-manifest.json"
    label_manifest_path.write_text(json.dumps({
        "status": "ready_for_probe_fit", "labels": str(labels_path),
        "class_definition": {"negative": "<=1.5", "positive": ">=2.5"},
    }), encoding="utf-8")
    trajectory_manifest_path = tmp_path / "trajectory-manifest.json"
    artifacts = {
        "trajectories": {"path": str(trajectories_path), "sha256": _sha256(trajectories_path)},
        "activations": {"path": str(activations_path), "sha256": _sha256(activations_path)},
        "activation_index": {"path": str(activation_index_path), "sha256": _sha256(activation_index_path)},
        "prefixes": {"path": str(prefixes_path), "sha256": _sha256(prefixes_path)},
    }
    trajectory_manifest_path.write_text(json.dumps({
        "status": "awaiting_independent_blind_annotations",
        "annotation_release_gate": {"passed": True}, "artifacts": artifacts,
    }), encoding="utf-8")
    fit_dir = tmp_path / "fit"
    fit = fit_probe(label_manifest_path, trajectory_manifest_path, fit_dir)
    assert fit["status"] == "frozen_before_evaluation"
    assert fit["incremental_gate"]["passed"]
    evaluation = evaluate_probe(fit_dir / "fit_manifest.json", tmp_path / "evaluation.json")
    assert evaluation["activation"]["auroc"] == 1.0
    assert evaluation["activation_minus_text_auroc"] > 0
    with pytest.raises(FileExistsError, match="single-release"):
        evaluate_probe(fit_dir / "fit_manifest.json", tmp_path / "evaluation.json")
