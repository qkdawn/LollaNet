"""Train the 20-dimensional soft thought-model activation router."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .model import MicroconceptRouter, legal_contributions, output_probabilities, soft_activation_loss
from .soft_data import DenseRouteDataBundle, DenseRouterDataset, DenseRouterExample, load_dense_score_data


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _loader(rows: Iterable[DenseRouterExample], batch_size: int, shuffle: bool = False) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(DenseRouterDataset(rows), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _binary_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    tp = int(np.logical_and(predicted == 1, actual == 1).sum())
    fp = int(np.logical_and(predicted == 1, actual == 0).sum())
    fn = int(np.logical_and(predicted == 0, actual == 1).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "positives": int(actual.sum())}


def _evaluate(model: MicroconceptRouter, rows: tuple[DenseRouterExample, ...], batch_size: int, device: torch.device, *, activation_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predicted_parts: list[np.ndarray] = []
    contribution_parts: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for batch in _loader(rows, batch_size):
            device_batch = {key: value.to(device) for key, value in batch.items()}
            output = model(device_batch)
            predicted_parts.append(output_probabilities(output)["all_activation"].cpu().numpy())
            contribution_parts.extend(legal_contributions(output, model))
    predicted = np.concatenate(predicted_parts, axis=0)
    target = np.stack([row.activation_targets for row in rows])
    mask = np.stack([row.activation_mask for row in rows]).astype(bool)
    weight = np.stack([row.activation_weights for row in rows])
    error = predicted - target
    masked_error = error[mask]
    effective = weight * mask
    weighted_denominator = float(effective.sum())
    threshold_target = (target >= activation_threshold).astype(np.int64)
    threshold_predicted = (predicted >= activation_threshold).astype(np.int64)
    binary = _binary_metrics(threshold_predicted[mask], threshold_target[mask]) if mask.any() else {"precision": None, "recall": None, "f1": None, "positives": 0}
    per_model: dict[str, Any] = {}
    for position, model_id in enumerate(model.route_models):
        model_mask = mask[:, position]
        per_model[model_id] = {
            "supervised": int(model_mask.sum()),
            "mae": float(np.abs(error[model_mask, position]).mean()) if model_mask.any() else None,
            "target_mean": float(target[model_mask, position].mean()) if model_mask.any() else None,
            "predicted_mean": float(predicted[model_mask, position].mean()) if model_mask.any() else None,
        }
    predictions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        model_rows = []
        for position, model_id in enumerate(model.route_models):
            model_rows.append({
                "model_id": model_id,
                "predicted_activation": float(predicted[row_index, position]),
                "target_activation": float(target[row_index, position]) if mask[row_index, position] else None,
                "loss_mask": bool(mask[row_index, position]),
                "loss_weight": float(weight[row_index, position]),
            })
        selected = [
            item["model_id"]
            for item in sorted(model_rows, key=lambda item: (-item["predicted_activation"], item["model_id"]))
            if item["predicted_activation"] >= activation_threshold
        ]
        predictions.append({
            "episode_id": row.episode_id,
            "split": row.split,
            "model_activations": model_rows,
            "selected_models_at_threshold": selected,
            "no_model_above_threshold": not selected,
            "microconcept_to_model_contributions": contribution_parts[row_index],
        })
    metrics = {
        "episodes": len(rows),
        "model_dimensions": len(model.route_models),
        "supervised_dimensions": int(mask.sum()),
        "masked_dimensions": int(mask.size - mask.sum()),
        "supervised_dimension_rate": float(mask.mean()),
        "masked_mae": float(np.abs(masked_error).mean()) if masked_error.size else None,
        "masked_rmse": float(np.sqrt(np.square(masked_error).mean())) if masked_error.size else None,
        "weighted_mae": float((np.abs(error) * effective).sum() / weighted_denominator) if weighted_denominator else None,
        "activation_threshold": activation_threshold,
        "high_activation": binary,
        "per_model": per_model,
        "label_boundary": "Weak dual-teacher numeric supervision; exploratory only",
    }
    return metrics, predictions


def _without_task_relevance(rows: tuple[DenseRouterExample, ...]) -> tuple[DenseRouterExample, ...]:
    values = []
    for row in rows:
        numeric = row.numeric.copy()
        numeric[:, 2] = 0.0
        values.append(DenseRouterExample(
            episode_id=row.episode_id,
            split=row.split,
            status=row.status,
            direction=row.direction,
            numeric=numeric,
            activation_targets=row.activation_targets,
            activation_mask=row.activation_mask,
            activation_weights=row.activation_weights,
        ))
    return tuple(values)


def _task_relevance_sensitivity(model: MicroconceptRouter, rows: tuple[DenseRouterExample, ...], normal_predictions: list[dict[str, Any]], batch_size: int, device: torch.device, *, activation_threshold: float) -> dict[str, Any]:
    _, ablated_predictions = _evaluate(model, _without_task_relevance(rows), batch_size, device, activation_threshold=activation_threshold)
    absolute_changes = []
    set_changes = 0
    for normal, ablated in zip(normal_predictions, ablated_predictions, strict=True):
        normal_scores = np.asarray([item["predicted_activation"] for item in normal["model_activations"]])
        ablated_scores = np.asarray([item["predicted_activation"] for item in ablated["model_activations"]])
        absolute_changes.extend(np.abs(normal_scores - ablated_scores).tolist())
        set_changes += normal["selected_models_at_threshold"] != ablated["selected_models_at_threshold"]
    return {
        "episodes": len(rows),
        "mean_absolute_activation_change_when_zeroed": float(np.mean(absolute_changes)) if absolute_changes else None,
        "max_absolute_activation_change_when_zeroed": float(np.max(absolute_changes)) if absolute_changes else None,
        "selected_set_change_rate_when_zeroed": set_changes / len(rows) if rows else None,
    }


def _train_epoch(model: MicroconceptRouter, rows: tuple[DenseRouterExample, ...], optimizer: torch.optim.Optimizer, batch_size: int, device: torch.device, training: dict[str, Any]) -> dict[str, float]:
    model.train()
    accumulated = {"total": 0.0, "activation": 0.0, "attention_alignment": 0.0, "supervised_weight": 0.0}
    batches = 0
    for batch in _loader(rows, batch_size, shuffle=True):
        device_batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(device_batch)
        loss, items = soft_activation_loss(
            output,
            device_batch,
            activation_weight=float(training.get("activation_loss_weight", 1.0)),
            attention_alignment_weight=float(training.get("attention_alignment_loss_weight", 0.0)),
            huber_beta=float(training.get("huber_beta", 0.1)),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        for key, value in items.items():
            accumulated[key] += value
        batches += 1
    return {key: value / max(1, batches) for key, value in accumulated.items()}


def _checkpoint_score(metrics: dict[str, Any]) -> float:
    weighted_mae = metrics.get("weighted_mae")
    f1 = metrics.get("high_activation", {}).get("f1")
    return -(float(weighted_mae) if weighted_mae is not None else 1.0) + 0.25 * (float(f1) if f1 is not None else 0.0)


def run(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    inference = config["inference"]
    _seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    bundle: DenseRouteDataBundle = load_dense_score_data(Path(args.dataset_root))
    output_dir = Path(args.output_root) / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "predictions").mkdir()
    (output_dir / "reports").mkdir()
    _atomic_json(output_dir / "run_config.json", config)
    _atomic_json(output_dir / "source_audit.json", bundle.audit)
    model = MicroconceptRouter(
        bundle.graph,
        bundle.support,
        len(bundle.direction_vocab),
        hidden_dim=int(config["architecture"]["node_hidden_dim"]),
        relation_embedding_dim=int(config["architecture"]["relation_embedding_dim"]),
        model_embedding_dim=int(config["architecture"]["model_embedding_dim"]),
        task_relevance_attention_strength=float(config["architecture"].get("task_relevance_attention_strength", 0.15)),
        task_relevance_residual_strength=float(config["architecture"].get("task_relevance_residual_strength", 0.25)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    history = []
    best_score = float("-inf")
    best_epoch = 0
    threshold = float(inference["activation_threshold"])
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = _train_epoch(model, bundle.train, optimizer, int(training["batch_size"]), device, training)
        development, _ = _evaluate(model, bundle.development, int(training["batch_size"]), device, activation_threshold=threshold)
        score = _checkpoint_score(development)
        history.append({"epoch": epoch, "train_loss": train_loss, "development": development})
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "config": config,
                "support": asdict(bundle.support),
                "direction_vocab": bundle.direction_vocab,
                "route_models": model.route_models,
                "target_type": "20-dimensional-soft-activation",
            }, output_dir / "checkpoints" / "best.pt")
    checkpoint = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    report: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "history": history,
        "device": str(device),
        "teacher_target_source": "model-scores-v1",
        "target_type": "20-dimensional-soft-activation",
        "architecture_contract": {
            "factual_path_independent_of_task_relevance": True,
            "task_relevance_role": "bounded_attention_and_focused_residual_modulation",
            "task_relevance_can_erase_factual_path": False,
            "attention_alignment_loss_weight": float(training.get("attention_alignment_loss_weight", 0.0)),
        },
    }
    for split_name, rows in (("train", bundle.train), ("development", bundle.development)):
        metrics, predictions = _evaluate(model, rows, int(training["batch_size"]), device, activation_threshold=threshold)
        report[split_name] = metrics
        report.setdefault("task_relevance_sensitivity", {})[split_name] = _task_relevance_sensitivity(
            model, rows, predictions, int(training["batch_size"]), device, activation_threshold=threshold,
        )
        _atomic_jsonl(output_dir / "predictions" / f"{split_name}.jsonl", predictions)
    report["limitations"] = [
        "This is a microconcept-state-to-20-model activation router, not yet the end-to-end fact encoder.",
        "Targets are Weak dual-teacher numeric judgments, not human Gold labels.",
        "A masked dimension means the two teachers differed by more than the declared score-gap threshold; it is not a negative label.",
        "All 20 models remain in the output layer, but reliable high-score examples are still required for positive coverage.",
        "Sealed test data is not read by this training entry point.",
    ]
    _atomic_json(output_dir / "reports" / "training_report.json", report)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_soft_scores_v1.yaml"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    output = run(args)
    print(f"Trained dense microconcept router: {output}")


if __name__ == "__main__":
    main()
