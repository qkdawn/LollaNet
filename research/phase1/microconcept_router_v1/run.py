"""Train Microconcept Router V1 without consuming raw facts or sealed labels by default."""

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

from .data import IGNORE_INDEX, RouteDataBundle, RouterDataset, RouterExample, load_route_data, serializable_predictions
from .model import MicroconceptRouter, legal_contributions, output_probabilities, router_loss


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


def _loader(rows: Iterable[RouterExample], batch_size: int, shuffle: bool = False) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(RouterDataset(rows), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _binary_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    tp = int(np.logical_and(predicted == 1, actual == 1).sum())
    fp = int(np.logical_and(predicted == 1, actual == 0).sum())
    fn = int(np.logical_and(predicted == 0, actual == 1).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "positives": int(actual.sum())}


def _macro_f1(predicted: np.ndarray, actual: np.ndarray, class_count: int) -> float | None:
    values = []
    for class_id in range(class_count):
        metrics = _binary_metrics((predicted == class_id).astype(np.int64), (actual == class_id).astype(np.int64))
        if metrics["positives"]:
            values.append(float(metrics["f1"] or 0.0))
    return float(np.mean(values)) if values else None


def _evaluate(
    model: MicroconceptRouter,
    rows: tuple[RouterExample, ...],
    batch_size: int,
    device: torch.device,
    *,
    auxiliary_threshold: float,
    abstain_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    primary_parts: list[np.ndarray] = []
    auxiliary_parts: list[np.ndarray] = []
    abstain_parts: list[np.ndarray] = []
    contribution_parts: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for batch in _loader(rows, batch_size):
            device_batch = {key: value.to(device) for key, value in batch.items()}
            output = model(device_batch)
            probabilities = output_probabilities(output)
            primary_parts.append(probabilities["primary"].cpu().numpy())
            auxiliary_parts.append(probabilities["auxiliary"].cpu().numpy())
            abstain_parts.append(probabilities["abstain"].cpu().numpy())
            contribution_parts.extend(legal_contributions(output, model))
    probability_output = {
        "primary": np.concatenate(primary_parts, axis=0),
        "auxiliary": np.concatenate(auxiliary_parts, axis=0),
        "abstain": np.concatenate(abstain_parts, axis=0),
    }
    primary_target = np.array([row.primary_target for row in rows], dtype=np.int64)
    auxiliary_target = np.stack([row.auxiliary_targets for row in rows])
    abstain_target = np.array([row.abstain for row in rows], dtype=np.int64)
    primary_mask = primary_target != IGNORE_INDEX
    primary_predicted = probability_output["primary"].argmax(axis=1)
    auxiliary_predicted = (probability_output["auxiliary"] >= auxiliary_threshold).astype(np.int64)
    abstain_predicted = (probability_output["abstain"] >= abstain_threshold).astype(np.int64)

    predictions = serializable_predictions(
        rows,
        probability_output,
        model.support,
        auxiliary_threshold=auxiliary_threshold,
        abstain_threshold=abstain_threshold,
    )
    for prediction, contributions in zip(predictions, contribution_parts, strict=True):
        prediction["microconcept_to_model_contributions"] = contributions

    # Route-level metrics are computed on the constrained serialized decision,
    # not on independent logits. This catches the failure mode where each head
    # looks reasonable but the final primary/auxiliary/abstain route is wrong.
    route_exact: list[bool] = []
    primary_abstain_exact: list[bool] = []
    auxiliary_set_exact: list[bool] = []
    supported_route_mask: list[bool] = []
    for row, prediction in zip(rows, predictions, strict=True):
        expected_primary = None if row.abstain or row.primary_target == IGNORE_INDEX else model.support.primary_models[row.primary_target]
        expected_auxiliary = set(
            model.support.auxiliary_models[position]
            for position, value in enumerate(row.auxiliary_targets)
            if value >= 0.5
        ) if not row.abstain else set()
        predicted_auxiliary = set(prediction["auxiliary_models"])
        abstain_ok = bool(prediction["abstain"]) == bool(row.abstain)
        primary_ok = bool(row.abstain) or (
            row.primary_target != IGNORE_INDEX and prediction["primary_model"] == expected_primary
        )
        auxiliary_ok = predicted_auxiliary == expected_auxiliary
        primary_abstain_exact.append(abstain_ok and primary_ok)
        auxiliary_set_exact.append(abstain_ok and auxiliary_ok)
        route_exact.append(abstain_ok and primary_ok and auxiliary_ok)
        supported_route_mask.append(bool(row.abstain) or row.primary_target != IGNORE_INDEX)

    route_array = np.asarray(route_exact, dtype=bool)
    primary_abstain_array = np.asarray(primary_abstain_exact, dtype=bool)
    auxiliary_set_array = np.asarray(auxiliary_set_exact, dtype=bool)
    supported_array = np.asarray(supported_route_mask, dtype=bool)
    metrics = {
        "episodes": len(rows),
        "primary_evaluated": int(primary_mask.sum()),
        "primary_masked_unsupported_target": int((~primary_mask).sum()),
        "primary_top1_accuracy": float((primary_predicted[primary_mask] == primary_target[primary_mask]).mean()) if primary_mask.any() else None,
        "primary_macro_f1": _macro_f1(primary_predicted[primary_mask], primary_target[primary_mask], len(model.support.primary_models)) if primary_mask.any() else None,
        "auxiliary": _binary_metrics(auxiliary_predicted.reshape(-1), auxiliary_target.astype(np.int64).reshape(-1)),
        "auxiliary_set_exact": float(auxiliary_set_array.mean()) if len(auxiliary_set_array) else None,
        "abstain": _binary_metrics(abstain_predicted, abstain_target),
        "abstain_brier": float(np.mean((probability_output["abstain"] - abstain_target) ** 2)),
        "primary_abstain_exact": float(primary_abstain_array.mean()) if len(primary_abstain_array) else None,
        "route_exact": float(route_array.mean()) if len(route_array) else None,
        "route_exact_supported": float(route_array[supported_array].mean()) if supported_array.any() else None,
        "label_boundary": "Weak manifest-bound LLM supervision; exploratory only",
    }
    return metrics, predictions


def _without_task_relevance(rows: tuple[RouterExample, ...]) -> tuple[RouterExample, ...]:
    """Return the same examples with only the task-relevance channel ablated."""
    ablated: list[RouterExample] = []
    for row in rows:
        numeric = row.numeric.copy()
        numeric[:, 2] = 0.0
        ablated.append(RouterExample(
            episode_id=row.episode_id,
            split=row.split,
            status=row.status,
            direction=row.direction,
            numeric=numeric,
            abstain=row.abstain,
            primary_target=row.primary_target,
            auxiliary_targets=row.auxiliary_targets,
        ))
    return tuple(ablated)


def _task_relevance_sensitivity(
    model: MicroconceptRouter,
    rows: tuple[RouterExample, ...],
    normal_metrics: dict[str, Any],
    normal_predictions: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    *,
    auxiliary_threshold: float,
    abstain_threshold: float,
) -> dict[str, Any]:
    ablated_metrics, ablated_predictions = _evaluate(
        model,
        _without_task_relevance(rows),
        batch_size,
        device,
        auxiliary_threshold=auxiliary_threshold,
        abstain_threshold=abstain_threshold,
    )
    count = len(normal_predictions)
    route_changes = 0
    abstain_changes = 0
    primary_changes = 0
    auxiliary_changes = 0
    for normal, ablated in zip(normal_predictions, ablated_predictions, strict=True):
        normal_route = (normal["abstain"], normal["primary_model"], tuple(normal["auxiliary_models"]))
        ablated_route = (ablated["abstain"], ablated["primary_model"], tuple(ablated["auxiliary_models"]))
        route_changes += normal_route != ablated_route
        abstain_changes += normal["abstain"] != ablated["abstain"]
        primary_changes += normal["primary_model"] != ablated["primary_model"]
        auxiliary_changes += tuple(normal["auxiliary_models"]) != tuple(ablated["auxiliary_models"])
    rate = lambda value: value / count if count else None
    return {
        "episodes": count,
        "route_change_rate_when_zeroed": rate(route_changes),
        "abstain_change_rate_when_zeroed": rate(abstain_changes),
        "primary_change_rate_when_zeroed": rate(primary_changes),
        "auxiliary_set_change_rate_when_zeroed": rate(auxiliary_changes),
        "normal_route_exact": normal_metrics.get("route_exact"),
        "zeroed_route_exact": ablated_metrics.get("route_exact"),
        "route_exact_delta_normal_minus_zeroed": (
            normal_metrics["route_exact"] - ablated_metrics["route_exact"]
            if normal_metrics.get("route_exact") is not None and ablated_metrics.get("route_exact") is not None
            else None
        ),
    }


def _checkpoint_score(metrics: dict[str, Any]) -> float:
    """Score the final constrained route, while ignoring unlearnable targets."""
    values = {
        "route": metrics.get("route_exact_supported"),
        "primary_abstain": metrics.get("primary_abstain_exact"),
        "auxiliary": metrics.get("auxiliary_set_exact"),
        "abstain_f1": (metrics.get("abstain") or {}).get("f1"),
    }
    return float(
        0.45 * (values["route"] if values["route"] is not None else 0.0)
        + 0.25 * (values["primary_abstain"] if values["primary_abstain"] is not None else 0.0)
        + 0.20 * (values["auxiliary"] if values["auxiliary"] is not None else 0.0)
        + 0.10 * (values["abstain_f1"] if values["abstain_f1"] is not None else 0.0)
    )

def _train_epoch(
    model: MicroconceptRouter,
    rows: tuple[RouterExample, ...],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    config: dict[str, Any],
    auxiliary_pos_weight: torch.Tensor,
    primary_class_weight: torch.Tensor,
    abstain_pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.train()
    accumulated: dict[str, float] = {"total": 0.0, "primary": 0.0, "auxiliary": 0.0, "abstain": 0.0, "attention_alignment": 0.0}
    batches = 0
    for batch in _loader(rows, batch_size, shuffle=True):
        device_batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(device_batch)
        loss, items = router_loss(
            output,
            device_batch,
            primary_weight=float(config["primary_loss_weight"]),
            auxiliary_weight=float(config["auxiliary_loss_weight"]),
            abstain_weight=float(config["abstain_loss_weight"]),
            auxiliary_pos_weight=auxiliary_pos_weight,
            primary_class_weight=primary_class_weight,
            abstain_pos_weight=abstain_pos_weight,
            attention_alignment_weight=float(config.get("attention_alignment_loss_weight", 0.0)),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        for key, value in items.items():
            accumulated[key] += value
        batches += 1
    return {key: value / max(1, batches) for key, value in accumulated.items()}


def _auxiliary_pos_weight(rows: tuple[RouterExample, ...], device: torch.device) -> torch.Tensor:
    matrix = np.stack([row.auxiliary_targets for row in rows])
    positives = matrix.sum(axis=0)
    weights = np.clip((len(rows) - positives) / np.maximum(positives, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _primary_class_weight(rows: tuple[RouterExample, ...], class_count: int, device: torch.device) -> torch.Tensor:
    targets = [row.primary_target for row in rows if row.primary_target != IGNORE_INDEX]
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=class_count).astype(np.float32)
    total = float(counts.sum())
    weights = np.ones(class_count, dtype=np.float32)
    present = counts > 0
    if total and present.any():
        weights[present] = total / (float(present.sum()) * counts[present])
        weights[present] = np.clip(weights[present], 0.5, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _abstain_pos_weight(rows: tuple[RouterExample, ...], device: torch.device) -> torch.Tensor:
    positives = float(sum(row.abstain for row in rows))
    negatives = float(len(rows) - positives)
    # A bounded positive weight prevents the tiny pilot from collapsing to the
    # majority non-abstain class while avoiding an unstable infinite weight.
    value = float(np.clip(negatives / max(positives, 1.0), 1.0, 10.0))
    return torch.tensor(value, dtype=torch.float32, device=device)


def run(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    inference = config["inference"]
    include_sealed = bool(args.evaluate_sealed)
    if include_sealed and not args.confirm_sealed_evaluation:
        raise ValueError("sealed evaluation requires --confirm-sealed-evaluation")
    _seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    teacher_target_source = args.teacher_target_source or config.get("teacher_target_source", "review-v1")
    bundle = load_route_data(
        Path(args.dataset_root), include_sealed=include_sealed,
        minimum_primary_support=int(training["minimum_train_primary_support"]),
        minimum_auxiliary_support=int(training["minimum_train_auxiliary_support"]),
        teacher_target_source=teacher_target_source,
    )
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
        bundle.graph, bundle.support, len(bundle.direction_vocab),
        hidden_dim=int(config["architecture"]["node_hidden_dim"]),
        relation_embedding_dim=int(config["architecture"]["relation_embedding_dim"]),
        model_embedding_dim=int(config["architecture"]["model_embedding_dim"]),
        task_relevance_attention_strength=float(config["architecture"].get("task_relevance_attention_strength", 0.15)),
        task_relevance_residual_strength=float(config["architecture"].get("task_relevance_residual_strength", 0.25)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    auxiliary_pos_weight = _auxiliary_pos_weight(bundle.train, device)
    primary_class_weight = _primary_class_weight(bundle.train, len(bundle.support.primary_models), device)
    abstain_pos_weight = _abstain_pos_weight(bundle.train, device)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = _train_epoch(
            model,
            bundle.train,
            optimizer,
            int(training["batch_size"]),
            device,
            training,
            auxiliary_pos_weight,
            primary_class_weight,
            abstain_pos_weight,
        )
        development, _ = _evaluate(
            model,
            bundle.development,
            int(training["batch_size"]),
            device,
            auxiliary_threshold=float(inference["auxiliary_threshold"]),
            abstain_threshold=float(inference["abstain_threshold"]),
        )
        score = _checkpoint_score(development)
        history.append({"epoch": epoch, "train_loss": train_loss, "development": development})
        if score > best_score:
            best_score, best_epoch = float(score), epoch
            torch.save({
                "state_dict": model.state_dict(), "config": config, "support": asdict(bundle.support),
                "direction_vocab": bundle.direction_vocab, "route_models": model.route_models,
            }, output_dir / "checkpoints" / "best.pt")
    checkpoint = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    reports: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "history": history,
        "device": str(device),
        "teacher_target_source": teacher_target_source,
        "training_balance": {
            "primary_class_weight": primary_class_weight.detach().cpu().tolist(),
            "abstain_pos_weight": float(abstain_pos_weight.detach().cpu()),
            "auxiliary_pos_weight": auxiliary_pos_weight.detach().cpu().tolist(),
            "attention_alignment_loss_weight": float(training.get("attention_alignment_loss_weight", 0.0)),
        },
    }
    for split_name, rows in (("train", bundle.train), ("development", bundle.development), ("sealed_test", bundle.sealed_test)):
        if not rows:
            continue
        metrics, predictions = _evaluate(model, rows, int(training["batch_size"]), device, auxiliary_threshold=float(inference["auxiliary_threshold"]), abstain_threshold=float(inference["abstain_threshold"]))
        reports[split_name] = metrics
        reports.setdefault("task_relevance_sensitivity", {})[split_name] = _task_relevance_sensitivity(
            model,
            rows,
            metrics,
            predictions,
            int(training["batch_size"]),
            device,
            auxiliary_threshold=float(inference["auxiliary_threshold"]),
            abstain_threshold=float(inference["abstain_threshold"]),
        )
        _atomic_jsonl(output_dir / "predictions" / f"{split_name}.jsonl", predictions)
    target_description = (
        "dual-adjudicator exact-consensus plus parent accepted-review"
        if teacher_target_source == "adjudication-v1"
        else "accepted-review"
    )
    reports["limitations"] = [
        "This is a microconcept-state-to-model router, not an end-to-end fact encoder.",
        f"Labels are Weak {target_description} LLM supervision, not human Gold labels.",
        "Only models meeting train-only support thresholds produce numeric predictions; other model roles remain unsupported/null.",
        "Sealed test labels are not read unless explicit opt-in confirmation is provided.",
    ]
    _atomic_json(output_dir / "reports" / "training_report.json", reports)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_v1.yaml"))
    parser.add_argument("--teacher-target-source", choices=("review-v1", "adjudication-v1"), default=None)
    parser.add_argument("--evaluate-sealed", action="store_true")
    parser.add_argument("--confirm-sealed-evaluation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    output = run(args)
    print(f"Trained microconcept router: {output}")


if __name__ == "__main__":
    main()
