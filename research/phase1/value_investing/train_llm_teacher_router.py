"""Train a neural router from blinded GPT teacher pseudo-labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_fscore_support
import yaml

try:
    from .download_free_pilot import external_output
    from .router_data import RouterPreprocessor, load_records
    from .validate_llm_teacher_labels import ALLOWED_MODELS, MODEL_ASSETS
except ImportError:
    from download_free_pilot import external_output
    from router_data import RouterPreprocessor, load_records
    from validate_llm_teacher_labels import ALLOWED_MODELS, MODEL_ASSETS


SEEDS = (11, 23, 37, 53, 71)
MINIMUM_POSITIVES = {"train": 20, "development": 5, "sealed_test": 5}


def _torch():
    import torch
    from torch import nn
    return torch, nn


def set_seed(seed: int) -> None:
    torch, _nn = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_teacher_examples(candidate_path: Path, teacher_path: Path) -> tuple[dict[str, list[tuple[dict[str, Any], dict[str, Any]]]], str]:
    records, fingerprint = load_records(candidate_path)
    by_id = {record["episode_id"]: record for record in records}
    labels = [json.loads(line) for line in teacher_path.read_text(encoding="utf-8").splitlines()]
    groups = {split: [] for split in ("train", "development", "sealed_test")}
    for label in labels:
        record = by_id.get(label["episode_id"])
        if record is None or label["abstain"]:
            continue
        groups[record["_split"]].append((record, label))
    return groups, fingerprint


def select_trainable_models(groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> tuple[list[str], dict[str, dict[str, int]], list[str]]:
    counts = {model: {split: sum(model in label["selected_models"] for _record, label in examples) for split, examples in groups.items()} for model in sorted(ALLOWED_MODELS)}
    trainable = [model for model in sorted(ALLOWED_MODELS) if all(counts[model][split] >= minimum for split, minimum in MINIMUM_POSITIVES.items())]
    unsupported = sorted(ALLOWED_MODELS - set(trainable))
    return trainable, counts, unsupported


def build_arrays(examples: list[tuple[dict[str, Any], dict[str, Any]]], preprocessor: RouterPreprocessor, models: list[str]) -> dict[str, np.ndarray]:
    records = [record for record, _label in examples]
    values, sectors, _missing = preprocessor.transform(records)
    activation = np.zeros((len(examples), len(models)), dtype=np.float32)
    rank = np.zeros_like(activation)
    rank_mask = np.zeros_like(activation)
    count = np.zeros(len(examples), dtype=np.int64)
    confidence = np.zeros(len(examples), dtype=np.float32)
    model_index = {model: index for index, model in enumerate(models)}
    for row, (_record, label) in enumerate(examples):
        filtered = [model for model in label["ordered_models"] if model in model_index]
        count[row] = min(len(filtered), 7)
        confidence[row] = float(label["confidence"])
        denominator = max(len(filtered) - 1, 1)
        for position, model in enumerate(filtered):
            column = model_index[model]
            activation[row, column] = 1.0
            rank[row, column] = position / denominator
            rank_mask[row, column] = 1.0
    return {"values": values, "sectors": sectors, "activation": activation, "rank": rank, "rank_mask": rank_mask, "count": count, "confidence": confidence}


def build_model(input_dim: int, sector_count: int, model_count: int):
    torch, nn = _torch()

    class TeacherRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(sector_count, 8)
            self.encoder = nn.Sequential(nn.Linear(input_dim + 8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
            self.activation_head = nn.Linear(32, model_count)
            self.rank_head = nn.Linear(32, model_count)
            self.count_head = nn.Linear(32, 8)

        def forward(self, values, sectors):
            hidden = self.encoder(torch.cat((values, self.embedding(sectors)), dim=1))
            return self.activation_head(hidden), torch.sigmoid(self.rank_head(hidden)), self.count_head(hidden)

    return TeacherRouter()


def predict(model, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch, _nn = _torch()
    model.eval()
    with torch.no_grad():
        activation, rank, count = model(torch.tensor(data["values"]), torch.tensor(data["sectors"]))
    return torch.sigmoid(activation).numpy(), rank.numpy(), torch.softmax(count, dim=1).numpy()


def macro_ap(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean([average_precision_score(y[:, index], probabilities[:, index]) for index in range(y.shape[1])]))


def train_one(train: dict[str, np.ndarray], development: dict[str, np.ndarray], input_dim: int, sector_count: int, model_count: int, seed: int):
    torch, _nn = _torch()
    set_seed(seed)
    model = build_model(input_dim, sector_count, model_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tensors = {name: torch.tensor(value) for name, value in train.items()}
    positive = tensors["activation"].sum(0)
    negative = len(train["activation"]) - positive
    pos_weight = torch.clamp(negative / torch.clamp(positive, min=1), max=20)
    best_state, best_score, patience = None, -np.inf, 0
    for _epoch in range(300):
        model.train()
        optimizer.zero_grad()
        activation_logits, predicted_rank, count_logits = model(tensors["values"], tensors["sectors"])
        activation_loss = torch.nn.functional.binary_cross_entropy_with_logits(activation_logits, tensors["activation"], pos_weight=pos_weight, reduction="none").mean(1)
        activation_loss = (activation_loss * tensors["confidence"]).mean()
        rank_loss = torch.nn.functional.smooth_l1_loss(predicted_rank, tensors["rank"], reduction="none")
        rank_loss = (rank_loss * tensors["rank_mask"] * tensors["confidence"].unsqueeze(1)).sum() / torch.clamp(tensors["rank_mask"].sum(), min=1)
        count_loss = torch.nn.functional.cross_entropy(count_logits, tensors["count"], reduction="none")
        count_loss = (count_loss * tensors["confidence"]).mean()
        (activation_loss + 0.25 * rank_loss + 0.2 * count_loss).backward()
        optimizer.step()
        probability, _rank, _count = predict(model, development)
        score = macro_ap(development["activation"], probability)
        if score > best_score + 1e-6:
            best_score, patience = score, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    return model, best_score


def thresholds(development: dict[str, np.ndarray], probabilities: np.ndarray, models: list[str]) -> dict[str, float]:
    result = {}
    for index, model in enumerate(models):
        candidates = np.linspace(0.05, 0.95, 19)
        scores = [f1_score(development["activation"][:, index], probabilities[:, index] >= threshold, zero_division=0) for threshold in candidates]
        result[model] = float(candidates[int(np.argmax(scores))])
    return result


def decode(probability: np.ndarray, rank: np.ndarray, count_probability: np.ndarray, models: list[str]) -> list[str]:
    amount = int(np.argmax(count_probability))
    amount = min(max(amount, 0), min(7, len(models)))
    selected = np.argsort(-probability)[:amount]
    ordered = sorted(selected, key=lambda index: rank[index])
    return [models[index] for index in ordered]


def evaluate(data: dict[str, np.ndarray], probabilities: np.ndarray, ranks: np.ndarray, counts: np.ndarray, models: list[str], teacher_examples: list[tuple[dict[str, Any], dict[str, Any]]], calibrated: dict[str, float]) -> dict[str, Any]:
    per_model = {}
    for index, model in enumerate(models):
        y, p = data["activation"][:, index], probabilities[:, index]
        precision, recall, f1, _ = precision_recall_fscore_support(y, p >= calibrated[model], average="binary", zero_division=0)
        per_model[model] = {"positives": int(y.sum()), "pr_auc": float(average_precision_score(y, p)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "brier": float(brier_score_loss(y, p))}
    exact_set = exact_order = pair_correct = pair_total = count_correct = 0
    for row, (_record, label) in enumerate(teacher_examples):
        predicted = decode(probabilities[row], ranks[row], counts[row], models)
        actual = [model for model in label["ordered_models"] if model in models]
        exact_set += set(predicted) == set(actual)
        exact_order += predicted == actual
        count_correct += len(predicted) == len(actual)
        predicted_position = {model: index for index, model in enumerate(predicted)}
        for left in range(len(actual)):
            for right in range(left + 1, len(actual)):
                if actual[left] in predicted_position and actual[right] in predicted_position:
                    pair_total += 1
                    pair_correct += predicted_position[actual[left]] < predicted_position[actual[right]]
    return {"per_model": per_model, "macro_pr_auc": float(np.mean([item["pr_auc"] for item in per_model.values()])), "exact_selected_set_rate": exact_set / len(teacher_examples), "exact_ordered_route_rate": exact_order / len(teacher_examples), "predicted_count_accuracy": count_correct / len(teacher_examples), "pairwise_order_accuracy_on_jointly_selected": pair_correct / pair_total if pair_total else None}


def logistic_baseline(train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray], models: list[str]) -> dict[str, float]:
    result = {}
    for index, model_name in enumerate(models):
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
        classifier.fit(train["values"], train["activation"][:, index])
        probability = classifier.predict_proba(evaluation["values"])[:, 1]
        result[model_name] = float(average_precision_score(evaluation["activation"][:, index], probability))
    return result


def run(candidate_path: Path, teacher_path: Path, output: Path) -> dict[str, Any]:
    candidate_path, teacher_path, output = map(external_output, (candidate_path, teacher_path, output))
    groups, fingerprint = load_teacher_examples(candidate_path, teacher_path)
    models, counts, unsupported = select_trainable_models(groups)
    if not models:
        raise ValueError("no teacher-labeled models meet the train/dev/test support thresholds")
    preprocessor = RouterPreprocessor.fit([record for record, _label in groups["train"]])
    arrays = {split: build_arrays(examples, preprocessor, models) for split, examples in groups.items()}
    trained = [train_one(arrays["train"], arrays["development"], arrays["train"]["values"].shape[1], len(preprocessor.sectors), len(models), seed) for seed in SEEDS]
    development_predictions = [predict(model, arrays["development"]) for model, _score in trained]
    test_predictions = [predict(model, arrays["sealed_test"]) for model, _score in trained]
    dev_probability = np.mean([item[0] for item in development_predictions], axis=0)
    test_probability = np.mean([item[0] for item in test_predictions], axis=0)
    test_rank = np.mean([item[1] for item in test_predictions], axis=0)
    test_count = np.mean([item[2] for item in test_predictions], axis=0)
    calibrated = thresholds(arrays["development"], dev_probability, models)
    report = {
        "experiment": "phase1_llm_teacher_router_v1",
        "teacher_labels": "gpt-5.6-sol pseudo-labels; not human ground truth",
        "candidate_data_fingerprint_sha256": fingerprint,
        "split_sizes_non_abstained": {split: len(examples) for split, examples in groups.items()},
        "trainable_models": models,
        "unsupported_models": {model: {"reason": "insufficient teacher-positive support", "positive_counts": counts[model]} for model in unsupported},
        "minimum_positive_support": MINIMUM_POSITIVES,
        "positive_counts": counts,
        "preprocessor": preprocessor.to_dict(),
        "development_macro_pr_auc_by_seed": {str(seed): float(score) for seed, (_model, score) in zip(SEEDS, trained)},
        "thresholds": calibrated,
        "sealed_test": evaluate(arrays["sealed_test"], test_probability, test_rank, test_count, models, groups["sealed_test"], calibrated),
        "logistic_baseline_sealed_test_pr_auc": logistic_baseline(arrays["train"], arrays["sealed_test"], models),
        "limitations": ["Teacher labels come from one LLM family and are pseudo-labels.", "The 1,000 episodes were stratified for label diversity and are not population representative.", "Free market data uses an unverified current-ticker historical mapping.", "Unsupported model outputs must remain null, not zero."],
    }
    output.mkdir(parents=True, exist_ok=True)
    torch, _nn = _torch()
    for seed, (model, _score) in zip(SEEDS, trained):
        torch.save(model.state_dict(), output / f"teacher_router_seed_{seed}.pt")
    predictions = []
    for row, (record, _label) in enumerate(groups["sealed_test"]):
        route = decode(test_probability[row], test_rank[row], test_count[row], models)
        all_probabilities = {model: (float(test_probability[row, models.index(model)]) if model in models else None) for model in sorted(ALLOWED_MODELS)}
        predictions.append({"episode_id": record["episode_id"], "ordered_models": route, "activation_probabilities": all_probabilities, "unsupported_models": unsupported})
    (output / "sealed_test_predictions.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--teacher-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.candidates, args.teacher_labels, args.output)
    print(f"Trained {len(report['trainable_models'])} teacher-supervised model heads; output: {external_output(args.output)}")


if __name__ == "__main__":
    main()
