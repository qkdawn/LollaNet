"""Train and evaluate the Phase 1 partial-supervision neural router on CPU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, mean_absolute_error, precision_recall_fscore_support
from scipy.stats import spearmanr
import yaml

try:
    from .download_free_pilot import external_output
    from .router_data import ROUTE_RULES, RouterPreprocessor, audit_records, load_records, targets
    from .router_decode import decode_route
except ImportError:
    from download_free_pilot import external_output
    from router_data import ROUTE_RULES, RouterPreprocessor, audit_records, load_records, targets
    from router_decode import decode_route


ROOT = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 53, 71)
LAMBDAS = (0.0, 0.05, 0.1, 0.2)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Install research/phase1/value_investing/requirements-router.txt before training.") from error
    return torch, nn


def set_seed(seed: int) -> None:
    torch, _nn = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(input_dim: int, sector_count: int):
    torch, nn = _torch()

    class Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(sector_count, 8)
            self.encoder = nn.Sequential(nn.Linear(input_dim + 8, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 32), nn.ReLU())
            self.route = nn.Linear(32, len(ROUTE_RULES))
            self.return_head = nn.Linear(32, 1)

        def forward(self, values, sectors):
            hidden = self.encoder(torch.cat((values, self.embedding(sectors)), dim=1))
            return self.route(hidden), self.return_head(hidden).squeeze(1)

    return Router()


@dataclass
class Predictions:
    rule: np.ndarray
    returns: np.ndarray


def predict(model, values: np.ndarray, sectors: np.ndarray) -> Predictions:
    torch, _nn = _torch()
    model.eval()
    with torch.no_grad():
        logits, predicted_return = model(torch.tensor(values), torch.tensor(sectors))
    return Predictions(torch.sigmoid(logits).cpu().numpy(), predicted_return.cpu().numpy())


def train_one(train: dict[str, np.ndarray], development: dict[str, np.ndarray], input_dim: int, sector_count: int, seed: int, return_weight: float):
    torch, _nn = _torch()
    set_seed(seed)
    model = build_model(input_dim, sector_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_values = torch.tensor(train["values"])
    train_sectors = torch.tensor(train["sectors"])
    route_y, route_mask = torch.tensor(train["route_y"]), torch.tensor(train["route_mask"])
    return_y, return_mask = torch.tensor(train["return_y"]), torch.tensor(train["return_mask"])
    positive = (route_y * route_mask).sum(0)
    negative = route_mask.sum(0) - positive
    pos_weight = torch.clamp(negative / torch.clamp(positive, min=1), max=30)
    best_state, best_score, patience = None, -np.inf, 0
    for _epoch in range(250):
        model.train()
        optimizer.zero_grad()
        logits, predicted_return = model(train_values, train_sectors)
        route_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, route_y, pos_weight=pos_weight, reduction="none")
        route_loss = (route_loss * route_mask).sum() / torch.clamp(route_mask.sum(), min=1)
        return_loss = torch.nn.functional.smooth_l1_loss(predicted_return, return_y, reduction="none")
        return_loss = (return_loss * return_mask).sum() / torch.clamp(return_mask.sum(), min=1)
        (route_loss + return_weight * return_loss).backward()
        optimizer.step()
        probabilities = predict(model, development["values"], development["sectors"]).rule
        score = macro_pr_auc(development["route_y"], development["route_mask"], probabilities)
        if score > best_score + 1e-6:
            best_score, patience = score, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model, best_score


def macro_pr_auc(y: np.ndarray, mask: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    for index in range(len(ROUTE_RULES)):
        observed = mask[:, index].astype(bool)
        scores.append(average_precision_score(y[observed, index], probabilities[observed, index]))
    return float(np.mean(scores))


def thresholds_from_development(data: dict[str, np.ndarray], probabilities: np.ndarray) -> dict[str, float]:
    thresholds = {}
    for index, rule in enumerate(ROUTE_RULES):
        observed = data["route_mask"][:, index].astype(bool)
        candidates = np.linspace(0.05, 0.95, 19)
        scores = [f1_score(data["route_y"][observed, index], probabilities[observed, index] >= threshold, zero_division=0) for threshold in candidates]
        thresholds[rule] = float(candidates[int(np.argmax(scores))])
    return thresholds


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for start in np.linspace(0, 0.9, bins):
        selected = (probabilities >= start) & (probabilities < start + 0.1)
        if selected.any():
            result += selected.mean() * abs(y[selected].mean() - probabilities[selected].mean())
    return float(result)


def rule_metrics(data: dict[str, np.ndarray], probabilities: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    result = {}
    for index, rule in enumerate(ROUTE_RULES):
        observed = data["route_mask"][:, index].astype(bool)
        y, p = data["route_y"][observed, index], probabilities[observed, index]
        precision, recall, f1, _support = precision_recall_fscore_support(y, p >= thresholds[rule], average="binary", zero_division=0)
        result[rule] = {"observed": int(observed.sum()), "positives": int(y.sum()), "pr_auc": float(average_precision_score(y, p)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)}
    return result


def graph_metrics(records: list[dict[str, Any]], probabilities: np.ndarray, thresholds: dict[str, float], rules: dict[str, Any]) -> dict[str, float]:
    exact, edge_tp, edge_fp, edge_fn = 0, 0, 0, 0
    for record, row in zip(records, probabilities):
        predicted = decode_route(dict(zip(ROUTE_RULES, map(float, row))), thresholds, rules)
        actual_edges = {tuple(edge) for edge in record["route_labels"]["conditional_edges"]}
        predicted_edges = {tuple(edge) for edge in predicted["conditional_edges"]}
        exact += predicted["ordered_models"] == record["route_labels"]["ordered_models"]
        edge_tp += len(actual_edges & predicted_edges)
        edge_fp += len(predicted_edges - actual_edges)
        edge_fn += len(actual_edges - predicted_edges)
    precision = edge_tp / (edge_tp + edge_fp) if edge_tp + edge_fp else 0.0
    recall = edge_tp / (edge_tp + edge_fn) if edge_tp + edge_fn else 0.0
    return {"route_exact_match": exact / len(records), "edge_precision": precision, "edge_recall": recall, "edge_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def return_metrics(data: dict[str, np.ndarray], predictions: np.ndarray, mean_std: tuple[float, float] | None) -> dict[str, Any]:
    observed = data["return_mask"].astype(bool)
    if not observed.any() or mean_std is None:
        return {"observed": 0}
    y = data["return_y"][observed] * mean_std[1] + mean_std[0]
    p = predictions[observed] * mean_std[1] + mean_std[0]
    return {"observed": int(observed.sum()), "mae_log_return": float(mean_absolute_error(y, p)), "spearman": float(spearmanr(y, p).statistic), "directional_accuracy": float(np.mean((y >= 0) == (p >= 0))), "status": "exploratory_price_only"}


def fit_logistic(train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray]) -> np.ndarray:
    probability = np.zeros((len(evaluation["values"]), len(ROUTE_RULES)), dtype=np.float32)
    for index in range(len(ROUTE_RULES)):
        observed = train["route_mask"][:, index].astype(bool)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
        model.fit(train["values"][observed], train["route_y"][observed, index])
        probability[:, index] = model.predict_proba(evaluation["values"])[:, 1]
    return probability


def package(records: list[dict[str, Any]], preprocessor: RouterPreprocessor, return_clip=None, return_mean_std=None) -> tuple[dict[str, np.ndarray], tuple[float, float] | None, tuple[float, float] | None]:
    values, sectors, _missing = preprocessor.transform(records)
    route_y, route_mask, return_y, return_mask, clip, mean_std = targets(records, return_clip)
    if return_mean_std is not None:
        # targets computes mean/std from its own split; convert to the train-established scale.
        raw = return_y * (mean_std[1] if mean_std else 1) + (mean_std[0] if mean_std else 0)
        return_y = (raw - return_mean_std[0]) / return_mean_std[1]
        mean_std = return_mean_std
    return {"values": values, "sectors": sectors, "route_y": route_y, "route_mask": route_mask, "return_y": return_y, "return_mask": return_mask}, clip, mean_std


def run(input_path: Path, output: Path) -> dict[str, Any]:
    input_path, output = external_output(input_path), external_output(output)
    records, fingerprint = load_records(input_path)
    audit = audit_records(records)
    groups = {split: [record for record in records if record["_split"] == split] for split in ("train", "development", "sealed_test")}
    preprocessor = RouterPreprocessor.fit(groups["train"])
    train, clip, mean_std = package(groups["train"], preprocessor)
    development, _clip, _mean_std = package(groups["development"], preprocessor, clip, mean_std)
    sealed, _clip, _mean_std = package(groups["sealed_test"], preprocessor, clip, mean_std)
    output.mkdir(parents=True, exist_ok=True)
    grid: dict[float, list[Any]] = {}
    for weight in LAMBDAS:
        grid[weight] = [train_one(train, development, train["values"].shape[1], len(preprocessor.sectors), seed, weight) for seed in SEEDS]
    selected_weight = max(LAMBDAS, key=lambda weight: float(np.mean([item[1] for item in grid[weight]])))
    models = [item[0] for item in grid[selected_weight]]
    dev_probabilities = np.mean([predict(model, development["values"], development["sectors"]).rule for model in models], axis=0)
    test_predictions = [predict(model, sealed["values"], sealed["sectors"]) for model in models]
    test_rule = np.mean([item.rule for item in test_predictions], axis=0)
    test_return = np.mean([item.returns for item in test_predictions], axis=0)
    thresholds = thresholds_from_development(development, dev_probabilities)
    rules = yaml.safe_load((ROOT / "route_rules.yaml").read_text(encoding="utf-8"))
    logistic = fit_logistic(train, sealed)
    report = {"experiment": "phase1_partial_supervision_router_v1", "data_fingerprint_sha256": fingerprint, "limitations": ["Rule labels are pre-registered-rule supervision, not human causal ground truth.", "Yahoo Finance price history uses an unverified current-ticker mapping.", "Price outcomes omit complete delisting returns and cannot support investment-performance claims."], "audit": audit, "preprocessor": preprocessor.to_dict(), "return_target": {"clip_log_return": clip, "mean_std": mean_std, "status": "exploratory_price_only"}, "selected_return_loss_weight": selected_weight, "selection_development_macro_pr_auc": {str(weight): float(np.mean([item[1] for item in grid[weight]])) for weight in LAMBDAS}, "thresholds": thresholds, "sealed_test": {"neural_rules": rule_metrics(sealed, test_rule, thresholds), "neural_graph": graph_metrics(groups["sealed_test"], test_rule, thresholds, rules), "neural_return": return_metrics(sealed, test_return, mean_std), "logistic_rules": rule_metrics(sealed, logistic, thresholds)}, "oracle": "The pre-registered rule engine is a deterministic supervision ceiling, not a learned baseline."}
    for index, model in enumerate(models):
        torch, _nn = _torch()
        torch.save(model.state_dict(), output / f"router_seed_{SEEDS[index]}.pt")
    (output / "run_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    predictions = []
    for record, probability, return_prediction in zip(groups["sealed_test"], test_rule, test_return):
        decoded = decode_route(dict(zip(ROUTE_RULES, map(float, probability))), thresholds, rules)
        predictions.append({"episode_id": record["episode_id"], "rule_probabilities": dict(zip(ROUTE_RULES, map(float, probability))), "predicted_standardized_log_return": float(return_prediction), "decoded_route": decoded})
    (output / "sealed_test_predictions.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="External market-valued candidate JSONL")
    parser.add_argument("--output", required=True, type=Path, help="External run directory")
    args = parser.parse_args()
    report = run(args.input, args.output)
    print(f"Completed router experiment with lambda={report['selected_return_loss_weight']}; results: {external_output(args.output)}")


if __name__ == "__main__":
    main()
