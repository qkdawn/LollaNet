"""Masked loss, calibration, metrics, deterministic rule projection, and interventions."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .data import (
    APPLICABILITY_VALUES, CONCEPT_IDS, DIRECTION_VALUES, STATUS_VALUES, Example,
    frozen_hash_text,
)
from .model import select_node_logits

IGNORE = -100


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target != IGNORE
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], target[mask])


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ordinal CE plus expected-rank error, only over genuinely observed targets."""
    mask = target != IGNORE
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected, y = logits[mask], target[mask]
    ce = nn.functional.cross_entropy(selected, y)
    values = torch.arange(selected.shape[-1], dtype=selected.dtype, device=selected.device)
    expected = (torch.softmax(selected, dim=-1) * values).sum(dim=-1)
    return ce + 0.25 * nn.functional.smooth_l1_loss(expected, y.to(expected.dtype))


def multitask_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float], include_task: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    selected = select_node_logits(outputs, batch["concept"])
    parts = {
        "status": masked_cross_entropy(selected["status"], batch["status"]),
        "intensity": ordinal_loss(selected["intensity"], batch["intensity"]),
        "direction": masked_cross_entropy(selected["direction"], batch["direction"]),
        "applicability": masked_cross_entropy(selected["applicability"], batch["applicability"]),
        "relevance": ordinal_loss(selected["relevance"], batch["relevance"]),
    }
    total = parts["status"] + weights["intensity"] * parts["intensity"] + weights["direction"] * parts["direction"]
    if include_task:
        total = total + weights["applicability"] * parts["applicability"] + weights["relevance"] * parts["relevance"]
    return total, {key: float(value.detach().cpu()) for key, value in parts.items()}


@dataclass(frozen=True)
class Temperatures:
    status: float = 1.0
    intensity: float = 1.0
    direction: float = 1.0
    applicability: float = 1.0
    relevance: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in ("status", "intensity", "direction", "applicability", "relevance")}


def _probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = logits / float(temperature)
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def select_temperatures(predictions: list[dict[str, Any]], grid: Iterable[float]) -> Temperatures:
    values: dict[str, float] = {}
    for target in ("status", "intensity", "direction", "applicability", "relevance"):
        usable = [row for row in predictions if row[f"target_{target}"] != IGNORE]
        if not usable:
            values[target] = 1.0
            continue
        y = np.array([row[f"target_{target}"] for row in usable])
        logits = np.array([row[f"logits_{target}"] for row in usable], dtype=np.float64)
        best = min(grid, key=lambda t: float(np.mean(np.sum((_probs(logits, t) - np.eye(logits.shape[1])[y]) ** 2, axis=1))))
        values[target] = float(best)
    return Temperatures(**values)


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float | None:
    if len(targets) == 0:
        return None
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    total = len(targets)
    error = 0.0
    for low, high in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def _classification_metrics(rows: list[dict[str, Any]], target: str, labels: tuple[str, ...], temperature: float) -> dict[str, Any]:
    usable = [row for row in rows if row[f"target_{target}"] != IGNORE]
    if not usable:
        return {"n": 0, "macro_f1": None, "brier": None, "ece": None, "precision": {}, "recall": {}}
    y = np.asarray([row[f"target_{target}"] for row in usable], dtype=int)
    prob = _probs(np.asarray([row[f"logits_{target}"] for row in usable], dtype=float), temperature)
    pred = prob.argmax(axis=1)
    precision: dict[str, float] = {}; recall: dict[str, float] = {}; support: dict[str, int] = {}; f1s: list[float] = []
    for index, label in enumerate(labels):
        tp = int(np.sum((pred == index) & (y == index))); fp = int(np.sum((pred == index) & (y != index))); fn = int(np.sum((pred != index) & (y == index)))
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        precision[label] = p; recall[label] = r; support[label] = int(np.sum(y == index)); f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return {
        "n": int(len(y)), "macro_f1": float(np.mean(f1s)),
        "brier": float(np.mean(np.sum((prob - np.eye(len(labels))[y]) ** 2, axis=1))), "ece": ece(prob, y),
        "precision": precision, "recall": recall, "support": support,
    }

def _quadratic_kappa(y: np.ndarray, pred: np.ndarray, classes: int = 4) -> float | None:
    if len(y) == 0 or (len(np.unique(y)) == 1 and len(np.unique(pred)) == 1):
        return None
    observed = np.zeros((classes, classes), dtype=float)
    for actual, estimated in zip(y, pred):
        observed[int(actual), int(estimated)] += 1
    observed /= observed.sum()
    actual_hist = np.bincount(y, minlength=classes) / len(y); pred_hist = np.bincount(pred, minlength=classes) / len(pred)
    expected = np.outer(actual_hist, pred_hist)
    grid = np.arange(classes, dtype=float); weights = ((grid[:, None] - grid[None, :]) ** 2) / max((classes - 1) ** 2, 1)
    denom = float((weights * expected).sum())
    return float(1 - (weights * observed).sum() / denom) if denom > 0 else None


def _ordinal_metrics(rows: list[dict[str, Any]], target: str, temperature: float) -> dict[str, Any]:
    usable = [row for row in rows if row[f"target_{target}"] != IGNORE]
    if not usable:
        return {"n": 0, "mae": None, "weighted_kappa": None}
    y = np.asarray([row[f"target_{target}"] for row in usable], dtype=int)
    pred = _probs(np.asarray([row[f"logits_{target}"] for row in usable], dtype=float), temperature).argmax(axis=1)
    return {"n": int(len(y)), "mae": float(np.abs(y - pred).mean()), "weighted_kappa": _quadratic_kappa(y, pred)}

def evaluate_predictions(rows: list[dict[str, Any]], temperatures: Temperatures, split: str, label_scope: str) -> dict[str, Any]:
    report = {
        "split": split, "label_scope": label_scope, "rows": len(rows), "episodes": len({row["episode_id"] for row in rows}),
        "status": _classification_metrics(rows, "status", STATUS_VALUES, temperatures.status),
        "direction": _classification_metrics(rows, "direction", DIRECTION_VALUES, temperatures.direction),
        "applicability": _classification_metrics(rows, "applicability", APPLICABILITY_VALUES, temperatures.applicability),
        "intensity": _ordinal_metrics(rows, "intensity", temperatures.intensity),
        "relevance": _ordinal_metrics(rows, "relevance", temperatures.relevance),
    }
    per_concept: dict[str, Any] = {}
    for concept in CONCEPT_IDS:
        subset = [row for row in rows if row["microconcept_id"] == concept]
        per_concept[concept] = {
            "status": _classification_metrics(subset, "status", STATUS_VALUES, temperatures.status),
            "intensity": _ordinal_metrics(subset, "intensity", temperatures.intensity),
            "relevance": _ordinal_metrics(subset, "relevance", temperatures.relevance),
        }
    report["per_concept"] = per_concept
    # Low frequency means fewer than ten observed (non-unknown) status rows in
    # this split.  We report separately rather than silently treating it as zero.
    report["low_frequency_concepts"] = [name for name, item in per_concept.items() if item["status"]["n"] < 10]
    return report


def prediction_rows(model: torch.nn.Module, batches: Iterable[dict[str, Any]], device: torch.device, examples: list[Example], temperatures: Temperatures | None = None) -> list[dict[str, Any]]:
    model.eval(); output: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for raw_batch in batches:
            size = len(raw_batch["examples"])
            tensors = {key: value.to(device) for key, value in raw_batch.items() if isinstance(value, torch.Tensor)}
            selected = select_node_logits(model(tensors), tensors["concept"])
            for index in range(size):
                example = raw_batch["examples"][index]
                item: dict[str, Any] = {
                    "example_id": example.example_id, "episode_id": example.episode_id, "accession": example.accession,
                    "available_at": example.available_at, "year": example.year, "microconcept_id": example.concept_id,
                    "task_variant": example.task_variant, "label_tier": example.effective_tier,
                    "target_status": int(tensors["status"][index].cpu()), "target_intensity": int(tensors["intensity"][index].cpu()),
                    "target_direction": int(tensors["direction"][index].cpu()), "target_applicability": int(tensors["applicability"][index].cpu()),
                    "target_relevance": int(tensors["relevance"][index].cpu()),
                    "activation": float(selected["activation"][index].cpu()),
                }
                for target in ("status", "intensity", "direction", "applicability", "relevance"):
                    item[f"logits_{target}"] = [float(v) for v in selected[target][index].cpu().tolist()]
                output.append(item)
            offset += size
    return output


def rule_projection_rows(examples: list[Example]) -> list[dict[str, Any]]:
    """Deterministic projection baseline, deliberately retained even when it wins.

    The current pilot labels originate from a deterministic F+C projection.  This
    baseline is therefore expected to be a strong (often tautological) comparator,
    not evidence of learned generalisation.
    """
    from research.phase1.microconcepts.microconcept_labels_v1 import Card, Episode, decide_state, load_cards, project_signals
    cards = load_cards()
    claims_by_accession: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        for claim in example.claims:
            claims_by_accession[example.accession].append({
                "claim_key": claim.claim_key, "claim_id": claim.claim_id, "accession": claim.accession,
                "available_at": claim.available_at, "polarity": claim.polarity, "time_horizon": claim.time_horizon,
                "strength": claim.strength, "confidence": claim.confidence, "section": "MD&A", "start_char": 0,
                "end_char": 1, "text_sha256": "rule-baseline-not-used-as-input", "extractor_version": "rule-baseline",
            })
    output: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for example in examples:
        key = (example.episode_id, example.concept_id)
        if key not in cache:
            episode = Episode(example.episode_id, example.accession, example.available_at, "10-Q", "SIC-unknown", example.factors, {name: "encoded_factor" for name in example.factors}, {})
            signals, reasons = project_signals(episode, cards[example.concept_id], claims_by_accession)
            cache[key], _ = decide_state(episode, cards[example.concept_id], signals, reasons)
        projected = cache[key]
        status = projected["status"]
        direction = projected["direction"]
        # q/g templates are an explicit deterministic task comparator, not a
        # label lookup.  Exact natural-language template matching is intentionally
        # conservative; unmatched tasks become unknown.
        unrelated = "文档归档" in example.question or "文档管理流程" in example.goal
        app = "not_applicable" if unrelated else "applicable"
        relevance = 0 if unrelated else (3 if status != "unknown" else 2)
        def one_hot(value: int, size: int) -> list[float]:
            return [12.0 if i == value else -12.0 for i in range(size)]
        output.append({
            "example_id": example.example_id, "episode_id": example.episode_id, "accession": example.accession, "available_at": example.available_at,
            "year": example.year, "microconcept_id": example.concept_id, "task_variant": example.task_variant, "label_tier": example.effective_tier,
            "target_status": STATUS_VALUES.index(example.status) if example.status in STATUS_VALUES else IGNORE,
            "target_intensity": example.intensity if example.intensity is not None else IGNORE,
            "target_direction": DIRECTION_VALUES.index(example.direction) if example.direction in DIRECTION_VALUES else IGNORE,
            "target_applicability": APPLICABILITY_VALUES.index(example.applicability) if example.applicability in APPLICABILITY_VALUES else IGNORE,
            "target_relevance": example.relevance if example.relevance is not None else IGNORE,
            "logits_status": one_hot(STATUS_VALUES.index(status) if status in STATUS_VALUES else 0, 2),
            "logits_intensity": one_hot(int(projected["intensity"]), 4),
            "logits_direction": one_hot(DIRECTION_VALUES.index(direction) if direction in DIRECTION_VALUES else 0, 4),
            "logits_applicability": one_hot(APPLICABILITY_VALUES.index(app), 2),
            "logits_relevance": one_hot(relevance, 4),
            "activation": float(0.95 if app == "applicable" and status != "unknown" else 0.05),
        })
    return output


def task_pair_metrics(predictions: list[dict[str, Any]], temperatures: Temperatures) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["episode_id"], row["microconcept_id"])].append(row)
    differences: list[float] = []; aligned: list[bool] = []
    for pair in grouped.values():
        focused = next((x for x in pair if x["task_variant"] == "focused"), None)
        unrelated = next((x for x in pair if x["task_variant"] == "unrelated"), None)
        if focused is None or unrelated is None:
            continue
        fp = _probs(np.array([focused["logits_relevance"]]), temperatures.relevance)[0]
        up = _probs(np.array([unrelated["logits_relevance"]]), temperatures.relevance)[0]
        fexp = float(np.dot(fp, np.arange(4))); uexp = float(np.dot(up, np.arange(4)))
        differences.append(fexp - uexp); aligned.append(fexp > uexp)
    return {"paired_rows": len(differences), "goal_sensitivity_mean_relevance_delta": float(np.mean(differences)) if differences else None, "goal_sensitivity_alignment": float(np.mean(aligned)) if aligned else None}
