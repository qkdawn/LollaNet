"""Manifest-bound dense score data for the microconcept-to-model router."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from research.phase1.mechanism_questions_v1.model_score_chain_v1 import (
    MODEL_SCORE_MANIFEST_FILENAME,
    ModelScoreChainError,
    verify_model_score_manifest,
)
from research.phase1.mechanism_questions_v1.question_set_v1 import ValidationError
from .data import (
    STATUS_INDEX,
    SupportSet,
    SemanticGraph,
    RouterDataError,
    _direction_vocab,
    _fingerprint,
    _validate_states,
    load_semantic_graph,
)


@dataclass(frozen=True)
class DenseRouterExample:
    episode_id: str
    split: str
    status: np.ndarray
    direction: np.ndarray
    numeric: np.ndarray
    activation_targets: np.ndarray
    activation_mask: np.ndarray
    activation_weights: np.ndarray


@dataclass(frozen=True)
class DenseRouteDataBundle:
    graph: SemanticGraph
    support: SupportSet
    direction_vocab: tuple[str, ...]
    train: tuple[DenseRouterExample, ...]
    development: tuple[DenseRouterExample, ...]
    audit: dict[str, Any]


def _encode_dense_example(episode: dict[str, Any], target: dict[str, Any], split: str, graph: SemanticGraph, direction_vocab: tuple[str, ...]) -> DenseRouterExample:
    state_by_id = _validate_states(episode, graph)
    direction_index = {name: position for position, name in enumerate(direction_vocab)}
    status = np.array([STATUS_INDEX[state_by_id[concept]["status"]] for concept in graph.concept_ids], dtype=np.int64)
    direction = np.array([
        direction_index.get(state_by_id[concept]["direction"], direction_index["<UNK>"])
        for concept in graph.concept_ids
    ], dtype=np.int64)
    numeric = np.array([
        [
            state_by_id[concept]["intensity"] / 3.0,
            state_by_id[concept]["temporal_persistence"] / 3.0,
            state_by_id[concept]["task_relevance"] / 3.0,
        ]
        for concept in graph.concept_ids
    ], dtype=np.float32)
    target_map = target.get("activation_targets")
    mask_map = target.get("activation_loss_mask")
    weight_map = target.get("activation_loss_weight")
    if not isinstance(target_map, dict) or not isinstance(mask_map, dict) or not isinstance(weight_map, dict):
        raise RouterDataError(f"dense target {target.get('target_id')} lacks per-model maps")
    if set(target_map) != set(graph.model_ids) or set(mask_map) != set(graph.model_ids) or set(weight_map) != set(graph.model_ids):
        raise RouterDataError("dense targets must cover exactly all fixed thought models")
    activation_targets = np.zeros(len(graph.model_ids), dtype=np.float32)
    activation_mask = np.zeros(len(graph.model_ids), dtype=np.float32)
    activation_weights = np.zeros(len(graph.model_ids), dtype=np.float32)
    for position, model_id in enumerate(graph.model_ids):
        supervised = mask_map[model_id]
        value = target_map[model_id]
        weight = weight_map[model_id]
        if not isinstance(supervised, bool):
            raise RouterDataError(f"dense target mask must be boolean for {model_id}")
        if supervised:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise RouterDataError(f"invalid normalized activation target for {model_id}")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 < float(weight) <= 1.0:
                raise RouterDataError(f"invalid activation supervision weight for {model_id}")
            activation_targets[position] = float(value)
            activation_mask[position] = 1.0
            activation_weights[position] = float(weight)
        else:
            if value is not None or float(weight) != 0.0:
                raise RouterDataError(f"masked activation dimension must have null target and zero weight for {model_id}")
    if not activation_mask.any():
        raise RouterDataError(f"Episode {episode['episode_id']} has no supervised model dimensions")
    return DenseRouterExample(
        episode_id=str(episode["episode_id"]),
        split=split,
        status=status,
        direction=direction,
        numeric=numeric,
        activation_targets=activation_targets,
        activation_mask=activation_mask,
        activation_weights=activation_weights,
    )


def _coverage(rows: Iterable[DenseRouterExample], model_ids: tuple[str, ...]) -> dict[str, Any]:
    rows = tuple(rows)
    result: dict[str, Any] = {}
    for position, model_id in enumerate(model_ids):
        supervised = sum(bool(row.activation_mask[position]) for row in rows)
        useful = sum(bool(row.activation_mask[position] and row.activation_targets[position] >= 0.5) for row in rows)
        strong = sum(bool(row.activation_mask[position] and row.activation_targets[position] >= 0.75) for row in rows)
        disputed = len(rows) - supervised
        result[model_id] = {
            "supervised": supervised,
            "disputed_masked": disputed,
            "useful_or_higher_target": useful,
            "strong_or_higher_target": strong,
        }
    return result


def load_dense_score_data(dataset_root: Path) -> DenseRouteDataBundle:
    """Load all-model soft targets; sealed_test is intentionally unsupported in V1."""
    dataset_root = Path(dataset_root)
    graph = load_semantic_graph()
    score_dir = dataset_root / "labels" / "train_development" / "model_scores"
    manifest_path = score_dir / MODEL_SCORE_MANIFEST_FILENAME
    aggregate_path = score_dir / "aggregated_model_scores.jsonl"
    try:
        manifest, paths, verified = verify_model_score_manifest(
            manifest_path,
            dataset_root=dataset_root,
            expected_aggregate_path=aggregate_path,
        )
    except (ModelScoreChainError, ValidationError, OSError, KeyError, TypeError, ValueError) as error:
        raise RouterDataError(f"untrusted dense model-score chain: {error}") from error
    episodes = {row["episode_id"]: row for row in verified["episodes"]}
    targets = {row["episode_id"]: row for row in verified["aggregates"]}
    if len(targets) != len(verified["aggregates"]):
        raise RouterDataError("duplicate dense model-score target")
    if set(targets) - set(episodes):
        raise RouterDataError("dense model-score targets reference unknown Episodes")
    train_episodes = {episode_id: row for episode_id, row in episodes.items() if row.get("split") == "train" and episode_id in targets}
    development_episodes = {episode_id: row for episode_id, row in episodes.items() if row.get("split") == "development" and episode_id in targets}
    unexpected_splits = sorted({row.get("split") for row in episodes.values()} - {"train", "development"})
    if unexpected_splits:
        raise RouterDataError(f"dense train_development score chain contains unexpected splits: {unexpected_splits}")
    if not train_episodes or not development_episodes:
        raise RouterDataError("dense score training requires non-empty train and development targets")
    direction_vocab = _direction_vocab(train_episodes.values())
    encode = lambda rows, split: tuple(
        _encode_dense_example(row, targets[row["episode_id"]], split, graph, direction_vocab)
        for row in sorted(rows.values(), key=lambda item: item["episode_id"])
    )
    train = encode(train_episodes, "train")
    development = encode(development_episodes, "development")
    train_coverage = _coverage(train, graph.model_ids)
    development_coverage = _coverage(development, graph.model_ids)
    useful_counts = Counter({model_id: item["useful_or_higher_target"] for model_id, item in train_coverage.items()})
    # The network always emits all 20 dimensions. These counts are diagnostic,
    # not gates that remove long-tail models from the output layer.
    support = SupportSet(
        primary_models=graph.model_ids,
        auxiliary_models=graph.model_ids,
        primary_counts=dict(sorted(useful_counts.items())),
        auxiliary_counts=dict(sorted(useful_counts.items())),
        unsupported_primary_targets=0,
    )
    supervised = sum(int(row.activation_mask.sum()) for row in (*train, *development))
    total = (len(train) + len(development)) * len(graph.model_ids)
    audit = {
        "dataset_root": str(dataset_root),
        "teacher_target_source": "model-scores-v1",
        "label_tier": "Weak",
        "router_input": [
            "microconcept_states.status",
            "microconcept_states.intensity",
            "microconcept_states.direction",
            "microconcept_states.temporal_persistence",
            "microconcept_states.task_relevance",
        ],
        "target": "20-dimensional normalized thought-model activation vector",
        "aggregation": "per-model arithmetic mean when teacher score gap is within threshold; otherwise per-model loss mask",
        "fixed_models": len(graph.model_ids),
        "counts": {"train": len(train), "development": len(development)},
        "supervised_dimensions": supervised,
        "total_dimensions": total,
        "supervised_dimension_rate": supervised / total if total else 0.0,
        "train_coverage_by_model": train_coverage,
        "development_coverage_by_model": development_coverage,
        "manifest": manifest,
        "fingerprint_sha256": _fingerprint([manifest_path, *(paths[role] for role in sorted(paths))]),
        "sealed_test_read": False,
    }
    return DenseRouteDataBundle(graph, support, direction_vocab, train, development, audit)


class DenseRouterDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: Iterable[DenseRouterExample]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "status": torch.tensor(row.status, dtype=torch.long),
            "direction": torch.tensor(row.direction, dtype=torch.long),
            "numeric": torch.tensor(row.numeric, dtype=torch.float32),
            "activation_target": torch.tensor(row.activation_targets, dtype=torch.float32),
            "activation_mask": torch.tensor(row.activation_mask, dtype=torch.float32),
            "activation_weight": torch.tensor(row.activation_weights, dtype=torch.float32),
        }
