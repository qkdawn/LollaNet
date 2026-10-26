"""Data loading and input-contract enforcement for Microconcept Router V1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from research.phase1.mechanism_questions_v1.llm_teacher_v1 import INITIAL_PROMPT_VERSION, REVIEW_PROMPT_VERSION
from research.phase1.mechanism_questions_v1.question_set_v1 import ValidationError, load_semantic_assets
from research.phase1.mechanism_questions_v1.teacher_chain_v1 import (
    MANIFEST_FILENAME,
    TeacherChainError,
    validate_manifest_contents,
    verify_manifest_files,
)
from research.phase1.mechanism_questions_v1.adjudication_chain_v1 import (
    ADJUDICATION_MANIFEST_FILENAME,
    AdjudicationChainError,
    verify_adjudication_manifest,
)

IGNORE_INDEX = -100
STATUS_VOCAB = ("unknown", "supported", "candidate", "mixed", "contradicted")
STATUS_INDEX = {name: index for index, name in enumerate(STATUS_VOCAB)}
ROUTER_PROHIBITED_FIELDS = frozenset({
    "facts", "structured_evidence", "source_refs", "case_id", "domain", "question", "goal",
    "constraints", "primary_model", "auxiliary_models", "route_labels", "outcomes",
})


class RouterDataError(ValueError):
    """Raised when the V1 data contract is violated."""


@dataclass(frozen=True)
class SemanticGraph:
    concept_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    edge_relation: np.ndarray  # [models, concepts], -1 means illegal edge


@dataclass(frozen=True)
class RouterExample:
    episode_id: str
    split: str
    status: np.ndarray
    direction: np.ndarray
    numeric: np.ndarray
    abstain: int
    primary_target: int
    auxiliary_targets: np.ndarray


@dataclass(frozen=True)
class SupportSet:
    primary_models: tuple[str, ...]
    auxiliary_models: tuple[str, ...]
    primary_counts: dict[str, int]
    auxiliary_counts: dict[str, int]
    unsupported_primary_targets: int

    @property
    def route_models(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.primary_models) | set(self.auxiliary_models)))


@dataclass(frozen=True)
class RouteDataBundle:
    graph: SemanticGraph
    support: SupportSet
    direction_vocab: tuple[str, ...]
    train: tuple[RouterExample, ...]
    development: tuple[RouterExample, ...]
    sealed_test: tuple[RouterExample, ...]
    audit: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RouterDataError(f"missing required data file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RouterDataError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def _fingerprint(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_semantic_graph() -> SemanticGraph:
    """Read only the fixed public relation index; no route data can add edges."""
    assets = load_semantic_assets()
    root = Path(__file__).resolve().parents[3]
    index_path = root / "research" / "phase1" / "microconcepts" / "microconcept_index.yaml"
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    concept_ids = tuple(item["id"] for item in index["concepts"])
    model_ids = tuple(sorted(assets["models"]))
    relation_ids = tuple(sorted({edge["relation"] for edge in index["edges"] if edge["target"] in set(model_ids)}))
    relation_index = {relation: position for position, relation in enumerate(relation_ids)}
    edge_relation = np.full((len(model_ids), len(concept_ids)), -1, dtype=np.int64)
    concept_index = {concept: position for position, concept in enumerate(concept_ids)}
    model_index = {model: position for position, model in enumerate(model_ids)}
    for edge in index["edges"]:
        source, target, relation = edge["source"], edge["target"], edge["relation"]
        if source in concept_index and target in model_index:
            edge_relation[model_index[target], concept_index[source]] = relation_index[relation]
    if (edge_relation >= 0).sum() == 0:
        raise RouterDataError("fixed relation index contains no microconcept-to-model edges")
    if any(not np.any(edge_relation[position] >= 0) for position in range(len(model_ids))):
        missing = [model for position, model in enumerate(model_ids) if not np.any(edge_relation[position] >= 0)]
        raise RouterDataError(f"models without a legal microconcept edge: {missing}")
    return SemanticGraph(concept_ids, model_ids, relation_ids, edge_relation)


def _rows_by_split(rows: Iterable[dict[str, Any]], *, cohort: str) -> dict[str, dict[str, dict[str, Any]]]:
    allowed = {"train", "development"} if cohort == "train_development" else {"sealed_test"}
    result = {split: {} for split in allowed}
    for row in rows:
        episode_id = row.get("episode_id")
        split = row.get("split")
        if not isinstance(episode_id, str) or not episode_id:
            raise RouterDataError(f"{cohort} episode is missing episode_id")
        if split not in allowed:
            raise RouterDataError(f"{cohort} manifest contains unexpected split {split!r}")
        if episode_id in result[split]:
            raise RouterDataError(f"duplicate episode_id {episode_id}")
        result[split][episode_id] = row
    return result


def _load_verified_cohort(dataset_root: Path, cohort: str, *, teacher_target_source: str = "review-v1") -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]], list[Path], dict[str, Any]]:
    if teacher_target_source == "adjudication-v1":
        if cohort != "train_development":
            raise RouterDataError("adjudication-v1 is unavailable for sealed_test until a sealed child chain is explicitly created")
        label_dir = dataset_root / "labels" / cohort
        child_manifest_path = label_dir / "adjudication" / ADJUDICATION_MANIFEST_FILENAME
        accepted_path = label_dir / "adjudication" / "accepted_adjudication.jsonl"
        try:
            child_manifest, child_paths, child_verified = verify_adjudication_manifest(
                child_manifest_path, dataset_root=dataset_root, expected_accepted_path=accepted_path,
            )
        except (AdjudicationChainError, TeacherChainError, ValidationError, OSError, KeyError, TypeError, ValueError) as error:
            raise RouterDataError(f"untrusted {cohort} adjudication child chain: {error}") from error
        episodes = _rows_by_split(child_verified["episodes"], cohort=cohort)
        labels: dict[str, dict[str, Any]] = {}
        for label in child_verified["accepted"]:
            if label.get("label_pass") != "adjudication" or label.get("adjudication_status") != "accepted":
                raise RouterDataError("finalized adjudication labels must be accepted adjudication rows")
            if label.get("label_tier") != "Weak":
                raise RouterDataError("adjudication-v1 expects only explicitly declared Weak labels")
            episode_id = label.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id or episode_id in labels:
                raise RouterDataError("accepted adjudication labels must have unique episode_id values")
            labels[episode_id] = label
        parent_path = (child_manifest_path.parent / child_manifest["parent_chain"]["path"]).resolve()
        parent_manifest = child_verified["parent_manifest"]
        verified_paths = [child_manifest_path, parent_path, *(child_verified["parent_verified"]["paths"][role] for role in sorted(child_verified["parent_verified"]["paths"])), *(child_paths[role] for role in sorted(child_paths))]
        deduped_paths = list(dict.fromkeys(verified_paths))
        return episodes, labels, deduped_paths, {"source": "adjudication-v1", "child_manifest": child_manifest, "parent_manifest": parent_manifest}
    if teacher_target_source != "review-v1":
        raise RouterDataError(f"unsupported teacher_target_source: {teacher_target_source}")
    label_dir = dataset_root / "labels" / cohort
    accepted_path = label_dir / "accepted_review.jsonl"
    manifest_path = label_dir / MANIFEST_FILENAME
    try:
        manifest, paths = verify_manifest_files(
            manifest_path, dataset_root=dataset_root, expected_accepted_path=accepted_path,
        )
        verified = validate_manifest_contents(manifest, paths)
        expected_prompts = {"initial": [INITIAL_PROMPT_VERSION], "review": [REVIEW_PROMPT_VERSION]}
        if manifest.get("prompt_versions") != expected_prompts:
            raise TeacherChainError("teacher-chain manifest does not use the current initial/review prompt versions")
    except (TeacherChainError, ValidationError, OSError, KeyError, TypeError, ValueError) as error:
        raise RouterDataError(f"untrusted {cohort} teacher-label chain: {error}") from error
    episodes = _rows_by_split(verified["episodes"], cohort=cohort)
    labels: dict[str, dict[str, Any]] = {}
    for label in verified["accepted"]:
        if label.get("label_pass") != "review" or label.get("review_status") != "accepted":
            raise RouterDataError("finalized accepted labels must be accepted review rows")
        if label.get("label_tier") != "Weak":
            raise RouterDataError("V1 expects only explicitly declared Weak accepted-review labels")
        episode_id = label.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RouterDataError("accepted review label missing episode_id")
        if episode_id in labels:
            raise RouterDataError(f"duplicate accepted review label for {episode_id}")
        labels[episode_id] = label
    verified_paths = [manifest_path, *(paths[role] for role in sorted(paths))]
    return episodes, labels, verified_paths, manifest

def _direction_vocab(train_episodes: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    values = {"<UNK>"}
    for episode in train_episodes:
        for state in episode.get("microconcept_states", []):
            direction = state.get("direction")
            if not isinstance(direction, str) or not direction:
                raise RouterDataError(f"episode {episode.get('episode_id')} has no direction for a microconcept")
            values.add(direction)
    return tuple(sorted(values))


def _validate_states(episode: dict[str, Any], graph: SemanticGraph) -> dict[str, dict[str, Any]]:
    states = episode.get("microconcept_states")
    if not isinstance(states, list):
        raise RouterDataError(f"episode {episode.get('episode_id')} has no microconcept_states list")
    by_id = {state.get("microconcept_id"): state for state in states if isinstance(state, dict)}
    if set(by_id) != set(graph.concept_ids) or len(by_id) != len(graph.concept_ids):
        raise RouterDataError(f"episode {episode.get('episode_id')} must contain every one of the fixed microconcepts exactly once")
    for concept_id, state in by_id.items():
        if state.get("status") not in STATUS_INDEX:
            raise RouterDataError(f"episode {episode.get('episode_id')} has invalid status for {concept_id}: {state.get('status')!r}")
        for key in ("intensity", "temporal_persistence", "task_relevance"):
            value = state.get(key)
            if not isinstance(value, int) or not 0 <= value <= 3:
                raise RouterDataError(f"episode {episode.get('episode_id')} has invalid {key} for {concept_id}")
    return by_id


def build_support_set(labels: Iterable[dict[str, Any]], train_episode_ids: set[str], graph: SemanticGraph, minimum_primary_support: int, minimum_auxiliary_support: int) -> SupportSet:
    primary = Counter()
    auxiliary = Counter()
    for label in labels:
        if label["episode_id"] not in train_episode_ids or label.get("abstain"):
            continue
        selected = label.get("primary_model")
        if selected not in graph.model_ids:
            raise RouterDataError(f"unknown primary model {selected!r}")
        primary[selected] += 1
        for model_id in label.get("auxiliary_models", []):
            if model_id not in graph.model_ids:
                raise RouterDataError(f"unknown auxiliary model {model_id!r}")
            auxiliary[model_id] += 1
    primary_models = tuple(sorted(model for model, count in primary.items() if count >= minimum_primary_support))
    auxiliary_models = tuple(sorted(model for model, count in auxiliary.items() if count >= minimum_auxiliary_support))
    if not primary_models:
        raise RouterDataError("no primary model meets the train-only support threshold")
    if not auxiliary_models:
        raise RouterDataError("no auxiliary model meets the train-only support threshold")
    unsupported = sum(
        1 for label in labels
        if label["episode_id"] in train_episode_ids and not label.get("abstain") and label.get("primary_model") not in primary_models
    )
    return SupportSet(primary_models, auxiliary_models, dict(sorted(primary.items())), dict(sorted(auxiliary.items())), unsupported)


def _router_input_signature(episode: dict[str, Any], graph: SemanticGraph) -> tuple[tuple[Any, ...], ...]:
    """Return the complete Router-visible state signature in fixed node order."""
    state_by_id = _validate_states(episode, graph)
    return tuple(
        (
            concept,
            state_by_id[concept]["status"],
            state_by_id[concept]["intensity"],
            state_by_id[concept]["direction"],
            state_by_id[concept]["temporal_persistence"],
            state_by_id[concept]["task_relevance"],
        )
        for concept in graph.concept_ids
    )


def _route_target_signature(label: dict[str, Any]) -> tuple[bool, str | None, tuple[str, ...]]:
    return (
        bool(label.get("abstain")),
        None if label.get("abstain") else label.get("primary_model"),
        () if label.get("abstain") else tuple(sorted(label.get("auxiliary_models", []))),
    )


def _validate_input_label_consistency(episodes: Iterable[dict[str, Any]], labels: dict[str, dict[str, Any]], graph: SemanticGraph, *, cohort: str) -> None:
    """Reject unlearnable cases where identical allowed inputs imply different targets."""
    routes_by_input: dict[tuple[tuple[Any, ...], ...], set[tuple[bool, str | None, tuple[str, ...]]]] = {}
    examples_by_input: dict[tuple[tuple[Any, ...], ...], list[str]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        signature = _router_input_signature(episode, graph)
        routes_by_input.setdefault(signature, set()).add(_route_target_signature(labels[episode_id]))
        examples_by_input.setdefault(signature, []).append(episode_id)
    conflicts = [(signature, routes_by_input[signature], examples_by_input[signature]) for signature in routes_by_input if len(routes_by_input[signature]) > 1]
    if conflicts:
        _, routes, episode_ids = conflicts[0]
        raise RouterDataError(
            f"{cohort} has {len(conflicts)} identical Router-input signature(s) with conflicting route labels; "
            "the microconcept states must be question-conditioned before Router training. "
            f"Example episodes: {episode_ids[:4]}; routes: {sorted(routes)}"
        )


def _encode_example(episode: dict[str, Any], label: dict[str, Any], split: str, graph: SemanticGraph, support: SupportSet, direction_vocab: tuple[str, ...]) -> RouterExample:
    state_by_id = _validate_states(episode, graph)
    direction_index = {name: position for position, name in enumerate(direction_vocab)}
    status = np.array([STATUS_INDEX[state_by_id[concept]["status"]] for concept in graph.concept_ids], dtype=np.int64)
    direction = np.array([direction_index.get(state_by_id[concept]["direction"], direction_index["<UNK>"]) for concept in graph.concept_ids], dtype=np.int64)
    numeric = np.array([
        [state_by_id[concept]["intensity"] / 3.0, state_by_id[concept]["temporal_persistence"] / 3.0, state_by_id[concept]["task_relevance"] / 3.0]
        for concept in graph.concept_ids
    ], dtype=np.float32)
    abstain = int(bool(label.get("abstain")))
    primary_index = {model: position for position, model in enumerate(support.primary_models)}
    primary_target = IGNORE_INDEX if abstain else primary_index.get(label.get("primary_model"), IGNORE_INDEX)
    auxiliary_targets = np.zeros(len(support.auxiliary_models), dtype=np.float32)
    if not abstain:
        auxiliary_index = {model: position for position, model in enumerate(support.auxiliary_models)}
        for model_id in label.get("auxiliary_models", []):
            if model_id in auxiliary_index:
                auxiliary_targets[auxiliary_index[model_id]] = 1.0
    return RouterExample(str(episode["episode_id"]), split, status, direction, numeric, abstain, primary_target, auxiliary_targets)


def load_route_data(dataset_root: Path, *, include_sealed: bool = False, minimum_primary_support: int = 20, minimum_auxiliary_support: int = 20, enforce_input_label_consistency: bool = True, teacher_target_source: str = "review-v1") -> RouteDataBundle:
    """Load an explicitly selected, manifest-bound Weak route target source."""
    dataset_root = Path(dataset_root)
    graph = load_semantic_graph()
    train_development, labels, verified_paths, train_development_manifest = _load_verified_cohort(dataset_root, "train_development", teacher_target_source=teacher_target_source)
    train = train_development["train"]
    development = train_development["development"]
    sealed: dict[str, dict[str, Any]] = {}
    sealed_manifest: dict[str, Any] | None = None
    if include_sealed:
        sealed_cohort, sealed_labels, sealed_paths, sealed_manifest = _load_verified_cohort(dataset_root, "sealed_test")
        sealed = sealed_cohort["sealed_test"]
        overlap = set(labels) & set(sealed_labels)
        if overlap:
            raise RouterDataError(f"duplicate accepted labels across cohorts: {sorted(overlap)[:3]}")
        labels.update(sealed_labels)
        verified_paths.extend(sealed_paths)
    source_splits = {"train": train, "development": development, "sealed_test": sealed}
    expected_ids = set(train) | set(development) | set(sealed)
    unexpected = set(labels) - expected_ids
    if unexpected:
        raise RouterDataError(f"labels reference episodes outside requested split view: {len(unexpected)}")

    # The finalized chain must cover every Episode, but only independently accepted
    # review rows are supervision.  review_required rows remain bound by the
    # manifest and are intentionally excluded rather than converted into targets.
    accepted_splits = {
        split: {episode_id: row for episode_id, row in rows.items() if episode_id in labels}
        for split, rows in source_splits.items()
    }
    train = accepted_splits["train"]
    development = accepted_splits["development"]
    sealed = accepted_splits["sealed_test"]
    if not train:
        raise RouterDataError("no accepted review labels are available for training")
    if not development:
        raise RouterDataError("no accepted review labels are available for development evaluation")

    if enforce_input_label_consistency:
        _validate_input_label_consistency(train.values(), labels, graph, cohort="train")
        _validate_input_label_consistency(development.values(), labels, graph, cohort="development")
        if sealed:
            _validate_input_label_consistency(sealed.values(), labels, graph, cohort="sealed_test")
    direction_vocab = _direction_vocab(train.values())
    support = build_support_set(labels.values(), set(train), graph, minimum_primary_support, minimum_auxiliary_support)
    encode = lambda rows, split: tuple(_encode_example(row, labels[row["episode_id"]], split, graph, support, direction_vocab) for row in rows.values())
    paths = verified_paths
    source_counts = {split: len(rows) for split, rows in source_splits.items()}
    accepted_counts = {split: len(rows) for split, rows in accepted_splits.items()}
    review_required_counts = {split: source_counts[split] - accepted_counts[split] for split in source_counts}
    audit = {
        "dataset_root": str(dataset_root),
        "requested_splits": ["train", "development"] + (["sealed_test"] if include_sealed else []),
        "label_tier": "Weak",
        "teacher_target_source": teacher_target_source,
        "router_input": ["microconcept_states.status", "microconcept_states.intensity", "microconcept_states.direction", "microconcept_states.temporal_persistence", "microconcept_states.task_relevance"],
        "router_prohibited_fields": sorted(ROUTER_PROHIBITED_FIELDS),
        "fingerprint_sha256": _fingerprint(paths),
        "teacher_chain_manifests": {
            "train_development": train_development_manifest,
            **({"sealed_test": sealed_manifest} if sealed_manifest is not None else {}),
        },
        "counts": accepted_counts,
        "teacher_chain_episode_counts": source_counts,
        "accepted_supervision_counts": accepted_counts,
        "excluded_supervision_counts": review_required_counts,
        "accepted_supervision_total": sum(accepted_counts.values()),
        "excluded_supervision_total": sum(review_required_counts.values()),
        # Backward-compatible names for review-v1 consumers. For adjudication-v1,
        # the excluded rows are exact-route disagreements rather than unreviewed rows.
        "review_required_excluded_counts": review_required_counts,
        "review_required_excluded_total": sum(review_required_counts.values()),
        "fixed_nodes": len(graph.concept_ids),
        "fixed_models": len(graph.model_ids),
        "direction_vocabulary_size_train_only": len(direction_vocab),
        "input_label_consistency_enforced": enforce_input_label_consistency,
        "primary_supported_models": list(support.primary_models),
        "auxiliary_supported_models": list(support.auxiliary_models),
        "primary_train_counts": support.primary_counts,
        "auxiliary_train_counts": support.auxiliary_counts,
        "primary_targets_masked_for_insufficient_train_support": support.unsupported_primary_targets,
    }
    if teacher_target_source == "adjudication-v1":
        child_manifest = train_development_manifest["child_manifest"]
        audit["adjudication_summary"] = {
            "parent_accepted": len(child_manifest["parent_accepted_episode_ids"]),
            "eligible_for_dual_adjudication": len(child_manifest["eligible_episode_ids"]),
            "exact_route_consensus": len(child_manifest["consensus_episode_ids"]),
            "route_disagreement_excluded": len(child_manifest["disagreement_episode_ids"]),
            "final_accepted": len(child_manifest["accepted_episode_ids"]),
            "parent_identity_overlap": child_manifest["parent_identity_overlap"],
        }
        audit["exclusion_reason"] = "dual_adjudicator_route_disagreement"
    else:
        audit["exclusion_reason"] = "parent_review_not_accepted"
    return RouteDataBundle(graph, support, direction_vocab, encode(train, "train"), encode(development, "development"), encode(sealed, "sealed_test"), audit)


class RouterDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: Iterable[RouterExample]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "status": torch.tensor(row.status, dtype=torch.long),
            "direction": torch.tensor(row.direction, dtype=torch.long),
            "numeric": torch.tensor(row.numeric, dtype=torch.float32),
            "abstain": torch.tensor(row.abstain, dtype=torch.float32),
            "primary": torch.tensor(row.primary_target, dtype=torch.long),
            "auxiliary": torch.tensor(row.auxiliary_targets, dtype=torch.float32),
        }


def serializable_predictions(rows: Iterable[RouterExample], output: dict[str, np.ndarray], support: SupportSet, *, auxiliary_threshold: float, abstain_threshold: float) -> list[dict[str, Any]]:
    """Turn numeric router outputs into constrained, auditable selections."""
    predictions: list[dict[str, Any]] = []
    primary_probs = output["primary"]
    auxiliary_probs = output["auxiliary"]
    abstain_probs = output["abstain"]
    for index, row in enumerate(rows):
        abstain_probability = float(abstain_probs[index])
        primary_model: str | None = None
        auxiliary: list[str] = []
        if abstain_probability < abstain_threshold:
            primary_model = support.primary_models[int(primary_probs[index].argmax())]
            ranked_auxiliary = sorted(zip(support.auxiliary_models, auxiliary_probs[index]), key=lambda pair: (-float(pair[1]), pair[0]))
            auxiliary = [model for model, probability in ranked_auxiliary if float(probability) >= auxiliary_threshold and model != primary_model][:2]
        predictions.append({
            "episode_id": row.episode_id,
            "split": row.split,
            "abstain_probability": abstain_probability,
            "abstain": abstain_probability >= abstain_threshold,
            "primary_model": primary_model,
            "auxiliary_models": auxiliary,
            "primary_activation": {model: float(primary_probs[index, position]) for position, model in enumerate(support.primary_models)},
            "auxiliary_activation": {model: float(auxiliary_probs[index, position]) for position, model in enumerate(support.auxiliary_models)},
            "unsupported_primary_models": [],
            "unsupported_auxiliary_models": [],
        })
    return predictions
