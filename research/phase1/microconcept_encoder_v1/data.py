"""Data contracts, leakage guards, deterministic splits, and vectorisation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
MICRO_ROOT = REPO_ROOT / "research" / "phase1" / "microconcepts"
FEATURE_DICTIONARY = REPO_ROOT / "research" / "phase1" / "value_investing" / "feature_dictionary.yaml"
CLAIM_TYPES = MICRO_ROOT / "qualitative_claim_types.yaml"
MICRO_INDEX = MICRO_ROOT / "microconcept_index.yaml"
CARDS = MICRO_ROOT / "cards"

FORBIDDEN_INPUT_KEYS = frozenset({
    "route_labels", "exploratory_outcome", "issuer_name", "ticker_at_collection",
    "raw_mda_text", "text", "outcome", "thought_model_routing", "llm_steering",
})
MONEY_FIELDS = frozenset({"ttm_operating_cash_flow", "ttm_capex", "ttm_free_cash_flow", "market_cap"})
STATUS_VALUES = ("supported", "contradicted")
DIRECTION_VALUES = ("increasing", "decreasing", "stable", "mixed")
APPLICABILITY_VALUES = ("applicable", "not_applicable")


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _load_feature_names() -> tuple[str, ...]:
    data = yaml.safe_load(FEATURE_DICTIONARY.read_text(encoding="utf-8"))
    names: list[str] = []
    for group in ("financial_features", "market_state"):
        for item in data.get(group, []):
            name = item.get("name")
            if name and name != "sector_code":
                names.append(str(name))
    if len(set(names)) != len(names):
        raise ValueError("feature dictionary contains duplicate numeric field")
    return tuple(names)


def _load_claim_vocab() -> dict[str, int]:
    data = yaml.safe_load(CLAIM_TYPES.read_text(encoding="utf-8"))
    ids = [str(item["id"]) for item in data.get("claims", [])]
    return {value: index + 1 for index, value in enumerate(ids)}  # zero is padding/unknown


def _load_concepts() -> tuple[str, ...]:
    ids: list[str] = []
    for card_path in sorted(CARDS.glob("*.md")):
        text = card_path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"invalid card frontmatter: {card_path}")
        _, payload, _ = text.split("---\n", 2)
        card = yaml.safe_load(payload)
        ids.append(str(card["id"]))
    if len(ids) != 11 or len(set(ids)) != 11:
        raise ValueError(f"expected fixed 11 concepts, got {ids}")
    return tuple(sorted(ids))


FEATURE_NAMES = _load_feature_names()
CLAIM_VOCAB = _load_claim_vocab()
CONCEPT_IDS = _load_concepts()
CONCEPT_INDEX = {name: index for index, name in enumerate(CONCEPT_IDS)}


def _load_static_candidate_relations() -> tuple[dict[str, Any], ...]:
    """Read the fixed semantic registry for audit only; it is never learned or routed."""
    index = yaml.safe_load(MICRO_INDEX.read_text(encoding="utf-8"))
    relations: list[dict[str, Any]] = []
    for item in index.get("by_model", []):
        model_id = str(item.get("model_id"))
        for target in item.get("microconcepts", []):
            concept_id = str(target.get("microconcept_id"))
            if concept_id in CONCEPT_INDEX:
                relations.append({"model_id": model_id, "microconcept_id": concept_id, "relation": str(target.get("relation"))})
    return tuple(sorted(relations, key=lambda row: (row["microconcept_id"], row["model_id"], row["relation"])))


STATIC_CANDIDATE_RELATIONS = _load_static_candidate_relations()


@dataclass(frozen=True)
class EpisodeInput:
    episode_id: str
    accession: str
    available_at: str
    year: int
    form: str
    cik_for_split: str
    factors: dict[str, float | None]


@dataclass(frozen=True)
class ClaimInput:
    claim_key: str
    claim_id: str
    accession: str
    available_at: str
    polarity: str
    time_horizon: str
    strength: str
    confidence: float


@dataclass(frozen=True)
class Example:
    example_id: str
    episode_id: str
    accession: str
    available_at: str
    year: int
    cik_for_split: str
    concept_id: str
    task_label_id: str
    question: str
    goal: str
    task_variant: str
    factors: dict[str, float | None]
    claims: tuple[ClaimInput, ...]
    status: str | None
    intensity: int | None
    direction: str | None
    applicability: str | None
    relevance: int | None
    evidence_label_tier: str
    task_label_tier: str
    evidence_label_source: str
    task_label_source: str
    evidence_review_status: str
    task_review_status: str
    evidence_dispute_status: str
    task_dispute_status: str

    @property
    def effective_tier(self) -> str:
        # The weaker tier governs a multi-head row.  The order is intentional:
        # Gold requires both state and task targets to have qualifying review.
        ranks = {"Weak": 0, "Silver": 1, "Gold": 2}
        return min((self.evidence_label_tier, self.task_label_tier), key=lambda item: ranks.get(item, -1))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _read_episodes(candidates_path: Path) -> dict[str, EpisodeInput]:
    episodes: dict[str, EpisodeInput] = {}
    for line_number, raw in enumerate(_read_jsonl(candidates_path), 1):
        # Deliberately project only the explicitly allowlisted source fields.  The
        # raw row can contain forbidden fields, but none is inspected here.
        episode_id = raw.get("episode_id")
        filing = raw.get("filing")
        availability = raw.get("availability")
        financial = raw.get("financial_features")
        market = raw.get("market_state")
        cik = raw.get("cik")
        if not (isinstance(episode_id, str) and isinstance(filing, dict) and isinstance(availability, dict)
                and isinstance(financial, dict) and isinstance(market, dict) and isinstance(cik, str)):
            raise ValueError(f"candidate row {line_number}: missing allowlisted fields")
        accession = filing.get("accession")
        available_at = availability.get("public_available_at")
        form = filing.get("form")
        if not all(isinstance(v, str) and v for v in (accession, available_at, form)):
            raise ValueError(f"candidate row {line_number}: invalid filing metadata")
        try:
            year = int(available_at[:4])
        except ValueError as exc:
            raise ValueError(f"candidate row {line_number}: invalid available_at") from exc
        merged = {**financial, **market}
        factors = {name: _as_float(merged.get(name)) for name in FEATURE_NAMES}
        if episode_id in episodes:
            raise ValueError(f"duplicate episode_id {episode_id}")
        episodes[episode_id] = EpisodeInput(episode_id, accession, available_at, year, form, cik, factors)
    if not episodes:
        raise ValueError("empty candidate file")
    return episodes


def _read_claims(claims_path: Path) -> dict[str, tuple[ClaimInput, ...]]:
    required = {"claim_key", "claim_id", "accession", "available_at", "polarity", "time_horizon", "strength", "confidence"}
    by_accession: dict[str, list[ClaimInput]] = defaultdict(list)
    for line_number, raw in enumerate(_read_jsonl(claims_path), 1):
        missing = required - set(raw)
        if missing:
            raise ValueError(f"claims row {line_number}: missing {sorted(missing)}")
        claim_id = raw["claim_id"]
        if claim_id not in CLAIM_VOCAB:
            raise ValueError(f"claims row {line_number}: uncontrolled claim_id {claim_id!r}")
        values = [raw[key] for key in ("claim_key", "claim_id", "accession", "available_at", "polarity", "time_horizon", "strength")]
        if not all(isinstance(v, str) and v for v in values):
            raise ValueError(f"claims row {line_number}: invalid string field")
        confidence = _as_float(raw["confidence"])
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError(f"claims row {line_number}: invalid confidence")
        by_accession[raw["accession"]].append(ClaimInput(
            raw["claim_key"], claim_id, raw["accession"], raw["available_at"], raw["polarity"],
            raw["time_horizon"], raw["strength"], confidence,
        ))
    return {key: tuple(sorted(value, key=lambda c: (c.available_at, c.claim_key))) for key, value in by_accession.items()}


def _genuine_gold(tier: str, annotator: Any, review: Any, dispute: Any) -> bool:
    """Machine-only `Gold` text is not accepted as Gold training evidence."""
    if tier != "Gold" or not isinstance(annotator, str) or annotator.startswith("system:"):
        return False
    return str(review) in {"human_confirmed", "adjudicated", "approved"} and str(dispute) in {"none", "resolved", "adjudicated"}


def _tier_or_downgrade(raw_tier: Any, annotator: Any, review: Any, dispute: Any) -> str:
    tier = str(raw_tier)
    if tier == "Gold" and not _genuine_gold(tier, annotator, review, dispute):
        return "Silver"  # conservative; exact source remains in the manifest
    if tier not in {"Weak", "Silver", "Gold"}:
        raise ValueError(f"unknown label tier {tier!r}")
    return tier


@dataclass(frozen=True)
class EncoderData:
    examples: tuple[Example, ...]
    episodes: dict[str, EpisodeInput]
    source_fingerprints: dict[str, dict[str, str]]
    source_audit: dict[str, Any]


def build_encoder_data(candidates_path: Path, claims_path: Path, labels_dir: Path) -> EncoderData:
    """Build joined task rows while retaining CIK exclusively for split grouping."""
    states_path = labels_dir / "episode_microconcept_labels.jsonl"
    tasks_path = labels_dir / "task_relevance_labels.jsonl"
    for path in (candidates_path, claims_path, states_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes = _read_episodes(candidates_path)
    claims_by_accession = _read_claims(claims_path)
    states: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(states_path):
        label_id = row.get("label_id")
        episode_id = row.get("episode_id")
        concept_id = row.get("microconcept_id")
        if not (isinstance(label_id, str) and isinstance(episode_id, str) and concept_id in CONCEPT_INDEX):
            raise ValueError("invalid evidence label identity")
        if episode_id not in episodes:
            raise ValueError(f"evidence label references absent episode {episode_id}")
        if label_id in states:
            raise ValueError(f"duplicate state label {label_id}")
        states[label_id] = row

    examples: list[Example] = []
    for task in _read_jsonl(tasks_path):
        state_id = task.get("state_label_id")
        state = states.get(state_id)
        if state is None:
            raise ValueError(f"task references unknown state label {state_id!r}")
        episode = episodes[state["episode_id"]]
        if task.get("episode_id") != episode.episode_id or task.get("microconcept_id") != state.get("microconcept_id"):
            raise ValueError("task/state Episode or concept mismatch")
        if state.get("accession") != episode.accession:
            raise ValueError("state accession mismatch")
        if str(state.get("available_at")) != episode.available_at:
            raise ValueError("state availability mismatch")
        joined_claims = tuple(item for item in claims_by_accession.get(episode.accession, ()) if item.available_at <= episode.available_at)
        # Check would catch a malformed future claim before it can reach a model.
        if any(item.available_at > episode.available_at for item in joined_claims):
            raise AssertionError("future claim escaped availability guard")
        status = state.get("status") if state.get("status") in {"supported", "contradicted"} else None
        intensity = state.get("intensity") if status is not None and state.get("intensity") in {0, 1, 2, 3} else None
        direction = state.get("direction") if state.get("direction") in DIRECTION_VALUES else None
        applicability = task.get("applicability") if task.get("applicability") in APPLICABILITY_VALUES else None
        relevance = task.get("relevance") if applicability is not None and task.get("relevance") in {0, 1, 2, 3} else None
        for text_name, text in (("question", task.get("question")), ("goal", task.get("goal"))):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"invalid {text_name}")
        examples.append(Example(
            example_id=str(task["task_label_id"]), episode_id=episode.episode_id, accession=episode.accession,
            available_at=episode.available_at, year=episode.year, cik_for_split=episode.cik_for_split,
            concept_id=str(state["microconcept_id"]), task_label_id=str(task["task_label_id"]),
            question=str(task["question"]), goal=str(task["goal"]), task_variant=str(task.get("task_variant", "unspecified")),
            factors=episode.factors, claims=joined_claims, status=status, intensity=intensity, direction=direction,
            applicability=applicability, relevance=relevance,
            evidence_label_tier=_tier_or_downgrade(state.get("label_tier"), state.get("annotator_id"), state.get("review_status"), state.get("dispute_status")),
            task_label_tier=_tier_or_downgrade(task.get("label_tier"), task.get("annotator_id"), task.get("review_status"), task.get("dispute_status")),
            evidence_label_source=str(state.get("label_source")), task_label_source=str(task.get("label_source")),
            evidence_review_status=str(state.get("review_status")), task_review_status=str(task.get("review_status")),
            evidence_dispute_status=str(state.get("dispute_status")), task_dispute_status=str(task.get("dispute_status")),
        ))
    if not examples:
        raise ValueError("no joined task examples")
    fingerprints = {
        "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
        "claims": {"path": str(claims_path), "sha256": sha256_file(claims_path)},
        "state_labels": {"path": str(states_path), "sha256": sha256_file(states_path)},
        "task_labels": {"path": str(tasks_path), "sha256": sha256_file(tasks_path)},
        "feature_dictionary": {"path": str(FEATURE_DICTIONARY), "sha256": sha256_file(FEATURE_DICTIONARY)},
        "claim_types": {"path": str(CLAIM_TYPES), "sha256": sha256_file(CLAIM_TYPES)},
        "microconcept_index": {"path": str(MICRO_INDEX), "sha256": sha256_file(MICRO_INDEX)},
    }
    return EncoderData(tuple(sorted(examples, key=lambda e: e.example_id)), episodes, fingerprints, {
        "forbidden_model_inputs": sorted(FORBIDDEN_INPUT_KEYS),
        "feature_names": list(FEATURE_NAMES), "claim_vocab": sorted(CLAIM_VOCAB), "concept_ids": list(CONCEPT_IDS),
        "static_candidate_relations": list(STATIC_CANDIDATE_RELATIONS),
        "static_candidate_relation_policy": "Frozen registry metadata only; no graph message passing, dynamic edge learning, route labels, or model routing input.",
        "total_examples": len(examples), "unique_episodes": len({e.episode_id for e in examples}),
        "gold_examples_after_human_guard": sum(e.effective_tier == "Gold" for e in examples),
        "source_tiers": dict(Counter(e.effective_tier for e in examples)),
    })


@dataclass(frozen=True)
class SplitBundle:
    train: tuple[Example, ...]
    development: tuple[Example, ...]
    issuer_heldout: tuple[Example, ...]
    sealed_temporal: tuple[Example, ...]
    ignored: tuple[Example, ...]
    heldout_ciks_hash: str
    audit: dict[str, Any]


def make_splits(data: EncoderData, issuer_fraction: float = 0.2, split_seed: str = "microconcept-encoder-v1-issuer-20260719") -> SplitBundle:
    """Freeze a temporal split and a CIK-group holdout without feature exposure."""
    if not 0 < issuer_fraction < 0.5:
        raise ValueError("issuer_fraction must be within (0, .5)")
    eligible_ciks = sorted({e.cik_for_split for e in data.examples if 2017 <= e.year <= 2021})
    ranked = sorted(eligible_ciks, key=lambda cik: (stable_hash({"seed": split_seed, "cik": cik}), cik))
    held_count = max(1, round(len(ranked) * issuer_fraction)) if ranked else 0
    heldout = frozenset(ranked[:held_count])
    train: list[Example] = []; development: list[Example] = []; issuer: list[Example] = []; sealed: list[Example] = []; ignored: list[Example] = []
    for row in data.examples:
        if row.year in {2022, 2023}:
            sealed.append(row)
        elif row.year in {2017, 2018, 2019, 2020, 2021} and row.cik_for_split in heldout:
            issuer.append(row)
        elif row.year in {2017, 2018, 2019, 2020}:
            train.append(row)
        elif row.year == 2021:
            development.append(row)
        else:
            ignored.append(row)
    train_ciks = {e.cik_for_split for e in train}
    issuer_ciks = {e.cik_for_split for e in issuer}
    if train_ciks & issuer_ciks:
        raise AssertionError("issuer leakage into train")
    audit = {
        "split_seed": split_seed, "issuer_holdout_fraction": issuer_fraction,
        "counts": {"train": len(train), "development": len(development), "issuer_heldout": len(issuer), "sealed_temporal": len(sealed), "ignored": len(ignored)},
        "episodes": {name: len({e.episode_id for e in rows}) for name, rows in (("train",train),("development",development),("issuer_heldout",issuer),("sealed_temporal",sealed),("ignored",ignored))},
        "years": {name: dict(sorted(Counter(e.year for e in rows).items())) for name, rows in (("train",train),("development",development),("issuer_heldout",issuer),("sealed_temporal",sealed),("ignored",ignored))},
        "issuer_counts": {"train": len(train_ciks), "development": len({e.cik_for_split for e in development}), "issuer_heldout": len(issuer_ciks)},
        "issuer_overlap_train_heldout": len(train_ciks & issuer_ciks),
        "heldout_cik_count": len(heldout),
        "labeled_sealed_targets": sum(e.status is not None or e.applicability is not None for e in sealed),
    }
    return SplitBundle(tuple(train), tuple(development), tuple(issuer), tuple(sealed), tuple(ignored), stable_hash(sorted(heldout)), audit)


@dataclass
class TrainOnlyNormalizer:
    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: Iterable[Example]) -> "TrainOnlyNormalizer":
        unique: dict[str, Example] = {}
        for row in rows:
            unique.setdefault(row.episode_id, row)
        values = np.array([[np.nan if r.factors.get(name) is None else float(r.factors.get(name)) for name in FEATURE_NAMES] for r in unique.values()], dtype=np.float64)
        values = _log_money(values)
        # Compute per column to avoid treating an entirely missing training
        # field as an observed zero and to avoid all-NaN runtime warnings.
        lower = np.zeros(values.shape[1], dtype=np.float64); upper = np.zeros_like(lower); median = np.zeros_like(lower)
        for index in range(values.shape[1]):
            observed = values[:, index][np.isfinite(values[:, index])]
            if observed.size:
                lower[index], upper[index] = np.quantile(observed, [0.01, 0.99])
                median[index] = np.median(observed)
        # All-null columns remain valid/masked.  Their fill and scale are neutral.
        clipped = np.clip(values, lower, upper)
        filled = np.where(np.isnan(clipped), median, clipped)
        mean = filled.mean(axis=0); std = filled.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return cls(lower, upper, median, mean, std)

    def transform(self, rows: Iterable[Example]) -> tuple[np.ndarray, np.ndarray]:
        rows = list(rows)
        raw = np.array([[np.nan if r.factors.get(name) is None else float(r.factors.get(name)) for name in FEATURE_NAMES] for r in rows], dtype=np.float64)
        raw = _log_money(raw)
        missing = np.isnan(raw).astype(np.float32)
        clipped = np.clip(raw, self.lower, self.upper)
        filled = np.where(np.isnan(clipped), self.median, clipped)
        normalized = ((filled - self.mean) / self.std).astype(np.float32)
        return np.concatenate((normalized, missing), axis=1), missing

    def to_dict(self) -> dict[str, Any]:
        return {"feature_names": list(FEATURE_NAMES), "money_fields": sorted(MONEY_FIELDS), **{key: getattr(self, key).tolist() for key in ("lower", "upper", "median", "mean", "std")}}


def _log_money(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, name in enumerate(FEATURE_NAMES):
        if name in MONEY_FIELDS:
            result[:, index] = np.sign(result[:, index]) * np.log1p(np.abs(result[:, index]))
    return result


def encode_claims(rows: Iterable[Example], max_claims: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed structured-claim tensor; no raw MD&A words or offsets become inputs."""
    rows = list(rows)
    claim_id = np.zeros((len(rows), max_claims), dtype=np.int64)
    polarity = np.zeros_like(claim_id); horizon = np.zeros_like(claim_id); strength = np.zeros_like(claim_id)
    confidence = np.zeros((len(rows), max_claims), dtype=np.float32)
    maps = {
        "polarity": {"positive": 1, "negative": 2, "mixed": 3, "unknown": 4},
        "horizon": {"near_term": 1, "medium_term": 2, "long_term": 3, "unspecified": 4},
        "strength": {"weak": 1, "medium": 2, "strong": 3, "unknown": 4},
    }
    for i, row in enumerate(rows):
        for j, claim in enumerate(row.claims[:max_claims]):
            claim_id[i,j] = CLAIM_VOCAB.get(claim.claim_id, 0)
            polarity[i,j] = maps["polarity"].get(claim.polarity, 4)
            horizon[i,j] = maps["horizon"].get(claim.time_horizon, 4)
            strength[i,j] = maps["strength"].get(claim.strength, 4)
            confidence[i,j] = claim.confidence
    return claim_id, polarity, horizon, strength, confidence


def frozen_hash_text(question: str, goal: str, dim: int = 96) -> np.ndarray:
    """A deterministic, frozen character/token n-gram encoder for lightweight q/g input."""
    vector = np.zeros(dim, dtype=np.float32)
    for prefix, value in (("q", question), ("g", goal)):
        normalized = " ".join(value.lower().split())
        units = [*normalized]
        tokens = normalized.split() + ["".join(units[i:i+2]) for i in range(max(0, len(units)-1))]
        for token in tokens:
            if not token.strip():
                continue
            digest = sha256(f"frozen-hash-text-v1|{prefix}|{token}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def encode_text(rows: Iterable[Example], dim: int = 96, shuffle_pairs: bool = False, seed: int = 0) -> np.ndarray:
    rows = list(rows)
    pairs = [(row.question, row.goal) for row in rows]
    if shuffle_pairs:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pairs))
        pairs = [pairs[index] for index in perm]
    return np.stack([frozen_hash_text(question, goal, dim) for question, goal in pairs]).astype(np.float32)


def target_arrays(rows: Iterable[Example]) -> dict[str, np.ndarray]:
    rows = list(rows)
    status = np.array([STATUS_VALUES.index(row.status) if row.status in STATUS_VALUES else -100 for row in rows], dtype=np.int64)
    intensity = np.array([row.intensity if row.intensity is not None else -100 for row in rows], dtype=np.int64)
    direction = np.array([DIRECTION_VALUES.index(row.direction) if row.direction in DIRECTION_VALUES else -100 for row in rows], dtype=np.int64)
    app = np.array([APPLICABILITY_VALUES.index(row.applicability) if row.applicability in APPLICABILITY_VALUES else -100 for row in rows], dtype=np.int64)
    relevance = np.array([row.relevance if row.relevance is not None else -100 for row in rows], dtype=np.int64)
    concepts = np.array([CONCEPT_INDEX[row.concept_id] for row in rows], dtype=np.int64)
    return {"status": status, "intensity": intensity, "direction": direction, "applicability": app, "relevance": relevance, "concept": concepts}


def serializable_split_rows(bundle: SplitBundle) -> Iterable[dict[str, Any]]:
    for split_name, rows in (("train", bundle.train), ("development", bundle.development), ("issuer_heldout", bundle.issuer_heldout), ("sealed_temporal", bundle.sealed_temporal), ("ignored", bundle.ignored)):
        for row in rows:
            yield {"split": split_name, "example_id": row.example_id, "episode_id": row.episode_id, "accession": row.accession, "available_at": row.available_at, "year": row.year, "microconcept_id": row.concept_id, "task_variant": row.task_variant, "effective_label_tier": row.effective_tier, "cik_sha256": stable_hash(row.cik_for_split)}
