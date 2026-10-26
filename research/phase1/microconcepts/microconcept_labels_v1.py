"""Deterministic, evidence-bound Episode microconcept labels v1.

This module deliberately has no dependency on routing, outcome, encoder, or LLM
code. Candidate records are projected onto an allowlisted representation before
any selection or annotation happens.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent
VERSION = "microconcept-labels-v1.0.0"
ANNOTATION_VERSION = "microconcept-evidence-projection-v1.0.0"
SELECTION_SEED = "microconcept-labels-v1-20260719"
ANNOTATED_AT = "2026-07-19T00:00:00Z"
PILOT_SIZE = 400
FROZEN_TEST_SIZE = 100
FROZEN_TEST_YEARS = frozenset({"2022", "2023"})
PILOT_MAX_YEAR = "2021"
LABEL_SOURCE = "deterministic_evidence_projection_v1"
TASK_LABEL_SOURCE = "deterministic_task_template_v1"
ANNOTATOR_ID = "system:microconcept-pilot-rule-builder"

FORBIDDEN_SOURCE_FIELDS = frozenset({
    "route_labels", "exploratory_outcome", "issuer_name", "ticker_at_collection",
    "raw_mda_text", "text", "outcome", "thought_model_routing", "llm_steering",
})
ALLOWED_STATUSES = frozenset({"supported", "contradicted", "unknown"})
ALLOWED_DIRECTIONS = frozenset({"increasing", "decreasing", "stable", "mixed", "unknown"})
ALLOWED_TIERS = frozenset({"Silver", "Weak", "Gold"})
BOUNDARY_CONCEPTS = frozenset({"constraint_pressure", "conversion_gap", "countervailing_pressure"})


@dataclass(frozen=True)
class Card:
    concept_id: str
    definition: str
    factors: tuple[str, ...]
    claims: tuple[str, ...]
    version: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    accession: str
    available_at: str
    form: str
    sic_division: str
    factors: dict[str, Any]
    factor_sources: dict[str, str]
    missing_reasons: dict[str, Any]

    @property
    def year(self) -> str:
        return self.available_at[:4]

    @property
    def stratum(self) -> tuple[str, str, str]:
        return self.year, self.form, self.sic_division


@dataclass(frozen=True)
class Signal:
    orientation: str  # support, oppose, mixed, context
    evidence_ref: dict[str, Any]
    weight: int = 1
    origin: str = "factor"


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def sic_division(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return f"SIC-{digits[:2]}" if len(digits) >= 2 else "SIC-unknown"


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} has no YAML frontmatter")
    _, payload, _ = text.split("---\n", 2)
    result = yaml.safe_load(payload)
    if not isinstance(result, dict):
        raise ValueError(f"{path} frontmatter is not a mapping")
    return result


def load_cards(card_root: Path = ROOT / "cards") -> dict[str, Card]:
    cards: dict[str, Card] = {}
    for path in sorted(card_root.glob("*.md")):
        raw = frontmatter(path)
        card = Card(
            concept_id=str(raw["id"]), definition=str(raw["definition"]),
            factors=tuple(item["id"] for item in raw["quantitative_evidence"]["factors"]),
            claims=tuple(raw["qualitative_evidence"]["claim_types"]), version=str(raw["version"]),
        )
        if card.concept_id in cards:
            raise ValueError(f"duplicate card ID: {card.concept_id}")
        cards[card.concept_id] = card
    if len(cards) != 11:
        raise ValueError(f"expected exactly 11 candidate cards, got {len(cards)}")
    return cards


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric == numeric else None
    return None


def project_episode(raw: dict[str, Any]) -> Episode:
    """Return an allowlisted view and intentionally never inspect forbidden values."""
    episode_id = raw.get("episode_id")
    filing = raw.get("filing")
    availability = raw.get("availability")
    financial = raw.get("financial_features")
    market = raw.get("market_state")
    if not isinstance(episode_id, str) or not isinstance(filing, dict) or not isinstance(availability, dict):
        raise ValueError("candidate record lacks required allowlisted identity/filing fields")
    if not isinstance(financial, dict) or not isinstance(market, dict):
        raise ValueError(f"{episode_id} lacks allowed factor mappings")
    accession, form, available_at = filing.get("accession"), filing.get("form"), availability.get("public_available_at")
    if not all(isinstance(item, str) and item for item in (accession, form, available_at)):
        raise ValueError(f"{episode_id} has invalid filing/public availability metadata")
    factors = dict(financial)
    factor_sources = {name: "financial_features" for name in financial}
    factors.update(market)
    factor_sources.update({name: "market_state" for name in market})
    missing_reasons = raw.get("missing_reasons")
    return Episode(
        episode_id=episode_id, accession=accession, available_at=available_at, form=form,
        sic_division=sic_division(raw.get("sic")), factors=factors, factor_sources=factor_sources,
        missing_reasons=dict(missing_reasons) if isinstance(missing_reasons, dict) else {},
    )


def load_episodes(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            episodes.append(project_episode(raw))
    if not episodes:
        raise ValueError(f"no Episodes in {path}")
    if len({item.episode_id for item in episodes}) != len(episodes):
        raise ValueError("duplicate episode_id in candidate input")
    return episodes


def load_claims(path: Path, allowed_claim_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    required = {"claim_key", "claim_id", "accession", "available_at", "section", "start_char", "end_char", "text_sha256", "extractor_version", "polarity", "strength", "confidence", "label_source"}
    by_accession: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            claim = json.loads(line)
            if not isinstance(claim, dict) or required - set(claim):
                raise ValueError(f"{path}:{line_number} missing structured-claim fields")
            if claim["claim_id"] not in allowed_claim_ids:
                raise ValueError(f"{path}:{line_number} has uncontrolled claim_id {claim['claim_id']!r}")
            if not isinstance(claim["available_at"], str) or not isinstance(claim["accession"], str):
                raise ValueError(f"{path}:{line_number} has invalid claim time/accession")
            by_accession[claim["accession"]].append(claim)
    for accession in by_accession:
        by_accession[accession].sort(key=lambda row: (row["start_char"], row["end_char"], row["claim_key"]))
    return dict(by_accession)


def _ordered(items: Iterable[Episode], namespace: str) -> list[Episode]:
    return sorted(items, key=lambda item: (stable_hash({"seed": SELECTION_SEED, "namespace": namespace, "episode_id": item.episode_id}), item.episode_id))


def stratified_select(items: Iterable[Episode], target: int, namespace: str) -> list[Episode]:
    """Deterministic round-robin sample across year × form × SIC division."""
    buckets: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
    for item in items:
        buckets[item.stratum].append(item)
    for key in buckets:
        buckets[key] = _ordered(buckets[key], f"{namespace}:{'|'.join(key)}")
    ordered_keys = sorted(buckets, key=lambda key: stable_hash({"seed": SELECTION_SEED, "namespace": namespace, "stratum": key}))
    selected: list[Episode] = []
    cursor = 0
    while len(selected) < target and ordered_keys:
        next_keys: list[tuple[str, str, str]] = []
        for key in ordered_keys:
            values = buckets[key]
            if cursor < len(values) and len(selected) < target:
                selected.append(values[cursor])
            if cursor + 1 < len(values):
                next_keys.append(key)
        ordered_keys = next_keys
        cursor += 1
    return sorted(selected, key=lambda item: item.episode_id)


def strata_distribution(items: Iterable[Episode]) -> dict[str, dict[str, int]]:
    materialized = list(items)
    return {
        "year": dict(sorted(Counter(row.year for row in materialized).items())),
        "form": dict(sorted(Counter(row.form for row in materialized).items())),
        "sic_division": dict(sorted(Counter(row.sic_division for row in materialized).items())),
    }


def select_pilot_and_freeze(episodes: list[Episode], claims_by_accession: dict[str, list[dict[str, Any]]]) -> tuple[list[Episode], list[Episode], dict[str, Any]]:
    frozen_pool = [item for item in episodes if item.year in FROZEN_TEST_YEARS]
    frozen = stratified_select(frozen_pool, FROZEN_TEST_SIZE, "sealed-temporal-test")
    frozen_ids = {item.episode_id for item in frozen}
    pre_cutoff = [item for item in episodes if item.year <= PILOT_MAX_YEAR and item.episode_id not in frozen_ids]
    claim_core = [item for item in pre_cutoff if item.accession in claims_by_accession]
    if len(claim_core) > PILOT_SIZE:
        claim_core = stratified_select(claim_core, PILOT_SIZE, "claim-covered-pilot")
    supplemental_needed = PILOT_SIZE - len(claim_core)
    core_ids = {row.episode_id for row in claim_core}
    supplemental_pool = [item for item in pre_cutoff if item.episode_id not in core_ids]
    supplemental = stratified_select(supplemental_pool, supplemental_needed, "nonclaim-pilot-supplement")
    pilot = sorted([*claim_core, *supplemental], key=lambda item: item.episode_id)
    if len(pilot) != PILOT_SIZE or len({item.episode_id for item in pilot}) != PILOT_SIZE:
        raise ValueError("could not assemble a unique 400-Episode pilot")
    metadata = {
        "selection_seed": SELECTION_SEED, "pilot_size": len(pilot), "frozen_time_test_size": len(frozen),
        "frozen_time_test_years": sorted(FROZEN_TEST_YEARS), "pilot_max_year": PILOT_MAX_YEAR,
        "pilot_claim_covered_core": len(claim_core), "pilot_missing_claim_supplement": len(supplemental),
        "pilot_strata": strata_distribution(pilot), "frozen_test_strata": strata_distribution(frozen),
    }
    return pilot, frozen, metadata


def _factor_polarity(factor_id: str, value: float) -> tuple[str, str, int] | None:
    """Return generic economic polarity: plus, minus, or context; fixed thresholds only."""
    rules: dict[str, tuple[float, float, bool]] = {
        "ttm_revenue_growth": (0.10, -0.05, True),
        "gross_margin_change": (0.02, -0.02, True),
        "current_ratio": (1.50, 1.00, True),
        "receivables_growth_minus_revenue_growth": (-0.05, 0.10, False),
        "inventory_growth_minus_revenue_growth": (-0.05, 0.10, False),
        "ttm_operating_margin": (0.10, 0.00, True),
        "net_debt_to_ebitda": (2.00, 4.00, False),
        "interest_coverage": (5.00, 2.00, True),
        "earnings_surprise_zscore": (1.00, -1.00, True),
        "earnings_volatility": (0.08, 0.20, False),
        "free_cash_flow_yield": (0.05, 0.00, True),
        "valuation_percentile": (0.35, 0.65, False),
        "trailing_drawdown": (-0.10, -0.30, True),
        "roic": (0.10, 0.00, True),
        "free_cash_flow_positive_years": (3.00, 1.00, True),
    }
    if factor_id in {"ttm_operating_cash_flow", "ttm_free_cash_flow"}:
        if value > 0:
            return "plus", f"{factor_id}>0", 1
        if value < 0:
            return "minus", f"{factor_id}<0", 1
        return None
    if factor_id == "ttm_capex":
        # Currency amount is not cross-issuer comparable; retain it only as a context reference.
        return ("context", "ttm_capex_observed_not_cross_issuer_normalized", 0) if value >= 0 else None
    if factor_id not in rules:
        return None
    favorable, adverse, normal_order = rules[factor_id]
    if normal_order:
        if value >= favorable:
            return "plus", f"{factor_id}>={favorable}", 1
        if value <= adverse:
            return "minus", f"{factor_id}<={adverse}", 1
    else:
        if value <= favorable:
            return "plus", f"{factor_id}<={favorable}", 1
        if value >= adverse:
            return "minus", f"{factor_id}>={adverse}", 1
    return None


def _factor_ref(episode: Episode, factor_id: str, value: float, rule_id: str, signal: str) -> dict[str, Any]:
    return {
        "kind": "factor", "episode_id": episode.episode_id, "factor_id": factor_id,
        "source_field": episode.factor_sources[factor_id], "value": value,
        "available_at": episode.available_at, "rule_id": rule_id, "signal": signal,
    }


def _claim_ref(claim: dict[str, Any], signal: str) -> dict[str, Any]:
    return {
        "kind": "mda_claim", "claim_key": claim["claim_key"], "claim_id": claim["claim_id"],
        "accession": claim["accession"], "section": claim["section"], "start_char": claim["start_char"],
        "end_char": claim["end_char"], "text_sha256": claim["text_sha256"],
        "extractor_version": claim["extractor_version"], "available_at": claim["available_at"],
        "polarity": claim["polarity"], "strength": claim["strength"], "confidence": claim["confidence"],
        "signal": signal,
    }


def factor_signals(episode: Episode, card: Card) -> list[tuple[str, Signal]]:
    result: list[tuple[str, Signal]] = []
    for factor_id in card.factors:
        value = _number(episode.factors.get(factor_id))
        if value is None:
            continue
        classification = _factor_polarity(factor_id, value)
        if classification is None:
            continue
        generic, rule_id, weight = classification
        result.append((generic, Signal(generic, _factor_ref(episode, factor_id, value, rule_id, generic), weight, "factor")))
    return result


def claim_signals(episode: Episode, card: Card, claims_by_accession: dict[str, list[dict[str, Any]]]) -> list[tuple[str, Signal]]:
    result: list[tuple[str, Signal]] = []
    for claim in claims_by_accession.get(episode.accession, []):
        if claim["claim_id"] not in card.claims or claim["available_at"] > episode.available_at:
            continue
        polarity = claim["polarity"]
        if polarity == "positive":
            generic = "plus"
        elif polarity == "negative":
            generic = "minus"
        elif polarity == "mixed":
            generic = "mixed"
        else:
            generic = "context"
        # Existing structured claims are Silver evidence. They can corroborate a
        # matching factor direction but can never decide an Episode state alone.
        result.append((generic, Signal(generic, _claim_ref(claim, generic), 1, "mda_claim")))
    return result


def _simple_projection(concept_id: str, factors: list[tuple[str, Signal]], claims: list[tuple[str, Signal]]) -> tuple[list[Signal], list[str]]:
    """Project generic polarity onto favorable/adverse concept definitions conservatively."""
    favorable = concept_id in {"capacity_headroom", "evidence_quality", "relative_gradient", "resource_balance", "temporal_persistence"}
    adverse = concept_id in {"constraint_pressure", "conversion_gap", "regime_proximity"}
    if not (favorable or adverse):
        return [], ["unsupported_projection_family"]
    projected: list[Signal] = []
    factor_directions = {direction for direction, _ in factors if direction in {"plus", "minus"}}
    for direction, signal in factors:
        if direction == "context":
            projected.append(Signal("context", signal.evidence_ref, 0, signal.origin))
        elif direction in {"plus", "minus"}:
            support = direction == "plus" if favorable else direction == "minus"
            projected.append(Signal("support" if support else "oppose", signal.evidence_ref, signal.weight, signal.origin))
    for direction, signal in claims:
        if direction == "mixed":
            projected.append(Signal("mixed", signal.evidence_ref, 0, signal.origin))
        elif direction in factor_directions:
            support = direction == "plus" if favorable else direction == "minus"
            projected.append(Signal("support" if support else "oppose", signal.evidence_ref, signal.weight, signal.origin))
        else:
            projected.append(Signal("context", signal.evidence_ref, 0, signal.origin))
    return projected, []


def project_signals(episode: Episode, card: Card, claims_by_accession: dict[str, list[dict[str, Any]]]) -> tuple[list[Signal], list[str]]:
    factors = factor_signals(episode, card)
    claims = claim_signals(episode, card, claims_by_accession)
    if card.concept_id in {"capacity_headroom", "constraint_pressure", "conversion_gap", "evidence_quality", "regime_proximity", "relative_gradient", "resource_balance", "temporal_persistence"}:
        return _simple_projection(card.concept_id, factors, claims)
    if card.concept_id == "countervailing_pressure":
        plus = [signal for direction, signal in factors if direction == "plus"]
        minus = [signal for direction, signal in factors if direction == "minus"]
        projected: list[Signal] = [Signal("context", signal.evidence_ref, 0, signal.origin) for direction, signal in factors if direction == "context"]
        projected += [Signal("context", signal.evidence_ref, 0, signal.origin) for _, signal in claims]
        if plus and minus:
            projected += [Signal("support", item.evidence_ref, item.weight, item.origin) for item in plus[:2] + minus[:2]]
        elif len(plus) >= 3:
            projected += [Signal("oppose", item.evidence_ref, item.weight, item.origin) for item in plus[:3]]
        elif len(minus) >= 3:
            projected += [Signal("oppose", item.evidence_ref, item.weight, item.origin) for item in minus[:3]]
        return projected, ["countervailing_units_not_netted"]
    if card.concept_id == "feedback_polarity":
        # A loop requires a direct reinvestment/capital-allocation disclosure and
        # corroborating operating factors; correlations alone are not causal proof.
        reinforcing_claims = [signal for direction, signal in claims if direction in {"plus", "minus"} and signal.evidence_ref["claim_id"] in {"reinvestment_quality", "capital_allocation_change"}]
        directional_factors = [signal for direction, signal in factors if direction in {"plus", "minus"}]
        projected = [Signal("context", signal.evidence_ref, 0, signal.origin) for _, signal in factors + claims]
        if reinforcing_claims and len(directional_factors) >= 2:
            projected += [Signal("support", item.evidence_ref, item.weight, item.origin) for item in directional_factors[:2] + reinforcing_claims[:1]]
        return projected, ["feedback_requires_disclosure_and_corroboration"]
    if card.concept_id == "outcome_asymmetry":
        # No permitted snapshot identifies a future outcome distribution.
        projected = [Signal("context", signal.evidence_ref, 0, signal.origin) for _, signal in factors + claims]
        return projected, ["outcome_distribution_not_identified_from_permitted_snapshot"]
    raise ValueError(f"unhandled card {card.concept_id}")


def decide_state(episode: Episode, card: Card, signals: list[Signal], decision_reasons: list[str]) -> tuple[dict[str, Any], list[str]]:
    support = [item for item in signals if item.orientation == "support"]
    oppose = [item for item in signals if item.orientation == "oppose"]
    mixed = [item for item in signals if item.orientation == "mixed"]
    context = [item for item in signals if item.orientation == "context"]
    support_score, oppose_score = sum(item.weight for item in support), sum(item.weight for item in oppose)
    review_reasons = list(decision_reasons)
    if support_score and oppose_score:
        status, intensity, direction = "unknown", 0, "mixed"
        review_reasons.append("evidence_conflict")
    elif mixed:
        status, intensity, direction = "unknown", 0, "mixed"
        review_reasons.append("mixed_claim_or_signal")
    elif support_score >= 2:
        status, intensity, direction = "supported", min(3, support_score), "increasing"
    elif oppose_score >= 2:
        status, intensity, direction = "contradicted", min(3, oppose_score), "decreasing"
    else:
        status, intensity, direction = "unknown", 0, "unknown"
        review_reasons.append("insufficient_correlated_evidence")
    evidence_refs = [item.evidence_ref for item in [*support, *oppose, *mixed, *context]]
    if not evidence_refs:
        # Keep an auditable observed factor reference in neutral regions. Missing
        # values still remain unknown rather than becoming a negative example.
        for factor_id in card.factors:
            value = _number(episode.factors.get(factor_id))
            if value is not None:
                evidence_refs = [_factor_ref(episode, factor_id, value, "context_observed_no_threshold_trigger", "context")]
                break
    if not evidence_refs:
        review_reasons.append("all_card_evidence_missing")
    evidence_refs = sorted(evidence_refs, key=lambda row: (row["kind"], row.get("factor_id", ""), row.get("claim_key", "")))
    corroborated = max(support_score, oppose_score) >= 2 and not mixed and not (support_score and oppose_score)
    tier = "Silver" if corroborated else "Weak"
    confidence = round(0.72 if intensity == 3 else 0.66 if intensity == 2 else 0.45 if direction == "mixed" else 0.40, 2)
    if card.concept_id in BOUNDARY_CONCEPTS:
        review_reasons.append("boundary_sensitive_concept")
    if status != "unknown" and max(support_score, oppose_score) == 2:
        review_reasons.append("minimum_corroboration")
    review_reasons = sorted(set(review_reasons))
    record = {
        "label_id": stable_hash({"version": VERSION, "episode_id": episode.episode_id, "microconcept_id": card.concept_id}),
        "episode_id": episode.episode_id, "microconcept_id": card.concept_id, "card_version": card.version,
        "accession": episode.accession, "form": episode.form, "sic_division": episode.sic_division,
        "available_at": episode.available_at, "status": status, "intensity": intensity, "direction": direction,
        "confidence": confidence, "evidence_refs": evidence_refs, "label_tier": tier, "label_source": LABEL_SOURCE,
        "annotator_id": ANNOTATOR_ID, "annotation_version": ANNOTATION_VERSION, "annotated_at": ANNOTATED_AT,
        "review_status": "pending_review" if review_reasons else "not_queued",
        "dispute_status": "needs_human_review" if review_reasons else "none", "decision_reasons": review_reasons,
    }
    return record, review_reasons


def build_state_labels(pilot: Iterable[Episode], cards: dict[str, Card], claims_by_accession: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    for episode in sorted(pilot, key=lambda item: item.episode_id):
        for concept_id, card in sorted(cards.items()):
            signals, decision_reasons = project_signals(episode, card, claims_by_accession)
            label, reasons = decide_state(episode, card, signals, decision_reasons)
            labels.append(label)
            if reasons:
                queue.append({
                    "queue_id": stable_hash({"version": VERSION, "state_label_id": label["label_id"], "reasons": reasons}),
                    "state_label_id": label["label_id"], "episode_id": episode.episode_id,
                    "microconcept_id": concept_id,
                    "priority": "high" if "evidence_conflict" in reasons or concept_id in BOUNDARY_CONCEPTS else "normal",
                    "reason_codes": reasons, "current_status": label["status"], "current_intensity": label["intensity"],
                    "evidence_refs": label["evidence_refs"],
                    "suggested_action": "Verify card boundary, evidence direction, and whether status should remain masked.",
                    "queue_status": "open", "created_at": ANNOTATED_AT,
                })
    return labels, queue


def task_templates(available_at: str, card: Card, status: str) -> list[tuple[str, str, str, str, int]]:
    date = available_at[:10]
    focused_question = f"截至 {date} 的可用披露和财务证据中，{card.definition}是否应成为该发行人近期基本面分析的重点？"
    focused_goal = "识别需要进一步核验的经营、流动性或现金转换风险与机会，并决定后续研究优先级。"
    focused_relevance = 3 if status != "unknown" else 2
    unrelated_question = f"截至 {date}，该申报文件是否采用了统一的内部文档归档命名格式？"
    unrelated_goal = "仅核对文档管理流程合规性，不评估发行人的经营、财务或市场状态。"
    return [
        ("focused", focused_question, focused_goal, "applicable", focused_relevance),
        ("unrelated", unrelated_question, unrelated_goal, "not_applicable", 0),
    ]


def build_task_labels(state_labels: Iterable[dict[str, Any]], cards: dict[str, Card]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for state in sorted(state_labels, key=lambda item: item["label_id"]):
        card = cards[state["microconcept_id"]]
        for variant, question, goal, applicability, relevance in task_templates(state["available_at"], card, state["status"]):
            tasks.append({
                "task_label_id": stable_hash({"version": VERSION, "state_label_id": state["label_id"], "task_variant": variant}),
                "state_label_id": state["label_id"], "episode_id": state["episode_id"], "microconcept_id": state["microconcept_id"],
                "task_variant": variant, "question": question, "goal": goal,
                "applicability": applicability, "relevance": relevance,
                "label_tier": "Silver" if variant == "focused" else "Weak", "label_source": TASK_LABEL_SOURCE,
                "annotator_id": ANNOTATOR_ID, "annotation_version": ANNOTATION_VERSION, "annotated_at": ANNOTATED_AT,
                "review_status": state["review_status"], "dispute_status": state["dispute_status"],
            })
    return tasks


def frozen_test_rows(frozen: Iterable[Episode]) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id, "accession": row.accession, "available_at": row.available_at,
        "form": row.form, "sic_division": row.sic_division, "split": "sealed_temporal_test",
        "frozen_at": ANNOTATED_AT, "selection_seed": SELECTION_SEED,
    } for row in sorted(frozen, key=lambda item: item.episode_id)]


def distribution_report(labels: list[dict[str, Any]], tasks: list[dict[str, Any]], queue: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    per_concept: dict[str, Any] = {}
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_concept[row["microconcept_id"]].append(row)
    for concept_id, rows in sorted(by_concept.items()):
        per_concept[concept_id] = {
            "status": dict(sorted(Counter(row["status"] for row in rows).items())),
            "intensity": dict(sorted(Counter(str(row["intensity"]) for row in rows).items())),
            "direction": dict(sorted(Counter(row["direction"] for row in rows).items())),
            "label_tier": dict(sorted(Counter(row["label_tier"] for row in rows).items())),
            "review_pending": sum(row["review_status"] == "pending_review" for row in rows),
        }
    # Co-labeling is only an audit clue, not human annotator agreement.
    concepts = sorted(by_concept)
    by_episode: dict[str, dict[str, str]] = defaultdict(dict)
    for row in labels:
        by_episode[row["episode_id"]][row["microconcept_id"]] = row["status"]
    co_status: dict[str, int] = {}
    for left_index, left in enumerate(concepts):
        for right in concepts[left_index + 1:]:
            co_status[f"{left}|{right}"] = sum(
                states.get(left) == states.get(right) and states.get(left) != "unknown"
                for states in by_episode.values()
            )
    reason_counts = Counter(reason for row in queue for reason in row["reason_codes"])
    return {
        "dataset_version": VERSION, "selection": selection,
        "label_counts": {"evidence_state": len(labels), "task_relevance": len(tasks), "episodes": len(by_episode)},
        "label_tier_counts": {
            "evidence_state": dict(sorted(Counter(row["label_tier"] for row in labels).items())),
            "task_relevance": dict(sorted(Counter(row["label_tier"] for row in tasks).items())),
            "combined": dict(sorted(Counter(row["label_tier"] for row in [*labels, *tasks]).items())),
        },
        "ai_only_label_counts": {
            "Silver": sum(row["label_tier"] == "Silver" for row in [*labels, *tasks]),
            "Weak": sum(row["label_tier"] == "Weak" for row in [*labels, *tasks]),
            "total": len(labels) + len(tasks),
        },
        "human_gold_count": 0,
        "agreement": {"status": "not_available", "reason": "No human-confirmed or two-person-arbitrated Gold labels exist in v1.", "human_gold_count": 0},
        "status_distribution": dict(sorted(Counter(row["status"] for row in labels).items())),
        "per_microconcept": per_concept,
        "task_distribution": {
            "applicability": dict(sorted(Counter(row["applicability"] for row in tasks).items())),
            "relevance": dict(sorted(Counter(str(row["relevance"]) for row in tasks).items())),
        },
        "review_queue": {"count": len(queue), "reason_counts": dict(sorted(reason_counts.items()))},
        "mutual_confusion_audit": {
            "interpretation": "same non-unknown status co-label counts; review with queue rather than call it confusion without human labels",
            "same_nonunknown_status_pairs": dict(sorted(co_status.items())),
            "boundary_concepts": {concept: per_concept[concept] for concept in sorted(BOUNDARY_CONCEPTS)},
        },
        "error_types_and_limitations": [
            "Missing evidence remains unknown and is never a contradicted/negative label.",
            "MD&A claims are inherited deterministic Silver evidence, not human Gold assertions.",
            "Outcome asymmetry is masked because a permitted snapshot does not identify a future outcome distribution.",
            "Constraint pressure, conversion gap, and countervailing pressure require human boundary review.",
            "No Gold labels exist; data supports pretraining or exploratory experiments only, not final effectiveness proof.",
        ],
    }


def annotation_guideline(cards: dict[str, Card]) -> str:
    concepts = "\n".join(f"- `{card.concept_id}` — {card.definition}" for _, card in sorted(cards.items()))
    return f"""# Episode-level Microconcept Annotation Guideline v1

**Dataset version:** `{VERSION}`  
**Automatic annotation version:** `{ANNOTATION_VERSION}`

## Scope and non-goals

This is an evidence-bound pilot for the fixed eleven candidate cards. It does
not train a microconcept encoder and does not use model-routing labels, future
outcomes, issuer identity, or raw MD&A text. `route_labels` and
`exploratory_outcome` are prohibited inputs. Every output is provisional
`Silver` or `Weak`; an automatic process must never emit `Gold`.

## Fixed concepts

{concepts}

## Evidence-state target (independent of q/g)

`status=supported` or `contradicted` requires two corroborating, time-available
directional signals. `unknown` means evidence is absent, neutral, single-source,
incompatible, or too weak. Missingness is a mask, never a negative label.
`intensity` is 0 for unknown; 1–3 represent corroboration. `direction` is a
current evidence implication, not a future-outcome prediction. Conflicts use
`unknown`, 0, and `mixed` pending review.

Each factor reference records source/value/time/fixed guardrail. Each MD&A
reference retains claim key, locator offsets, text hash, extractor version, and
time. A claim can corroborate a factor but cannot decide a state alone. Currency
amounts are never ranked across issuers.

## Boundary safeguards

- **constraint_pressure:** a poor ratio is not automatically a bottleneck.
- **conversion_gap:** growth or accounting profit alone does not establish a
  cash-conversion gap or fraud.
- **countervailing_pressure:** require opposing evidence; do not net
  incompatible financial units as physical forces.
- **feedback_polarity:** requires relevant disclosure plus operating-factor
  corroboration; correlation is not causal feedback.
- **outcome_asymmetry:** stays unknown without a permitted outcome-distribution
  source.

## q/g-conditioned relevance

Every factual label has one focused fundamental-analysis task and one explicitly
unrelated document-archiving task. Both reference the same `state_label_id`, so
facts never change with q/g while relevance does. Unknown factual state may
still be applicable; it never implies not applicable.

## Review and Gold promotion

Conflicting, weak, missing, and boundary-sensitive cases go to the independent
review queue. Only documented human confirmation or two-person arbitration may
promote a label to Gold. This automatic pilot has no Gold labels or human
agreement; it is suitable only for pretraining/exploration, not final
model-effectiveness proof.
"""


def build_manifest(candidates_path: Path, claims_path: Path, cards: dict[str, Card], selection: dict[str, Any], files: Iterable[Path]) -> dict[str, Any]:
    return {
        "dataset_version": VERSION, "annotation_version": ANNOTATION_VERSION, "generated_at": ANNOTATED_AT,
        "selection_seed": SELECTION_SEED,
        "source_files": {
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "mda_claims": {"path": str(claims_path), "sha256": sha256_file(claims_path)},
            "cards": {"path": str(ROOT / "cards"), "ids": sorted(cards), "versions": {key: card.version for key, card in sorted(cards.items())}},
        },
        "selection": selection, "prohibited_inputs": sorted(FORBIDDEN_SOURCE_FIELDS),
        "labeling_boundary": "No routing labels, future outcomes, issuer identity, raw MD&A text, encoder training, or LLM steering were used.",
        "outputs": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(files)},
        "recoverability": {"write_mode": "atomic per artifact", "rerun_policy": "deterministic selection and stable IDs; --resume may repair an interrupted run after source-hash verification"},
        "gold_policy": "Automatic builder emitted no Gold labels. Human confirmation or two-person arbitration is required for Gold.",
        "suitability": "Silver/Weak only: pretraining and exploratory experiments only; not final effectiveness proof.",
    }


def validate_label_shape(label: dict[str, Any], cards: dict[str, Card]) -> list[str]:
    errors: list[str] = []
    required = {"label_id", "episode_id", "microconcept_id", "status", "intensity", "direction", "evidence_refs", "available_at", "label_tier", "label_source", "annotator_id", "annotation_version", "annotated_at", "review_status", "dispute_status"}
    if missing := required - set(label):
        return [f"missing fields: {sorted(missing)}"]
    if label["microconcept_id"] not in cards:
        errors.append("microconcept_id not in fixed cards")
    if label["status"] not in ALLOWED_STATUSES:
        errors.append("invalid status")
    if label["intensity"] not in {0, 1, 2, 3}:
        errors.append("invalid intensity")
    if label["direction"] not in ALLOWED_DIRECTIONS:
        errors.append("invalid direction")
    if label["label_tier"] not in ALLOWED_TIERS or label["label_tier"] == "Gold":
        errors.append("automatic label may not be Gold")
    if label["status"] == "unknown" and label["intensity"] != 0:
        errors.append("unknown must have intensity 0")
    if label["status"] != "unknown" and not label["evidence_refs"]:
        errors.append("non-unknown label lacks evidence refs")
    if not isinstance(label["evidence_refs"], list):
        errors.append("evidence_refs must be a list")
    for ref in label["evidence_refs"]:
        if ref.get("kind") == "factor":
            needed = {"kind", "episode_id", "factor_id", "source_field", "value", "available_at", "rule_id", "signal"}
            if needed - set(ref):
                errors.append("invalid factor evidence ref")
            elif ref["episode_id"] != label["episode_id"]:
                errors.append("factor evidence episode mismatch")
        elif ref.get("kind") == "mda_claim":
            needed = {"kind", "claim_key", "claim_id", "accession", "section", "start_char", "end_char", "text_sha256", "extractor_version", "available_at", "polarity", "strength", "confidence", "signal"}
            if needed - set(ref):
                errors.append("invalid mda claim evidence ref")
            elif ref["available_at"] > label["available_at"]:
                errors.append("claim evidence is after Episode availability")
            elif not (isinstance(ref["start_char"], int) and isinstance(ref["end_char"], int) and ref["start_char"] < ref["end_char"]):
                errors.append("invalid claim evidence offsets")
        else:
            errors.append("unrecognized evidence reference kind")
    return errors


def validate_task_shape(task: dict[str, Any], state_ids: set[str]) -> list[str]:
    errors: list[str] = []
    required = {"task_label_id", "state_label_id", "episode_id", "microconcept_id", "task_variant", "question", "goal", "applicability", "relevance", "label_tier", "label_source", "annotator_id", "annotation_version", "annotated_at", "review_status", "dispute_status"}
    if missing := required - set(task):
        return [f"missing fields: {sorted(missing)}"]
    if task["state_label_id"] not in state_ids:
        errors.append("unknown state_label_id")
    if task["applicability"] not in {"applicable", "not_applicable", "unknown"}:
        errors.append("invalid applicability")
    if task["relevance"] not in {0, 1, 2, 3}:
        errors.append("invalid relevance")
    if task["label_tier"] == "Gold":
        errors.append("automatic task label may not be Gold")
    return errors

