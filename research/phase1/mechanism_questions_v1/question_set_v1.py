"""Builders and validators for the LollaNet mechanism question-set V1.

This module deliberately contains no LLM invocation and never changes the fixed
microconcept or thinking-model assets.  It turns independently curated public
fact packs into route-blind Episodes and validates external weak-teacher labels.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE / "contracts" / "mechanism_question_v1_contract.yaml"
SELECTION_PATH = PACKAGE / "contracts" / "selection_contract.yaml"
MICRO_INDEX_PATH = ROOT / "research" / "phase1" / "microconcepts" / "microconcept_index.yaml"
CARDS_DIR = ROOT / "research" / "phase1" / "microconcepts" / "cards"
MODELS_DIR = ROOT / "research" / "phase1" / "models"

DOMAINS = ("business_finance", "software_engineering", "product_organization", "science_public_systems")
QUESTION_KINDS = (
    "mechanism_diagnosis", "decision_tradeoff", "evidence_discrimination", "conditional_or_counterfactual",
)
STATE_STATUSES = {"supported", "contradicted", "unknown", "mixed", "candidate"}
ACTIVE_STATUSES = {"supported", "candidate", "mixed"}
STATE_CHANGE_AXES = {"status", "intensity", "direction", "temporal_persistence", "task_relevance"}
BOUNDED_STATE_AXES = {"intensity", "temporal_persistence", "task_relevance"}

# Episode ``task_relevance`` combines an explicit question-kind profile with
# evidence-derived salience from the fact state. A Router which may not read
# question text still receives task demand, without collapsing every fact under
# one question kind to the same near-fixed vector.
TASK_RELEVANCE_PROFILES: dict[str, dict[str, int]] = {
    "mechanism_diagnosis_v1": {
        "capacity_headroom": 2, "constraint_pressure": 3, "conversion_gap": 2,
        "countervailing_pressure": 0, "evidence_quality": 1, "feedback_polarity": 3,
        "outcome_asymmetry": 2, "regime_proximity": 2, "relative_gradient": 2,
        "resource_balance": 1, "temporal_persistence": 1,
    },
    "decision_tradeoff_v1": {
        "capacity_headroom": 3, "constraint_pressure": 3, "conversion_gap": 1,
        "countervailing_pressure": 2, "evidence_quality": 1, "feedback_polarity": 1,
        "outcome_asymmetry": 3, "regime_proximity": 3, "relative_gradient": 0,
        "resource_balance": 2, "temporal_persistence": 1,
    },
    "evidence_discrimination_v1": {
        "capacity_headroom": 1, "constraint_pressure": 1, "conversion_gap": 2,
        "countervailing_pressure": 3, "evidence_quality": 3, "feedback_polarity": 1,
        "outcome_asymmetry": 1, "regime_proximity": 2, "relative_gradient": 2,
        "resource_balance": 0, "temporal_persistence": 2,
    },
    "conditional_or_counterfactual_v1": {
        "capacity_headroom": 2, "constraint_pressure": 2, "conversion_gap": 2,
        "countervailing_pressure": 1, "evidence_quality": 1, "feedback_polarity": 3,
        "outcome_asymmetry": 2, "regime_proximity": 3, "relative_gradient": 1,
        "resource_balance": 2, "temporal_persistence": 3,
    },
}
PROFILE_BY_QUESTION_KIND = {
    "mechanism_diagnosis": "mechanism_diagnosis_v1",
    "decision_tradeoff": "decision_tradeoff_v1",
    "evidence_discrimination": "evidence_discrimination_v1",
    "conditional_or_counterfactual": "conditional_or_counterfactual_v1",
}
FORBIDDEN_FIELDS = {
    "route_labels", "thought_model_routing", "primary_model", "auxiliary_models", "near_miss_models",
    "exploratory_outcome", "outcome", "future_outcome", "issuer_name", "ticker_at_collection", "raw_mda_text",
}

class ValidationError(ValueError):
    """Raised when an artifact violates the public research contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValidationError(f"microconcept card lacks YAML front matter: {path}")
    _, raw, _ = text.split("---", 2)
    return yaml.safe_load(raw) or {}


@lru_cache(maxsize=1)
def load_semantic_assets() -> dict[str, Any]:
    """Load read-only V1 semantic assets and derive the legal concept-model edges."""
    index = load_yaml(MICRO_INDEX_PATH)
    concepts = index.get("concepts", [])
    concept_ids = {item["id"] for item in concepts}
    cards = {_front_matter(path)["id"]: _front_matter(path) for path in CARDS_DIR.glob("*.md")}
    models: dict[str, dict[str, Any]] = {}
    for path in MODELS_DIR.glob("*.yaml"):
        model = load_yaml(path)
        model_id = model.get("id") or model.get("model_id")
        if not model_id:
            raise ValidationError(f"model asset lacks id: {path}")
        models[model_id] = model
    edges: dict[str, list[dict[str, str]]] = defaultdict(list)
    permitted_relations = {"requires", "supports", "modulates", "challenges"}
    for concept in concepts:
        concept_id = concept["id"]
        indexed_models = set(concept.get("models", []))
        card_relations = cards[concept_id].get("model_relations", [])
        relation_models = {item.get("model_id") for item in card_relations}
        if indexed_models != relation_models:
            raise ValidationError(f"model relation index and card disagree for {concept_id}")
        for relation in card_relations:
            model_id, kind = relation.get("model_id"), relation.get("relation")
            if model_id not in models or kind not in permitted_relations:
                raise ValidationError(f"invalid fixed relation for {concept_id}: {model_id}/{kind}")
            edges[concept_id].append({"model_id": model_id, "relation": kind})
    if len(concept_ids) != 11 or len(cards) != 11:
        raise ValidationError("V1 requires exactly 11 fixed microconcept assets")
    if len(models) != 20:
        raise ValidationError("V1 requires exactly 20 fixed model assets")
    if set(cards) != concept_ids:
        raise ValidationError("microconcept cards and index disagree")
    return {"concept_ids": concept_ids, "cards": cards, "models": models, "edges": dict(edges)}


def _require(value: Any, message: str) -> None:
    if value is None or value == "" or value == []:
        raise ValidationError(message)


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"invalid {field}: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _walk_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{path}.{key}" if path else key
            if key in FORBIDDEN_FIELDS:
                raise ValidationError(f"forbidden field in fact pack: {current}")
            _walk_forbidden(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def _model_terms(assets: dict[str, Any]) -> set[str]:
    terms = set(assets["models"])
    for model in assets["models"].values():
        name = model.get("name")
        if isinstance(name, str) and len(name) > 2:
            terms.add(name.lower())
    return terms


def _validate_question_does_not_leak_route(variant: dict[str, Any], assets: dict[str, Any]) -> None:
    text = " ".join(str(variant.get(field, "")) for field in ("question", "goal", "constraints")).lower()
    for forbidden in _model_terms(assets):
        if forbidden.lower() in text:
            raise ValidationError(f"question leaks fixed model identifier/name: {forbidden}")


def _validate_counterfactual(variant: dict[str, Any], facts_by_id: dict[str, dict[str, Any]], microconcept_ids: set[str] | None = None) -> None:
    delta = variant.get("counterfactual_delta")
    if delta is None:
        return
    if not isinstance(delta, dict):
        raise ValidationError("counterfactual_delta must be an object or null")
    changes = delta.get("changes", [])
    if not 1 <= len(changes) <= 2:
        raise ValidationError("controlled counterfactual must modify one or two explicit facts")
    touched: set[str] = set()
    for change in changes:
        fact_id = change.get("fact_id")
        if fact_id not in facts_by_id:
            raise ValidationError("counterfactual_delta references an unknown fact")
        if not change.get("field") or "before" not in change or "after" not in change:
            raise ValidationError("each counterfactual change needs fact_id, field, before, after")
        field = change["field"]
        fact = facts_by_id[fact_id]
        if field not in fact:
            raise ValidationError("counterfactual_delta field must exist on the referenced fact")
        if change["before"] != fact[field]:
            raise ValidationError("counterfactual before value must match the referenced fact field")
        if change["before"] == change["after"]:
            raise ValidationError("counterfactual change must actually change a declared value")
        touched.add(fact_id)
    if len(touched) > 2:
        raise ValidationError("counterfactual may modify no more than two facts")
    state_changes = delta.get("microconcept_state_changes", [])
    if not isinstance(state_changes, list) or not state_changes:
        raise ValidationError("controlled counterfactual must declare at least one microconcept state or relevance change")
    changed_axes: set[tuple[str, str]] = set()
    for change in state_changes:
        concept_id = change.get("microconcept_id")
        if microconcept_ids is not None and concept_id not in microconcept_ids:
            raise ValidationError("counterfactual state change references an unknown microconcept")
        axis = change.get("axis")
        if axis not in STATE_CHANGE_AXES or "before" not in change or "after" not in change:
            raise ValidationError(
                "counterfactual state change needs microconcept_id, a supported state axis, before, and after"
            )
        key = (concept_id, axis)
        if key in changed_axes:
            raise ValidationError("counterfactual may change each microconcept axis at most once")
        changed_axes.add(key)
        if change["before"] == change["after"]:
            raise ValidationError("counterfactual state change must change the declared axis")
        if axis == "status" and change["after"] not in STATE_STATUSES:
            raise ValidationError("counterfactual status after value is invalid")
        if axis in BOUNDED_STATE_AXES and (
            isinstance(change["after"], bool) or not isinstance(change["after"], int) or not 0 <= change["after"] <= 3
        ):
            raise ValidationError(f"counterfactual {axis} after value must be an integer from 0 through 3")
        if axis == "direction" and (not isinstance(change["after"], str) or not change["after"].strip()):
            raise ValidationError("counterfactual direction after value must be a non-empty string")
    if not any(axis != "task_relevance" for _, axis in changed_axes):
        raise ValidationError("controlled counterfactual must declare at least one non-task microconcept state change")


def _validate_task_conditioning(variant: dict[str, Any], microconcept_ids: set[str]) -> None:
    profile = variant.get("task_relevance_profile")
    expected_profile = PROFILE_BY_QUESTION_KIND.get(variant.get("question_kind"))
    if profile != expected_profile:
        raise ValidationError("question variant must declare the canonical task_relevance_profile for its question_kind")
    if profile not in TASK_RELEVANCE_PROFILES:
        raise ValidationError("question variant has an unknown task_relevance_profile")
    if variant.get("task_focus_microconcept_id") not in microconcept_ids:
        raise ValidationError("question variant task_focus_microconcept_id must be a fixed microconcept")


def task_relevance_basis(
    variant: dict[str, Any], fact_states: Iterable[dict[str, Any]], microconcept_ids: Iterable[str],
) -> dict[str, Any]:
    """Explain the fact- and question-conditioned relevance assigned to every concept.

    The question-kind profile expresses task demand, while the case-level state
    contributes evidence-derived salience. Keeping both components in the
    Episode audit trail prevents four question profiles from becoming four
    near-constant Router inputs across otherwise different facts.
    """
    ids = tuple(sorted(microconcept_ids))
    _validate_task_conditioning(variant, set(ids))
    profile = deepcopy(TASK_RELEVANCE_PROFILES[variant["task_relevance_profile"]])
    if set(profile) != set(ids):
        raise ValidationError("task relevance profile does not cover exactly the fixed microconcepts")
    states = {row.get("microconcept_id"): row for row in fact_states}
    if set(states) != set(ids) or len(states) != len(ids):
        raise ValidationError("task relevance conditioning requires exactly the fixed fact microconcept states")
    focus = variant["task_focus_microconcept_id"]
    components = []
    vector: dict[str, int] = {}
    for concept_id in ids:
        state = states[concept_id]
        fact_level = state.get("task_relevance")
        if isinstance(fact_level, bool) or not isinstance(fact_level, int) or not 0 <= fact_level <= 3:
            raise ValidationError("fact microconcept task_relevance must be an integer from 0 through 3")
        profile_level = profile[concept_id]
        # The rounded midpoint keeps task demand influential without discarding
        # factual salience. Unknown states remain usable for evidence questions,
        # but cannot look fully fact-supported only because a profile is high.
        combined = (profile_level + fact_level + 1) // 2
        if state.get("status") == "unknown":
            combined = min(combined, 1 if profile_level <= 1 else 2)
        focus_bonus = int(concept_id == focus and fact_level >= 2 and combined < 3)
        final = min(3, combined + focus_bonus)
        vector[concept_id] = final
        components.append({
            "microconcept_id": concept_id,
            "question_profile_level": profile_level,
            "fact_relevance_level": fact_level,
            "fact_status": state.get("status"),
            "focus_bonus": focus_bonus,
            "final_task_relevance": final,
        })
    return {
        "profile": variant["task_relevance_profile"],
        "focus_microconcept_id": focus,
        "components": components,
        "vector": {concept_id: vector[concept_id] for concept_id in ids},
    }


def task_relevance_vector(
    variant: dict[str, Any], microconcept_ids: Iterable[str], fact_states: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Return the auditable fact- and question-conditioned relevance vector."""
    return task_relevance_basis(variant, fact_states, microconcept_ids)["vector"]


def question_conditioned_microconcept_states(
    fact_states: list[dict[str, Any]], variant: dict[str, Any], assets: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply task demand and declared counterfactual state changes to fact states.

    The evidence state axes are copied unchanged for ordinary questions. A
    controlled counterfactual may additionally change any declared state axis;
    its ``before`` value must match the state after task conditioning, not an
    unconditioned or inferred value.
    """
    assets = assets or load_semantic_assets()
    by_id = {row.get("microconcept_id"): deepcopy(row) for row in fact_states}
    if set(by_id) != assets["concept_ids"] or len(by_id) != len(assets["concept_ids"]):
        raise ValidationError("question conditioning requires exactly the fixed microconcept states")
    vector = task_relevance_vector(variant, assets["concept_ids"], by_id.values())
    for concept_id, state in by_id.items():
        state["task_relevance"] = vector[concept_id]
    delta = variant.get("counterfactual_delta")
    if delta is not None:
        for change in delta["microconcept_state_changes"]:
            concept_id, axis = change["microconcept_id"], change["axis"]
            state = by_id[concept_id]
            if state[axis] != change["before"]:
                raise ValidationError(
                    f"counterfactual microconcept before value does not match question-conditioned state: {concept_id}.{axis}"
                )
            state[axis] = change["after"]
            if state["status"] in ACTIVE_STATUSES and not state.get("evidence_refs"):
                raise ValidationError("counterfactual cannot activate an evidence-free microconcept state")
    return [by_id[concept_id] for concept_id in sorted(by_id)]


def validate_fact_pack(pack: dict[str, Any], assets: dict[str, Any] | None = None) -> None:
    """Validate one public fact pack without inspecting or producing route labels."""
    assets = assets or load_semantic_assets()
    _walk_forbidden(pack)
    required_fields = ("case_id", "domain", "as_of", "public_available_at", "source_refs", "facts", "factual_uncertainties", "structured_evidence", "microconcept_states", "question_variants")
    missing = [field for field in required_fields if field not in pack or pack[field] is None]
    if missing:
        raise ValidationError(f"fact pack missing {', '.join(missing)}")
    for field in ("case_id", "domain", "as_of", "public_available_at", "source_refs", "facts", "microconcept_states", "question_variants"):
        _require(pack.get(field), f"fact pack missing {field}")
    if not isinstance(pack["factual_uncertainties"], list) or not isinstance(pack["structured_evidence"], list):
        raise ValidationError("factual_uncertainties and structured_evidence must be lists")
    if pack["domain"] not in DOMAINS:
        raise ValidationError(f"unknown domain: {pack['domain']}")
    as_of = _parse_time(pack["as_of"], "as_of")
    public_at = _parse_time(pack["public_available_at"], "public_available_at")
    if public_at > as_of:
        raise ValidationError("public_available_at must be no later than the fact-pack as_of cutoff")
    sources = pack["source_refs"]
    source_ids = {source.get("source_id") for source in sources}
    if None in source_ids or len(source_ids) != len(sources):
        raise ValidationError("source_refs need unique non-empty source_id values")
    for source in sources:
        _require(source.get("title"), "source_refs requires title")
        url = source.get("url", "")
        if not isinstance(url, str) or not re.match(r"https?://", url):
            raise ValidationError("public source_refs requires an http(s) URL")
        if _parse_time(source.get("published_at"), "source.published_at") > as_of:
            raise ValidationError("source published after the fact-pack cutoff")
    fact_ids: set[str] = set()
    for fact in pack["facts"]:
        fact_id = fact.get("fact_id")
        _require(fact_id, "every fact needs fact_id")
        if fact_id in fact_ids:
            raise ValidationError("fact ids must be unique within a case")
        fact_ids.add(fact_id)
        _require(fact.get("statement"), "every fact needs a statement")
        refs = fact.get("evidence_refs", [])
        if not refs or not set(refs).issubset(source_ids):
            raise ValidationError("fact evidence_refs must point to source_refs")
        if _parse_time(fact.get("available_at"), "fact.available_at") > as_of:
            raise ValidationError("fact became public after the fact-pack cutoff")
    states = pack["microconcept_states"]
    ids = [state.get("microconcept_id") for state in states]
    if set(ids) != assets["concept_ids"] or len(ids) != 11:
        raise ValidationError("each pack must provide exactly one state for each fixed microconcept")
    for state in states:
        if state.get("status") not in STATE_STATUSES:
            raise ValidationError("invalid microconcept state status")
        for axis in ("intensity", "temporal_persistence", "task_relevance"):
            value = state.get(axis)
            if not isinstance(value, int) or not 0 <= value <= 3:
                raise ValidationError(f"microconcept {axis} must be an integer from 0 through 3")
        _require(state.get("direction"), "microconcept direction is required")
        refs = state.get("evidence_refs", [])
        if not set(refs).issubset(fact_ids):
            raise ValidationError("microconcept evidence_refs must point to fact ids")
        if state["status"] in ACTIVE_STATUSES and not refs:
            raise ValidationError("active microconcept states need cited fact evidence")
    variants = pack["question_variants"]
    kinds = [variant.get("question_kind") for variant in variants]
    if len(variants) != 4 or set(kinds) != set(QUESTION_KINDS):
        raise ValidationError("each pack must contain exactly the four V1 question variants")
    for variant in variants:
        for field in ("question", "goal", "constraints"):
            _require(variant.get(field), f"question variant missing {field}")
        if not isinstance(variant.get("cross_domain_transfer"), bool):
            raise ValidationError("cross_domain_transfer must be boolean")
        if "transfer_rationale" not in variant:
            raise ValidationError("question variant missing transfer_rationale")
        transfer = variant.get("transfer_rationale")
        if variant["cross_domain_transfer"]:
            if not isinstance(transfer, dict) or transfer.get("source_domain") not in DOMAINS or transfer["source_domain"] == pack["domain"] or not isinstance(transfer.get("structural_mapping"), str) or not transfer["structural_mapping"].strip() or not set(transfer.get("evidence_refs", [])).issubset(fact_ids) or not transfer.get("evidence_refs"):
                raise ValidationError("cross-domain transfer needs a distinct source domain, structural mapping, and fact evidence")
        elif transfer is not None:
            raise ValidationError("non-transfer variants must set transfer_rationale to null")
        _validate_question_does_not_leak_route(variant, assets)
        _validate_task_conditioning(variant, assets["concept_ids"])
        _validate_counterfactual(variant, {fact["fact_id"]: fact for fact in pack["facts"]}, assets["concept_ids"])
        # This validates counterfactual ``before`` against the profile-conditioned state.
        question_conditioned_microconcept_states(pack["microconcept_states"], variant, assets)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)

def split_case_packs(case_ids: Iterable[str], require_v1_size: bool = False) -> dict[str, str]:
    """Deterministically split by fact pack, never by Episode."""
    values = list(case_ids)
    ordered = sorted(set(values), key=lambda item: hashlib.sha256(item.encode("utf-8")).hexdigest())
    if len(ordered) != len(values):
        raise ValidationError("case_id values must be unique")
    if require_v1_size and len(ordered) != 160:
        raise ValidationError("full V1 requires exactly 160 fact packs")
    if len(ordered) == 160:
        train_n, dev_n = 112, 24
    else:
        train_n, dev_n = round(len(ordered) * 0.70), round(len(ordered) * 0.15)
        if len(ordered) >= 3:
            train_n = min(train_n, len(ordered) - 2)
            dev_n = min(dev_n, len(ordered) - train_n - 1)
    return {case_id: "train" if index < train_n else "development" if index < train_n + dev_n else "sealed_test" for index, case_id in enumerate(ordered)}


def candidate_subgraph(states: list[dict[str, Any]], assets: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return only legal fixed concept-model candidate edges for the router."""
    assets = assets or load_semantic_assets()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for state in states:
        concept_id = state["microconcept_id"]
        if state["task_relevance"] <= 0:
            continue
        nodes.append({
            "microconcept_id": concept_id,
            "status": state["status"], "intensity": state["intensity"], "direction": state["direction"],
            "temporal_persistence": state["temporal_persistence"], "task_relevance": state["task_relevance"],
            "evidence_refs": list(state.get("evidence_refs", [])),
        })
        for edge in assets["edges"].get(concept_id, []):
            key = (concept_id, edge["model_id"], edge["relation"])
            if key not in seen:
                edges.append({"microconcept_id": concept_id, **edge})
                seen.add(key)
    return {"microconcept_nodes": nodes, "edges": edges, "candidate_model_ids": sorted({edge["model_id"] for edge in edges})}


def build_episodes(packs: Iterable[dict[str, Any]], require_v1_size: bool = False) -> list[dict[str, Any]]:
    assets = load_semantic_assets()
    pack_list = list(packs)
    case_ids = [pack.get("case_id") for pack in pack_list]
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError("case_id values must be unique")
    for pack in pack_list:
        validate_fact_pack(pack, assets)
    if require_v1_size:
        domain_counts = Counter(pack["domain"] for pack in pack_list)
        if len(pack_list) != 160 or any(domain_counts[domain] != 40 for domain in DOMAINS):
            raise ValidationError("full V1 requires 160 packs, 40 in every fixed domain")
        if any(pack.get("example") or pack.get("fictional") for pack in pack_list):
            raise ValidationError("full V1 cannot contain fictional or example fact packs")
    splits = split_case_packs(case_ids, require_v1_size=require_v1_size)
    episodes: list[dict[str, Any]] = []
    for pack in pack_list:
        for variant in pack["question_variants"]:
            relevance_basis = task_relevance_basis(variant, pack["microconcept_states"], assets["concept_ids"])
            episode_states = question_conditioned_microconcept_states(pack["microconcept_states"], variant, assets)
            graph = candidate_subgraph(episode_states, assets)
            episodes.append({
                "episode_id": f"{pack['case_id']}::{variant['question_kind']}",
                "case_id": pack["case_id"], "domain": pack["domain"], "split": splits[pack["case_id"]],
                "question_kind": variant["question_kind"], "question": variant["question"], "goal": variant["goal"],
                "constraints": variant["constraints"], "public_available_at": pack["public_available_at"],
                "as_of": pack["as_of"], "source_refs": deepcopy(pack["source_refs"]), "facts": deepcopy(pack["facts"]),
                "factual_uncertainties": deepcopy(pack["factual_uncertainties"]),
                "structured_evidence": deepcopy(pack["structured_evidence"]),
                "fact_microconcept_states": deepcopy(pack["microconcept_states"]),
                "task_relevance_profile": variant["task_relevance_profile"],
                "task_focus_microconcept_id": variant["task_focus_microconcept_id"],
                "task_relevance_basis": relevance_basis,
                "microconcept_states": episode_states, "candidate_subgraph": graph,
                "cross_domain_transfer": variant["cross_domain_transfer"],
                "transfer_rationale": deepcopy(variant.get("transfer_rationale")),
                "counterfactual_delta": deepcopy(variant.get("counterfactual_delta")),
            })
    validate_episodes(episodes, require_v1_size=require_v1_size)
    return episodes


def validate_episodes(episodes: Iterable[dict[str, Any]], require_v1_size: bool = False) -> None:
    assets = load_semantic_assets()
    rows = list(episodes)
    ids = [row.get("episode_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValidationError("episode_id values must be unique")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("case_id")].append(row)
        for field in ("episode_id", "case_id", "domain", "split", "question_kind", "question", "goal", "constraints", "as_of", "public_available_at", "source_refs", "facts", "fact_microconcept_states", "task_relevance_profile", "task_focus_microconcept_id", "task_relevance_basis", "microconcept_states", "candidate_subgraph"):
            _require(row.get(field), f"episode missing {field}")
        if row["domain"] not in DOMAINS or row["split"] not in {"train", "development", "sealed_test"}:
            raise ValidationError("episode domain or split invalid")
        _validate_question_does_not_leak_route(row, assets)
        if "cross_domain_transfer" not in row or not isinstance(row["cross_domain_transfer"], bool) or "transfer_rationale" not in row:
            raise ValidationError("episode missing valid cross-domain-transfer fields")
        transfer = row.get("transfer_rationale")
        if row["cross_domain_transfer"]:
            if not isinstance(transfer, dict) or transfer.get("source_domain") not in DOMAINS or transfer["source_domain"] == row["domain"] or not isinstance(transfer.get("structural_mapping"), str) or not transfer["structural_mapping"].strip():
                raise ValidationError("episode cross-domain transfer rationale is invalid")
        elif transfer is not None:
            raise ValidationError("non-transfer episode must not carry a transfer rationale")
        as_of = _parse_time(row["as_of"], "episode.as_of")
        if _parse_time(row["public_available_at"], "episode.public_available_at") > as_of:
            raise ValidationError("episode public availability is after its cutoff")
        source_ids = {source.get("source_id") for source in row["source_refs"]}
        if None in source_ids or len(source_ids) != len(row["source_refs"]):
            raise ValidationError("episode source_refs need unique non-empty source ids")
        for source in row["source_refs"]:
            if not source.get("title") or not re.match(r"https?://", source.get("url", "")):
                raise ValidationError("episode sources must retain public title and http(s) URL")
            if _parse_time(source.get("published_at"), "episode.source.published_at") > as_of:
                raise ValidationError("episode source was published after its cutoff")
        fact_ids = set()
        for fact in row["facts"]:
            if not fact.get("fact_id") or fact["fact_id"] in fact_ids or not fact.get("statement"):
                raise ValidationError("episode facts must have unique id and non-empty statement")
            fact_ids.add(fact["fact_id"])
            if not fact.get("evidence_refs") or not set(fact["evidence_refs"]).issubset(source_ids):
                raise ValidationError("episode fact evidence references are invalid")
            if _parse_time(fact.get("available_at"), "episode.fact.available_at") > as_of:
                raise ValidationError("episode fact became public after its cutoff")
        state_ids = [state.get("microconcept_id") for state in row["microconcept_states"]]
        fact_state_ids = [state.get("microconcept_id") for state in row["fact_microconcept_states"]]
        if set(state_ids) != assets["concept_ids"] or len(state_ids) != 11 or set(fact_state_ids) != assets["concept_ids"] or len(fact_state_ids) != 11:
            raise ValidationError("episode must retain exactly the fixed 11 fact and question-conditioned microconcept states")
        if row.get("cross_domain_transfer") and not set(transfer.get("evidence_refs", [])).issubset(fact_ids):
            raise ValidationError("episode transfer rationale must cite episode facts")
        _validate_task_conditioning(row, assets["concept_ids"])
        _validate_counterfactual(row, {fact["fact_id"]: fact for fact in row["facts"]}, assets["concept_ids"])
        expected_basis = task_relevance_basis(row, row["fact_microconcept_states"], assets["concept_ids"])
        if json.dumps(row["task_relevance_basis"], ensure_ascii=False, sort_keys=True) != json.dumps(expected_basis, ensure_ascii=False, sort_keys=True):
            raise ValidationError("episode task_relevance_basis must exactly explain its fact- and question-conditioned vector")
        expected_states = question_conditioned_microconcept_states(row["fact_microconcept_states"], row, assets)
        if json.dumps(row["microconcept_states"], ensure_ascii=False, sort_keys=True) != json.dumps(expected_states, ensure_ascii=False, sort_keys=True):
            raise ValidationError("episode microconcept_states must exactly apply its question conditioning and declared counterfactual changes")
        graph = row["candidate_subgraph"]
        expected_graph = candidate_subgraph(expected_states, assets)
        if json.dumps(graph, ensure_ascii=False, sort_keys=True) != json.dumps(expected_graph, ensure_ascii=False, sort_keys=True):
            raise ValidationError("episode candidate_subgraph must be recomputed from question-conditioned microconcept states")
    for case_id, variants in groups.items():
        if len(variants) != 4 or {item["question_kind"] for item in variants} != set(QUESTION_KINDS):
            raise ValidationError(f"case {case_id} does not have exactly four V1 variants")
        if len({item["split"] for item in variants}) != 1:
            raise ValidationError("all variants of a fact pack must remain in the same split")
        for field in ("domain", "as_of", "public_available_at", "source_refs", "facts", "factual_uncertainties", "structured_evidence", "fact_microconcept_states"):
            fingerprints = {json.dumps(item.get(field), ensure_ascii=False, sort_keys=True) for item in variants}
            if len(fingerprints) != 1:
                raise ValidationError(f"all variants of a fact pack must keep {field} unchanged")
        relevance_vectors = {
            tuple(
                (state["microconcept_id"], state["task_relevance"])
                for state in sorted(item["microconcept_states"], key=lambda state: state["microconcept_id"])
            )
            for item in variants
        }
        if len(relevance_vectors) != 4:
            raise ValidationError("the four question variants must expose four distinct task_relevance vectors")
    if require_v1_size:
        if len(groups) != 160 or len(rows) != 640:
            raise ValidationError("full V1 requires exactly 160 cases and 640 episodes")
        if len({row["question"] for row in rows}) < 160:
            raise ValidationError("full V1 requires at least 160 distinct question texts")
        counts = Counter(item["split"] for item in rows)
        if counts != Counter({"train": 448, "development": 96, "sealed_test": 96}):
            raise ValidationError("full V1 split sizes are incorrect")
        domain_counts = Counter(item["domain"] for item in rows)
        if any(domain_counts[domain] != 160 for domain in DOMAINS):
            raise ValidationError("full V1 must contain 40 packs / 160 episodes per domain")


def router_view(episode: dict[str, Any]) -> dict[str, Any]:
    """Strict Router input: task text, facts, sources, identity, and outcomes stay hidden."""
    allowed = ("episode_id", "microconcept_states", "candidate_subgraph")
    view = {key: deepcopy(episode[key]) for key in allowed if key in episode}
    forbidden = {
        "question", "goal", "constraints", "facts", "structured_evidence", "source_refs", "case_id",
        "issuer_name", "ticker_at_collection", "outcome", "future_outcome",
    }
    if forbidden & set(view):
        raise AssertionError("router view accidentally exposed prohibited fields")
    return view


def teacher_view(episode: dict[str, Any]) -> dict[str, Any]:
    """Teacher-visible route packet input; task text is allowed but fact/source evidence is not."""
    view = router_view(episode)
    for field in ("question", "goal", "constraints"):
        if field in episode:
            view[field] = deepcopy(episode[field])
    forbidden = {"facts", "structured_evidence", "source_refs", "case_id", "issuer_name", "ticker_at_collection", "outcome", "future_outcome"}
    if forbidden & set(view):
        raise AssertionError("teacher view accidentally exposed prohibited evidence fields")
    return view


def model_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(model.get(key, [] if key.endswith("signals") else "")) for key in (
        "id", "name", "definition", "mechanism", "positive_task_signals", "negative_task_signals", "preconditions",
        "failure_conditions", "prediction_targets", "action_affordances",
    )}


def model_selection_contracts(candidate_model_ids: Iterable[str], assets: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Derive label-time selection contracts from read-only cards and the fixed edge index.

    The contract retains global hard requirements even when the current
    candidate subgraph was entered through another concept. It never invents a
    relation missing from the fixed cards/index.
    """
    assets = assets or load_semantic_assets()
    selected = set(candidate_model_ids)
    concepts_by_model: dict[str, list[str]] = defaultdict(list)
    for concept_id, edges in assets["edges"].items():
        for edge in edges:
            if edge["model_id"] in selected:
                concepts_by_model[edge["model_id"]].append(concept_id)
    result = []
    for model_id in sorted(selected):
        model = assets["models"][model_id]
        support = sorted(concepts_by_model[model_id])
        kinds = {concept_id: next(edge["relation"] for edge in assets["edges"][concept_id] if edge["model_id"] == model_id) for concept_id in support}
        near_miss = sorted({edge["model_id"] for concept_id in support for edge in assets["edges"].get(concept_id, []) if edge["model_id"] != model_id and edge["model_id"] in selected})
        result.append({
            "model_id": model_id,
            "primary_condition": deepcopy(model.get("positive_task_signals", [])),
            "required_microconcepts": sorted(concept_id for concept_id, relation in kinds.items() if relation == "requires"),
            "supporting_microconcepts": sorted(concept_id for concept_id, relation in kinds.items() if relation == "supports"),
            "modulators": sorted(concept_id for concept_id, relation in kinds.items() if relation == "modulates"),
            "challenge_conditions": {"microconcepts": sorted(concept_id for concept_id, relation in kinds.items() if relation == "challenges"), "card_conditions": deepcopy(model.get("negative_task_signals", [])) + deepcopy(model.get("failure_conditions", []))},
            "near_miss_models": near_miss,
            "expected_output": {"prediction_targets": deepcopy(model.get("prediction_targets", [])), "action_affordances": deepcopy(model.get("action_affordances", []))},
        })
    return result


def teacher_packet(episode: dict[str, Any], mode: str, initial_label: dict[str, Any] | None = None, assets: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create an audit-ready external packet; no API call is made by this function."""
    if mode not in {"initial", "review"}:
        raise ValidationError("teacher packet mode must be initial or review")
    assets = assets or load_semantic_assets()
    view = teacher_view(episode)
    candidate_ids = view["candidate_subgraph"].get("candidate_model_ids", [])
    packet = {
        "packet_version": "mechanism-question-v1.1.0", "mode": mode,
        "episode": view,
        "candidate_models": [model_summary(assets["models"][model_id]) for model_id in candidate_ids],
        "model_selection_contracts": model_selection_contracts(candidate_ids, assets),
        "instructions": "Produce a Weak route only. Do not infer facts outside this packet. Select one primary model or abstain; cite legal microconcepts, name every near miss with an exclusion reason, assess challenge conditions, and give a next verification step. Return only the requested JSON schema.",
    }
    if mode == "review":
        if initial_label is None:
            raise ValidationError("review packet requires an initial label")
        packet["initial_label"] = deepcopy(initial_label)
        packet["instructions"] = "Independently challenge the proposed Weak route against the selection contract. Mark accepted only when the exact route survives; otherwise use review_required and explain the violation. " + packet["instructions"]
    return packet

def _episode_lookup(episodes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values = list(episodes)
    lookup = {episode["episode_id"]: episode for episode in values}
    if len(lookup) != len(values):
        raise ValidationError("episode_id values must be unique")
    return lookup


def _label_route(label: dict[str, Any]) -> tuple[Any, tuple[str, ...]]:
    return label.get("primary_model"), tuple(sorted(label.get("auxiliary_models", [])))


def validate_teacher_label(label: dict[str, Any], episode: dict[str, Any], assets: dict[str, Any] | None = None) -> None:
    """Validate syntax, legal edge use, and selection-contract constraints for one pass."""
    assets = assets or load_semantic_assets()
    for field in ("label_id", "episode_id", "label_pass", "teacher_model", "teacher_version", "prompt_version", "label_timestamp", "confidence", "abstain", "label_tier", "cited_microconcept_ids", "cited_evidence_refs", "near_miss_models", "near_miss_exclusions", "challenge_assessment", "addressed_challenge_microconcept_ids", "review_status"):
        if field not in label or label[field] is None:
            raise ValidationError(f"teacher label missing {field}")
    if label["label_tier"] != "Weak":
        raise ValidationError("teacher labels must be explicitly tiered Weak")
    if label["episode_id"] != episode["episode_id"]:
        raise ValidationError("teacher label episode_id does not match episode")
    if label["label_pass"] not in {"initial", "review"}:
        raise ValidationError("label_pass must be initial or review")
    if not isinstance(label["teacher_model"], str) or not label["teacher_model"].strip():
        raise ValidationError("teacher_model must record the external LLM model identifier")
    if not isinstance(label["teacher_version"], str) or not label["teacher_version"].strip():
        raise ValidationError("teacher_version must record the external LLM version or deployment")
    if not isinstance(label["confidence"], (int, float)) or not 0 <= label["confidence"] <= 1:
        raise ValidationError("label confidence must be between zero and one")
    _parse_time(label["label_timestamp"], "label_timestamp")
    if label["review_status"] not in ({"proposed"} if label["label_pass"] == "initial" else {"accepted", "review_required"}):
        raise ValidationError("label review_status is inconsistent with label_pass")
    graph = episode["candidate_subgraph"]
    candidate_ids = set(graph.get("candidate_model_ids", []))
    edges = {(item["microconcept_id"], item["model_id"], item["relation"]) for item in graph.get("edges", [])}
    state_by_id = {state["microconcept_id"]: state for state in episode["microconcept_states"]}
    cited_concepts = set(label["cited_microconcept_ids"])
    if not cited_concepts or not cited_concepts.issubset(state_by_id):
        raise ValidationError("label must cite existing microconcept ids")
    cited_facts = set(label["cited_evidence_refs"])
    fact_ids = {fact["fact_id"] for fact in episode["facts"]}
    if not cited_facts.issubset(fact_ids):
        raise ValidationError("label cited_evidence_refs must point to episode fact ids")
    visible_evidence = {ref for concept_id in cited_concepts for ref in state_by_id[concept_id].get("evidence_refs", [])}
    if not cited_facts.issubset(visible_evidence):
        raise ValidationError("label may only cite evidence exposed by its cited microconcept states")
    for concept_id in cited_concepts:
        if state_by_id[concept_id]["status"] not in ACTIVE_STATUSES:
            raise ValidationError("cited microconcept must have an active state")
    primary = label.get("primary_model")
    auxiliary = label.get("auxiliary_models", [])
    near_miss = label["near_miss_models"]
    if not isinstance(auxiliary, list) or len(auxiliary) > 2 or len(set(auxiliary)) != len(auxiliary):
        raise ValidationError("auxiliary_models must contain zero to two unique models")
    if not isinstance(near_miss, list) or not near_miss:
        raise ValidationError("every label needs at least one near_miss_model")
    exclusions = label["near_miss_exclusions"]
    if not isinstance(exclusions, dict) or set(exclusions) != set(near_miss) or any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()):
        raise ValidationError("every near miss needs a non-empty exclusion reason")
    if not isinstance(label["challenge_assessment"], str) or not label["challenge_assessment"].strip():
        raise ValidationError("label needs an explicit challenge assessment")
    if label["abstain"]:
        if primary or auxiliary:
            raise ValidationError("abstaining labels cannot select primary or auxiliary models")
        _require(label.get("abstain_reason"), "abstaining label requires abstain_reason")
    else:
        if primary not in candidate_ids:
            raise ValidationError("primary model is not in the legal candidate subgraph")
        if any(model_id not in candidate_ids for model_id in auxiliary):
            raise ValidationError("auxiliary model is not in the legal candidate subgraph")
        if primary in auxiliary:
            raise ValidationError("primary model cannot also be auxiliary")
        legal_primary_edge = any(concept_id in cited_concepts and model_id == primary and relation in {"requires", "supports"} for concept_id, model_id, relation in edges)
        if not legal_primary_edge:
            raise ValidationError("selected primary must cite a legally connected requires/supports microconcept")
        for concept_id, static_edges in assets["edges"].items():
            if any(edge["model_id"] == primary and edge["relation"] == "requires" for edge in static_edges):
                if concept_id not in cited_concepts or state_by_id[concept_id]["status"] not in ACTIVE_STATUSES:
                    raise ValidationError("a required microconcept was not supported and cited for the primary model")
    selected = ({primary} if primary else set()) | set(auxiliary)
    for model_id in selected:
        if not any(concept_id in cited_concepts and connected_model == model_id for concept_id, connected_model, _ in edges):
            raise ValidationError("each selected model must cite a legally connected microconcept")
    active_challenges = {concept_id for concept_id, model_id, relation in edges if primary and model_id == primary and relation == "challenges" and state_by_id[concept_id]["status"] in ACTIVE_STATUSES}
    addressed_challenges = set(label["addressed_challenge_microconcept_ids"])
    if not addressed_challenges.issubset(state_by_id):
        raise ValidationError("addressed challenge microconcept ids must exist in the episode")
    if not active_challenges.issubset(addressed_challenges):
        raise ValidationError("active challenge microconcepts cannot be silently ignored")
    if any(model_id not in candidate_ids for model_id in near_miss):
        raise ValidationError("near_miss model is not legal for this episode")
    if selected & set(near_miss):
        raise ValidationError("near_miss model cannot be selected")
    roles = label.get("auxiliary_model_roles", {})
    if not isinstance(roles, dict) or not set(roles).issubset(auxiliary):
        raise ValidationError("auxiliary_model_roles must only assign selected auxiliary models")
    if sum(role == "epistemic_guard" for role in roles.values()) > 1:
        raise ValidationError("at most one auxiliary epistemic guard is allowed")
    _require(label.get("route_rationale") or label.get("abstain_reason"), "label needs route rationale or abstain reason")
    _require(label.get("next_verification"), "every label needs a next verification step")


def validate_teacher_labels(labels: Iterable[dict[str, Any]], episodes: Iterable[dict[str, Any]], require_complete: bool = False) -> None:
    """Validate two-pass external Weak labels and initial/review consistency."""
    assets = load_semantic_assets()
    lookup = _episode_lookup(episodes)
    values = list(labels)
    label_ids = [label.get("label_id") for label in values]
    if len(label_ids) != len(set(label_ids)):
        raise ValidationError("label_id values must be unique")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for label in values:
        if label.get("episode_id") not in lookup:
            raise ValidationError("teacher label references an unknown episode")
        validate_teacher_label(label, lookup[label["episode_id"]], assets)
        current = grouped[label["episode_id"]]
        if label["label_pass"] in current:
            raise ValidationError("only one label per pass is allowed for an episode")
        current[label["label_pass"]] = label
    for episode_id, passes in grouped.items():
        if set(passes) != {"initial", "review"}:
            raise ValidationError(f"episode {episode_id} must have both initial and review labels")
        initial, review = passes["initial"], passes["review"]
        if review["review_status"] == "accepted" and _label_route(initial) != _label_route(review):
            raise ValidationError("accepted review must retain the exact initial route")
    if require_complete and set(grouped) != set(lookup):
        missing = sorted(set(lookup) - set(grouped))
        raise ValidationError(f"full V1 requires both label passes for every episode; missing {len(missing)}")


def accepted_reviews(labels: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [label for label in labels if label.get("label_pass") == "review" and label.get("review_status") == "accepted"]


def coverage_report(episodes: Iterable[dict[str, Any]], labels: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report V1 support/coverage without relabeling unsupported models as negatives."""
    lookup = _episode_lookup(episodes)
    accepted = accepted_reviews(labels)
    models: dict[str, dict[str, Any]] = defaultdict(lambda: {"primary": 0, "total": 0, "domains": set()})
    abstain = 0
    transfers = 0
    for label in accepted:
        episode = lookup.get(label.get("episode_id"))
        if not episode:
            continue
        transfers += int(bool(episode.get("cross_domain_transfer")))
        if label.get("abstain"):
            abstain += 1
            continue
        for model_id in [label.get("primary_model")] + list(label.get("auxiliary_models", [])):
            if model_id:
                models[model_id]["total"] += 1
                models[model_id]["domains"].add(episode["domain"])
        if label.get("primary_model"):
            models[label["primary_model"]]["primary"] += 1
    asset_ids = sorted(load_semantic_assets()["models"])
    per_model = {}
    for model_id in asset_ids:
        item = models[model_id]
        supported = item["primary"] >= 20 and item["total"] >= 40 and len(item["domains"]) >= 3
        per_model[model_id] = {"primary_examples": item["primary"], "total_examples": item["total"], "domains": sorted(item["domains"]), "status": "supported" if supported else "unsupported"}
    denominator = len(accepted)
    return {
        "accepted_review_labels": denominator, "abstain_rate": abstain / denominator if denominator else 0.0,
        "cross_domain_transfer_rate": transfers / denominator if denominator else 0.0,
        "models": per_model,
    }


def validate_quality_gates(report: dict[str, Any]) -> None:
    """Apply corpus gates after review acceptance; unsupported models stay unsupported."""
    if report["accepted_review_labels"] == 0:
        raise ValidationError("no review-accepted Weak labels are available for quality-gate validation")
    if report["abstain_rate"] < 0.15:
        raise ValidationError("accepted review labels must include at least a 15% abstain rate")
    if report["cross_domain_transfer_rate"] < 0.10:
        raise ValidationError("accepted review labels must include at least a 10% cross-domain-transfer rate")

