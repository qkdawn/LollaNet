"""External two-model LLM supervision for the mechanism Router V1.1.

This module contains no routing heuristic.  The initial model receives only a
route-blind teacher packet and proposes a Weak route.  A *different* audit model
receives the packet plus the initial label and may only accept or reject it.
Both local contract validation and provenance fingerprints are mandatory.

The command intentionally refuses to use one model identifier for both passes.
It does not print API keys and stores only response fingerprints, never raw API
responses, in the audit ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .teacher_chain_v1 import (
    canonical_sha256,
    make_teacher_identity,
    require_independent_teachers,
    teacher_identity_from_label,
)
from .question_set_v1 import (
    ValidationError,
    read_jsonl,
    validate_teacher_label,
    validate_teacher_labels,
    write_jsonl,
)

INITIAL_PROMPT_VERSION = "mechanism-question-v1.1.0-initial-llm"
REVIEW_PROMPT_VERSION = "mechanism-question-v1.1.0-independent-audit-llm"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_INITIAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "confidence", "abstain", "primary_model", "auxiliary_models", "auxiliary_model_roles",
        "near_miss_models", "near_miss_exclusion_items", "cited_microconcept_ids", "cited_evidence_refs",
        "route_rationale", "challenge_assessment", "addressed_challenge_microconcept_ids",
        "next_verification", "abstain_reason",
    ],
    "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "abstain": {"type": "boolean"},
        "primary_model": {"type": ["string", "null"]},
        "auxiliary_models": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "auxiliary_model_roles": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["model_id", "role"],
                "properties": {"model_id": {"type": "string"}, "role": {"type": "string"}},
            }, "maxItems": 2,
        },
        "near_miss_models": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "near_miss_exclusion_items": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["model_id", "reason"],
                "properties": {"model_id": {"type": "string"}, "reason": {"type": "string"}},
            }, "minItems": 1,
        },
        "cited_microconcept_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "cited_evidence_refs": {"type": "array", "items": {"type": "string"}},
        "route_rationale": {"type": "string"},
        "challenge_assessment": {"type": "string"},
        "addressed_challenge_microconcept_ids": {"type": "array", "items": {"type": "string"}},
        "next_verification": {"type": "string"},
        "abstain_reason": {"type": "string"},
    },
}

_REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["review_status", "confidence", "audit_rationale"],
    "properties": {
        "review_status": {"type": "string", "enum": ["accepted", "review_required"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "audit_rationale": {"type": "string"},
    },
}

_INITIAL_SYSTEM = """You are the initial, independent Weak-label teacher for a fixed semantic routing task.
Use only the supplied packet.  Do not infer outside facts, do not alter the candidate graph, and do not treat labels as Gold truth.
A non-abstaining route must select exactly one legal primary and zero to two legal auxiliaries.  Cite active legal microconcepts and every cited evidence reference must be exposed by them.  Return only JSON conforming to the supplied schema."""

_REVIEW_SYSTEM = """You are an independent audit model, not the initial route author.
Use only the review packet.  Test the initial route against legal edges, hard requires conditions, active challenge conditions, task-conditioned microconcept state, and evidence boundaries.  Return accepted only when the exact proposed route survives.  Otherwise return review_required.  Do not propose a replacement route.  Return only JSON conforming to the supplied schema."""


def _response_text(client: Any, *, model: str, system: str, packet: dict[str, Any], schema_name: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the Responses API and return parsed JSON plus non-sensitive audit metadata."""
    response = client.responses.create(
        model=model,
        store=False,
        instructions=system,
        input=_canonical_json(packet),
        text={"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
    )
    raw = getattr(response, "output_text", None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("external LLM returned no structured output text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError("external LLM output is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValidationError("external LLM output must be a JSON object")
    return payload, {
        "api_response_id": getattr(response, "id", None),
        "request_fingerprint_sha256": _sha256(packet),
        "raw_response_fingerprint_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _pairs_to_dict(items: Any, *, key: str, value: str, field: str) -> dict[str, str]:
    if not isinstance(items, list):
        raise ValidationError(f"{field} must be a list")
    result: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(key), str) or not item[key] or not isinstance(item.get(value), str) or not item[value].strip():
            raise ValidationError(f"{field} contains an invalid item")
        if item[key] in result:
            raise ValidationError(f"{field} contains duplicate {key}")
        result[item[key]] = item[value]
    return result


def initial_label_from_payload(
    episode: dict[str, Any], payload: dict[str, Any], *, model: str, teacher_version: str, audit: dict[str, Any],
    provider: str = "openai", deployment: str = "",
) -> dict[str, Any]:
    """Normalize one independently generated initial response and validate it locally."""
    identity = make_teacher_identity(provider=provider, model=model, deployment=deployment, version=teacher_version)
    roles = _pairs_to_dict(payload.get("auxiliary_model_roles", []), key="model_id", value="role", field="auxiliary_model_roles")
    exclusions = _pairs_to_dict(payload.get("near_miss_exclusion_items", []), key="model_id", value="reason", field="near_miss_exclusion_items")
    label = {
        "label_id": f"{episode['episode_id']}::initial", "episode_id": episode["episode_id"], "label_pass": "initial",
        "teacher_model": identity["model"], "teacher_version": identity["version"], "teacher_identity": identity,
        "prompt_version": INITIAL_PROMPT_VERSION,
        "label_timestamp": _timestamp(), "label_tier": "Weak", "confidence": payload.get("confidence"),
        "abstain": payload.get("abstain"), "primary_model": payload.get("primary_model"),
        "auxiliary_models": payload.get("auxiliary_models"), "auxiliary_model_roles": roles,
        "near_miss_models": payload.get("near_miss_models"), "near_miss_exclusions": exclusions,
        "cited_microconcept_ids": payload.get("cited_microconcept_ids"), "cited_evidence_refs": payload.get("cited_evidence_refs"),
        "route_rationale": payload.get("route_rationale"), "challenge_assessment": payload.get("challenge_assessment"),
        "addressed_challenge_microconcept_ids": payload.get("addressed_challenge_microconcept_ids"),
        "next_verification": payload.get("next_verification"), "abstain_reason": payload.get("abstain_reason"),
        "review_status": "proposed", "generation_method": "external independent LLM initial route",
        "provenance": {"teacher_pass": "initial", "teacher_identity": identity, **audit},
    }
    validate_teacher_label(label, episode)
    return label


def review_label_from_payload(
    episode: dict[str, Any], initial: dict[str, Any], payload: dict[str, Any], *, model: str, teacher_version: str, audit: dict[str, Any],
    provider: str = "openai", deployment: str = "",
) -> dict[str, Any]:
    """Create an audit-only review row; only an independent model can accept it."""
    initial_identity = teacher_identity_from_label(initial)
    review_identity = make_teacher_identity(provider=provider, model=model, deployment=deployment, version=teacher_version)
    require_independent_teachers(initial_identity, review_identity)
    if payload.get("review_status") not in {"accepted", "review_required"}:
        raise ValidationError("audit LLM must return accepted or review_required")
    rationale = payload.get("audit_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValidationError("audit LLM must provide a non-empty audit rationale")
    initial_hash = canonical_sha256(initial)
    review = deepcopy(initial)
    review.update({
        "label_id": f"{episode['episode_id']}::review", "label_pass": "review",
        "teacher_model": review_identity["model"], "teacher_version": review_identity["version"],
        "teacher_identity": review_identity, "prompt_version": REVIEW_PROMPT_VERSION, "label_timestamp": _timestamp(),
        "confidence": payload.get("confidence"), "review_status": payload["review_status"],
        "generation_method": "external independent LLM audit; exact initial route retained for accepted-review gating",
        "audit_rationale": rationale, "initial_label_sha256": initial_hash,
        "provenance": {"teacher_pass": "review", "teacher_identity": review_identity,
                       "initial_teacher_identity": initial_identity, "initial_label_sha256": initial_hash, **audit},
    })
    validate_teacher_label(review, episode)
    return review



def _ledger_entry(
    label: dict[str, Any], *, mode: str, audit: dict[str, Any], initial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "episode_id": label["episode_id"],
        "label_id": label["label_id"],
        "mode": mode,
        "teacher_identity": label["teacher_identity"],
        "teacher_model": label["teacher_model"],
        "teacher_version": label["teacher_version"],
        "prompt_version": label["prompt_version"],
        "label_sha256": canonical_sha256(label),
        **audit,
    }
    if mode == "review":
        if initial is None:
            raise ValidationError("review ledger entry requires the initial label")
        entry["initial_label_sha256"] = canonical_sha256(initial)
    return entry


def _load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {row.get("episode_id"): row for row in rows}
    if len(result) != len(rows) or None in result:
        raise ValidationError("episodes must have unique, non-empty episode_id values")
    return result


def _run_initial(
    packets: Iterable[dict[str, Any]], episodes: dict[str, dict[str, Any]], *, model: str, teacher_version: str, client: Any,
    provider: str = "openai", deployment: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels, ledger = [], []
    for packet in packets:
        if packet.get("mode") != "initial":
            raise ValidationError("initial run received a non-initial packet")
        episode_id = packet.get("episode", {}).get("episode_id")
        episode = episodes.get(episode_id)
        if episode is None:
            raise ValidationError("teacher packet references an unknown episode")
        payload, audit = _response_text(client, model=model, system=_INITIAL_SYSTEM, packet=packet, schema_name="mechanism_route_initial", schema=_INITIAL_RESPONSE_SCHEMA)
        label = initial_label_from_payload(episode, payload, model=model, teacher_version=teacher_version, audit=audit, provider=provider, deployment=deployment)
        labels.append(label)
        ledger.append(_ledger_entry(label, mode="initial", audit=audit))
    return labels, ledger


def _run_review(
    packets: Iterable[dict[str, Any]], episodes: dict[str, dict[str, Any]], *, model: str, teacher_version: str, client: Any,
    provider: str = "openai", deployment: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels, ledger = [], []
    for packet in packets:
        if packet.get("mode") != "review":
            raise ValidationError("review run received a non-review packet")
        episode_id = packet.get("episode", {}).get("episode_id")
        episode, initial = episodes.get(episode_id), packet.get("initial_label")
        if episode is None or not isinstance(initial, dict):
            raise ValidationError("review packet needs a known episode and an initial label")
        payload, audit = _response_text(client, model=model, system=_REVIEW_SYSTEM, packet=packet, schema_name="mechanism_route_audit", schema=_REVIEW_RESPONSE_SCHEMA)
        label = review_label_from_payload(episode, initial, payload, model=model, teacher_version=teacher_version, audit=audit, provider=provider, deployment=deployment)
        labels.append(label)
        ledger.append(_ledger_entry(label, mode="review", audit=audit, initial=initial))
    return labels, ledger


def _client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValidationError("OPENAI_API_KEY is not set; prepare packets now and run external labeling only after configuring a key")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ValidationError("openai Python package is required for external LLM labeling") from error
    return OpenAI()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("initial", "review"))
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-ledger", required=True, type=Path)
    parser.add_argument("--provider", default="openai", help="Normalized provider identity, for example openai.")
    parser.add_argument("--model", required=True, help="External model identifier. The audit provider/model identity must differ from the initial identity.")
    parser.add_argument("--deployment", default="", help="Optional deployment or endpoint identity.")
    parser.add_argument("--teacher-version", default="api-unpinned", help="Pinned model/deployment/version recorded in each label.")
    args = parser.parse_args()
    packets, episodes, client = read_jsonl(args.packets), _load_episodes(args.episodes), _client()
    if args.mode == "initial":
        labels, ledger = _run_initial(packets, episodes, model=args.model, teacher_version=args.teacher_version, client=client, provider=args.provider, deployment=args.deployment)
    else:
        labels, ledger = _run_review(packets, episodes, model=args.model, teacher_version=args.teacher_version, client=client, provider=args.provider, deployment=args.deployment)
        cohort_episodes = [episodes[packet["episode"]["episode_id"]] for packet in packets]
        validate_teacher_labels([packet["initial_label"] for packet in packets] + labels, cohort_episodes, require_complete=True)
    write_jsonl(args.output, labels)
    write_jsonl(args.audit_ledger, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
