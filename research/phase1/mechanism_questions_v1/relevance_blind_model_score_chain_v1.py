"""Tamper-evident relevance-blind dual-teacher all-model score chain.

The teacher-visible semantic payload is intentionally limited to raw facts, the
question, and the fixed twenty model cards. Router-derived relevance states,
microconcept states, candidate graphs, and selection contracts are never
materialized in a packet. Aggregation reuses the existing per-model mean/mask
implementation so teacher visibility is the experimental difference.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .adjudication_chain_v1 import load_verified_parent
from .model_score_chain_v1 import (
    DEFAULT_MAX_SCORE_GAP, MODEL_SCORE_POLICY_VERSION, MODEL_SCORE_SCALE_MAX,
    MODEL_SCORE_SCALE_MIN, _aggregate_pair, default_score_policy,
)
from .question_set_v1 import load_semantic_assets, model_summary, read_jsonl, write_jsonl
from .teacher_chain_v1 import (
    MANIFEST_FILENAME, TeacherChainError, canonical_sha256, file_descriptor,
    file_sha256, normalize_teacher_identity, teacher_identity_from_label,
    teacher_independence_key,
)

PACKAGE = Path(__file__).resolve().parent
VISIBILITY_CONTRACT_PATH = PACKAGE / "contracts" / "relevance_blind_model_score_v1_contract.yaml"
BLIND_SCORE_MANIFEST_FILENAME = "relevance_blind_model_score_manifest.json"
BLIND_SCORE_MANIFEST_SCHEMA_VERSION = "relevance-blind-thought-model-score-manifest-v1"
BLIND_SCORE_LABEL_SCHEMA_VERSION = "relevance-blind-thought-model-score-label-v1"
BLIND_SCORE_PACKET_VERSION = "mechanism-question-v1.2.0-relevance-blind-all-model-scores"
BLIND_SCORE_PROMPT_VERSION = "mechanism-question-v1.2.0-relevance-blind-dual-all-model-scores"
BLIND_VISIBILITY_CONTRACT_SCHEMA_VERSION = "relevance-blind-model-score-contract-v1"
REQUIRED_FILE_ROLES = frozenset({
    "score_packets", "teacher_scores", "score_audit_ledger", "aggregated_scores",
    "policy_contract", "prompt_contract",
})


class RelevanceBlindModelScoreChainError(TeacherChainError):
    """Raised when relevance-blind supervision is incomplete or leaks hints."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _unique(rows: Iterable[dict[str, Any]], *, key_fields: tuple[str, ...], description: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(not isinstance(value, str) or not value for value in key):
            raise RelevanceBlindModelScoreChainError(f"{description} row is missing {'/'.join(key_fields)}")
        if key in result:
            raise RelevanceBlindModelScoreChainError(f"duplicate {description} row for {key}")
        result[key] = row
    return result


def load_visibility_contract() -> dict[str, Any]:
    try:
        contract = yaml.safe_load(VISIBILITY_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RelevanceBlindModelScoreChainError("cannot load relevance-blind visibility contract") from error
    if not isinstance(contract, dict) or contract.get("schema_version") != BLIND_VISIBILITY_CONTRACT_SCHEMA_VERSION:
        raise RelevanceBlindModelScoreChainError("unsupported relevance-blind visibility contract")
    if contract.get("sealed_test_visible") is not False:
        raise RelevanceBlindModelScoreChainError("relevance-blind contract must keep sealed_test hidden")
    return contract


def default_blind_score_prompt_contract() -> dict[str, Any]:
    return {
        "schema_version": "relevance-blind-thought-model-score-prompt-contract-v1",
        "prompt_version": BLIND_SCORE_PROMPT_VERSION,
        "packet_version": BLIND_SCORE_PACKET_VERSION,
        "visibility_condition": "raw_facts_question_and_twenty_model_cards_only",
        "decision_subject": "independent_applicability_of_every_thought_model",
        "required_score_count": 20,
        "score_scale": {
            "0": "not applicable or prerequisites contradicted",
            "1": "weak background relevance; does not materially change the judgment",
            "2": "useful supporting lens; changes at least one interpretation or check",
            "3": "strongly applicable; materially changes explanation, prediction, evidence test, or action",
            "4": "essential; omitting it would make the judgment materially defective",
        },
        "scoring_rules": [
            "Score every model independently; do not force scores to sum to a constant.",
            "Multiple models may receive high scores for the same Episode.",
            "Use only the raw facts, question, and model cards in the saved packet.",
            "Do not reconstruct or assume task_relevance, microconcept states, or a candidate graph.",
            "A high score requires satisfied model preconditions and a concrete change to explanation, prediction, evidence discrimination, action, or stopping condition.",
            "Return exactly one numeric score for each supplied model_id.",
        ],
        "sealed_test_visible": False,
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def relevance_blind_raw_facts(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Return unconditioned facts without sources, identity, or route state."""
    required = ("facts", "factual_uncertainties", "structured_evidence")
    if any(field not in episode for field in required):
        raise RelevanceBlindModelScoreChainError("Episode is missing the raw factual payload")
    return {field: deepcopy(episode[field]) for field in required}


def validate_relevance_blind_packet(packet: Mapping[str, Any]) -> None:
    boundary = load_visibility_contract().get("packet_boundary")
    if not isinstance(boundary, dict):
        raise RelevanceBlindModelScoreChainError("visibility contract has no packet boundary")
    if set(packet) != set(boundary.get("exact_top_level_fields", [])):
        raise RelevanceBlindModelScoreChainError("packet top-level fields do not match the allowlist")
    raw_facts = packet.get("raw_facts")
    if not isinstance(raw_facts, dict) or set(raw_facts) != set(boundary.get("raw_fact_fields", [])):
        raise RelevanceBlindModelScoreChainError("raw_facts fields do not match the allowlist")
    leaked = set(boundary.get("prohibited_recursive_fields", [])).intersection(_walk_keys(packet))
    if leaked:
        raise RelevanceBlindModelScoreChainError(f"packet leaked prohibited fields: {', '.join(sorted(leaked))}")
    model_ids, models = packet.get("model_ids"), packet.get("models")
    if not isinstance(model_ids, list) or len(model_ids) != 20 or len(set(model_ids)) != 20:
        raise RelevanceBlindModelScoreChainError("packet must expose exactly 20 model ids")
    if not isinstance(models, list) or len(models) != 20:
        raise RelevanceBlindModelScoreChainError("packet must expose exactly 20 model cards")
    if [model.get("id") for model in models if isinstance(model, dict)] != model_ids:
        raise RelevanceBlindModelScoreChainError("model cards must follow model_ids exactly")
    expected_attestation = {
        "condition": "relevance_blind", "task_relevance_hidden": True,
        "microconcept_states_hidden": True, "candidate_graph_hidden": True,
        "selection_contracts_hidden": True,
    }
    if packet.get("visibility_attestation") != expected_attestation:
        raise RelevanceBlindModelScoreChainError("visibility attestation is invalid")


def _expected_packet(episode: Mapping[str, Any], *, source_hash: str, model_ids: list[str], model_cards: list[dict[str, Any]], prompt_contract: Mapping[str, Any]) -> dict[str, Any]:
    question = episode.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RelevanceBlindModelScoreChainError("Episode question must be non-empty")
    return {
        "packet_version": BLIND_SCORE_PACKET_VERSION,
        "prompt_version": BLIND_SCORE_PROMPT_VERSION,
        "episode_id": episode["episode_id"],
        "source_manifest_sha256": source_hash,
        "episode_sha256": canonical_sha256(episode),
        "raw_facts": relevance_blind_raw_facts(episode),
        "question": question,
        "model_ids": model_ids,
        "models": model_cards,
        "score_rubric": prompt_contract["score_scale"],
        "instructions": "Score all 20 models independently on the anchored 0..4 scale using only the raw facts, question, and model cards. Do not infer hidden task relevance, microconcept states, or a candidate graph. Multiple models may be high. Return only the requested JSON object.",
        "visibility_attestation": {
            "condition": "relevance_blind", "task_relevance_hidden": True,
            "microconcept_states_hidden": True, "candidate_graph_hidden": True,
            "selection_contracts_hidden": True,
        },
    }


def build_relevance_blind_model_score_packets(*, source_manifest_path: Path, dataset_root: Path, output_dir: Path, max_score_gap: float = DEFAULT_MAX_SCORE_GAP) -> dict[str, Any]:
    if float(max_score_gap) != DEFAULT_MAX_SCORE_GAP:
        raise RelevanceBlindModelScoreChainError("relevance-blind v1 fixes max_score_gap at 1.0")
    source_manifest, verified, _ = load_verified_parent(source_manifest_path, dataset_root=dataset_root)
    episodes = sorted(verified["episodes"], key=lambda row: row["episode_id"])
    if any(episode.get("split") == "sealed_test" for episode in episodes):
        raise RelevanceBlindModelScoreChainError("preparation refuses sealed_test Episodes")
    assets = load_semantic_assets()
    model_ids = sorted(assets["models"])
    if len(model_ids) != 20:
        raise RelevanceBlindModelScoreChainError("fixed semantic assets must contain exactly 20 models")
    model_cards = [model_summary(assets["models"][model_id]) for model_id in model_ids]
    source_hash = file_sha256(Path(source_manifest_path))
    prompt = default_blind_score_prompt_contract()
    packets = [_expected_packet(episode, source_hash=source_hash, model_ids=model_ids, model_cards=model_cards, prompt_contract=prompt) for episode in episodes]
    for packet in packets:
        validate_relevance_blind_packet(packet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "relevance_blind_model_score_packets.jsonl"
    policy_path = output_dir / "relevance_blind_model_score_policy.json"
    prompt_path = output_dir / "relevance_blind_model_score_prompt_contract.json"
    for path in (packet_path, policy_path, prompt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite relevance-blind artifact: {path}")
    write_jsonl(packet_path, packets)
    _write_json(policy_path, default_score_policy(max_score_gap=max_score_gap))
    _write_json(prompt_path, prompt)
    return {
        "source_manifest": source_manifest, "source_manifest_sha256": source_hash,
        "episode_ids": [row["episode_id"] for row in episodes], "model_ids": model_ids,
        "packets": packets, "packet_path": packet_path, "policy_path": policy_path,
        "prompt_contract_path": prompt_path,
    }


def make_relevance_blind_model_score_label(*, packet: Mapping[str, Any], slot: str, teacher_identity: Mapping[str, Any], scores: Mapping[str, float], confidence: float, rationale: str, raw_response: Any, api_response_id: str, run_id: str) -> dict[str, Any]:
    validate_relevance_blind_packet(packet)
    if slot not in {"a", "b"}:
        raise RelevanceBlindModelScoreChainError("teacher slot must be a or b")
    identity = normalize_teacher_identity(teacher_identity)
    model_ids = packet["model_ids"]
    if set(scores) != set(model_ids):
        raise RelevanceBlindModelScoreChainError("teacher scores must cover exactly every packet model_id")
    normalized_scores: dict[str, float] = {}
    for model_id in sorted(model_ids):
        score = scores[model_id]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not MODEL_SCORE_SCALE_MIN <= float(score) <= MODEL_SCORE_SCALE_MAX:
            raise RelevanceBlindModelScoreChainError(f"invalid score for {model_id}")
        normalized_scores[model_id] = float(score)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise RelevanceBlindModelScoreChainError("confidence must be between zero and one")
    if not isinstance(rationale, str) or not rationale.strip():
        raise RelevanceBlindModelScoreChainError("score label requires a non-empty rationale")
    return {
        "schema_version": BLIND_SCORE_LABEL_SCHEMA_VERSION,
        "label_id": f"relevance-blind-model-score::{packet['episode_id']}::{slot}",
        "episode_id": packet["episode_id"], "teacher_slot": slot,
        "teacher_identity": identity, "teacher_model": identity["model"],
        "teacher_version": identity["version"], "prompt_version": packet["prompt_version"],
        "label_timestamp": _timestamp(), "label_tier": "Weak",
        "visibility_condition": "relevance_blind", "confidence": float(confidence),
        "model_scores": normalized_scores, "rationale": rationale.strip(),
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "episode_sha256": packet["episode_sha256"],
        "packet_sha256": canonical_sha256(packet),
        "raw_response_fingerprint_sha256": canonical_sha256(raw_response),
        "api_response_id": str(api_response_id), "run_id": str(run_id),
    }


def relevance_blind_model_score_ledger_entry(label: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": label["episode_id"], "label_id": label["label_id"],
        "mode": "relevance_blind_all_model_scores", "visibility_condition": "relevance_blind",
        "teacher_slot": label["teacher_slot"], "teacher_identity": label["teacher_identity"],
        "teacher_model": label["teacher_model"], "teacher_version": label["teacher_version"],
        "prompt_version": label["prompt_version"], "label_sha256": canonical_sha256(label),
        "source_manifest_sha256": label["source_manifest_sha256"],
        "episode_sha256": label["episode_sha256"], "packet_sha256": label["packet_sha256"],
        "raw_response_fingerprint_sha256": label["raw_response_fingerprint_sha256"],
        "api_response_id": label["api_response_id"], "run_id": label["run_id"],
    }


def _validate_label(label: Mapping[str, Any], *, packet: Mapping[str, Any], source_hash: str) -> dict[str, str]:
    if label.get("schema_version") != BLIND_SCORE_LABEL_SCHEMA_VERSION or label.get("label_tier") != "Weak":
        raise RelevanceBlindModelScoreChainError("unsupported relevance-blind model score label")
    if label.get("visibility_condition") != "relevance_blind":
        raise RelevanceBlindModelScoreChainError("score label visibility condition mismatch")
    slot = label.get("teacher_slot")
    if slot not in {"a", "b"}:
        raise RelevanceBlindModelScoreChainError("invalid teacher slot")
    expected_label_id = f"relevance-blind-model-score::{packet['episode_id']}::{slot}"
    if label.get("label_id") != expected_label_id or label.get("episode_id") != packet["episode_id"]:
        raise RelevanceBlindModelScoreChainError("score label identity mismatch")
    if label.get("prompt_version") != BLIND_SCORE_PROMPT_VERSION:
        raise RelevanceBlindModelScoreChainError("score label prompt version mismatch")
    if label.get("source_manifest_sha256") != source_hash or label.get("episode_sha256") != packet["episode_sha256"]:
        raise RelevanceBlindModelScoreChainError("score label source binding mismatch")
    if label.get("packet_sha256") != canonical_sha256(packet):
        raise RelevanceBlindModelScoreChainError("score label packet binding mismatch")
    scores = label.get("model_scores")
    if not isinstance(scores, dict) or set(scores) != set(packet["model_ids"]):
        raise RelevanceBlindModelScoreChainError("score label must cover every model exactly once")
    for model_id, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not MODEL_SCORE_SCALE_MIN <= float(score) <= MODEL_SCORE_SCALE_MAX:
            raise RelevanceBlindModelScoreChainError(f"invalid score for {model_id}")
    confidence = label.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise RelevanceBlindModelScoreChainError("invalid score-label confidence")
    return teacher_identity_from_label(label)


def validate_relevance_blind_model_score_artifacts(*, source_manifest_path: Path, dataset_root: Path, packets: list[dict[str, Any]], labels: list[dict[str, Any]], ledger: list[dict[str, Any]], policy: Mapping[str, Any], prompt_contract: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest, verified, _ = load_verified_parent(source_manifest_path, dataset_root=dataset_root)
    episodes = {row["episode_id"]: row for row in verified["episodes"]}
    if any(episode.get("split") == "sealed_test" for episode in episodes.values()):
        raise RelevanceBlindModelScoreChainError("chain refuses sealed_test Episodes")
    try:
        expected_policy = default_score_policy(max_score_gap=float(policy.get("max_score_gap_for_supervision", -1)))
    except (TypeError, ValueError) as error:
        raise RelevanceBlindModelScoreChainError("invalid model score policy") from error
    if dict(policy) != expected_policy or float(policy["max_score_gap_for_supervision"]) != DEFAULT_MAX_SCORE_GAP:
        raise RelevanceBlindModelScoreChainError("policy must use the fixed relevance-blind v1 gap threshold of 1.0")
    if dict(prompt_contract) != default_blind_score_prompt_contract():
        raise RelevanceBlindModelScoreChainError("prompt contract does not match")
    source_hash = file_sha256(Path(source_manifest_path))
    assets = load_semantic_assets()
    expected_models = sorted(assets["models"])
    expected_cards = [model_summary(assets["models"][model_id]) for model_id in expected_models]
    packet_by_id = _unique(packets, key_fields=("episode_id",), description="relevance-blind score packet")
    if {key[0] for key in packet_by_id} != set(episodes):
        raise RelevanceBlindModelScoreChainError("score packets must cover every source Episode")
    for episode_id, episode in episodes.items():
        expected_packet = _expected_packet(episode, source_hash=source_hash, model_ids=expected_models, model_cards=expected_cards, prompt_contract=prompt_contract)
        validate_relevance_blind_packet(packet_by_id[(episode_id,)])
        if packet_by_id[(episode_id,)] != expected_packet:
            raise RelevanceBlindModelScoreChainError(f"score packet mismatch for {episode_id}")
    label_by_key = _unique(labels, key_fields=("episode_id", "teacher_slot"), description="relevance-blind score label")
    expected_keys = {(episode_id, slot) for episode_id in episodes for slot in ("a", "b")}
    if set(label_by_key) != expected_keys:
        raise RelevanceBlindModelScoreChainError("score labels must contain exactly two slots for every Episode")
    ledger_by_key = _unique(ledger, key_fields=("episode_id", "teacher_slot"), description="relevance-blind score ledger")
    if set(ledger_by_key) != expected_keys:
        raise RelevanceBlindModelScoreChainError("score ledger must cover every teacher label")
    identities_by_slot: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key in sorted(expected_keys):
        episode_id, slot = key
        label = label_by_key[key]
        identity = _validate_label(label, packet=packet_by_id[(episode_id,)], source_hash=source_hash)
        identities_by_slot[slot].add(teacher_independence_key(identity))
        if dict(ledger_by_key[key]) != relevance_blind_model_score_ledger_entry(label):
            raise RelevanceBlindModelScoreChainError("score audit ledger does not exactly bind its label")
    if any(len(values) != 1 for values in identities_by_slot.values()) or identities_by_slot["a"] == identities_by_slot["b"]:
        raise RelevanceBlindModelScoreChainError("teacher slots must be stable and use distinct provider/model identities")
    aggregates = []
    for episode_id in sorted(episodes):
        aggregate = _aggregate_pair(
            [label_by_key[(episode_id, "a")], label_by_key[(episode_id, "b")]],
            max_score_gap=float(policy["max_score_gap_for_supervision"]),
        )
        if aggregate["supervised_model_count"]:
            aggregates.append(aggregate)
    return {
        "source_manifest": source_manifest, "source_manifest_sha256": source_hash,
        "episodes": list(episodes.values()), "episode_ids": sorted(episodes),
        "model_ids": expected_models, "aggregates": aggregates,
        "aggregated_episode_ids": [row["episode_id"] for row in aggregates],
        "fully_disputed_episode_ids": sorted(set(episodes) - {row["episode_id"] for row in aggregates}),
        "teacher_identities": {slot: [list(value) for value in sorted(values)] for slot, values in sorted(identities_by_slot.items())},
    }


def finalize_relevance_blind_model_scores(*, source_manifest_path: Path, dataset_root: Path, packets_path: Path, labels_path: Path, audit_ledger_path: Path, policy_path: Path, prompt_contract_path: Path, output_dir: Path) -> dict[str, Any]:
    packets = read_jsonl(Path(packets_path))
    labels = read_jsonl(Path(labels_path))
    ledger = read_jsonl(Path(audit_ledger_path))
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    prompt = json.loads(Path(prompt_contract_path).read_text(encoding="utf-8"))
    summary = validate_relevance_blind_model_score_artifacts(
        source_manifest_path=source_manifest_path, dataset_root=dataset_root,
        packets=packets, labels=labels, ledger=ledger, policy=policy,
        prompt_contract=prompt,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "aggregated_relevance_blind_model_scores.jsonl"
    manifest_path = output_dir / BLIND_SCORE_MANIFEST_FILENAME
    if aggregate_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite finalized relevance-blind score artifacts")
    write_jsonl(aggregate_path, summary["aggregates"])
    paths_and_rows = {
        "score_packets": (Path(packets_path), len(packets)),
        "teacher_scores": (Path(labels_path), len(labels)),
        "score_audit_ledger": (Path(audit_ledger_path), len(ledger)),
        "aggregated_scores": (aggregate_path, len(summary["aggregates"])),
        "policy_contract": (Path(policy_path), None),
        "prompt_contract": (Path(prompt_contract_path), None),
    }
    source_path = Path(source_manifest_path).resolve()
    supervised_dimensions = sum(row["supervised_model_count"] for row in summary["aggregates"])
    total_dimensions = len(summary["episode_ids"]) * len(summary["model_ids"])
    manifest = {
        "schema_version": BLIND_SCORE_MANIFEST_SCHEMA_VERSION,
        "created_at": _timestamp(), "visibility_condition": "relevance_blind",
        "visibility_contract": {
            "schema_version": BLIND_VISIBILITY_CONTRACT_SCHEMA_VERSION,
            "sha256": file_sha256(VISIBILITY_CONTRACT_PATH),
        },
        "source_chain": {
            "schema_version": summary["source_manifest"]["schema_version"],
            "path": Path(os.path.relpath(source_path, output_dir.resolve())).as_posix(),
            "bytes": source_path.stat().st_size, "sha256": summary["source_manifest_sha256"],
        },
        "policy_version": policy["policy_version"], "prompt_version": prompt["prompt_version"],
        "files": {role: file_descriptor(path, relative_to=output_dir, rows=rows) for role, (path, rows) in paths_and_rows.items()},
        "episode_ids": summary["episode_ids"],
        "aggregated_episode_ids": summary["aggregated_episode_ids"],
        "fully_disputed_episode_ids": summary["fully_disputed_episode_ids"],
        "model_ids": summary["model_ids"], "teacher_identities": summary["teacher_identities"],
        "max_score_gap_for_supervision": policy["max_score_gap_for_supervision"],
        "supervised_dimensions": supervised_dimensions, "total_dimensions": total_dimensions,
        "supervised_dimension_rate": supervised_dimensions / total_dimensions if total_dimensions else 0.0,
        "label_tier": "Weak",
    }
    _write_json(manifest_path, manifest)
    return manifest


def verify_relevance_blind_model_score_manifest(manifest_path: Path, *, dataset_root: Path, expected_aggregate_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RelevanceBlindModelScoreChainError("relevance-blind model scores require a manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RelevanceBlindModelScoreChainError("manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BLIND_SCORE_MANIFEST_SCHEMA_VERSION:
        raise RelevanceBlindModelScoreChainError("unsupported relevance-blind manifest schema")
    if manifest.get("visibility_condition") != "relevance_blind":
        raise RelevanceBlindModelScoreChainError("manifest visibility condition mismatch")
    expected_visibility = {
        "schema_version": BLIND_VISIBILITY_CONTRACT_SCHEMA_VERSION,
        "sha256": file_sha256(VISIBILITY_CONTRACT_PATH),
    }
    if manifest.get("visibility_contract") != expected_visibility:
        raise RelevanceBlindModelScoreChainError("visibility contract hash binding failed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILE_ROLES:
        raise RelevanceBlindModelScoreChainError("manifest must bind every artifact exactly once")
    root = Path(dataset_root).resolve()
    paths: dict[str, Path] = {}
    for role in sorted(REQUIRED_FILE_ROLES):
        descriptor = files[role]
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise RelevanceBlindModelScoreChainError(f"descriptor {role} must use a relative path")
        path = (manifest_path.parent / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RelevanceBlindModelScoreChainError(f"file {role} escapes dataset root") from error
        if not path.is_file() or descriptor.get("bytes") != path.stat().st_size or descriptor.get("sha256") != file_sha256(path):
            raise RelevanceBlindModelScoreChainError(f"file integrity mismatch for {role}")
        paths[role] = path
    if expected_aggregate_path is not None and paths["aggregated_scores"] != Path(expected_aggregate_path).resolve():
        raise RelevanceBlindModelScoreChainError("aggregated score path does not match the expected path")
    source = manifest.get("source_chain")
    if not isinstance(source, dict) or set(source) != {"schema_version", "path", "bytes", "sha256"}:
        raise RelevanceBlindModelScoreChainError("invalid source_chain descriptor")
    relative = source.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RelevanceBlindModelScoreChainError("source manifest path must be relative")
    source_path = (manifest_path.parent / relative).resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise RelevanceBlindModelScoreChainError("source manifest escapes dataset root") from error
    if source_path.name != MANIFEST_FILENAME or not source_path.is_file():
        raise RelevanceBlindModelScoreChainError("source teacher-chain manifest is missing")
    if source.get("bytes") != source_path.stat().st_size or source.get("sha256") != file_sha256(source_path):
        raise RelevanceBlindModelScoreChainError("source manifest hash binding failed")
    packets = read_jsonl(paths["score_packets"])
    labels = read_jsonl(paths["teacher_scores"])
    ledger = read_jsonl(paths["score_audit_ledger"])
    aggregates = read_jsonl(paths["aggregated_scores"])
    policy = json.loads(paths["policy_contract"].read_text(encoding="utf-8"))
    prompt = json.loads(paths["prompt_contract"].read_text(encoding="utf-8"))
    summary = validate_relevance_blind_model_score_artifacts(
        source_manifest_path=source_path, dataset_root=dataset_root,
        packets=packets, labels=labels, ledger=ledger, policy=policy,
        prompt_contract=prompt,
    )
    if [canonical_sha256(row) for row in aggregates] != [canonical_sha256(row) for row in summary["aggregates"]]:
        raise RelevanceBlindModelScoreChainError("aggregated scores are not deterministic policy output")
    supervised_dimensions = sum(row["supervised_model_count"] for row in aggregates)
    total_dimensions = len(summary["episode_ids"]) * len(summary["model_ids"])
    expected_fields = {
        "episode_ids": summary["episode_ids"],
        "aggregated_episode_ids": summary["aggregated_episode_ids"],
        "fully_disputed_episode_ids": summary["fully_disputed_episode_ids"],
        "model_ids": summary["model_ids"], "teacher_identities": summary["teacher_identities"],
        "max_score_gap_for_supervision": policy["max_score_gap_for_supervision"],
        "supervised_dimensions": supervised_dimensions, "total_dimensions": total_dimensions,
        "supervised_dimension_rate": supervised_dimensions / total_dimensions if total_dimensions else 0.0,
        "policy_version": MODEL_SCORE_POLICY_VERSION, "prompt_version": BLIND_SCORE_PROMPT_VERSION,
        "label_tier": "Weak",
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise RelevanceBlindModelScoreChainError(f"manifest {field} does not match verified artifacts")
    for role, path in paths.items():
        expected_rows = files[role].get("rows")
        if expected_rows is not None and expected_rows != len(read_jsonl(path)):
            raise RelevanceBlindModelScoreChainError(f"manifest row count mismatch for {role}")
    return manifest, paths, {**summary, "aggregates": aggregates}

