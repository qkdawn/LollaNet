"""Tamper-evident dual-adjudicator child chain for disputed Weak routes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .llm_teacher_v1 import INITIAL_PROMPT_VERSION, REVIEW_PROMPT_VERSION
from .question_set_v1 import ACTIVE_STATUSES, read_jsonl, teacher_view, write_jsonl
from .teacher_chain_v1 import (
    MANIFEST_FILENAME, TeacherChainError, canonical_sha256, file_descriptor, file_sha256,
    normalize_teacher_identity, route_signature, teacher_identity_from_label,
    teacher_independence_key, validate_manifest_contents, verify_manifest_files,
)

ADJUDICATION_DIRNAME = "adjudication"
ADJUDICATION_MANIFEST_FILENAME = "adjudication_manifest.json"
ADJUDICATION_MANIFEST_SCHEMA_VERSION = "teacher-label-adjudication-manifest-v1"
ADJUDICATION_POLICY_SCHEMA_VERSION = "teacher-label-adjudication-policy-v1"
ADJUDICATION_PROMPT_CONTRACT_SCHEMA_VERSION = "teacher-label-adjudication-prompt-contract-v1"
ADJUDICATION_PACKET_VERSION = "mechanism-question-v1.1.0-route-relabel"
ADJUDICATION_PROMPT_VERSION = "mechanism-question-v1.1.0-dual-route-relabel"
ADJUDICATION_POLICY_VERSION = "mechanism-question-dual-consensus-v1"
REQUIRED_CHILD_FILE_ROLES = frozenset({
    "adjudication_packets", "adjudicator_labels", "adjudication_audit_ledger",
    "accepted_adjudications", "policy_contract", "prompt_contract",
})
ROUTE_FIELDS = ("abstain", "primary_model", "auxiliary_models")


class AdjudicationChainError(TeacherChainError):
    """Raised when the adjudication child chain is incomplete or untrusted."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _unique(rows: Iterable[dict[str, Any]], *, key_fields: tuple[str, ...], description: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(not isinstance(value, str) or not value for value in key):
            raise AdjudicationChainError(f"{description} row is missing {'/'.join(key_fields)}")
        if key in result:
            raise AdjudicationChainError(f"duplicate {description} row for {key}")
        result[key] = row
    return result


def _parent_objects(parent_verified: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = parent_verified["paths"]
    episodes = {row["episode_id"]: row for row in parent_verified["episodes"]}
    initial = {row["episode_id"]: row for row in read_jsonl(paths["initial_labels"])}
    review = {row["episode_id"]: row for row in read_jsonl(paths["review_labels"])}
    if set(episodes) != set(initial) or set(initial) != set(review):
        raise AdjudicationChainError("parent Episode/initial/review sets do not match")
    return episodes, initial, review


def load_verified_parent(parent_manifest_path: Path, *, dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    parent_manifest_path = Path(parent_manifest_path)
    manifest, paths = verify_manifest_files(
        parent_manifest_path, dataset_root=Path(dataset_root),
        expected_accepted_path=parent_manifest_path.parent / "accepted_review.jsonl",
    )
    verified = validate_manifest_contents(manifest, paths)
    expected = {"initial": [INITIAL_PROMPT_VERSION], "review": [REVIEW_PROMPT_VERSION]}
    if manifest.get("prompt_versions") != expected:
        raise AdjudicationChainError("parent chain does not use current prompt versions")
    return manifest, verified, paths


def default_policy(*, review_visibility: str = "route_blind", packet_consumption_attested: bool = True) -> dict[str, Any]:
    if review_visibility not in {"route_blind", "status_visible", "status_and_rationale_visible"}:
        raise AdjudicationChainError("unsupported review_visibility")
    return {
        "schema_version": ADJUDICATION_POLICY_SCHEMA_VERSION,
        "policy_version": ADJUDICATION_POLICY_VERSION,
        "eligible_cohort_rule": "all_parent_review_required",
        "final_acceptance_rule": "parent_review_accepted_or_two_adjudicator_exact_route_consensus",
        "missing_adjudication_action": "exclude",
        "disagreement_action": "exclude",
        "route_rewrite_allowed_for_review_required": True,
        "parent_accepted_route_rewrite_allowed": False,
        "required_adjudicator_count": 2,
        "adjudicator_independence_requirement": "distinct_normalized_provider_model_between_slots",
        "parent_teacher_identity_overlap": "allowed_but_disclosed",
        "review_visibility": review_visibility,
        "packet_consumption_attested": bool(packet_consumption_attested),
        "label_tier": "Weak",
    }


def default_prompt_contract() -> dict[str, Any]:
    return {
        "schema_version": ADJUDICATION_PROMPT_CONTRACT_SCHEMA_VERSION,
        "prompt_version": ADJUDICATION_PROMPT_VERSION,
        "packet_version": ADJUDICATION_PACKET_VERSION,
        "decision_subject": "route_from_question_conditioned_microconcept_state",
        "route_fields": list(ROUTE_FIELDS),
        "maximum_auxiliary_models": 2,
        "sealed_test_visible": False,
        "instructions": "Use only the saved packet. Select one legal primary plus up to two legal auxiliaries, or abstain. Do not preserve the parent route by default and do not infer outside facts.",
    }


def build_adjudication_packets(*, parent_manifest_path: Path, dataset_root: Path, output_dir: Path, review_visibility: str = "route_blind") -> dict[str, Any]:
    parent_manifest, verified, _ = load_verified_parent(parent_manifest_path, dataset_root=dataset_root)
    episodes, initial, review = _parent_objects(verified)
    parent_hash = file_sha256(Path(parent_manifest_path))
    eligible_ids = sorted(eid for eid, row in review.items() if row.get("review_status") == "review_required")
    packets = []
    for eid in eligible_ids:
        packet = {
            "packet_version": ADJUDICATION_PACKET_VERSION,
            "prompt_version": ADJUDICATION_PROMPT_VERSION,
            "episode_id": eid,
            "parent_manifest_sha256": parent_hash,
            "episode_sha256": canonical_sha256(episodes[eid]),
            "initial_label_sha256": canonical_sha256(initial[eid]),
            "review_label_sha256": canonical_sha256(review[eid]),
            "episode": teacher_view(episodes[eid]),
        }
        if review_visibility == "status_visible":
            packet["parent_review"] = {"review_status": review[eid]["review_status"]}
        elif review_visibility == "status_and_rationale_visible":
            packet["parent_review"] = {"review_status": review[eid]["review_status"], "audit_rationale": review[eid].get("audit_rationale", "")}
        packets.append(packet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "adjudication_packets.jsonl"
    policy_path = output_dir / "adjudication_policy.json"
    prompt_path = output_dir / "adjudication_prompt_contract.json"
    write_jsonl(packet_path, packets)
    _write_json(policy_path, default_policy(review_visibility=review_visibility))
    _write_json(prompt_path, default_prompt_contract())
    return {"parent_manifest": parent_manifest, "parent_manifest_sha256": parent_hash, "eligible_episode_ids": eligible_ids, "packet_path": packet_path, "policy_path": policy_path, "prompt_contract_path": prompt_path, "packets": packets}


def _validate_policy(policy: Mapping[str, Any]) -> None:
    expected = default_policy(review_visibility=str(policy.get("review_visibility", "")), packet_consumption_attested=bool(policy.get("packet_consumption_attested")))
    if dict(policy) != expected:
        raise AdjudicationChainError("adjudication policy is not the supported fail-closed V1 contract")
    if not policy.get("packet_consumption_attested"):
        raise AdjudicationChainError("accepted adjudication requires attested saved-packet consumption")


def _validate_prompt_contract(contract: Mapping[str, Any]) -> None:
    if dict(contract) != default_prompt_contract():
        raise AdjudicationChainError("adjudication prompt contract does not match V1")


def _validate_packet(packet: Mapping[str, Any], *, episode: Mapping[str, Any], initial: Mapping[str, Any], review: Mapping[str, Any], parent_hash: str, policy: Mapping[str, Any]) -> None:
    eid = episode["episode_id"]
    expected = {
        "packet_version": ADJUDICATION_PACKET_VERSION,
        "prompt_version": ADJUDICATION_PROMPT_VERSION,
        "episode_id": eid,
        "parent_manifest_sha256": parent_hash,
        "episode_sha256": canonical_sha256(episode),
        "initial_label_sha256": canonical_sha256(initial),
        "review_label_sha256": canonical_sha256(review),
        "episode": teacher_view(dict(episode)),
    }
    visibility = policy["review_visibility"]
    if visibility == "status_visible":
        expected["parent_review"] = {"review_status": review["review_status"]}
    elif visibility == "status_and_rationale_visible":
        expected["parent_review"] = {"review_status": review["review_status"], "audit_rationale": review.get("audit_rationale", "")}
    if dict(packet) != expected:
        raise AdjudicationChainError(f"packet is not canonical and parent-bound for {eid}")

def _validate_route(label: Mapping[str, Any], episode: Mapping[str, Any]) -> None:
    abstain, primary, auxiliary = route_signature(label)
    if len(auxiliary) > 2 or len(set(auxiliary)) != len(auxiliary):
        raise AdjudicationChainError("auxiliary_models must contain zero to two unique models")
    candidates = set(episode["candidate_subgraph"].get("candidate_model_ids", []))
    if abstain:
        if primary is not None or auxiliary:
            raise AdjudicationChainError("abstaining routes cannot select models")
        return
    if primary not in candidates or primary in auxiliary or any(model_id not in candidates for model_id in auxiliary):
        raise AdjudicationChainError("adjudication route selects an illegal candidate model")
    states = {row["microconcept_id"]: row for row in episode["microconcept_states"]}
    edges = episode["candidate_subgraph"].get("edges", [])
    if not any(edge.get("model_id") == primary and edge.get("relation") in {"requires", "supports"} and states.get(edge.get("microconcept_id"), {}).get("status") in ACTIVE_STATUSES for edge in edges):
        raise AdjudicationChainError("primary lacks an active requires/supports edge")
    for model_id in auxiliary:
        if not any(edge.get("model_id") == model_id and states.get(edge.get("microconcept_id"), {}).get("status") in ACTIVE_STATUSES for edge in edges):
            raise AdjudicationChainError("auxiliary lacks an active legal edge")


def _validate_adjudicator_label(label: Mapping[str, Any], *, episode: Mapping[str, Any], initial: Mapping[str, Any], review: Mapping[str, Any], packet: Mapping[str, Any], parent_hash: str) -> dict[str, str]:
    required = {
        "label_id", "episode_id", "label_pass", "adjudicator_slot", "teacher_identity", "teacher_model", "teacher_version",
        "prompt_version", "label_timestamp", "label_tier", "confidence", "abstain", "primary_model", "auxiliary_models",
        "rationale", "parent_manifest_sha256", "episode_sha256", "initial_label_sha256", "review_label_sha256", "packet_sha256",
        "raw_response_fingerprint_sha256", "api_response_id", "run_id",
    }
    missing = required - set(label)
    if missing:
        raise AdjudicationChainError(f"adjudicator label missing fields: {sorted(missing)}")
    if label.get("episode_id") != episode["episode_id"] or label.get("label_pass") != "adjudication" or label.get("adjudicator_slot") not in {"a", "b"}:
        raise AdjudicationChainError("adjudicator label identity is invalid")
    if label.get("prompt_version") != ADJUDICATION_PROMPT_VERSION or label.get("label_tier") != "Weak":
        raise AdjudicationChainError("adjudicator prompt/tier contract is invalid")
    confidence = label.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise AdjudicationChainError("confidence must be between zero and one")
    if not isinstance(label.get("rationale"), str) or not label["rationale"].strip():
        raise AdjudicationChainError("rationale must be non-empty")
    identity = teacher_identity_from_label(label)
    expected = {
        "parent_manifest_sha256": parent_hash,
        "episode_sha256": canonical_sha256(episode),
        "initial_label_sha256": canonical_sha256(initial),
        "review_label_sha256": canonical_sha256(review),
        "packet_sha256": canonical_sha256(packet),
    }
    for field, value in expected.items():
        if label.get(field) != value:
            raise AdjudicationChainError(f"label {field} is not bound to parent artifact")
    digest = label.get("raw_response_fingerprint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise AdjudicationChainError("raw_response_fingerprint_sha256 must be lowercase SHA-256")
    for field in ("api_response_id", "run_id"):
        if not isinstance(label.get(field), str) or not label[field]:
            raise AdjudicationChainError(f"label requires {field}")
    _validate_route(label, episode)
    return identity


def _validate_ledger_entry(entry: Mapping[str, Any], label: Mapping[str, Any]) -> None:
    expected = {
        "episode_id": label["episode_id"], "label_id": label["label_id"], "mode": "adjudication",
        "adjudicator_slot": label["adjudicator_slot"], "teacher_identity": label["teacher_identity"],
        "teacher_model": label["teacher_model"], "teacher_version": label["teacher_version"],
        "prompt_version": label["prompt_version"], "label_sha256": canonical_sha256(label),
        "parent_manifest_sha256": label["parent_manifest_sha256"], "episode_sha256": label["episode_sha256"],
        "initial_label_sha256": label["initial_label_sha256"], "review_label_sha256": label["review_label_sha256"],
        "packet_sha256": label["packet_sha256"], "raw_response_fingerprint_sha256": label["raw_response_fingerprint_sha256"],
        "api_response_id": label["api_response_id"], "run_id": label["run_id"],
    }
    for field, value in expected.items():
        if entry.get(field) != value:
            raise AdjudicationChainError(f"ledger {field} does not match label {label['label_id']}")


def make_adjudicator_label(*, packet: Mapping[str, Any], slot: str, teacher_identity: Mapping[str, Any], confidence: float, abstain: bool, primary_model: str | None, auxiliary_models: Iterable[str], rationale: str, raw_response: Any, api_response_id: str, run_id: str, label_timestamp: str | None = None) -> dict[str, Any]:
    identity = normalize_teacher_identity(teacher_identity)
    eid = str(packet["episode_id"])
    return {
        "label_id": f"adjudication-{slot}::{eid}", "episode_id": eid, "label_pass": "adjudication", "adjudicator_slot": slot,
        "teacher_identity": identity, "teacher_model": identity["model"], "teacher_version": identity["version"],
        "prompt_version": ADJUDICATION_PROMPT_VERSION, "label_timestamp": label_timestamp or _timestamp(), "label_tier": "Weak",
        "confidence": float(confidence), "abstain": bool(abstain), "primary_model": primary_model,
        "auxiliary_models": sorted(set(auxiliary_models)), "rationale": rationale,
        "parent_manifest_sha256": packet["parent_manifest_sha256"], "episode_sha256": packet["episode_sha256"],
        "initial_label_sha256": packet["initial_label_sha256"], "review_label_sha256": packet["review_label_sha256"],
        "packet_sha256": canonical_sha256(packet), "raw_response_fingerprint_sha256": canonical_sha256(raw_response),
        "api_response_id": api_response_id, "run_id": run_id,
    }


def adjudication_ledger_entry(label: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": label["episode_id"], "label_id": label["label_id"], "mode": "adjudication", "adjudicator_slot": label["adjudicator_slot"],
        "teacher_identity": label["teacher_identity"], "teacher_model": label["teacher_model"], "teacher_version": label["teacher_version"],
        "prompt_version": label["prompt_version"], "label_sha256": canonical_sha256(label), "parent_manifest_sha256": label["parent_manifest_sha256"],
        "episode_sha256": label["episode_sha256"], "initial_label_sha256": label["initial_label_sha256"], "review_label_sha256": label["review_label_sha256"],
        "packet_sha256": label["packet_sha256"], "raw_response_fingerprint_sha256": label["raw_response_fingerprint_sha256"],
        "api_response_id": label["api_response_id"], "run_id": label["run_id"],
    }


def _final_target_from_parent(review: Mapping[str, Any], parent_hash: str) -> dict[str, Any]:
    return {
        "label_id": f"adjudication-final::{review['episode_id']}", "episode_id": review["episode_id"], "label_pass": "adjudication",
        "adjudication_status": "accepted", "acceptance_basis": "parent_review_accepted", "label_tier": "Weak",
        "confidence": review["confidence"], "abstain": review["abstain"], "primary_model": review.get("primary_model"),
        "auxiliary_models": sorted(review.get("auxiliary_models", [])), "parent_manifest_sha256": parent_hash,
        "source_review_label_sha256": canonical_sha256(review), "adjudicator_label_ids": [], "adjudicator_label_sha256": [],
        "policy_version": ADJUDICATION_POLICY_VERSION,
    }


def _final_target_from_consensus(labels: list[Mapping[str, Any]], parent_hash: str) -> dict[str, Any]:
    pair = sorted(labels, key=lambda row: row["adjudicator_slot"])
    first = pair[0]
    return {
        "label_id": f"adjudication-final::{first['episode_id']}", "episode_id": first["episode_id"], "label_pass": "adjudication",
        "adjudication_status": "accepted", "acceptance_basis": "dual_adjudicator_exact_route_consensus", "label_tier": "Weak",
        "confidence": min(float(row["confidence"]) for row in pair), "abstain": first["abstain"], "primary_model": first.get("primary_model"),
        "auxiliary_models": sorted(first.get("auxiliary_models", [])), "parent_manifest_sha256": parent_hash,
        "source_review_label_sha256": first["review_label_sha256"], "adjudicator_label_ids": [row["label_id"] for row in pair],
        "adjudicator_label_sha256": [canonical_sha256(row) for row in pair], "policy_version": ADJUDICATION_POLICY_VERSION,
    }

def validate_adjudication_artifacts(*, parent_manifest_path: Path, dataset_root: Path, packets: list[dict[str, Any]], labels: list[dict[str, Any]], ledger: list[dict[str, Any]], policy: Mapping[str, Any], prompt_contract: Mapping[str, Any]) -> dict[str, Any]:
    parent_manifest, parent_verified, _ = load_verified_parent(parent_manifest_path, dataset_root=dataset_root)
    _validate_policy(policy)
    _validate_prompt_contract(prompt_contract)
    parent_hash = file_sha256(Path(parent_manifest_path))
    episodes, initial, review = _parent_objects(parent_verified)
    eligible_ids = sorted(eid for eid, row in review.items() if row.get("review_status") == "review_required")
    packet_by_id = _unique(packets, key_fields=("episode_id",), description="adjudication packet")
    if set(key[0] for key in packet_by_id) != set(eligible_ids):
        raise AdjudicationChainError("packets must cover exactly all parent review_required Episodes")
    for eid in eligible_ids:
        _validate_packet(packet_by_id[(eid,)], episode=episodes[eid], initial=initial[eid], review=review[eid], parent_hash=parent_hash, policy=policy)
    label_by_key = _unique(labels, key_fields=("episode_id", "adjudicator_slot"), description="adjudicator label")
    expected_keys = {(eid, slot) for eid in eligible_ids for slot in ("a", "b")}
    if set(label_by_key) != expected_keys:
        raise AdjudicationChainError("labels must contain exactly slots a and b for every eligible Episode")
    ledger_by_key = _unique(ledger, key_fields=("episode_id", "adjudicator_slot"), description="audit ledger")
    if set(ledger_by_key) != expected_keys:
        raise AdjudicationChainError("audit ledger must cover exactly both labels for every eligible Episode")
    identities_by_slot: dict[str, set[tuple[str, str]]] = defaultdict(set)
    parent_identity_keys = {teacher_independence_key(identity) for role in ("initial", "review") for identity in parent_manifest["teacher_identities"][role]}
    parent_overlap = {"a": False, "b": False}
    for eid, slot in sorted(expected_keys):
        label = label_by_key[(eid, slot)]
        identity = _validate_adjudicator_label(label, episode=episodes[eid], initial=initial[eid], review=review[eid], packet=packet_by_id[(eid,)], parent_hash=parent_hash)
        identity_key = teacher_independence_key(identity)
        identities_by_slot[slot].add(identity_key)
        parent_overlap[slot] = parent_overlap[slot] or identity_key in parent_identity_keys
        _validate_ledger_entry(ledger_by_key[(eid, slot)], label)
    if any(len(values) != 1 for values in identities_by_slot.values()) or identities_by_slot["a"] == identities_by_slot["b"]:
        raise AdjudicationChainError("adjudicator slots must each be stable and distinct provider/model identities")
    accepted: list[dict[str, Any]] = []
    consensus_ids: list[str] = []
    disagreement_ids: list[str] = []
    for eid in sorted(review):
        if review[eid]["review_status"] == "accepted":
            accepted.append(_final_target_from_parent(review[eid], parent_hash))
        else:
            pair = [label_by_key[(eid, "a")], label_by_key[(eid, "b")]]
            if route_signature(pair[0]) == route_signature(pair[1]):
                accepted.append(_final_target_from_consensus(pair, parent_hash))
                consensus_ids.append(eid)
            else:
                disagreement_ids.append(eid)
    return {
        "accepted": accepted, "accepted_episode_ids": [row["episode_id"] for row in accepted], "eligible_episode_ids": eligible_ids,
        "consensus_episode_ids": consensus_ids, "disagreement_episode_ids": disagreement_ids,
        "parent_accepted_episode_ids": sorted(parent_verified["accepted_episode_ids"]),
        "adjudicator_identities": {slot: [list(value) for value in sorted(values)] for slot, values in sorted(identities_by_slot.items())},
        "parent_identity_overlap": parent_overlap, "parent_manifest": parent_manifest, "parent_verified": parent_verified,
        "parent_manifest_sha256": parent_hash,
    }


def finalize_adjudication(*, parent_manifest_path: Path, dataset_root: Path, packets_path: Path, labels_path: Path, audit_ledger_path: Path, policy_path: Path, prompt_contract_path: Path, output_dir: Path) -> dict[str, Any]:
    packets = read_jsonl(Path(packets_path)); labels = read_jsonl(Path(labels_path)); ledger = read_jsonl(Path(audit_ledger_path))
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8")); prompt = json.loads(Path(prompt_contract_path).read_text(encoding="utf-8"))
    summary = validate_adjudication_artifacts(parent_manifest_path=parent_manifest_path, dataset_root=dataset_root, packets=packets, labels=labels, ledger=ledger, policy=policy, prompt_contract=prompt)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted_adjudication.jsonl"; write_jsonl(accepted_path, summary["accepted"])
    paths_and_rows = {
        "adjudication_packets": (Path(packets_path), len(packets)), "adjudicator_labels": (Path(labels_path), len(labels)),
        "adjudication_audit_ledger": (Path(audit_ledger_path), len(ledger)), "accepted_adjudications": (accepted_path, len(summary["accepted"])),
        "policy_contract": (Path(policy_path), None), "prompt_contract": (Path(prompt_contract_path), None),
    }
    parent_path = Path(parent_manifest_path).resolve()
    manifest = {
        "schema_version": ADJUDICATION_MANIFEST_SCHEMA_VERSION, "created_at": _timestamp(),
        "parent_chain": {"schema_version": summary["parent_manifest"]["schema_version"], "path": Path(os.path.relpath(parent_path, output_dir.resolve())).as_posix(), "bytes": parent_path.stat().st_size, "sha256": summary["parent_manifest_sha256"]},
        "policy_version": policy["policy_version"], "prompt_version": prompt["prompt_version"],
        "files": {role: file_descriptor(path, relative_to=output_dir, rows=rows) for role, (path, rows) in paths_and_rows.items()},
        "parent_episode_ids": summary["parent_manifest"]["episode_ids"], "parent_accepted_episode_ids": summary["parent_accepted_episode_ids"],
        "eligible_episode_ids": summary["eligible_episode_ids"], "consensus_episode_ids": summary["consensus_episode_ids"],
        "disagreement_episode_ids": summary["disagreement_episode_ids"], "accepted_episode_ids": summary["accepted_episode_ids"],
        "adjudicator_identities": summary["adjudicator_identities"], "parent_identity_overlap": summary["parent_identity_overlap"],
    }
    _write_json(output_dir / ADJUDICATION_MANIFEST_FILENAME, manifest)
    return manifest


def verify_adjudication_manifest(manifest_path: Path, *, dataset_root: Path, expected_accepted_path: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise AdjudicationChainError("adjudication-v1 requires a child manifest; no fallback is allowed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdjudicationChainError("adjudication manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ADJUDICATION_MANIFEST_SCHEMA_VERSION:
        raise AdjudicationChainError("unsupported adjudication manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_CHILD_FILE_ROLES:
        raise AdjudicationChainError("adjudication manifest must bind every child artifact exactly once")
    root = Path(dataset_root).resolve(); paths: dict[str, Path] = {}
    for role in sorted(REQUIRED_CHILD_FILE_ROLES):
        descriptor = files[role]; relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise AdjudicationChainError(f"adjudication descriptor {role} must use a relative path")
        path = (manifest_path.parent / relative).resolve()
        try: path.relative_to(root)
        except ValueError as error: raise AdjudicationChainError(f"adjudication file {role} escapes dataset root") from error
        if not path.is_file() or descriptor.get("bytes") != path.stat().st_size or descriptor.get("sha256") != file_sha256(path):
            raise AdjudicationChainError(f"adjudication file integrity mismatch for {role}")
        paths[role] = path
    if paths["accepted_adjudications"] != Path(expected_accepted_path).resolve():
        raise AdjudicationChainError("accepted_adjudications path does not match Router cohort path")
    parent = manifest.get("parent_chain")
    if not isinstance(parent, dict) or set(parent) != {"schema_version", "path", "bytes", "sha256"}:
        raise AdjudicationChainError("invalid parent_chain descriptor")
    relative = parent.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AdjudicationChainError("parent manifest path must be relative")
    parent_path = (manifest_path.parent / relative).resolve()
    try: parent_path.relative_to(root)
    except ValueError as error: raise AdjudicationChainError("parent manifest escapes dataset root") from error
    if parent_path.name != MANIFEST_FILENAME or not parent_path.is_file():
        raise AdjudicationChainError("parent teacher-chain manifest is missing")
    if parent.get("bytes") != parent_path.stat().st_size or parent.get("sha256") != file_sha256(parent_path):
        raise AdjudicationChainError("parent manifest hash binding failed")
    packets = read_jsonl(paths["adjudication_packets"]); labels = read_jsonl(paths["adjudicator_labels"]); ledger = read_jsonl(paths["adjudication_audit_ledger"]); accepted = read_jsonl(paths["accepted_adjudications"])
    policy = json.loads(paths["policy_contract"].read_text(encoding="utf-8")); prompt = json.loads(paths["prompt_contract"].read_text(encoding="utf-8"))
    summary = validate_adjudication_artifacts(parent_manifest_path=parent_path, dataset_root=dataset_root, packets=packets, labels=labels, ledger=ledger, policy=policy, prompt_contract=prompt)
    if [canonical_sha256(row) for row in accepted] != [canonical_sha256(row) for row in summary["accepted"]]:
        raise AdjudicationChainError("accepted_adjudication.jsonl is not deterministic policy output")
    expected_fields = {
        "parent_episode_ids": summary["parent_manifest"]["episode_ids"], "parent_accepted_episode_ids": summary["parent_accepted_episode_ids"],
        "eligible_episode_ids": summary["eligible_episode_ids"], "consensus_episode_ids": summary["consensus_episode_ids"],
        "disagreement_episode_ids": summary["disagreement_episode_ids"], "accepted_episode_ids": summary["accepted_episode_ids"],
        "adjudicator_identities": summary["adjudicator_identities"], "parent_identity_overlap": summary["parent_identity_overlap"],
        "policy_version": policy["policy_version"], "prompt_version": prompt["prompt_version"],
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected: raise AdjudicationChainError(f"manifest {field} does not match verified artifacts")
    for role, path in paths.items():
        expected_rows = files[role].get("rows")
        if expected_rows is not None and expected_rows != len(read_jsonl(path)): raise AdjudicationChainError(f"manifest row count mismatch for {role}")
    return manifest, paths, {**summary, "accepted": accepted, "episodes": summary["parent_verified"]["episodes"]}
