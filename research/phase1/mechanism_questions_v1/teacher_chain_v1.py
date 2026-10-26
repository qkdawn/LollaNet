"""Shared integrity contract for the two-pass Weak teacher label chain."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .question_set_v1 import ValidationError, validate_teacher_labels

MANIFEST_FILENAME = "teacher_chain_manifest.json"
MANIFEST_SCHEMA_VERSION = "teacher-label-chain-manifest-v1"
TEACHER_IDENTITY_SCHEMA_VERSION = "teacher-identity-v1"
REQUIRED_FILE_ROLES = frozenset({
    "episodes",
    "initial_labels",
    "review_labels",
    "initial_audit_ledger",
    "review_audit_ledger",
    "all_labels",
    "accepted_reviews",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TeacherChainError(ValidationError):
    """Raised when teacher provenance or finalized artifacts are not trustworthy."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalized_text(value: Any, field: str, *, casefold: bool) -> str:
    if not isinstance(value, str):
        raise TeacherChainError(f"teacher identity {field} must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise TeacherChainError(f"teacher identity {field} must be non-empty")
    return normalized.casefold() if casefold else normalized


def normalize_teacher_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    """Return the one canonical representation accepted by labels and manifests."""
    if not isinstance(identity, Mapping):
        raise TeacherChainError("teacher_identity must be a structured object")
    allowed = {"schema_version", "provider", "model", "deployment", "version"}
    if set(identity) != allowed:
        raise TeacherChainError("teacher_identity must contain exactly schema_version/provider/model/deployment/version")
    if identity.get("schema_version") != TEACHER_IDENTITY_SCHEMA_VERSION:
        raise TeacherChainError("unsupported teacher identity schema_version")
    deployment = identity.get("deployment")
    if not isinstance(deployment, str):
        raise TeacherChainError("teacher identity deployment must be a string")
    return {
        "schema_version": TEACHER_IDENTITY_SCHEMA_VERSION,
        "provider": _normalized_text(identity.get("provider"), "provider", casefold=True),
        "model": _normalized_text(identity.get("model"), "model", casefold=True),
        "deployment": " ".join(deployment.strip().split()).casefold(),
        "version": _normalized_text(identity.get("version"), "version", casefold=True),
    }


def make_teacher_identity(*, provider: str, model: str, deployment: str, version: str) -> dict[str, str]:
    return normalize_teacher_identity({
        "schema_version": TEACHER_IDENTITY_SCHEMA_VERSION,
        "provider": provider,
        "model": model,
        "deployment": deployment,
        "version": version,
    })


def teacher_identity_from_label(label: Mapping[str, Any]) -> dict[str, str]:
    raw_identity = label.get("teacher_identity")
    identity = normalize_teacher_identity(raw_identity)
    if dict(raw_identity) != identity:
        raise TeacherChainError("teacher_identity must already use its canonical normalized representation")
    if label.get("teacher_model") != identity["model"]:
        raise TeacherChainError("teacher_model must equal normalized teacher_identity.model")
    if label.get("teacher_version") != identity["version"]:
        raise TeacherChainError("teacher_version must equal normalized teacher_identity.version")
    return identity


def teacher_independence_key(identity: Mapping[str, Any]) -> tuple[str, str]:
    normalized = normalize_teacher_identity(identity)
    return normalized["provider"], normalized["model"]


def require_independent_teachers(initial: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    if teacher_independence_key(initial) == teacher_independence_key(review):
        raise TeacherChainError("initial and review teacher identities must use different provider/model identities")


def route_signature(label: Mapping[str, Any]) -> tuple[bool, str | None, tuple[str, ...]]:
    abstain = label.get("abstain")
    if not isinstance(abstain, bool):
        raise TeacherChainError("route signature requires a boolean abstain field")
    primary = label.get("primary_model")
    auxiliary = label.get("auxiliary_models")
    if not isinstance(auxiliary, list) or any(not isinstance(item, str) for item in auxiliary):
        raise TeacherChainError("route signature requires auxiliary_models to be a string list")
    return abstain, primary, tuple(sorted(auxiliary))


def require_exact_review_route(initial: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    if review.get("review_status") not in {"accepted", "review_required"}:
        raise TeacherChainError("review_status must be accepted or review_required")
    if route_signature(initial) != route_signature(review):
        raise TeacherChainError("review labels must retain the exact initial abstain/primary/auxiliary route")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TeacherChainError(f"{field} must be a lowercase canonical SHA-256 hex digest")
    return value


def _unique_rows(rows: Iterable[dict[str, Any]], *, field: str, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key:
            raise TeacherChainError(f"{description} row missing {field}")
        if key in result:
            raise TeacherChainError(f"duplicate {description} row for {key}")
        result[key] = row
    return result


def _validate_ledger(
    labels: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    *,
    mode: str,
    initial_by_episode: Mapping[str, dict[str, Any]],
) -> None:
    ledger_by_episode = _unique_rows(ledger, field="episode_id", description=f"{mode} audit ledger")
    labels_by_episode = _unique_rows(labels, field="episode_id", description=f"{mode} label")
    if set(ledger_by_episode) != set(labels_by_episode):
        raise TeacherChainError(f"{mode} audit ledger must cover exactly the {mode} labels")
    for episode_id, label in labels_by_episode.items():
        entry = ledger_by_episode[episode_id]
        if entry.get("mode") != mode or entry.get("label_id") != label.get("label_id"):
            raise TeacherChainError(f"{mode} audit ledger identity does not match label {episode_id}")
        identity = teacher_identity_from_label(label)
        raw_ledger_identity = entry.get("teacher_identity")
        ledger_identity = normalize_teacher_identity(raw_ledger_identity)
        if dict(raw_ledger_identity) != ledger_identity:
            raise TeacherChainError(f"{mode} audit ledger teacher identity is not normalized for {episode_id}")
        if ledger_identity != identity:
            raise TeacherChainError(f"{mode} audit ledger teacher identity does not match label {episode_id}")
        if entry.get("teacher_model") != ledger_identity["model"] or entry.get("teacher_version") != ledger_identity["version"]:
            raise TeacherChainError(f"{mode} audit ledger compatibility identity fields do not match for {episode_id}")
        if entry.get("prompt_version") != label.get("prompt_version"):
            raise TeacherChainError(f"{mode} audit ledger prompt_version does not match label {episode_id}")
        if _require_sha256(entry.get("label_sha256"), "ledger label_sha256") != canonical_sha256(label):
            raise TeacherChainError(f"{mode} audit ledger label hash does not match label {episode_id}")
        provenance = label.get("provenance")
        if not isinstance(provenance, dict):
            raise TeacherChainError(f"label {episode_id} is missing structured provenance")
        provenance_identity = normalize_teacher_identity(provenance.get("teacher_identity"))
        if provenance_identity != identity or dict(provenance.get("teacher_identity")) != provenance_identity:
            raise TeacherChainError(f"label provenance teacher identity does not match for {episode_id}")
        for field in ("request_fingerprint_sha256", "raw_response_fingerprint_sha256"):
            digest = _require_sha256(entry.get(field), f"ledger {field}")
            if digest != _require_sha256(provenance.get(field), f"label provenance {field}"):
                raise TeacherChainError(f"{mode} audit ledger {field} does not match label {episode_id}")
        if entry.get("api_response_id") != provenance.get("api_response_id"):
            raise TeacherChainError(f"{mode} audit ledger api_response_id does not match label {episode_id}")
        if mode == "review":
            initial = initial_by_episode[episode_id]
            expected_initial_hash = canonical_sha256(initial)
            provenance_initial_identity = normalize_teacher_identity(provenance.get("initial_teacher_identity"))
            expected_initial_identity = teacher_identity_from_label(initial)
            if provenance_initial_identity != expected_initial_identity or dict(provenance.get("initial_teacher_identity")) != provenance_initial_identity:
                raise TeacherChainError(f"review provenance initial teacher identity does not match for {episode_id}")
            if _require_sha256(label.get("initial_label_sha256"), "review initial_label_sha256") != expected_initial_hash:
                raise TeacherChainError(f"review label is not bound to the canonical initial label for {episode_id}")
            if _require_sha256(provenance.get("initial_label_sha256"), "review provenance initial_label_sha256") != expected_initial_hash:
                raise TeacherChainError(f"review provenance is not bound to the canonical initial label for {episode_id}")
            if _require_sha256(entry.get("initial_label_sha256"), "review ledger initial_label_sha256") != expected_initial_hash:
                raise TeacherChainError(f"review audit ledger is not bound to the canonical initial label for {episode_id}")


def validate_teacher_chain(
    episodes: list[dict[str, Any]],
    initial_labels: list[dict[str, Any]],
    review_labels: list[dict[str, Any]],
    initial_ledger: list[dict[str, Any]],
    review_ledger: list[dict[str, Any]],
    *,
    expected_initial_prompt: str | None = None,
    expected_review_prompt: str | None = None,
) -> dict[str, Any]:
    """Validate labels, exact-route immutability, identities, hashes, and ledgers."""
    validate_teacher_labels(initial_labels + review_labels, episodes, require_complete=True)
    initial_by_episode = _unique_rows(initial_labels, field="episode_id", description="initial label")
    review_by_episode = _unique_rows(review_labels, field="episode_id", description="review label")
    if set(initial_by_episode) != set(review_by_episode):
        raise TeacherChainError("initial and review label episode sets must match")
    for episode_id, initial in initial_by_episode.items():
        review = review_by_episode[episode_id]
        if initial.get("label_pass") != "initial" or review.get("label_pass") != "review":
            raise TeacherChainError("label files must not mix initial and review passes")
        initial_identity = teacher_identity_from_label(initial)
        review_identity = teacher_identity_from_label(review)
        require_independent_teachers(initial_identity, review_identity)
        require_exact_review_route(initial, review)
        expected_hash = canonical_sha256(initial)
        if review.get("initial_label_sha256") != expected_hash:
            raise TeacherChainError(f"review label is not bound to initial label {episode_id}")
        if expected_initial_prompt is not None and initial.get("prompt_version") != expected_initial_prompt:
            raise TeacherChainError("initial label prompt_version is not the finalized prompt version")
        if expected_review_prompt is not None and review.get("prompt_version") != expected_review_prompt:
            raise TeacherChainError("review label prompt_version is not the finalized prompt version")
    _validate_ledger(initial_labels, initial_ledger, mode="initial", initial_by_episode=initial_by_episode)
    _validate_ledger(review_labels, review_ledger, mode="review", initial_by_episode=initial_by_episode)
    accepted = [review_by_episode[key] for key in sorted(review_by_episode) if review_by_episode[key]["review_status"] == "accepted"]
    initial_identities = sorted({canonical_json(teacher_identity_from_label(row)) for row in initial_labels})
    review_identities = sorted({canonical_json(teacher_identity_from_label(row)) for row in review_labels})
    return {
        "accepted": accepted,
        "accepted_episode_ids": [row["episode_id"] for row in accepted],
        "episode_ids": sorted(initial_by_episode),
        "prompt_versions": {
            "initial": sorted({row["prompt_version"] for row in initial_labels}),
            "review": sorted({row["prompt_version"] for row in review_labels}),
        },
        "teacher_identities": {
            "initial": [json.loads(value) for value in initial_identities],
            "review": [json.loads(value) for value in review_identities],
        },
    }


def file_descriptor(path: Path, *, relative_to: Path, rows: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    relative = Path(os.path.relpath(path, Path(relative_to).resolve())).as_posix()
    descriptor: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TeacherChainError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise TeacherChainError(f"JSONL row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def verify_manifest_files(
    manifest_path: Path,
    *,
    dataset_root: Path,
    expected_accepted_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the manifest and every referenced byte hash before parsing labels."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise TeacherChainError("accepted reviews require a teacher-chain manifest; bare accepted_review.jsonl is forbidden")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TeacherChainError("teacher-chain manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TeacherChainError("unsupported teacher-chain manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILE_ROLES:
        raise TeacherChainError("teacher-chain manifest must bind every required artifact exactly once")
    root = Path(dataset_root).resolve()
    resolved: dict[str, Path] = {}
    for role in sorted(REQUIRED_FILE_ROLES):
        descriptor = files[role]
        if not isinstance(descriptor, dict):
            raise TeacherChainError(f"manifest file descriptor {role} must be an object")
        relative = descriptor.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise TeacherChainError(f"manifest file descriptor {role} must use a relative path")
        path = (manifest_path.parent / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise TeacherChainError(f"manifest file {role} escapes the dataset root") from error
        if not path.is_file():
            raise TeacherChainError(f"manifest file {role} is missing")
        if descriptor.get("bytes") != path.stat().st_size:
            raise TeacherChainError(f"manifest byte count mismatch for {role}")
        if _require_sha256(descriptor.get("sha256"), f"manifest {role} sha256") != file_sha256(path):
            raise TeacherChainError(f"manifest hash mismatch for {role}")
        resolved[role] = path
    if resolved["accepted_reviews"] != Path(expected_accepted_path).resolve():
        raise TeacherChainError("manifest accepted_reviews path does not match the Router cohort path")
    return manifest, resolved


def validate_manifest_contents(manifest: dict[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    """Parse already-hash-verified files and compare chain summary to the manifest."""
    episodes = read_jsonl(paths["episodes"])
    initial = read_jsonl(paths["initial_labels"])
    review = read_jsonl(paths["review_labels"])
    initial_ledger = read_jsonl(paths["initial_audit_ledger"])
    review_ledger = read_jsonl(paths["review_audit_ledger"])
    all_labels = read_jsonl(paths["all_labels"])
    accepted = read_jsonl(paths["accepted_reviews"])
    summary = validate_teacher_chain(episodes, initial, review, initial_ledger, review_ledger)
    if [canonical_sha256(row) for row in all_labels] != [canonical_sha256(row) for row in initial + review]:
        raise TeacherChainError("all_labels.jsonl does not exactly contain initial labels followed by review labels")
    if [canonical_sha256(row) for row in accepted] != [canonical_sha256(row) for row in summary["accepted"]]:
        raise TeacherChainError("accepted_review.jsonl does not exactly contain the accepted review labels")
    for field in ("accepted_episode_ids", "episode_ids", "prompt_versions", "teacher_identities"):
        if manifest.get(field) != summary[field]:
            raise TeacherChainError(f"teacher-chain manifest {field} does not match finalized artifacts")
    for role, path in paths.items():
        expected_rows = manifest["files"][role].get("rows")
        if expected_rows is not None and expected_rows != len(read_jsonl(path)):
            raise TeacherChainError(f"manifest row count mismatch for {role}")
    return {**summary, "episodes": episodes, "accepted": accepted, "paths": dict(paths)}
