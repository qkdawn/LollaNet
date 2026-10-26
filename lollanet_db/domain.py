from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

from .models import (
    CHANGE_KINDS,
    CLAIM_KINDS,
    EVIDENCE_STANCES,
    MODEL_KINDS,
    MODEL_STATUSES,
    RELATION_KINDS,
    ModelRef,
    VersionSpec,
)


MODEL_ID_RE = re.compile(r"^mm-[a-z0-9]+(?:-[a-z0-9]+)*$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_TRANSITIONS = {
    "candidate": {"candidate", "active", "contested", "deprecated"},
    "active": {"active", "contested", "deprecated"},
    "contested": {"candidate", "active", "contested", "deprecated"},
    "deprecated": {"candidate"},
}


class DomainError(ValueError):
    pass


def validate_version(version: VersionSpec) -> None:
    if not MODEL_ID_RE.fullmatch(version.ref.model_id):
        raise DomainError(f"invalid model id: {version.ref.model_id}")
    if version.ref.version_no < 1:
        raise DomainError("version number must be positive")
    if version.status not in MODEL_STATUSES:
        raise DomainError(f"invalid model status: {version.status}")
    if version.kind not in MODEL_KINDS:
        raise DomainError(f"invalid model kind: {version.kind}")
    if not version.name.strip() or not version.english_name.strip():
        raise DomainError(f"{version.ref} requires non-empty names")
    if not HASH_RE.fullmatch(version.content_hash):
        raise DomainError(f"{version.ref} requires a SHA-256 content hash")

    counts = Counter(claim.kind for claim in version.claims)
    missing = [kind for kind in CLAIM_KINDS if counts[kind] == 0]
    if missing:
        raise DomainError(f"{version.ref} missing required claims: {', '.join(missing)}")
    seen_claims: set[tuple[str, int]] = set()
    for claim in version.claims:
        if claim.kind not in CLAIM_KINDS or not claim.statement.strip() or claim.ordinal < 1:
            raise DomainError(f"{version.ref} has an invalid {claim.kind!r} claim")
        key = (claim.kind, claim.ordinal)
        if key in seen_claims:
            raise DomainError(f"{version.ref} has duplicate claim {key}")
        seen_claims.add(key)
        if claim.kind == "prediction" and not (claim.time_window or "").strip():
            raise DomainError(f"{version.ref} prediction requires a time window")
        if claim.kind != "prediction" and claim.time_window is not None:
            raise DomainError(f"{version.ref} only predictions may have a time window")

    stances = Counter(item.stance for item in version.evidence)
    for item in version.evidence:
        if item.stance not in EVIDENCE_STANCES:
            raise DomainError(f"{version.ref} has invalid evidence stance: {item.stance}")
        if (item.claim_kind, item.claim_ordinal) not in seen_claims:
            raise DomainError(f"{version.ref} evidence targets an unknown claim")
        if not item.source_ref.strip() or not item.summary.strip() or not item.rationale.strip():
            raise DomainError(f"{version.ref} has incomplete evidence")
    if version.status == "active" and (not stances["supports"] or not (stances["opposes"] or stances["limits"])):
        raise DomainError(f"active version {version.ref} requires support and opposing/limiting evidence")
    if version.status == "contested" and not (stances["opposes"] or stances["limits"]):
        raise DomainError(f"contested version {version.ref} requires opposing or limiting evidence")

    if len(version.categories) != 3:
        raise DomainError(f"{version.ref} requires source, discipline, and subdiscipline categories")
    expected_categories = ("source_category", "discipline", "subdiscipline")
    if tuple(level.kind for level in version.categories) != expected_categories:
        raise DomainError(f"{version.ref} category hierarchy is invalid")
    for level in version.categories:
        if not level.slug or not level.name.strip():
            raise DomainError(f"{version.ref} contains an empty category")

    for relation in version.relations:
        if relation.kind not in RELATION_KINDS or not relation.rationale.strip():
            raise DomainError(f"{version.ref} has an invalid formal relation")
        if relation.target.model_id == version.ref.model_id:
            raise DomainError(f"{version.ref} cannot relate to the same stable model")


def validate_change(
    kind: str,
    inputs: Sequence[ModelRef],
    input_statuses: Mapping[ModelRef, str],
    outputs: Sequence[VersionSpec],
) -> None:
    if kind not in CHANGE_KINDS:
        raise DomainError(f"invalid change kind: {kind}")
    if len(set(inputs)) != len(inputs):
        raise DomainError("change inputs must be unique")
    if len({output.ref for output in outputs}) != len(outputs):
        raise DomainError("change outputs must be unique")
    for output in outputs:
        validate_version(output)

    by_model = {output.ref.model_id: output for output in outputs}
    input_ids = {item.model_id for item in inputs}
    new_outputs = [output for output in outputs if output.ref.model_id not in input_ids]

    def require_successor(ref: ModelRef, status: str | None = None) -> VersionSpec:
        output = by_model.get(ref.model_id)
        if output is None or output.ref.version_no != ref.version_no + 1:
            raise DomainError(f"change needs successor version for {ref}")
        if status is not None and output.status != status:
            raise DomainError(f"successor of {ref} must be {status}")
        return output

    if kind == "new":
        if inputs or len(outputs) != 1 or outputs[0].ref.version_no != 1 or outputs[0].status != "candidate":
            raise DomainError("new requires zero inputs and one candidate version 1")
        return

    if not inputs or set(input_statuses) != set(inputs):
        raise DomainError(f"{kind} requires current input statuses")

    if kind == "amend":
        if len(inputs) != 1 or len(outputs) != 1 or new_outputs:
            raise DomainError("amend requires one input and one same-model output")
        successor = require_successor(inputs[0])
        if successor.status not in ALLOWED_TRANSITIONS[input_statuses[inputs[0]]]:
            raise DomainError(f"invalid status transition to {successor.status}")
    elif kind in {"contest", "deprecate"}:
        expected = "contested" if kind == "contest" else "deprecated"
        if new_outputs or len(outputs) != len(inputs):
            raise DomainError(f"{kind} requires one successor for every input")
        for ref in inputs:
            successor = require_successor(ref, expected)
            if successor.status not in ALLOWED_TRANSITIONS[input_statuses[ref]]:
                raise DomainError(f"invalid status transition to {successor.status}")
    elif kind == "split":
        if len(inputs) != 1 or len(new_outputs) < 2:
            raise DomainError("split requires one input and at least two new models")
        successor = require_successor(inputs[0], "deprecated")
        if successor.status not in ALLOWED_TRANSITIONS[input_statuses[inputs[0]]]:
            raise DomainError(f"invalid status transition to {successor.status}")
        if any(output.ref.version_no != 1 or output.status != "candidate" for output in new_outputs):
            raise DomainError("split results must be candidate version 1 models")
    elif kind == "merge":
        if len(inputs) < 2 or len(new_outputs) != 1:
            raise DomainError("merge requires at least two inputs and one new model")
        for ref in inputs:
            successor = require_successor(ref, "deprecated")
            if successor.status not in ALLOWED_TRANSITIONS[input_statuses[ref]]:
                raise DomainError(f"invalid status transition to {successor.status}")
        if new_outputs[0].ref.version_no != 1 or new_outputs[0].status != "candidate":
            raise DomainError("merged model must be candidate version 1")
