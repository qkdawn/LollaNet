from __future__ import annotations

from dataclasses import dataclass, field


CLAIM_KINDS = (
    "definition",
    "mechanism",
    "input",
    "transformation",
    "output",
    "premise",
    "boundary",
    "falsifier",
    "prediction",
)
MODEL_STATUSES = ("candidate", "active", "contested", "deprecated")
MODEL_KINDS = ("world", "reasoning", "action")
RELATION_KINDS = ("requires", "composes", "competes", "corrects", "specializes")
CHANGE_KINDS = ("new", "amend", "split", "merge", "contest", "deprecate")
EVIDENCE_STANCES = ("supports", "opposes", "limits")


@dataclass(frozen=True, order=True)
class ModelRef:
    model_id: str
    version_no: int

    def __str__(self) -> str:
        return f"{self.model_id}@{self.version_no}"


@dataclass(frozen=True)
class ClaimSpec:
    kind: str
    statement: str
    ordinal: int = 1
    time_window: str | None = None
    legacy_key: str | None = None


@dataclass(frozen=True)
class EvidenceSpec:
    source_ref: str
    summary: str
    stance: str
    claim_kind: str
    claim_ordinal: int = 1
    locator: str | None = None
    url: str | None = None
    rationale: str = "Public source mapped during model review."
    legacy_key: str | None = None


@dataclass(frozen=True)
class CategoryLevel:
    kind: str
    slug: str
    name: str


@dataclass(frozen=True)
class RelationSpec:
    target: ModelRef
    kind: str
    rationale: str
    legacy_source_path: str | None = None


@dataclass(frozen=True)
class VersionSpec:
    ref: ModelRef
    status: str
    kind: str
    formalism: bool
    name: str
    english_name: str
    created_on: str
    content_hash: str
    claims: tuple[ClaimSpec, ...]
    categories: tuple[CategoryLevel, ...]
    evidence: tuple[EvidenceSpec, ...] = ()
    relations: tuple[RelationSpec, ...] = ()
    legacy_source_path: str | None = None


@dataclass(frozen=True)
class LegacyLibrary:
    versions: tuple[VersionSpec, ...]
    discovery_hints_skipped: int
    source_files: int
    parser_warnings: tuple[str, ...] = ()


@dataclass
class InsertCounts:
    models: int = 0
    versions: int = 0
    claims: int = 0
    evidence: int = 0
    relations: int = 0
    categories: int = 0
    mappings: int = 0
    details: dict[str, int] = field(default_factory=dict)
