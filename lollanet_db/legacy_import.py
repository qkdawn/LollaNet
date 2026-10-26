from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .domain import DomainError, validate_version
from .models import (
    CategoryLevel,
    ClaimSpec,
    EvidenceSpec,
    LegacyLibrary,
    ModelRef,
    RelationSpec,
    VersionSpec,
)


FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
MODEL_REF_RE = re.compile(r"^(mm-[a-z0-9-]+)@([1-9][0-9]*)$")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")

SOURCE_SLUGS = {
    "哲学模型": "philosophy-models",
    "历史机制": "historical-mechanisms",
    "学科知识": "disciplinary-knowledge",
}
DISCIPLINE_SLUGS = {
    "哲学": "philosophy",
    "制度演化": "institutional-evolution",
    "数学": "mathematics",
    "物理学": "physics",
    "化学": "chemistry",
    "心理学": "psychology",
    "经济学": "economics",
    "概率与统计": "probability-statistics",
    "商学": "business",
    "工程学": "engineering",
    "生物学": "biology",
}


class LegacyImportError(ValueError):
    pass


@dataclass(frozen=True)
class ImportAudit:
    source_files: int
    models: int
    versions: int
    statuses: dict[str, int]
    claims: int
    evidence: int
    claim_evidence: int
    formal_relations: int
    categories: int
    discovery_hints_skipped: int
    skipped_items: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_files": self.source_files,
            "models": self.models,
            "versions": self.versions,
            "statuses": self.statuses,
            "claims": self.claims,
            "evidence": self.evidence,
            "claim_evidence": self.claim_evidence,
            "formal_relations": self.formal_relations,
            "categories": self.categories,
            "discovery_hints_skipped": self.discovery_hints_skipped,
            "skipped_items": list(self.skipped_items),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"


@dataclass
class _ParsedCard:
    version: VersionSpec
    related: tuple[str, ...]
    conflicts: tuple[str, ...]


def load_legacy_library(root: Path) -> tuple[LegacyLibrary, ImportAudit]:
    root = Path(root).resolve()
    model_root = root / "01-Models"
    paths = sorted(model_root.rglob("*.md"))
    if not paths:
        raise LegacyImportError(f"no public model cards found under {model_root}")

    parsed: list[_ParsedCard] = []
    errors: list[str] = []
    warnings: list[str] = []
    for path in paths:
        try:
            parsed.append(_parse_card(root, path, warnings))
        except (LegacyImportError, DomainError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.relative_to(root).as_posix()}: {error}")
    if errors:
        raise LegacyImportError("legacy import validation failed:\n- " + "\n- ".join(errors))

    versions = tuple(item.version for item in parsed)
    refs = {version.ref for version in versions}
    if len(refs) != len(versions):
        raise LegacyImportError("duplicate model version in legacy library")
    for version in versions:
        for relation in version.relations:
            if relation.target not in refs:
                raise LegacyImportError(f"{version.ref} references unknown formal target {relation.target}")

    discovery = _discovery_hint_count(parsed)
    status_counts = Counter(version.status for version in versions)
    category_slugs = {level.slug for version in versions for level in version.categories}
    evidence_count = sum(len(version.evidence) for version in versions)
    relation_count = sum(len(version.relations) for version in versions)
    claim_count = sum(len(version.claims) for version in versions)
    skipped = (
        "related/conflicts discovery hints were not imported as formal relations",
        "Markdown revision rows without historical bodies were not materialized as versions",
        "private proposals, cases, project records, Runtime, Context Packs, and research assets were not read",
    )
    library = LegacyLibrary(
        versions=versions,
        discovery_hints_skipped=discovery,
        source_files=len(paths),
        parser_warnings=tuple(warnings),
    )
    audit = ImportAudit(
        source_files=len(paths),
        models=len({version.ref.model_id for version in versions}),
        versions=len(versions),
        statuses={key: status_counts.get(key, 0) for key in ("candidate", "active", "contested", "deprecated")},
        claims=claim_count,
        evidence=evidence_count,
        claim_evidence=evidence_count,
        formal_relations=relation_count,
        categories=len(category_slugs),
        discovery_hints_skipped=discovery,
        skipped_items=skipped,
        warnings=tuple(warnings),
        errors=(),
    )
    return library, audit


def _parse_card(root: Path, path: Path, warnings: list[str]) -> _ParsedCard:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise LegacyImportError("missing YAML frontmatter")
    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise LegacyImportError("frontmatter must be a mapping")
    body = raw[match.end() :]
    model_id = _required_string(metadata, "id")
    version_no = metadata.get("version")
    if not isinstance(version_no, int):
        raise LegacyImportError("version must be an integer")
    ref = ModelRef(model_id, version_no)
    rel_path = path.relative_to(root).as_posix()

    sections = {
        "definition": _required_section(body, "一句话定义"),
        "mechanism": _required_section(body, "核心机制"),
        "input": _required_section(body, "输入"),
        "transformation": _required_section(body, "转化"),
        "output": _required_section(body, "输出"),
        "boundary": _required_section(body, "适用边界"),
        "falsifier": _required_section(body, "反证"),
        "prediction": _required_section(body, "可观察预测"),
    }
    premise = _section(body, "成立前提")
    if not premise:
        premise = _applicable_boundary(sections["boundary"])
        warnings.append(f"{ref}: premise classified from the existing applicable-boundary text")
    time_window = _prediction_window(sections["prediction"])
    claims = tuple(
        ClaimSpec(
            kind=kind,
            statement=premise if kind == "premise" else sections[kind],
            time_window=time_window if kind == "prediction" else None,
            legacy_key=f"{ref}:claim:{kind}:1",
        )
        for kind in (
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
    )
    evidence = _extract_evidence(body, ref)
    categories = _category_levels(metadata)
    relations = _formal_relations(metadata, rel_path)
    created = _date_string(metadata.get("created"), "created")
    version = VersionSpec(
        ref=ref,
        status=_required_string(metadata, "status"),
        kind=_required_string(metadata, "kind"),
        formalism=_required_bool(metadata, "formalism"),
        name=_required_string(metadata, "name"),
        english_name=_required_string(metadata, "english_name"),
        created_on=created,
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        claims=claims,
        categories=categories,
        evidence=evidence,
        relations=relations,
        legacy_source_path=rel_path,
    )
    validate_version(version)
    return _ParsedCard(
        version=version,
        related=tuple(_string_list(metadata.get("related"))),
        conflicts=tuple(_string_list(metadata.get("conflicts"))),
    )


def _section(body: str, title: str) -> str:
    headings = list(re.finditer(r"^(#{2,6})\s+(.+?)\s*$", body, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(2).strip() != title:
            continue
        level = len(heading.group(1))
        end = len(body)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        return body[heading.end() : end].strip()
    return ""


def _required_section(body: str, title: str) -> str:
    value = _section(body, title)
    if not _meaningful(value):
        raise LegacyImportError(f"missing or empty section: {title}")
    return value


def _meaningful(value: str) -> bool:
    return bool(re.sub(r"[\s\-*+:`#]", "", value))


def _applicable_boundary(boundary: str) -> str:
    lines = [line.strip() for line in boundary.splitlines() if line.strip()]
    applicable = [line for line in lines if "适用" in line and "不适用" not in line]
    return "\n".join(applicable) if applicable else boundary


def _prediction_window(prediction: str) -> str:
    for line in prediction.splitlines():
        if "时间窗" in line:
            return line.strip().lstrip("-* ")
    raise LegacyImportError("prediction does not identify a time window")


def _extract_evidence(body: str, ref: ModelRef) -> tuple[EvidenceSpec, ...]:
    evidence_text = _required_section(body, "证据")
    headings = list(re.finditer(r"^###\s+(.+?)\s*$", evidence_text, re.MULTILINE))
    results: list[EvidenceSpec] = []
    for index, match in enumerate(MARKDOWN_LINK_RE.finditer(evidence_text), start=1):
        heading = ""
        for candidate in headings:
            if candidate.start() > match.start():
                break
            heading = candidate.group(1).strip()
        line_start = evidence_text.rfind("\n", 0, match.start()) + 1
        line_end = evidence_text.find("\n", match.end())
        if line_end < 0:
            line_end = len(evidence_text)
        summary = evidence_text[line_start:line_end].strip().lstrip("-* ")
        stance = _evidence_stance(heading, summary)
        claim_kind = {"supports": "mechanism", "opposes": "falsifier", "limits": "boundary"}[stance]
        results.append(
            EvidenceSpec(
                source_ref=match.group(1).strip(),
                url=match.group(2).strip(),
                summary=summary,
                stance=stance,
                claim_kind=claim_kind,
                rationale=f"Legacy {stance} mapping from the model card evidence section.",
                legacy_key=f"{ref}:evidence:{index}",
            )
        )
    if not results:
        raise LegacyImportError("evidence section has no attributable HTTP(S) source")
    return tuple(results)


def _evidence_stance(heading: str, summary: str) -> str:
    text = f"{heading} {summary}"
    limit_tokens = (
        "边界",
        "限制",
        "不自动",
        "外推",
        "适用",
        "区分",
        "反转",
        "调节",
        "个体差异",
    )
    oppose_tokens = ("反证", "反对", "竞争解释", "相反", "失效")
    if any(token in text for token in limit_tokens):
        return "limits"
    if "反证" in heading or any(token in text for token in oppose_tokens):
        return "opposes"
    return "supports"


def _category_levels(metadata: dict[str, Any]) -> tuple[CategoryLevel, ...]:
    source = _required_string(metadata, "source_category")
    discipline = _required_string(metadata, "discipline")
    subdiscipline = _required_string(metadata, "subdiscipline")
    try:
        source_slug = SOURCE_SLUGS[source]
        discipline_slug = DISCIPLINE_SLUGS[discipline]
    except KeyError as error:
        raise LegacyImportError(f"unknown formal model category: {error.args[0]}") from error
    digest = hashlib.sha256(f"{source}/{discipline}/{subdiscipline}".encode("utf-8")).hexdigest()[:16]
    return (
        CategoryLevel("source_category", source_slug, source),
        CategoryLevel("discipline", discipline_slug, discipline),
        CategoryLevel("subdiscipline", f"subdiscipline-{digest}", subdiscipline),
    )


def _formal_relations(metadata: dict[str, Any], source_path: str) -> tuple[RelationSpec, ...]:
    raw = metadata.get("model_relations") or []
    if not isinstance(raw, list):
        raise LegacyImportError("model_relations must be an array")
    result: list[RelationSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LegacyImportError("formal relation must be an object")
        target = _required_string(item, "target")
        match = MODEL_REF_RE.fullmatch(target)
        if not match:
            raise LegacyImportError(f"invalid formal relation target: {target}")
        result.append(
            RelationSpec(
                target=ModelRef(match.group(1), int(match.group(2))),
                kind=_required_string(item, "type"),
                rationale=_required_string(item, "rationale"),
                legacy_source_path=source_path,
            )
        )
    return tuple(result)


def _discovery_hint_count(cards: list[_ParsedCard]) -> int:
    by_name = {card.version.name: card.version.ref for card in cards}
    by_id = {card.version.ref.model_id: card.version.ref for card in cards}
    edges: set[tuple[str, str, str]] = set()
    for card in cards:
        source = str(card.version.ref)
        for relation, values in (("related", card.related), ("contrasts", card.conflicts)):
            for value in values:
                target_ref = by_id.get(value) or by_name.get(value)
                if target_ref is None or target_ref.model_id == card.version.ref.model_id:
                    continue
                left, right = sorted((source, str(target_ref)))
                edges.add((left, relation, right))
    return len(edges)


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LegacyImportError(f"{key} must be a non-empty string")
    return value.strip()


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise LegacyImportError(f"{key} must be boolean")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LegacyImportError("discovery hints must be string arrays")
    return value


def _date_string(value: Any, key: str) -> str:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, str):
        return date.fromisoformat(value).isoformat()
    raise LegacyImportError(f"{key} must use YYYY-MM-DD")
