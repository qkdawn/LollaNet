"""Validate LollaNet Vault structure, links, and evolvable model cards."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple


REQUIRED_DIRS = [
    "00-Start",
    "01-Models",
    "02-Checklists",
    "03-Templates",
    "04-Prompts",
    "05-Maps",
    "06-Scripts",
    "07-Docs",
    "08-Examples",
    "09-Cases",
]

MODEL_FIELDS = [
    "type",
    "id",
    "name",
    "english_name",
    "discipline",
    "kind",
    "status",
    "version",
    "created",
    "updated",
    "conflicts",
    "supersedes",
]

MODEL_SECTIONS = [
    "## 一句话定义",
    "## 核心机制",
    "## 耐久机制",
    "## 时代性表现",
    "## 适用边界",
    "## AI 调用条件",
    "## 不应调用",
    "## 场景化决策提问",
    "## 典型误用",
    "## 反向思考",
    "## 经典案例",
    "## 支持证据",
    "## 反证",
    "## 失效条件",
    "## 可观察预测",
    "## 关联模型",
    "## AI 输出要求",
    "## 修订历史",
]

MODEL_KINDS = {"world", "reasoning", "action"}
MODEL_STATUSES = {"candidate", "active", "contested", "deprecated"}
MODEL_ID_RE = re.compile(r"^mm-[a-z0-9]+(?:-[a-z0-9]+)*$")
PROPOSAL_FIELDS = [
    "id",
    "proposal_type",
    "status",
    "target_model_ids",
    "created",
    "updated",
    "source_refs",
    "privacy",
]
PROPOSAL_TYPES = {"new", "amend", "split", "merge", "contest", "deprecate"}
PROPOSAL_STATUSES = {"proposed", "accepted", "rejected", "deferred"}
PROPOSAL_ID_RE = re.compile(r"^mp-[a-z0-9]+(?:-[a-z0-9]+)*$")
CASE_FIELDS = [
    "case_name",
    "domain",
    "created",
    "updated",
    "date_observed",
    "as_of",
    "privacy",
    "provenance",
    "verification_status",
    "source_refs",
    "related_models",
]
CASE_PROVENANCE = {"public-sourced", "private-material", "fictional"}
SOURCE_PLACEHOLDERS = {"待补充", "待补充。", "待验证", "unknown"}
CASE_SECTIONS = [
    "## 案例摘要",
    "## 背景",
    "## 时代与制度条件",
    "## 关键事实",
    "## 事实、推断与待验证项",
    "## 涉及模型",
    "## 模型解释",
    "## 失败机制或关键转折",
    "## 耐久机制",
    "## 竞争性解释与反证",
    "## 来源",
    "## 模型修改提案",
]
WIKI_LINK_RE = re.compile(r"(?<!!)\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


class ModelMetadata(NamedTuple):
    path: Path
    model_id: str
    name: str
    conflicts: tuple[str, ...]
    supersedes: tuple[str, ...]


def frontmatter_block(text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    return None if end < 0 else text[4:end]


def has_frontmatter(text: str) -> bool:
    return frontmatter_block(text) is not None


def frontmatter_value(block: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*?)[ \t]*$", block)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def frontmatter_list(block: str, key: str) -> tuple[str, ...] | None:
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*?)[ \t]*$", block)
    if not match:
        return None
    inline = match.group(1).strip()
    if inline == "[]" or inline == "":
        if inline == "[]":
            return ()
        tail = block[match.end() :]
        items: list[str] = []
        for line in tail.splitlines():
            item = re.fullmatch(r"\s{2,}-\s+(.+?)\s*", line)
            if item:
                items.append(item.group(1).strip('"\''))
                continue
            if line.strip():
                break
        return tuple(items)
    if inline.startswith("[") and inline.endswith("]"):
        body = inline[1:-1].strip()
        return () if not body else tuple(part.strip().strip('"\'') for part in body.split(","))
    return (inline.strip('"\''),)


def note_stem_index(root: Path, include_private: bool = False) -> set[str]:
    return {path.stem for path in root.rglob("*.md") if not is_ignored_path(path, root, include_private)}


def is_ignored_path(path: Path, root: Path, include_private: bool = False) -> bool:
    rel = path.relative_to(root)
    if any(part.startswith(".") for part in rel.parts):
        return True
    if any(part == "__pycache__" for part in rel.parts):
        return True
    if rel.parts and rel.parts[0] in {"07-Docs", "tests"}:
        return True
    if rel.parts and rel.parts[0] == "private" and not include_private:
        return True
    return False


def should_validate_note(path: Path, root: Path, include_private: bool = False) -> bool:
    rel = path.relative_to(root)
    if is_ignored_path(path, root, include_private):
        return False
    return rel.name != "PROJECT_SPEC.md"


def validate_model_card(path: Path, text: str, rel: Path, errors: list[str]) -> ModelMetadata | None:
    block = frontmatter_block(text)
    if block is None:
        return None

    values = {key: frontmatter_value(block, key) for key in MODEL_FIELDS}
    for key, value in values.items():
        if value is None:
            errors.append(f"{rel}: missing required model field {key}")
    required_scalars = [
        "type",
        "id",
        "name",
        "english_name",
        "discipline",
        "kind",
        "status",
        "version",
        "created",
        "updated",
    ]
    for key in required_scalars:
        if values[key] == "":
            errors.append(f"{rel}: required model field {key} must not be empty")

    for section in MODEL_SECTIONS:
        if section not in text:
            errors.append(f"{rel}: missing required model section {section}")

    if "## 使用记录" in text or "当前个人使用线索" in text:
        errors.append(f"{rel}: public model cards must not contain personal usage records")

    if values["type"] not in {None, "mental_model"}:
        errors.append(f"{rel}: model type must be mental_model")

    model_id = values["id"] or ""
    if model_id and not MODEL_ID_RE.fullmatch(model_id):
        errors.append(f"{rel}: invalid model id {model_id!r}; expected ASCII mm-* identifier")

    kind = values["kind"]
    if kind is not None and kind not in MODEL_KINDS:
        errors.append(f"{rel}: invalid model kind {kind!r}")

    status = values["status"]
    if status is not None and status not in MODEL_STATUSES:
        errors.append(f"{rel}: invalid model status {status!r}")

    version = values["version"]
    if version is not None and (not version.isdigit() or int(version) < 1):
        errors.append(f"{rel}: model version must be a positive integer")

    conflicts = frontmatter_list(block, "conflicts")
    supersedes = frontmatter_list(block, "supersedes")
    if conflicts is None:
        conflicts = ()
    if supersedes is None:
        supersedes = ()

    return ModelMetadata(path, model_id, values["name"] or "", conflicts, supersedes)


def validate_model_change_proposal(text: str, rel: Path, errors: list[str]) -> tuple[str, ...]:
    block = frontmatter_block(text)
    if block is None:
        return ()

    values = {key: frontmatter_value(block, key) for key in PROPOSAL_FIELDS}
    for key in PROPOSAL_FIELDS:
        if values[key] is None:
            errors.append(f"{rel}: missing required model-change proposal field {key}")

    for key in ["id", "proposal_type", "status", "created", "updated", "privacy"]:
        if values[key] == "":
            errors.append(f"{rel}: model-change proposal field {key} must not be empty")

    proposal_type = values["proposal_type"]
    if proposal_type is not None and proposal_type not in PROPOSAL_TYPES:
        errors.append(f"{rel}: invalid proposal_type {proposal_type!r}")

    status = values["status"]
    if status is not None and status not in PROPOSAL_STATUSES:
        errors.append(f"{rel}: invalid model-change proposal status {status!r}")

    privacy = values["privacy"]
    if privacy is not None and privacy != "private":
        errors.append(f"{rel}: model-change proposal privacy must be private")

    proposal_id = values["id"]
    if proposal_id and not PROPOSAL_ID_RE.fullmatch(proposal_id):
        errors.append(f"{rel}: invalid proposal id {proposal_id!r}; expected ASCII mp-* identifier")

    expected_root = len(rel.parts) >= 3 and rel.parts[:2] == ("private", "model-proposals")
    if not expected_root:
        errors.append(f"{rel}: model-change proposals may only exist under private/model-proposals/<status>/")
    elif rel.parts[2] not in PROPOSAL_STATUSES:
        errors.append(f"{rel}: unknown model-change proposal directory {rel.parts[2]!r}")
    elif status is not None and rel.parts[2] != status:
        errors.append(f"{rel}: proposal status {status!r} must match directory {rel.parts[2]!r}")

    targets = frontmatter_list(block, "target_model_ids")
    if proposal_type != "new" and targets is not None and not targets:
        errors.append(f"{rel}: non-new model-change proposals require target_model_ids")
    return targets or ()


def validate_case_card(text: str, rel: Path, errors: list[str]) -> None:
    block = frontmatter_block(text)
    if block is None:
        return

    values = {key: frontmatter_value(block, key) for key in CASE_FIELDS}
    for key, value in values.items():
        if value is None:
            errors.append(f"{rel}: missing required case field {key}")

    for section in CASE_SECTIONS:
        if section not in text:
            errors.append(f"{rel}: missing required case section {section}")

    provenance = values["provenance"]
    if provenance is not None and provenance not in CASE_PROVENANCE:
        errors.append(f"{rel}: invalid case provenance {provenance!r}")

    for key in ["created", "updated", "as_of"]:
        value = values[key]
        if not value:
            continue
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(f"{rel}: case {key} must use YYYY-MM-DD")

    observed = values["date_observed"]
    if observed and observed != "unknown":
        try:
            date.fromisoformat(observed)
        except ValueError:
            errors.append(f"{rel}: case date_observed must use YYYY-MM-DD or unknown")

    sources = frontmatter_list(block, "source_refs") or ()
    privacy = values["privacy"]
    verification = values["verification_status"]

    if provenance == "public-sourced":
        if not rel.parts or rel.parts[0] != "09-Cases":
            errors.append(f"{rel}: public evidence cases must exist under 09-Cases/")
        if privacy != "public":
            errors.append(f"{rel}: public evidence case privacy must be public")
        if verification != "sourced":
            errors.append(f"{rel}: public evidence case verification_status must be sourced")
        if frontmatter_value(block, "public_case") != "true":
            errors.append(f"{rel}: public evidence cases require public_case: true")
        if not sources or any(source in SOURCE_PLACEHOLDERS for source in sources):
            errors.append(f"{rel}: public evidence cases require attributable source_refs")
    elif provenance == "fictional":
        if len(rel.parts) < 2 or rel.parts[:2] != ("08-Examples", "cases"):
            errors.append(f"{rel}: fictional case examples must exist under 08-Examples/cases/")
        if privacy != "public":
            errors.append(f"{rel}: fictional case example privacy must be public")
        if frontmatter_value(block, "example") != "true" or frontmatter_value(block, "fictional") != "true":
            errors.append(f"{rel}: fictional case examples require example: true and fictional: true")
        if verification != "fictional":
            errors.append(f"{rel}: fictional case verification_status must be fictional")
        if sources:
            errors.append(f"{rel}: fictional case examples must not claim source_refs")
    elif provenance == "private-material":
        expected = (
            len(rel.parts) >= 6
            and rel.parts[0] == "private"
            and rel.parts[1] == "projects"
            and rel.parts[3] == "evidence"
            and rel.parts[4] == "cases"
        )
        if not expected:
            errors.append(f"{rel}: private cases must exist under private/projects/<id>/evidence/cases/")
        if privacy != "private":
            errors.append(f"{rel}: private case privacy must be private")


def validate_vault(vault_root: Path, include_private: bool = False) -> dict[str, object]:
    root = Path(vault_root)
    errors: list[str] = []
    warnings: list[str] = []

    for directory in REQUIRED_DIRS:
        if not (root / directory).is_dir():
            errors.append(f"missing required directory: {directory}")

    note_names = note_stem_index(root, include_private)
    model_metadata: list[ModelMetadata] = []
    proposal_targets: list[tuple[Path, tuple[str, ...]]] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if not should_validate_note(path, root, include_private):
            # Public proposals are forbidden even in otherwise ignored documentation folders.
            if rel.parts and rel.parts[0] not in {"private", "tests"} and not any(
                part.startswith(".") for part in rel.parts
            ):
                ignored_text = path.read_text(encoding="utf-8")
                ignored_block = frontmatter_block(ignored_text)
                if ignored_block:
                    ignored_type = frontmatter_value(ignored_block, "type")
                    if ignored_type == "model_change_proposal":
                        proposal_targets.append((rel, validate_model_change_proposal(ignored_text, rel, errors)))
                    elif ignored_type == "case":
                        validate_case_card(ignored_text, rel, errors)
            continue
        text = path.read_text(encoding="utf-8")
        block = frontmatter_block(text)
        if block is None:
            errors.append(f"{rel}: missing YAML frontmatter")
        elif frontmatter_value(block, "type") == "model_change_proposal":
            proposal_targets.append((rel, validate_model_change_proposal(text, rel, errors)))
        elif frontmatter_value(block, "type") == "case":
            validate_case_card(text, rel, errors)
        if rel.parts and rel.parts[0] == "01-Models" and path.suffix == ".md":
            metadata = validate_model_card(path, text, rel, errors)
            if metadata:
                model_metadata.append(metadata)
        for link in WIKI_LINK_RE.findall(text):
            if link not in note_names:
                errors.append(f"{rel}: broken wiki link [[{link}]]")

    id_to_path: dict[str, Path] = {}
    model_ids = {metadata.model_id for metadata in model_metadata if metadata.model_id}
    for metadata in model_metadata:
        if metadata.model_id in id_to_path:
            first = id_to_path[metadata.model_id].relative_to(root)
            rel = metadata.path.relative_to(root)
            errors.append(f"{rel}: duplicate model id {metadata.model_id!r}; first used by {first}")
        elif metadata.model_id:
            id_to_path[metadata.model_id] = metadata.path

        rel = metadata.path.relative_to(root)
        for target in metadata.conflicts:
            if target not in model_ids:
                errors.append(f"{rel}: unknown conflicts target {target!r}")
        for target in metadata.supersedes:
            if target not in model_ids:
                errors.append(f"{rel}: unknown supersedes id {target!r}")

    for rel, targets in proposal_targets:
        for target in targets:
            if target not in model_ids:
                errors.append(f"{rel}: unknown target_model_ids entry {target!r}")

    model_count = len(list((root / "01-Models").rglob("*.md"))) if (root / "01-Models").exists() else 0
    if model_count < 30:
        warnings.append(f"model card count is {model_count}; first-step target is 30")

    return {
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LollaNet Vault structure, links, and model evolution fields.")
    parser.add_argument("--root", default=".", help="Vault root path")
    parser.add_argument("--include-private", action="store_true", help="Also validate private local workspace")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_vault(Path(args.root), include_private=args.include_private)
    for error in result["errors"]:
        print(f"ERROR: {error}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Summary: {result['error_count']} error(s), {result['warning_count']} warning(s)")
    return 1 if result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
