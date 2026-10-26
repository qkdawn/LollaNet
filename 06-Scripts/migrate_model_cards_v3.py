"""Migrate legacy model cards to the compact mechanism-interface schema.

This is a one-time, deterministic structural migration. It does not invent
formal relations or change a model's substantive claims; it reuses the card's
existing mechanism, evidence, boundaries, predictions, and formalism.
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


TODAY = "2026-07-31"
MODEL_REF_RE = re.compile(r"^(mm-[a-z0-9-]+)@([1-9][0-9]*)$")


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if not match:
        raise ValueError("missing YAML frontmatter")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a mapping")
    return data, text[match.end() :]


def section(body: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$([\s\S]*?)(?=^## |\Z)",
        re.MULTILINE,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def any_level_section(body: str, heading: str) -> str:
    pattern = re.compile(
        rf"^(#{{2,6}}) {re.escape(heading)}\s*$([\s\S]*?)(?=^#{{2,6}} |\Z)",
        re.MULTILINE,
    )
    match = pattern.search(body)
    return match.group(2).strip() if match else ""


def first_paragraph(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[0] if paragraphs else ""


def compact_line(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text if len(text) <= limit else text[: limit - 1].rstrip("，。； ") + "。"


def bullet_payload(text: str, marker: str) -> str:
    match = re.search(rf"(?:^|\n)\s*[-*]\s*{re.escape(marker)}\s*[:：]\s*(.+)", text)
    return compact_line(match.group(1)) if match else ""


def prediction_lines(text: str) -> list[str]:
    preferred = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        if any(token in stripped for token in ("预测对象", "如果模型成立", "区分", "若", "应", "时间窗")):
            preferred.append(compact_line(stripped))
    if preferred:
        return preferred[:2]
    return [compact_line(line.strip()) for line in text.splitlines() if line.strip().startswith(("-", "*"))][:2]


def falsifier_lines(text: str, fallback: str) -> list[str]:
    preferred = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        if any(token in stripped for token in ("削弱", "替代解释", "转为", "失效", "不能证明", "不支持")):
            preferred.append(compact_line(stripped))
    if preferred:
        return preferred[:3]
    fallback_lines = [compact_line(line.strip()) for line in fallback.splitlines() if line.strip().startswith(("-", "*"))]
    return fallback_lines[:2] or ["若输出不随机制所需变量变化，降低该机制的置信度。"]


def yaml_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {key: yaml_value(item) for key, item in value.items()}
    return value


def frontmatter(data: dict[str, Any], version_map: dict[str, str]) -> str:
    keep = [
        "type", "id", "name", "english_name", "discipline", "source_category",
        "subdiscipline", "formalism", "kind", "status", "version", "created",
        "updated", "related", "conflicts", "model_relations", "supersedes",
    ]
    result: dict[str, Any] = {}
    for key in keep:
        if key in data:
            result[key] = yaml_value(data[key])
    model_id = str(result["id"])
    old_ref = f"{model_id}@{data['version']}"
    new_ref = version_map.get(old_ref, f"{model_id}@{int(data['version']) + 1}")
    result["version"] = int(new_ref.rsplit("@", 1)[1])
    result["updated"] = TODAY
    relations = result.get("model_relations") or []
    for relation in relations:
        target = relation.get("target") if isinstance(relation, dict) else None
        if isinstance(target, str) and target in version_map:
            relation["target"] = version_map[target]
    result["model_relations"] = relations
    dumped = yaml.safe_dump(result, allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip()
    return f"---\n{dumped}\n---\n"


def migrate_card(path: Path, version_map: dict[str, str]) -> bool:
    text = path.read_text(encoding="utf-8")
    data, body = split_frontmatter(text)
    if "## 机制接口" in body:
        return False

    name = str(data.get("name") or path.stem)
    one = section(body, "一句话定义")
    core = section(body, "核心机制")
    durable = section(body, "耐久机制")
    boundary = section(body, "适用边界")
    failure = section(body, "失效条件")
    support = section(body, "支持证据")
    counter = section(body, "反证")
    prediction = section(body, "可观察预测")
    formal = section(body, "形式化内容")
    revision = section(body, "修订历史")
    ai_output = section(body, "AI 输出要求")
    sources = section(body, "来源")
    classic = section(body, "经典案例")

    input_text = bullet_payload(ai_output, "关键变量")
    if not input_text:
        input_text = compact_line(first_paragraph(core)) or "核心机制所需的变量、前提和观察窗口。"

    transform = compact_line(first_paragraph(core)) or durable or "按该模型的核心机制将输入映射为可观察结果。"
    outputs = prediction_lines(prediction)
    if not outputs:
        outputs = ["输出应随核心机制所需变量变化，并在预定观察窗口复核。"]
    falsifiers = falsifier_lines(counter, boundary + "\n" + failure)

    evidence_parts: list[str] = []
    if support:
        evidence_parts.append("### 支持\n\n" + support)
    elif sources:
        evidence_parts.append("### 来源\n\n" + sources)
    elif classic:
        evidence_parts.append("### 既有证据\n\n" + classic)
    if counter:
        evidence_parts.append("### 反证与替代解释\n\n" + counter)
    if not evidence_parts:
        evidence_parts.append("- 当前卡片没有可回引证据；保持为待补充状态。")

    relation_items = data.get("model_relations") or []
    if relation_items:
        relations = []
        for item in relation_items:
            target = version_map.get(item["target"], item["target"])
            relations.append(
                f"- `{target}` via `{item['type']}`：{item.get('rationale', '').strip()}"
            )
        relation_text = "\n".join(relations)
    else:
        relation_text = "暂无正式关系；`related` 与 `conflicts` 仅用于发现候选，不表示任务中已激活。"

    history = revision or "| 日期 | 版本 | 状态 | 变更 | 提案 |\n| --- | --- | --- | --- | --- |"
    old_version = int(data["version"])
    proposal_id = "mp-2026-07-31-compact-model-card-batch-2"
    migration_row = f"| {TODAY} | {old_version + 1} | {data.get('status', 'candidate')} | 迁移为统一机制接口，保留原机制、证据、边界与预测 | {proposal_id} |"
    history_lines = history.splitlines()
    if len(history_lines) >= 2 and history_lines[0].lstrip().startswith("|") and "---" in history_lines[1]:
        history_lines.insert(2, migration_row)
        history = "\n".join(history_lines)

    parts = [
        f"# {name}",
        "## 一句话定义\n\n" + (one or "用一个可检验机制说明输入如何改变输出。"),
        "## 核心机制\n\n" + (core or durable or "该卡尚未提供核心机制描述。"),
        "## 机制接口",
        "### 输入\n\n" + input_text,
        "### 转化\n\n" + transform,
        "### 输出\n\n" + "\n".join(f"- {line.lstrip('-* ')}" for line in outputs),
        "### 反证\n\n" + "\n".join(f"- {line.lstrip('-* ')}" for line in falsifiers),
        "## 适用边界\n\n" + (boundary or "- 适用边界需要结合任务事实判断。"),
    ]
    if failure:
        parts.append("### 失效条件\n\n" + failure)
    parts.extend([
        "## 证据\n\n" + "\n\n".join(evidence_parts),
        "## 可观察预测\n\n" + (prediction or "- 对象与时间窗：在预定观察期检查输出是否随机制变量变化。\n- 区分替代解释：若不随机制变量变化，降低本模型置信度。"),
        "## 正式关系\n\n" + relation_text,
    ])
    if str(data.get("formalism", "false")).lower() == "true":
        parts.append("## 形式化内容\n\n" + (formal or "形式化内容待补充；不得在缺少定义和前提时伪造公式。"))
    parts.append("## 修订历史\n\n" + history)
    new_text = frontmatter(data, version_map) + "\n" + "\n\n".join(parts).rstrip() + "\n"
    path.write_text(new_text, encoding="utf-8")
    return True


def refine_duplicate_input(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    _, body = split_frontmatter(text)
    core = any_level_section(body, "核心机制")
    current_input = any_level_section(body, "输入")
    if not core or re.sub(r"\s+", " ", core) != re.sub(r"\s+", " ", current_input):
        return False
    definition = any_level_section(body, "定义与符号")
    if definition:
        replacement = compact_line(first_paragraph(definition))
    else:
        replacement = compact_line(re.split(r"(?<=。)", first_paragraph(core), maxsplit=1)[0])
    pattern = re.compile(r"(^### 输入[ \t]*$)\r?\n[\s\S]*?(?=^### |^## |\Z)", re.MULTILINE)
    updated, count = pattern.subn(r"\1\n\n" + replacement + "\n", text, count=1)
    if count != 1:
        raise ValueError(f"could not replace input section: {path}")
    path.write_text(updated, encoding="utf-8")
    return True


def normalize_markdown_spacing(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = re.sub(r"\n+(?=^#{1,6} )", "\n\n", text, flags=re.MULTILINE)
    updated = re.sub(r"^(#{1,6} [^\n]+)\n+", r"\1\n\n", updated, flags=re.MULTILINE)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main(root: Path) -> int:
    paths = sorted((root / "01-Models").rglob("*.md"))
    cards: list[tuple[Path, dict[str, Any], str]] = []
    for path in paths:
        data, body = split_frontmatter(path.read_text(encoding="utf-8"))
        cards.append((path, data, body))
    version_map = {
        f"{data['id']}@{data['version']}": f"{data['id']}@{int(data['version']) + 1}"
        for _, data, body in cards
        if "## 机制接口" not in body
    }
    changed = sum(migrate_card(path, version_map) for path, _, _ in cards)
    refined = sum(refine_duplicate_input(path) for path, _, _ in cards)
    formatted = sum(normalize_markdown_spacing(path) for path, _, _ in cards)
    print(
        f"migrated={changed} skipped={len(cards) - changed} "
        f"refined_inputs={refined} formatted={formatted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()))
