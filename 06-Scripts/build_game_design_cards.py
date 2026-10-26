"""Build game-design adapter cards from the public LollaNet model library."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


TRANSFER_MODES = {"direct", "structural", "mechanic-only"}
DESIGN_LENSES = {
    "player-behavior",
    "core-loop",
    "combat",
    "economy",
    "progression",
    "level-design",
    "social",
    "content",
    "production",
    "live-ops",
    "world-rules",
}
LENS_LABELS = {
    "player-behavior": "玩家行为",
    "core-loop": "核心循环",
    "combat": "战斗系统",
    "economy": "经济与数值",
    "progression": "成长系统",
    "level-design": "关卡设计",
    "social": "社交系统",
    "content": "内容与叙事",
    "production": "研发与制作",
    "live-ops": "运营与长期生态",
    "world-rules": "世界规则与模拟",
}
MODE_LABELS = {
    "direct": "直接使用",
    "structural": "结构映射",
    "mechanic-only": "玩法机制",
}
MODE_WARNINGS = {
    "direct": "仍需用玩法数据验证，不能仅凭模型名称断言玩家会按预期行动。",
    "structural": "必须明确映射状态、输入与输出；类比本身不是玩家心理或市场事实。",
    "mechanic-only": "只用于规则、模拟或题材构造；不得据此推断乐趣、留存或付费。",
}


def parse_frontmatter(text: str, path: Path) -> dict:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        block = text.split("\n---\n", 1)[0][4:]
    except IndexError as exc:
        raise ValueError(f"{path}: malformed YAML frontmatter") from exc
    data = yaml.safe_load(block)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    return data


def extract_definition(text: str, path: Path) -> str:
    marker = "## 一句话定义"
    if marker not in text:
        raise ValueError(f"{path}: missing {marker}")
    section = text.split(marker, 1)[1]
    for line in section.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    raise ValueError(f"{path}: empty {marker}")


def load_models(root: Path) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for path in sorted((root / "01-Models").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = parse_frontmatter(text, path)
        model_id = metadata.get("id")
        if not model_id:
            raise ValueError(f"{path}: missing model id")
        if model_id in models:
            raise ValueError(f"duplicate model id: {model_id}")
        models[model_id] = {
            "id": model_id,
            "name": metadata["name"],
            "version": metadata["version"],
            "status": metadata["status"],
            "discipline": metadata["discipline"],
            "definition": extract_definition(text, path),
            "path": path.relative_to(root).as_posix(),
        }
    return models


def load_adapters(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("adapters"), list):
        raise ValueError(f"{path}: expected an adapters list")
    return data["adapters"]


def validate_adapters(models: dict[str, dict], adapters: list[dict]) -> None:
    required = {
        "source_model_id",
        "transfer_mode",
        "design_lenses",
        "translation",
        "decision",
        "prototype",
        "measure",
        "falsifier",
        "boundary",
    }
    seen: set[str] = set()
    errors: list[str] = []
    for index, adapter in enumerate(adapters, start=1):
        if not isinstance(adapter, dict):
            errors.append(f"adapter {index}: must be a mapping")
            continue
        missing = required - set(adapter)
        if missing:
            errors.append(f"adapter {index}: missing {sorted(missing)}")
            continue
        model_id = adapter["source_model_id"]
        if model_id in seen:
            errors.append(f"duplicate adapter: {model_id}")
        seen.add(model_id)
        if model_id not in models:
            errors.append(f"unknown source model: {model_id}")
        if adapter["transfer_mode"] not in TRANSFER_MODES:
            errors.append(f"{model_id}: invalid transfer_mode")
        lenses = adapter["design_lenses"]
        if not isinstance(lenses, list) or not lenses:
            errors.append(f"{model_id}: design_lenses must be a non-empty list")
        elif invalid := set(lenses) - DESIGN_LENSES:
            errors.append(f"{model_id}: invalid design_lenses {sorted(invalid)}")
        for field in required - {"source_model_id", "transfer_mode", "design_lenses"}:
            value = adapter[field]
            if not isinstance(value, str) or len(value.strip()) < 8:
                errors.append(f"{model_id}: {field} is too short")

    missing_models = set(models) - seen
    extra_models = seen - set(models)
    if missing_models:
        errors.append(f"missing adapters: {sorted(missing_models)}")
    if extra_models:
        errors.append(f"extra adapters: {sorted(extra_models)}")
    if errors:
        raise ValueError("\n".join(errors))


def quote_yaml(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_card(model: dict, adapter: dict) -> str:
    title = model["name"]
    lenses = "、".join(LENS_LABELS[item] for item in adapter["design_lenses"])
    source_link = f"../../../../{model['path']}"
    return f'''---
type: game_design_card
source_model_id: {model["id"]}
source_model_version: {model["version"]}
source_model_status: {model["status"]}
source_discipline: {quote_yaml(model["discipline"])}
transfer_mode: {adapter["transfer_mode"]}
design_lenses: {quote_yaml(adapter["design_lenses"])}
generated: true
---

# {title}

> 母模型：[{model['name']}](<{source_link}>)。证据、公式和正式边界以母模型为准。

## 母模型

- 来源：`{model['id']}@{model['version']}`（`{model['status']}`）
- 学科：{model['discipline']}
- 原始定义：{model['definition']}
- 使用方式：{MODE_LABELS[adapter['transfer_mode']]}

## 设计机制

{adapter['translation']}

## 适用域

{lenses}

## 关键决策

{adapter['decision']}

## 最小原型

{adapter['prototype']}

## 观察与反证

- 主要观察：{adapter['measure']}
- 最小反证：{adapter['falsifier']}
- 消融检查：移除本模型后，如果原型、预测或行动阈值都不变，则不应调用本卡。

## 误用边界

{adapter['boundary']} {MODE_WARNINGS[adapter['transfer_mode']]}
'''


def render_index(models: dict[str, dict], adapters: list[dict]) -> str:
    mode_counts = Counter(item["transfer_mode"] for item in adapters)
    discipline_counts = Counter(models[item["source_model_id"]]["discipline"] for item in adapters)
    lines = [
        "---",
        "type: game_design_index",
        "generated_from: 01-Models",
        "generated: true",
        "---",
        "",
        "# LollaNet 游戏策划模型",
        "",
        "本目录包含全部公共模型在游戏策划中的直接用法。这里不修改母模型、不登记正式关系，也不假定模型对所有游戏都有效。",
        "",
        "## 使用规则",
        "",
        "1. 先提出具体策划决策和最少事实，再选卡；不要按学科整包调用。",
        "2. 每次只保留会改变机制、预测、原型或行动阈值的卡。",
        "3. 直接使用类可进入策划实验；结构映射类必须明确变量；玩法机制类只用于规则、模拟或题材。",
        "4. 证据、公式和正式边界回到母卡；策划数据不能自动修改公共模型。",
        "",
        "## 覆盖",
        "",
        f"- 总卡数：{len(adapters)}",
        f"- 直接使用：{mode_counts['direct']}",
        f"- 结构映射：{mode_counts['structural']}",
        f"- 玩法机制：{mode_counts['mechanic-only']}",
        "",
        "## 按学科浏览",
        "",
    ]
    for discipline in sorted(discipline_counts):
        lines.append(f"### {discipline}（{discipline_counts[discipline]}）")
        lines.append("")
        items = [item for item in adapters if models[item["source_model_id"]]["discipline"] == discipline]
        for item in sorted(items, key=lambda value: models[value["source_model_id"]]["name"]):
            model = models[item["source_model_id"]]
            lines.append(
                f"- [{model['name']}](<Cards/{model['discipline']}/{model['name']}.md>)："
                f"{MODE_LABELS[item['transfer_mode']]}；"
                f"{'、'.join(LENS_LABELS[lens] for lens in item['design_lenses'])}"
            )
        lines.append("")
    return "\n".join(lines)


def expected_outputs(root: Path, models: dict[str, dict], adapters: list[dict]) -> dict[Path, str]:
    base = root / "05-Maps" / "Game Design"
    outputs = {base / "README.md": render_index(models, adapters)}
    for adapter in adapters:
        model = models[adapter["source_model_id"]]
        path = base / "Cards" / model["discipline"] / f"{model['name']}.md"
        outputs[path] = render_card(model, adapter)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="LollaNet repository root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write generated adapter cards")
    mode.add_argument("--check", action="store_true", help="check generated cards without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_path = root / "05-Maps" / "Game Design" / "adapters.yaml"
    models = load_models(root)
    adapters = load_adapters(data_path)
    validate_adapters(models, adapters)
    outputs = expected_outputs(root, models, adapters)

    if args.write:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"Generated {len(outputs) - 1} game-design cards and 1 index.")
        return 0

    mismatches = []
    for path, content in outputs.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            mismatches.append(path.relative_to(root).as_posix())
    cards_dir = root / "05-Maps" / "Game Design" / "Cards"
    actual_cards = set(cards_dir.rglob("*.md")) if cards_dir.exists() else set()
    expected_cards = set(outputs) - {root / "05-Maps" / "Game Design" / "README.md"}
    extra = actual_cards - expected_cards
    if mismatches or extra:
        for path in mismatches:
            print(f"STALE: {path}", file=sys.stderr)
        for path in sorted(extra):
            print(f"EXTRA: {path.relative_to(root).as_posix()}", file=sys.stderr)
        return 1
    print(f"Verified {len(expected_cards)} game-design cards and 1 index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
