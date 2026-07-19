"""Generate the read-only Phase 1 model asset index."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    from .validate_models import validate
except ImportError:  # Supports direct execution: python research/phase1/generate_index.py
    from validate_models import validate


ROOT = Path(__file__).resolve().parent
DOMAIN_NAMES = {"math": "数学", "physics": "物理", "chemistry": "化学", "engineering": "工程", "finance": "金融"}


def load_models() -> list[dict[str, Any]]:
    result = []
    for path in sorted((ROOT / "models").glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            result.append(yaml.safe_load(handle))
    return result


def render(models: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        "type: research_model_index",
        "scope: phase1_independent_assets",
        "generated_by: research/phase1/generate_index.py",
        "---",
        "",
        "# Phase 1 Model Index",
        "",
        "由 `generate_index.py` 自动生成；仅供浏览，不是训练结果。",
        "",
    ]
    for domain, chinese_name in DOMAIN_NAMES.items():
        domain_models = [model for model in models if model["domain"] == domain]
        lines.extend([f"## {chinese_name}", "", "| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |", "| --- | --- | --- | --- | --- |"])
        for model in domain_models:
            types = ", ".join(sorted({field["type"] for field in model["input_fields"]}))
            compatible = ", ".join(model["composition"]["compatible_with"])
            lines.append(f"| {model['name']} (`{model['id']}`) | {model['kind']} | {', '.join(model['functional_roles'])} | {types} | {compatible} |")
        lines.append("")
    lines.extend(["## 组合关系", ""])
    for model in models:
        for pattern in model["composition"]["composition_patterns"]:
            lines.append(f"- `{model['id']}`: {' -> '.join(pattern['route'])}。{pattern['purpose']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    errors = validate()
    if errors:
        raise SystemExit("\n".join(errors))
    (ROOT / "MODEL_INDEX.md").write_text(render(load_models()), encoding="utf-8")
    print("Generated research/phase1/MODEL_INDEX.md")


if __name__ == "__main__":
    main()
