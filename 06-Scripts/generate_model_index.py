"""Generate the static model index from canonical model metadata."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

try:
    from model_taxonomy import SOURCE_CATEGORIES
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_taxonomy import SOURCE_CATEGORIES


def scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*?)\s*$", text)
    return match.group(1).strip().strip('"\'') if match else ""


def collect_models(vault_root: Path) -> dict[str, dict[str, dict[str, list[dict[str, str]]]]]:
    grouped: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in sorted((Path(vault_root) / "01-Models").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        source_category = scalar(text, "source_category")
        discipline = scalar(text, "discipline")
        subdiscipline = scalar(text, "subdiscipline")
        if not source_category or not discipline or not subdiscipline:
            raise ValueError(f"model is missing taxonomy metadata: {path}")
        grouped[source_category][discipline][subdiscipline].append(
            {
                "name": scalar(text, "name") or path.stem,
                "kind": scalar(text, "kind"),
                "status": scalar(text, "status"),
            }
        )
    return grouped


def render_index(vault_root: Path, today: str | None = None) -> str:
    today = today or date.today().isoformat()
    grouped = collect_models(vault_root)
    lines = [
        "---",
        "type: index",
        "title: LollaNet Model Index",
        f"updated: {today}",
        "---",
        "",
        "# LollaNet Model Index",
        "",
        "> 本索引由 `06-Scripts/generate_model_index.py` 根据模型 Frontmatter 生成。它用于导航，不是固定模型路由。",
        "",
    ]
    seen_categories = set()
    for source_category in SOURCE_CATEGORIES:
        if source_category not in grouped:
            continue
        seen_categories.add(source_category)
        lines.append(f"## {source_category}")
        lines.append("")
        for discipline in sorted(grouped[source_category]):
            lines.append(f"### {discipline}")
            lines.append("")
            for subdiscipline in sorted(grouped[source_category][discipline]):
                lines.append(f"#### {subdiscipline}")
                lines.append("")
                for model in sorted(grouped[source_category][discipline][subdiscipline], key=lambda item: item["name"]):
                    suffix = f" (`{model['kind']}`, `{model['status']}`)"
                    lines.append(f"- [[{model['name']}]]{suffix}")
                lines.append("")
    unknown_categories = sorted(set(grouped) - seen_categories)
    if unknown_categories:
        raise ValueError(f"unknown source categories in model cards: {', '.join(unknown_categories)}")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the static LollaNet model index.")
    parser.add_argument("--root", default=".", help="Vault root path")
    parser.add_argument("--apply", action="store_true", help="Write 00-Start/Model Index.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    rendered = render_index(root)
    if args.apply:
        target = root / "00-Start" / "Model Index.md"
        target.write_text(rendered, encoding="utf-8")
        print(target)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
