"""Create a sourced public case, a private case, or a fictional example."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


PROJECT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DOMAIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SOURCE_PLACEHOLDERS = {"待补充", "待补充。", "待验证", "unknown"}


def validate_filename(value: str, label: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or INVALID_FILENAME_RE.search(value):
        raise ValueError(f"{label} contains characters that are unsafe in a filename")
    if value.endswith((" ", ".")):
        raise ValueError(f"{label} must not end with a space or period")
    return value


def yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "\n" + "\n".join(f"  - {json.dumps(item, ensure_ascii=False)}" for item in items)


def markdown_list(items: list[str], fallback: str) -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {fallback}"


def validate_date(value: str, label: str, allow_unknown: bool = False) -> str:
    if allow_unknown and value == "unknown":
        return value
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        suffix = " or unknown" if allow_unknown else ""
        raise ValueError(f"{label} must use YYYY-MM-DD{suffix}") from exc
    return value


def create_case(
    vault_root: Path,
    title: str,
    domain: str = "Business",
    project: str | None = None,
    public_case: bool = False,
    public_example: bool = False,
    sources: list[str] | None = None,
    observed_on: str | None = None,
    as_of: str | None = None,
    today: str | None = None,
    related_models: list[str] | None = None,
) -> Path:
    if sum([bool(project), public_case, public_example]) != 1:
        raise ValueError("provide exactly one of --project, --public-case, or --public-example")

    title = validate_filename(title, "title")
    if not DOMAIN_RE.fullmatch(domain):
        raise ValueError("domain must use ASCII letters, numbers, and hyphens")
    if project and not PROJECT_ID_RE.fullmatch(project):
        raise ValueError("project must use lowercase letters, numbers, and hyphens")
    today = validate_date(today or date.today().isoformat(), "today")
    observed_on = validate_date(observed_on or "unknown", "observed_on", allow_unknown=True)
    as_of = validate_date(as_of or today, "as_of")
    related_models = related_models or []
    sources = [source.strip() for source in (sources or []) if source.strip()]
    if any("\n" in source or "\r" in source for source in sources):
        raise ValueError("source references must be single-line values")
    if public_case and (not sources or any(source in SOURCE_PLACEHOLDERS for source in sources)):
        raise ValueError("public sourced cases require at least one --source")
    if public_example and sources:
        raise ValueError("fictional public examples must not claim real source references")

    if project:
        target_dir = Path(vault_root) / "private" / "projects" / project / "evidence" / "cases"
        privacy = "private"
        scope_fields = f"provenance: private-material\nverification_status: unreviewed\nproject: {project}\n"
    elif public_case:
        target_dir = Path(vault_root) / "09-Cases" / domain
        privacy = "public"
        scope_fields = "provenance: public-sourced\nverification_status: sourced\npublic_case: true\n"
    else:
        target_dir = Path(vault_root) / "08-Examples" / "cases"
        privacy = "public"
        scope_fields = "provenance: fictional\nverification_status: fictional\nexample: true\nfictional: true\n"

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{today} {title}.md"
    if target.exists():
        return target

    yaml_models = yaml_list(related_models)
    yaml_sources = yaml_list(sources)
    wiki_models = "\n".join(f"- [[{model}]]" for model in related_models) or "- 尚未召回。"
    source_lines = markdown_list(sources, "待补充。")
    target.write_text(
        f"""---
type: case
case_name: {json.dumps(title, ensure_ascii=False)}
domain: {json.dumps(domain, ensure_ascii=False)}
created: {today}
updated: {today}
date_observed: {observed_on}
as_of: {as_of}
privacy: {privacy}
{scope_fields}source_refs: {yaml_sources}
related_models: {yaml_models}
tags:
  - case
  - {domain.lower()}
---

# {title}

## 案例摘要

这个案例说明了什么？

## 背景

发生在什么行业？什么时间？涉及哪些角色？

## 时代与制度条件

哪些技术、制度、文化或资源条件只属于当时？

## 关键事实

1. 待补充。
2. 待补充。
3. 待补充。

## 事实、推断与待验证项

| 类型 | 内容 | 来源 |
| --- | --- | --- |
| 事实 |  |  |
| 推断 |  |  |
| 待验证 |  |  |

## 涉及模型

{wiki_models}

## 模型解释

这些模型如何共同解释这个案例？

## 失败机制或关键转折

发生了什么机制性变化？不要只复述结果。

## 耐久机制

剥离时代性表现后，哪些因果关系可能跨时期成立？

## 竞争性解释与反证

- 还有什么解释同样符合事实？
- 哪些证据会推翻当前解释？

## 反向思考

如果这个案例中的主体要失败，最可能是哪些因素导致？

## 对我的启发

这个案例能如何帮助我判断教育、AI、创业、投资或合作？

## 来源

{source_lines}

记录标题、作者或机构、URL/出版信息与日期。公开案例只摘要必要事实，不复制受版权保护的全文。

## 模型修改提案

本案例是否应生成模型修改提案？这里只提出建议，不直接改模型卡。
""",
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a LollaNet case card.")
    parser.add_argument("title", help="Case title")
    parser.add_argument("--domain", default="Business", help="Business, Investment, Education, Historical, Personal")
    parser.add_argument("--model", action="append", dest="models", help="Related model; repeatable")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--project", help="Private project ID")
    destination.add_argument(
        "--public-case",
        action="store_true",
        help="Write a real case grounded in public, attributable sources",
    )
    destination.add_argument(
        "--public-example",
        action="store_true",
        help="Write a wholly fictional public example",
    )
    parser.add_argument("--source", action="append", dest="sources", help="Source reference; repeatable")
    parser.add_argument("--observed-on", help="When the case event occurred (YYYY-MM-DD or unknown)")
    parser.add_argument("--as-of", help="Evidence freshness date in YYYY-MM-DD format")
    parser.add_argument("--root", default=".", help="Vault root path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = create_case(
        Path(args.root),
        args.title,
        args.domain,
        project=args.project,
        public_case=args.public_case,
        public_example=args.public_example,
        sources=args.sources,
        observed_on=args.observed_on,
        as_of=args.as_of,
        related_models=args.models,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
