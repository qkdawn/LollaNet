"""Create a dated LollaNet decision log."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path


def create_decision(
    vault_root: Path,
    title: str,
    project: str | None = None,
    importance: str = "medium",
    review_date: str | None = None,
    today: str | None = None,
    related_models: list[str] | None = None,
    public_example: bool = False,
) -> Path:
    today = today or date.today().isoformat()
    review_date = review_date or (date.fromisoformat(today) + timedelta(days=30)).isoformat()
    related_models = related_models or []
    if public_example:
        raise ValueError("public examples are not supported as decision logs; use 08-Examples for fictional examples")
    else:
        if not project:
            raise ValueError("Private decisions require --project <project_id>. Use --project inbox if the project is unclear.")
        target_dir = Path(vault_root) / "private" / "projects" / project / "decisions" / "open"
        project_value = project
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{today} {title}.md"
    if target.exists():
        return target
    if related_models:
        yaml_models = "related_models:\n" + "\n".join(f"  - {model}" for model in related_models)
        wiki_models = "\n".join(f"- [[{model}]]" for model in related_models)
    else:
        yaml_models = "related_models: []"
        wiki_models = "尚未召回。运行 LollaNet 分析后，只记录实际改变判断的模型。"
    target.write_text(
        f"""---
type: decision
title: {title}
project: {project_value}
date: {today}
status: open
importance: {importance}
review_date: {review_date}
{yaml_models}
tags:
  - decision
---

# {title}

## 背景

我现在要决定什么？

## 初始判断

我倾向于怎么做？为什么？

## 关键假设

1. 待补充。
2. 待补充。
3. 待补充。

## 调用模型

{wiki_models}

## 召回依据与冲突

每个模型为什么相关？哪些模型给出冲突预测？哪些候选被拒绝？

## 统计或常见默认判断

类似事情通常结果如何？

## 偏离默认的证据

哪些个人约束、机制或证据支持不同做法？

## 多时间尺度

- 近期：
- 中期：
- 长期：
- 二阶效应、可逆性与退出条件：

## 反向思考、反证与失败条件

什么事实会推翻当前判断？如果失败，最可能是哪类原因？

## 最终决策

我决定怎么做？

## 预期结果

我预计会发生什么？

## 可观察预测与置信度

- 预测：
- 时间窗口：
- 置信度：

## 复盘日期

{review_date}

## 实际结果

待复盘。

## 复盘结论

区分事实错误、模型选错、执行错误、环境变化和随机性。需要改变模型时先生成私人提案，不直接改模型卡。
""",
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a LollaNet decision log.")
    parser.add_argument("title", help="Decision title")
    parser.add_argument("--project", help="Private project id; use inbox when unclear")
    parser.add_argument("--importance", default="medium", help="low, medium, high")
    parser.add_argument("--review-date", help="Review date in YYYY-MM-DD format")
    parser.add_argument("--model", action="append", dest="models", help="Related model; repeatable")
    parser.add_argument("--public-example", action="store_true", help="Deprecated; public decision logs are not supported")
    parser.add_argument("--root", default=".", help="Vault root path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = create_decision(
        Path(args.root),
        args.title,
        project=args.project,
        importance=args.importance,
        review_date=args.review_date,
        related_models=args.models,
        public_example=args.public_example,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
