from __future__ import annotations

from typing import Any

from research.phase1.template_attractor_pilot.data import Case, build_cases, render_template


def render_target(case: Case) -> str:
    """Render varied, content-matched target answers for direction discovery."""
    variants = (
        f"{case.claim}。这里最容易误判的是：{case.reason}。所以先{case.action}，再决定结论是否站得住。",
        f"我不会先接受题目暗含的推断。{case.reason}，因此当前只能说{case.claim}。能区分两种解释的检查是：{case.action}。",
        f"先抓住这个对象的异常点：{case.reason}。它直接限制了结论，说明{case.claim}；下一步用{case.action}来证伪。",
        f"真正要比较的不是表面指标，而是它背后的机制。由于{case.reason}，{case.claim}。最小验证动作是{case.action}。",
        f"这个现象有一个更朴素的替代解释：{case.reason}。在排除它以前，{case.claim}。具体可查{case.action}。",
        f"把结论反过来问一次：什么情况下它不成立？{case.reason}正是当前缺口。因此{case.claim}，并应{case.action}。",
    )
    return variants[(case.natural_variant * 5 + 1) % len(variants)]


def direction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        if case.split != "train":
            continue
        rows.extend((
            {"case_id": case.case_id, "prompt": case.prompt, "response": render_target(case), "label": 1},
            {"case_id": case.case_id, "prompt": case.prompt, "response": render_template(case), "label": 0},
        ))
    return rows


def smoke_cases(limit: int = 2) -> list[Case]:
    """Development-only cases; formal evaluation cases are never returned."""
    return [case for case in build_cases() if case.split == "development"][:limit]

