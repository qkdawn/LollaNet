from __future__ import annotations

from itertools import product
from typing import Any, Iterable

from research.phase1.template_attractor_pilot.data import Case, build_cases


AXES = ("specificity", "independence", "non_template")


def _propositions(case: Case, specificity: int) -> tuple[str, str, str]:
    if specificity:
        return case.claim, case.reason, case.action
    return (
        "单凭题目中的表面变化还不能得到可靠结论",
        "样本构成、测量口径或其他替代解释仍未排除",
        "补充能够区分竞争解释的对照信息",
    )


def render_factorial(
    case: Case,
    specificity: int,
    independence: int,
    non_template: int,
) -> str:
    claim, reason, action = _propositions(case, specificity)
    if independence:
        opening = "先不接受问题暗含的结论。"
        statements = (
            f"目前最多只能暂定：{claim}。",
            f"一个必须先排除的替代解释是：{reason}。",
            f"真正能区分这些解释的检查是：{action}。",
        )
    else:
        opening = "题目中的变化已经给出了清楚的判断方向。"
        statements = (
            f"可以直接认为：{claim}。",
            f"原因就是：{reason}。",
            f"接下来只需用{action}确认这个判断。",
        )
    if not non_template:
        return (
            "这是一个需要从多个维度综合分析的问题。"
            f"第一，{statements[0]}第二，{statements[1]}第三，{statements[2]}"
            "综上所述，应当结合具体情况作出全面判断。"
        )
    variants = (
        f"{opening}{statements[1]}所以{statements[0]}{statements[2]}",
        f"{statements[1]}这使得{statements[0]}{opening}{statements[2]}",
        f"{opening}{statements[0]}关键不在继续解释，而在验证：{action}。因为{reason}。",
        f"先看缺口：{reason}。因此{claim}。可直接检验的是{action}。",
    )
    return variants[case.natural_variant % len(variants)]


def factorial_rows(splits: Iterable[str] = ("train", "development")) -> list[dict[str, Any]]:
    allowed = set(splits)
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        if case.split not in allowed or case.split == "evaluation":
            continue
        for specificity, independence, non_template in product((0, 1), repeat=3):
            factors = {
                "specificity": specificity,
                "independence": independence,
                "non_template": non_template,
            }
            rows.append({
                "record_id": f"{case.case_id}-s{specificity}i{independence}n{non_template}",
                "case_id": case.case_id,
                "split": case.split,
                "prompt": case.prompt,
                "response": render_factorial(case, specificity, independence, non_template),
                "anchors": list(case.anchors),
                **factors,
            })
    return rows


def matched_factor_contrasts(
    rows: list[dict[str, Any]],
    activations: dict[str, Any],
    axis: str,
) -> dict[str, Any]:
    import numpy as np

    if axis not in AXES:
        raise ValueError(f"unknown factor axis: {axis}")
    other_axes = tuple(value for value in AXES if value != axis)
    grouped: dict[tuple[Any, ...], dict[int, int]] = {}
    for index, row in enumerate(rows):
        key = (row["case_id"], *(int(row[value]) for value in other_axes))
        grouped.setdefault(key, {})[int(row[axis])] = index
    differences = []
    positive = []
    negative = []
    for pair in grouped.values():
        if set(pair) != {0, 1}:
            continue
        negative.append(activations[pair[0]])
        positive.append(activations[pair[1]])
        differences.append(activations[pair[1]] - activations[pair[0]])
    if not differences:
        raise ValueError(f"no complete contrasts for {axis}")
    return {
        "differences": np.stack(differences).astype(np.float32),
        "positive": np.stack(positive).astype(np.float32),
        "negative": np.stack(negative).astype(np.float32),
    }

