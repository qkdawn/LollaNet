from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def family_accuracy(rows: list[dict[str, Any]], families: list[str]) -> dict[str, float]:
    counts = Counter(str(row["model_family"]) for row in rows)
    if set(counts) != set(families) or any(counts[family] == 0 for family in families):
        raise ValueError("prediction rows do not cover every frozen model family")
    return {
        family: float(np.mean([bool(row["correct"]) for row in rows if row["model_family"] == family]))
        for family in families
    }


def summarize_condition(rows: list[dict[str, Any]], families: list[str]) -> dict[str, Any]:
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("condition predictions contain duplicate records")
    by_family = family_accuracy(rows, families)
    return {
        "record_count": len(rows),
        "family_accuracy": by_family,
        "macro_family_accuracy": float(np.mean(list(by_family.values()))),
        "parse_failure_rate": float(np.mean([not bool(row["parse_ok"]) for row in rows])),
    }


def choose_grid_setting(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("development grid is empty")
    identities = [(int(row["layer"]), float(row["alpha"])) for row in summaries]
    if len(identities) != len(set(identities)):
        raise ValueError("development grid contains duplicate settings")
    return min(
        summaries,
        key=lambda row: (
            -float(row["macro_family_accuracy"]),
            float(row["alpha"]),
            int(row["layer"]),
        ),
    )


def specificity_result(
    base_accuracy: float,
    correct_accuracy: float,
    wrong_accuracy: float,
    random_accuracy: float,
) -> dict[str, Any]:
    correct_delta = float(correct_accuracy - base_accuracy)
    wrong_delta = float(wrong_accuracy - base_accuracy)
    random_delta = float(random_accuracy - base_accuracy)
    passed = correct_delta > wrong_delta and correct_delta > random_delta
    return {
        "base_macro_family_accuracy": float(base_accuracy),
        "correct_macro_family_accuracy": float(correct_accuracy),
        "wrong_macro_family_accuracy": float(wrong_accuracy),
        "random_macro_family_accuracy": float(random_accuracy),
        "correct_delta": correct_delta,
        "wrong_delta": wrong_delta,
        "random_delta": random_delta,
        "passed": passed,
    }

