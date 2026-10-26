"""Audit fact-specific task relevance and applied counterfactual microstates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .question_set_v1 import QUESTION_KINDS, ValidationError, read_jsonl

NON_TASK_AXES = {"status", "intensity", "direction", "temporal_persistence"}


def _task_vector(episode: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    basis_vector = episode.get("task_relevance_basis", {}).get("vector")
    if isinstance(basis_vector, dict):
        return tuple(sorted((concept_id, int(value)) for concept_id, value in basis_vector.items()))
    return tuple(
        (state["microconcept_id"], state["task_relevance"])
        for state in sorted(episode["microconcept_states"], key=lambda item: item["microconcept_id"])
    )


def audit_question_conditioning(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return corpus-level evidence for fact specificity and delta application."""
    rows = list(episodes)
    if any(row.get("split") == "sealed_test" for row in rows):
        raise ValidationError("question-conditioning audit accepts open train/development Episodes only")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("question_kind")].append(row)

    relevance: dict[str, dict[str, Any]] = {}
    for kind in QUESTION_KINDS:
        kind_rows = grouped.get(kind, [])
        vectors = [_task_vector(row) for row in kind_rows]
        counts = Counter(vectors)
        pair_count = 0
        differing_pairs = 0
        for left, right in combinations(vectors, 2):
            pair_count += 1
            differing_pairs += int(left != right)
        relevance[kind] = {
            "episode_count": len(kind_rows),
            "unique_vector_count": len(counts),
            "dominant_vector_count": max(counts.values(), default=0),
            "dominant_vector_share": max(counts.values(), default=0) / len(vectors) if vectors else 0.0,
            "same_kind_pair_count": pair_count,
            "same_kind_differing_pair_count": differing_pairs,
            "same_kind_differing_pair_rate": differing_pairs / pair_count if pair_count else 0.0,
        }

    declared_axes = Counter()
    declared_changes = 0
    applied_changes = 0
    counterfactual_episodes = 0
    counterfactual_with_non_task_delta = 0
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        delta = row.get("counterfactual_delta")
        if not delta:
            continue
        counterfactual_episodes += 1
        changes = delta.get("microconcept_state_changes", [])
        if any(change.get("axis") in NON_TASK_AXES for change in changes):
            counterfactual_with_non_task_delta += 1
        states = {state["microconcept_id"]: state for state in row["microconcept_states"]}
        for change in changes:
            declared_changes += 1
            axis = change.get("axis")
            declared_axes[axis] += 1
            state = states.get(change.get("microconcept_id"))
            actual = None if state is None else state.get(axis)
            if actual == change.get("after"):
                applied_changes += 1
            else:
                mismatches.append({
                    "episode_id": row.get("episode_id"),
                    "microconcept_id": change.get("microconcept_id"),
                    "axis": axis,
                    "expected_after": change.get("after"),
                    "actual": actual,
                })

    fact_specific = all(
        item["episode_count"] <= 1
        or (item["unique_vector_count"] >= 2 and item["same_kind_differing_pair_rate"] >= 0.25)
        for item in relevance.values()
    )
    no_dominant_fixed_template = all(
        item["episode_count"] < 4 or item["dominant_vector_share"] <= 0.75
        for item in relevance.values()
    )
    all_counterfactuals_have_non_task_delta = (
        counterfactual_episodes > 0 and counterfactual_episodes == counterfactual_with_non_task_delta
    )
    all_declared_changes_applied = (
        counterfactual_episodes > 0 and declared_changes == applied_changes and not mismatches
    )

    return {
        "artifact": "mechanism-question-conditioning-audit-v1",
        "episodes": len(rows),
        "question_kind_task_relevance": relevance,
        "counterfactuals": {
            "episode_count": counterfactual_episodes,
            "with_non_task_state_delta": counterfactual_with_non_task_delta,
            "declared_state_changes": declared_changes,
            "applied_state_changes": applied_changes,
            "declared_axes": dict(sorted(declared_axes.items())),
            "mismatches": mismatches,
        },
        "checks": {
            "same_kind_fact_specific_task_relevance": fact_specific,
            "no_near_fixed_question_kind_template": no_dominant_fixed_template,
            "all_counterfactuals_declare_non_task_state_delta": all_counterfactuals_have_non_task_delta,
            "all_declared_counterfactual_state_changes_applied": all_declared_changes_applied,
        },
    }


def validate_question_conditioning_audit(report: dict[str, Any]) -> None:
    failed = [name for name, passed in report.get("checks", {}).items() if not passed]
    if failed:
        raise ValidationError(f"question-conditioning audit failed: {', '.join(failed)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = audit_question_conditioning(read_jsonl(args.episodes))
    validate_question_conditioning_audit(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
