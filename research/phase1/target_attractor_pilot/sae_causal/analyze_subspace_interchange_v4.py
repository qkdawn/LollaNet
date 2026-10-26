from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES


METHODS = ("sae", "sae_random", "das", "das_random")
MATCHED_RANDOM = {"sae": "sae_random", "das": "das_random"}


def _baseline_by_target(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "baseline":
            continue
        key = (str(row["pair_id"]), str(row["target_class"]))
        if key in baselines:
            raise ValueError(f"duplicate baseline for {key}")
        baselines[key] = row
    return baselines


def _norms_are_matched(
    selected: list[dict[str, Any]], random_selected: list[dict[str, Any]]
) -> bool:
    random_by_swap = {
        (str(row["pair_id"]), str(row["source_class"]), str(row["target_class"])): row
        for row in random_selected
    }
    if len(random_by_swap) != len(random_selected) or len(selected) != len(random_selected):
        return False
    for row in selected:
        key = (
            str(row["pair_id"]),
            str(row["source_class"]),
            str(row["target_class"]),
        )
        control = random_by_swap.get(key)
        if control is None or not math.isclose(
            float(row["residual_update_l2"]),
            float(control["residual_update_l2"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    return True


def analyze(
    rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]
) -> dict[str, Any]:
    baselines = _baseline_by_target(rows)
    axes: dict[str, Any] = {}
    for axis in AXES:
        axis_rows = [
            row
            for row in rows
            if row.get("condition") == "subspace_interchange"
            and row.get("axis") == axis
        ]
        present_methods = {str(row.get("method")) for row in axis_rows}
        if present_methods != set(METHODS):
            raise ValueError(f"axis {axis} does not contain all four methods")

        conditions: dict[str, Any] = {}
        for method in METHODS:
            selected = [row for row in axis_rows if row["method"] == method]
            oriented: dict[str, list[float]] = {
                outcome: [] for outcome in SCORE_COLUMNS
            }
            raw_quality: dict[str, list[float]] = {
                outcome: [] for outcome in ("usefulness", "validity")
            }
            for row in selected:
                baseline = baselines[(str(row["pair_id"]), str(row["target_class"]))]
                patched_scores = scores[str(row["record_id"])]
                baseline_scores = scores[str(baseline["record_id"])]
                high_to_low = (
                    row["source_class"] == "high" and row["target_class"] == "low"
                )
                low_to_high = (
                    row["source_class"] == "low" and row["target_class"] == "high"
                )
                if not (high_to_low or low_to_high):
                    raise ValueError(f"invalid swap direction in {row['record_id']}")
                for outcome in SCORE_COLUMNS:
                    patched_minus_baseline = (
                        patched_scores[outcome] - baseline_scores[outcome]
                    )
                    oriented[outcome].append(
                        patched_minus_baseline if high_to_low else -patched_minus_baseline
                    )
                for outcome in raw_quality:
                    raw_quality[outcome].append(
                        patched_scores[outcome] - baseline_scores[outcome]
                    )
            conditions[method] = {
                "records": len(selected),
                "oriented_effects": {
                    outcome: float(np.mean(values))
                    for outcome, values in oriented.items()
                },
                "mean_raw_quality_change": {
                    outcome: float(np.mean(values))
                    for outcome, values in raw_quality.items()
                },
                "mean_residual_update_l2": float(
                    np.mean([float(row["residual_update_l2"]) for row in selected])
                ),
            }

        comparisons: dict[str, Any] = {}
        for method, random_method in MATCHED_RANDOM.items():
            target = conditions[method]["oriented_effects"][axis]
            random_target = conditions[random_method]["oriented_effects"][axis]
            other_effect = max(
                abs(conditions[method]["oriented_effects"][other])
                for other in AXES
                if other != axis
            )
            cross_axis_ratio = other_effect / abs(target) if abs(target) > 1e-12 else None
            usefulness_drop = max(
                0.0, -conditions[method]["mean_raw_quality_change"]["usefulness"]
            )
            validity_drop = max(
                0.0, -conditions[method]["mean_raw_quality_change"]["validity"]
            )
            selected = [row for row in axis_rows if row["method"] == method]
            random_selected = [
                row for row in axis_rows if row["method"] == random_method
            ]
            positive_target = target > 0
            better_than_random = target > random_target
            cross_axis_pass = cross_axis_ratio is not None and cross_axis_ratio <= 0.5
            quality_pass = usefulness_drop <= 0.25 and validity_drop <= 0.25
            norm_matched = _norms_are_matched(selected, random_selected)
            comparisons[method] = {
                "matched_random_method": random_method,
                "target_effect": target,
                "matched_random_target_effect": random_target,
                "difference_vs_matched_random": target - random_target,
                "positive_oriented_target_effect": positive_target,
                "better_than_matched_random": better_than_random,
                "cross_axis_ratio": cross_axis_ratio,
                "cross_axis_ratio_at_most_0_5": cross_axis_pass,
                "usefulness_drop": usefulness_drop,
                "validity_drop": validity_drop,
                "quality_drop_at_most_0_25": quality_pass,
                "residual_update_norm_matched": norm_matched,
                "gate_pass": (
                    positive_target
                    and better_than_random
                    and cross_axis_pass
                    and quality_pass
                    and norm_matched
                ),
            }
        axes[axis] = {"conditions": conditions, "comparisons": comparisons}

    return {
        "type": "sae_vs_das_subspace_interchange_smoke_analysis_v4",
        "status": "complete",
        "axes": axes,
        "evidence_boundary": (
            "Two-pair development engineering smoke with model-assisted blind scores. "
            "It can compare subspace actuators with norm-matched random controls but "
            "cannot establish a causal subspace."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.trajectories.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = _load_scores(args.scores, args.mapping)
    result = analyze(rows, scores)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({axis: result["axes"][axis]["comparisons"] for axis in AXES}, indent=2))


if __name__ == "__main__":
    main()
