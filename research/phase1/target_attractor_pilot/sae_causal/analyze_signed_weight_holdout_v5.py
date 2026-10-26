from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES
from .signed_dose_train_v4 import DIRECTIONS
from .signed_weight_holdout_v5 import ALPHAS, AXIS, METHODS


def _effect(
    row: dict[str, Any], baseline: dict[str, Any], scores: dict[str, dict[str, float]], outcome: str
) -> float:
    patched = scores[str(row["record_id"])][outcome]
    base = scores[str(baseline["record_id"])][outcome]
    return patched - base if row["direction"] == "high-to-low" else base - patched


def analyze(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    baselines = {
        (str(row["pair_id"]), str(row["direction"])): row
        for row in rows if row["condition"] == "baseline"
    }
    directions: dict[str, Any] = {}
    for direction, _, _ in DIRECTIONS:
        direction_rows = [
            row for row in rows
            if row["condition"] == "signed_weight_holdout" and row["direction"] == direction
        ]
        doses: dict[str, Any] = {}
        for alpha in ALPHAS:
            effects: dict[str, dict[str, float]] = {}
            for method in METHODS:
                method_rows = [
                    row for row in direction_rows
                    if row["method"] == method and float(row["alpha"]) == alpha
                ]
                if len(method_rows) != 2:
                    raise ValueError(f"incomplete {direction}/{method}/alpha={alpha}")
                effects[method] = {
                    outcome: float(np.mean([
                        _effect(
                            row,
                            baselines[(str(row["pair_id"]), direction)],
                            scores,
                            outcome,
                        )
                        for row in method_rows
                    ]))
                    for outcome in SCORE_COLUMNS
                }
            candidate = effects["signed_weighted"]
            target = candidate[AXIS]
            other = max(abs(candidate[name]) for name in AXES if name != AXIS)
            candidate_rows = [
                row for row in direction_rows
                if row["method"] == "signed_weighted" and float(row["alpha"]) == alpha
            ]
            raw_usefulness = float(np.mean([
                scores[str(row["record_id"])]["usefulness"]
                - scores[str(baselines[(str(row["pair_id"]), direction)]["record_id"])]["usefulness"]
                for row in candidate_rows
            ]))
            raw_validity = float(np.mean([
                scores[str(row["record_id"])]["validity"]
                - scores[str(baselines[(str(row["pair_id"]), direction)]["record_id"])]["validity"]
                for row in candidate_rows
            ]))
            equal_target = effects["equal_weight"][AXIS]
            random_target = effects["random_weighted"][AXIS]
            cross_ratio = other / abs(target) if abs(target) > 1e-12 else None
            usefulness_drop = max(0.0, -raw_usefulness)
            validity_drop = max(0.0, -raw_validity)
            gate = bool(
                target > 0
                and target > equal_target
                and target > random_target
                and cross_ratio is not None
                and cross_ratio <= 0.5
                and usefulness_drop <= 0.25
                and validity_drop <= 0.25
            )
            doses[str(alpha)] = {
                "records_per_method": 2,
                "signed_weighted_target_effect": target,
                "equal_weight_target_effect": equal_target,
                "random_weighted_target_effect": random_target,
                "difference_vs_equal_weight": target - equal_target,
                "difference_vs_random_weighted": target - random_target,
                "cross_axis_ratio": cross_ratio,
                "usefulness_drop": usefulness_drop,
                "validity_drop": validity_drop,
                "signed_weighted_oriented_effects": candidate,
                "passes_discovery_gate": gate,
            }
        directions[direction] = {"doses": doses}
    passes = [
        {"direction": direction, "alpha": alpha}
        for direction, result in directions.items()
        for alpha, dose in result["doses"].items()
        if dose["passes_discovery_gate"]
    ]
    return {
        "type": "signed_relative_feature_weight_holdout_analysis_v5",
        "status": "complete",
        "axis": AXIS,
        "directions": directions,
        "gate_passes": passes,
        "evidence_boundary": (
            "Two train-internal-holdout pairs with model-assisted blind scoring. "
            "This is discovery-only and is not development or formal evaluation evidence."
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
        json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze(rows, _load_scores(args.scores, args.mapping))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["directions"], indent=2))


if __name__ == "__main__":
    main()
