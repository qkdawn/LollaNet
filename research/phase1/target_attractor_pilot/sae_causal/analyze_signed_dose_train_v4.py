from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES
from .signed_dose_train_v4 import DIRECTIONS, NONZERO_ALPHAS


MATCHED_RANDOM = {"sae": "sae_random", "das": "das_random"}


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
    axes: dict[str, Any] = {}
    for axis in AXES:
        axis_rows = [row for row in rows if row.get("axis") == axis and row["condition"] == "signed_dose"]
        directions: dict[str, Any] = {}
        for direction, _, _ in DIRECTIONS:
            direction_rows = [row for row in axis_rows if row["direction"] == direction]
            comparisons: dict[str, Any] = {}
            for method, random_method in MATCHED_RANDOM.items():
                doses: dict[str, Any] = {}
                candidate_targets = [0.0]
                random_targets = [0.0]
                for alpha in NONZERO_ALPHAS:
                    candidate = [
                        row for row in direction_rows
                        if row["method"] == method and float(row["alpha"]) == alpha
                    ]
                    random_rows = [
                        row for row in direction_rows
                        if row["method"] == random_method and float(row["alpha"]) == alpha
                    ]
                    if len(candidate) != len(random_rows) or not candidate:
                        raise ValueError(f"incomplete {axis}/{direction}/{method}/alpha={alpha}")
                    effects = {
                        outcome: float(np.mean([
                            _effect(row, baselines[(str(row["pair_id"]), direction)], scores, outcome)
                            for row in candidate
                        ]))
                        for outcome in SCORE_COLUMNS
                    }
                    random_effects = {
                        outcome: float(np.mean([
                            _effect(row, baselines[(str(row["pair_id"]), direction)], scores, outcome)
                            for row in random_rows
                        ]))
                        for outcome in SCORE_COLUMNS
                    }
                    target = effects[axis]
                    random_target = random_effects[axis]
                    candidate_targets.append(target)
                    random_targets.append(random_target)
                    other = max(abs(effects[name]) for name in AXES if name != axis)
                    raw_usefulness = float(np.mean([
                        scores[str(row["record_id"])]["usefulness"]
                        - scores[str(baselines[(str(row["pair_id"]), direction)]["record_id"])]["usefulness"]
                        for row in candidate
                    ]))
                    raw_validity = float(np.mean([
                        scores[str(row["record_id"])]["validity"]
                        - scores[str(baselines[(str(row["pair_id"]), direction)]["record_id"])]["validity"]
                        for row in candidate
                    ]))
                    doses[str(alpha)] = {
                        "records": len(candidate),
                        "target_effect": target,
                        "matched_random_target_effect": random_target,
                        "difference_vs_matched_random": target - random_target,
                        "cross_axis_ratio": other / abs(target) if abs(target) > 1e-12 else None,
                        "usefulness_drop": max(0.0, -raw_usefulness),
                        "validity_drop": max(0.0, -raw_validity),
                        "oriented_effects": effects,
                    }
                ordered_alphas = [0.0, *NONZERO_ALPHAS]
                order = np.argsort(ordered_alphas)
                x = np.asarray(ordered_alphas)[order]
                candidate_y = np.asarray(candidate_targets)[order]
                random_y = np.asarray(random_targets)[order]
                candidate_slope = float(np.polyfit(x, candidate_y, 1)[0])
                random_slope = float(np.polyfit(x, random_y, 1)[0])
                comparisons[method] = {
                    "matched_random_method": random_method,
                    "doses": doses,
                    "candidate_signed_dose_slope": candidate_slope,
                    "matched_random_signed_dose_slope": random_slope,
                    "slope_difference": candidate_slope - random_slope,
                }
            directions[direction] = comparisons
        axes[axis] = {"directions": directions}
    return {
        "type": "signed_dose_train_only_diagnostic_v4",
        "status": "complete",
        "axes": axes,
        "evidence_boundary": (
            "Train-only model-assisted discovery diagnostic. Probe-mediated DAS is not formal DAS. "
            "This cannot establish a causal subspace or formal evaluation result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = analyze(rows, _load_scores(args.scores, args.mapping))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({axis: result["axes"][axis]["directions"] for axis in AXES}, indent=2))


if __name__ == "__main__":
    main()
