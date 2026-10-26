from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES
from .signed_weight_holdout_v5 import ALPHAS


def _oriented_effect(row: dict[str, Any], baseline: dict[str, Any], scores: dict[str, dict[str, float]], outcome: str) -> float:
    changed = scores[str(row["record_id"])][outcome]
    base = scores[str(baseline["record_id"])][outcome]
    return changed - base if row["direction"] == "high-to-low" else base - changed


def analyze(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    baselines = {
        (str(row["axis"]), str(row["pair_id"]), str(row["direction"])): row
        for row in rows if row["condition"] == "baseline"
    }
    axes: dict[str, Any] = {}
    for axis in sorted({str(row["axis"]) for row in rows}):
        doses: dict[str, Any] = {}
        for alpha in ALPHAS:
            methods: dict[str, Any] = {}
            for method in ("distributed", "random"):
                selected = [
                    row for row in rows if row["condition"] != "baseline"
                    and row["axis"] == axis and row["method"] == method
                    and float(row["alpha"]) == alpha
                ]
                if len(selected) != 4:
                    raise ValueError(f"incomplete {axis}/{method}/alpha={alpha}")
                effects = {
                    outcome: float(np.mean([
                        _oriented_effect(
                            row, baselines[(axis, str(row["pair_id"]), str(row["direction"]))],
                            scores, outcome,
                        ) for row in selected
                    ])) for outcome in SCORE_COLUMNS
                }
                quality = {
                    outcome: float(np.mean([
                        scores[str(row["record_id"])][outcome]
                        - scores[str(baselines[(axis, str(row["pair_id"]), str(row["direction"]))]["record_id"])][outcome]
                        for row in selected
                    ])) for outcome in ("usefulness", "validity")
                }
                methods[method] = {
                    "oriented_effects": effects,
                    "target_effect": effects[axis],
                    "usefulness_drop": max(0.0, -quality["usefulness"]),
                    "validity_drop": max(0.0, -quality["validity"]),
                }
            candidate, random = methods["distributed"], methods["random"]
            target = candidate["target_effect"]
            other = max(abs(candidate["oriented_effects"][name]) for name in AXES if name != axis)
            candidate["cross_axis_ratio"] = other / abs(target) if abs(target) > 1e-12 else None
            candidate["difference_vs_random"] = target - random["target_effect"]
            candidate["passes_discovery_gate"] = bool(
                target > 0 and target > random["target_effect"]
                and candidate["cross_axis_ratio"] is not None
                and candidate["cross_axis_ratio"] <= 0.5
                and candidate["usefulness_drop"] <= 0.25
                and candidate["validity_drop"] <= 0.25
            )
            doses[str(alpha)] = methods
        axes[axis] = {"doses": doses}
    passes = [
        {"axis": axis, "alpha": alpha}
        for axis, result in axes.items() for alpha, dose in result["doses"].items()
        if dose["distributed"]["passes_discovery_gate"]
    ]
    return {
        "type": "distributed_causal_intervention_analysis_v7",
        "status": "complete", "axes": axes, "gate_passes": passes,
        "evidence_boundary": (
            "Two train-internal-holdout pairs per axis with model-assisted blind scoring. "
            "Discovery-only; not development or formal evaluation evidence."
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
    print(json.dumps({"gate_passes": result["gate_passes"]}, indent=2))


if __name__ == "__main__":
    main()
