from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES
from .signed_dose_train_v4 import DIRECTIONS
from .signed_weight_holdout_v5 import ALPHAS, AXIS
from .temporal_policy_holdout_v6 import POLICY_METHODS


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
    conditions: dict[str, Any] = {}
    for policy, method in POLICY_METHODS:
        key = f"{policy}__{method}"
        doses: dict[str, Any] = {}
        for alpha in ALPHAS:
            selected = [
                row for row in rows
                if row.get("policy") == policy and row.get("method") == method
                and float(row["alpha"]) == alpha
            ]
            if len(selected) != 4:
                raise ValueError(f"incomplete {key}/alpha={alpha}: {len(selected)} records")
            effects = {
                outcome: float(np.mean([
                    _effect(
                        row,
                        baselines[(str(row["pair_id"]), str(row["direction"]))],
                        scores,
                        outcome,
                    )
                    for row in selected
                ]))
                for outcome in SCORE_COLUMNS
            }
            target = effects[AXIS]
            other = max(abs(effects[name]) for name in AXES if name != AXIS)
            raw_quality = {
                outcome: float(np.mean([
                    scores[str(row["record_id"])][outcome]
                    - scores[str(baselines[(str(row["pair_id"]), str(row["direction"]))]["record_id"])][outcome]
                    for row in selected
                ]))
                for outcome in ("usefulness", "validity")
            }
            doses[str(alpha)] = {
                "records": len(selected),
                "oriented_effects": effects,
                "target_effect": target,
                "cross_axis_ratio": other / abs(target) if abs(target) > 1e-12 else None,
                "usefulness_drop": max(0.0, -raw_quality["usefulness"]),
                "validity_drop": max(0.0, -raw_quality["validity"]),
                "mean_actual_energy_ratio": float(np.mean([
                    float(row["applied_update_energy_l2"]) / float(row["energy_budget"])
                    for row in selected
                ])),
            }
        conditions[key] = {"policy": policy, "method": method, "doses": doses}

    gate_passes: list[dict[str, Any]] = []
    for policy in ("persistent", "closed_loop"):
        candidate = conditions[f"{policy}__signed_weighted"]
        random = conditions[f"{policy}__random_weighted"]
        for alpha in ALPHAS:
            dose = candidate["doses"][str(alpha)]
            random_target = random["doses"][str(alpha)]["target_effect"]
            equal_target = None
            if policy == "persistent":
                equal_target = conditions["persistent__equal_weight"]["doses"][str(alpha)]["target_effect"]
            gate = bool(
                dose["target_effect"] > 0
                and dose["target_effect"] > random_target
                and (equal_target is None or dose["target_effect"] > equal_target)
                and dose["cross_axis_ratio"] is not None
                and dose["cross_axis_ratio"] <= 0.5
                and dose["usefulness_drop"] <= 0.25
                and dose["validity_drop"] <= 0.25
            )
            dose["random_target_effect"] = random_target
            dose["equal_target_effect"] = equal_target
            dose["passes_discovery_gate"] = gate
            if gate:
                gate_passes.append({"policy": policy, "alpha": alpha})
    return {
        "type": "temporal_policy_holdout_analysis_v6",
        "status": "complete",
        "axis": AXIS,
        "conditions": conditions,
        "gate_passes": gate_passes,
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
    print(json.dumps({"gate_passes": result["gate_passes"]}, indent=2))


if __name__ == "__main__":
    main()
