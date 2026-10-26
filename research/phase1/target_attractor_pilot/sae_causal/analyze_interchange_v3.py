from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AXES


SCORE_COLUMNS = {
    "specificity": "specificity_0_4",
    "independence": "independence_0_4",
    "non_template": "non_template_0_4",
    "usefulness": "usefulness_0_4",
    "validity": "validity_0_4",
}


def _load_scores(scores_path: Path, mapping_path: Path) -> dict[str, dict[str, float]]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, float]] = {}
    with scores_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["specificity_0_4"]:
                raise ValueError("blind score sheet is incomplete")
            if int(row["exclude_0_1"] or 0):
                continue
            score = {
                name: float(row[column]) for name, column in SCORE_COLUMNS.items()
            }
            mapped = mapping[row["item_id"]]
            record_ids = mapped if isinstance(mapped, list) else [mapped]
            for record_id in record_ids:
                key = str(record_id)
                if key in output:
                    raise ValueError(f"record {key} has more than one blind score")
                output[key] = score
    return output


def analyze(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    baselines = {
        (str(row["pair_id"]), str(row["target_class"])): row
        for row in rows
        if row["condition"] == "baseline"
    }
    axes: dict[str, Any] = {}
    for axis in AXES:
        axis_rows = [
            row for row in rows
            if row["condition"] == "interchange" and row["axis"] == axis
        ]
        conditions: dict[str, Any] = {}
        for random_control, name in ((False, "candidate"), (True, "random")):
            selected = [row for row in axis_rows if bool(row["random_control"]) == random_control]
            oriented: dict[str, list[float]] = {name: [] for name in SCORE_COLUMNS}
            raw_quality: dict[str, list[float]] = {name: [] for name in ("usefulness", "validity")}
            for row in selected:
                baseline = baselines[(str(row["pair_id"]), str(row["target_class"]))]
                intervention_scores = scores[str(row["record_id"])]
                baseline_scores = scores[str(baseline["record_id"])]
                sign = 1.0 if row["source_class"] == "high" else -1.0
                for outcome in SCORE_COLUMNS:
                    difference = intervention_scores[outcome] - baseline_scores[outcome]
                    oriented[outcome].append(sign * difference)
                for outcome in raw_quality:
                    raw_quality[outcome].append(
                        intervention_scores[outcome] - baseline_scores[outcome]
                    )
            conditions[name] = {
                "records": len(selected),
                "oriented_effects": {
                    outcome: float(np.mean(values)) for outcome, values in oriented.items()
                },
                "mean_raw_quality_change": {
                    outcome: float(np.mean(values)) for outcome, values in raw_quality.items()
                },
                "mean_residual_update_l2": float(
                    np.mean([float(row["residual_update_l2"]) for row in selected])
                ),
            }
        target = conditions["candidate"]["oriented_effects"][axis]
        random_target = conditions["random"]["oriented_effects"][axis]
        other_effect = max(
            abs(conditions["candidate"]["oriented_effects"][other])
            for other in AXES if other != axis
        )
        cross_axis_ratio = other_effect / abs(target) if abs(target) > 1e-12 else None
        usefulness_drop = max(
            0.0, -conditions["candidate"]["mean_raw_quality_change"]["usefulness"]
        )
        validity_drop = max(
            0.0, -conditions["candidate"]["mean_raw_quality_change"]["validity"]
        )
        axes[axis] = {
            "candidate_feature": int(axis_rows[0]["candidate_feature"]),
            "random_feature": int(next(row for row in axis_rows if row["random_control"])["feature"]),
            "conditions": conditions,
            "engineering_checks": {
                "positive_oriented_target_effect": target > 0,
                "better_than_random": target > random_target,
                "cross_axis_ratio": cross_axis_ratio,
                "cross_axis_ratio_at_most_0_5": (
                    cross_axis_ratio is not None and cross_axis_ratio <= 0.5
                ),
                "usefulness_drop": usefulness_drop,
                "validity_drop": validity_drop,
                "quality_drop_at_most_0_25": usefulness_drop <= 0.25 and validity_drop <= 0.25,
            },
        }
    return {
        "type": "sae_interchange_engineering_smoke_analysis_v3b",
        "status": "complete",
        "axes": axes,
        "evidence_boundary": (
            "Two activation-selected development pairs per axis and model-assisted blind scores. "
            "This can validate or reject the interchange actuator but cannot establish a causal feature."
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
        if line
    ]
    scores = _load_scores(args.scores, args.mapping)
    result = analyze(rows, scores)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        axis: result["axes"][axis]["engineering_checks"] for axis in AXES
    }, indent=2))


if __name__ == "__main__":
    main()
