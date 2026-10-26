from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCORES = ("specificity", "independence", "non_template", "usefulness", "validity")
TARGET_AXES = ("specificity", "independence", "non_template")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_scores(path: Path, mapping: dict[str, str]) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 52 or len({row["item_id"] for row in rows}) != 52:
        raise ValueError("blind score file must contain 52 unique items")
    if {row["item_id"] for row in rows} != set(mapping):
        raise ValueError("blind score coverage differs from private mapping")
    output = {}
    for row in rows:
        values = {name: float(row[f"{name}_0_4"]) for name in SCORES}
        if any(value < 0 or value > 4 or value * 2 != round(value * 2) for value in values.values()):
            raise ValueError(f"invalid score for {row['item_id']}")
        if int(row["exclude_0_1"]) != 0:
            raise ValueError("engineering smoke contains an excluded score")
        output[mapping[row["item_id"]]] = values
    return output


def _slope(x: list[float], y: list[float]) -> float:
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.trajectories)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    scores = _read_scores(args.scores, mapping)
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    train_rho = {
        axis: float(candidates["candidates"][axis][0]["train_rho"])
        for axis in TARGET_AXES
    }
    baseline = {
        row["case_id"]: scores[row["record_id"]]
        for row in rows if row["condition"] == "baseline"
    }
    interventions = [row for row in rows if row["condition"] == "intervention"]
    detailed = []
    for row in interventions:
        item_scores = scores[row["record_id"]]
        base_scores = baseline[row["case_id"]]
        detailed.append({
            **{key: row[key] for key in (
                "record_id", "case_id", "axis", "candidate_feature", "feature",
                "random_control", "signed_strength",
            )},
            **{f"delta_{name}": item_scores[name] - base_scores[name] for name in SCORES},
        })

    summaries = {}
    for axis in TARGET_AXES:
        axis_rows = [row for row in detailed if row["axis"] == axis]
        candidate_rows = [row for row in axis_rows if not row["random_control"]]
        control_rows = [row for row in axis_rows if row["random_control"]]
        sign = 1.0 if train_rho[axis] >= 0 else -1.0
        candidate_x = [sign * row["signed_strength"] for row in candidate_rows]
        control_x = [sign * row["signed_strength"] for row in control_rows]
        target_candidate = [row[f"delta_{axis}"] for row in candidate_rows]
        target_control = [row[f"delta_{axis}"] for row in control_rows]
        target_abs = float(np.mean(np.abs(target_candidate)))
        other_abs = float(np.mean([
            abs(row[f"delta_{other}"])
            for row in candidate_rows for other in TARGET_AXES if other != axis
        ]))
        summaries[axis] = {
            "candidate_feature": int(candidate_rows[0]["candidate_feature"]),
            "candidate_train_rho": train_rho[axis],
            "candidate_expected_direction_slope": _slope(candidate_x, target_candidate),
            "matched_random_direction_slope": _slope(control_x, target_control),
            "mean_absolute_target_delta": target_abs,
            "mean_absolute_other_axis_delta": other_abs,
            "other_to_target_ratio": other_abs / max(target_abs, 1e-8),
            "mean_usefulness_delta": float(np.mean([row["delta_usefulness"] for row in candidate_rows])),
            "mean_validity_delta": float(np.mean([row["delta_validity"] for row in candidate_rows])),
            "candidate_changed_response_count": sum(
                any(abs(row[f"delta_{name}"]) > 0 for name in SCORES) for row in candidate_rows
            ),
            "engineering_smoke_pass": False,
        }
        summary = summaries[axis]
        summary["engineering_smoke_pass"] = bool(
            summary["candidate_expected_direction_slope"] > 0
            and summary["candidate_expected_direction_slope"] > abs(summary["matched_random_direction_slope"])
            and summary["other_to_target_ratio"] <= 0.5
            and summary["mean_usefulness_delta"] >= -0.25
            and summary["mean_validity_delta"] >= -0.25
        )

    report = {
        "type": "sae_causal_intervention_engineering_smoke_analysis_v2",
        "status": "engineering_smoke_only",
        "records": len(rows),
        "summaries": summaries,
        "detailed_deltas": detailed,
        "evidence_boundary": (
            "Four prompts, one candidate and one random control per axis, one absolute dose. "
            "This can reject a broken intervention path but cannot establish a causal feature."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
