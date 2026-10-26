from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_interchange_v3 import SCORE_COLUMNS, _load_scores
from .contracts import AXES
from .free_generation_window_v8 import METHODS, POLICIES


def analyze(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    baselines = {(str(row["axis"]), str(row["pair_id"])): row for row in rows if row["condition"] == "baseline"}
    axes: dict[str, Any] = {}; passes: list[dict[str, str]] = []
    for axis in sorted({str(row["axis"]) for row in rows}):
        policies: dict[str, Any] = {}
        for policy in POLICIES:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = [row for row in rows if row["condition"] != "baseline" and row["axis"] == axis and row["policy"] == policy and row["method"] == method]
                if len(selected) != 4: raise ValueError(f"incomplete {axis}/{policy}/{method}")
                oriented: dict[str, float] = {}
                for outcome in SCORE_COLUMNS:
                    values = []
                    for row in selected:
                        base = scores[str(baselines[(axis, str(row["pair_id"]))]["record_id"])][outcome]
                        changed = scores[str(row["record_id"])][outcome]
                        values.append(changed - base if row["direction"] == "high-to-low" else base - changed)
                    oriented[outcome] = float(np.mean(values))
                quality = {}
                for outcome in ("usefulness", "validity"):
                    quality[outcome] = float(np.mean([
                        scores[str(row["record_id"])][outcome] - scores[str(baselines[(axis, str(row["pair_id"]))]["record_id"])][outcome]
                        for row in selected
                    ]))
                methods[method] = {"oriented_effects": oriented, "target_effect": oriented[axis], "usefulness_drop": max(0.0, -quality["usefulness"]), "validity_drop": max(0.0, -quality["validity"])}
            candidate, random = methods["distributed"], methods["random"]
            target = candidate["target_effect"]; other = max(abs(candidate["oriented_effects"][name]) for name in AXES if name != axis)
            candidate["cross_axis_ratio"] = other / abs(target) if abs(target) > 1e-12 else None
            candidate["difference_vs_random"] = target - random["target_effect"]
            candidate["passes_discovery_gate"] = bool(target > 0 and target > random["target_effect"] and candidate["cross_axis_ratio"] is not None and candidate["cross_axis_ratio"] <= 0.5 and candidate["usefulness_drop"] <= 0.25 and candidate["validity_drop"] <= 0.25)
            if candidate["passes_discovery_gate"]: passes.append({"axis": axis, "policy": policy})
            policies[policy] = methods
        axes[axis] = {"policies": policies}
    return {"type": "free_generation_temporal_window_analysis_v8", "status": "complete", "axes": axes, "gate_passes": passes, "evidence_boundary": "Two train-internal-holdout pairs per axis; model-assisted blind discovery scores only."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--trajectories", type=Path, required=True); parser.add_argument("--scores", type=Path, required=True); parser.add_argument("--mapping", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); rows = [json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = analyze(rows, _load_scores(args.scores, args.mapping)); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({"gate_passes": result["gate_passes"]}, indent=2))


if __name__ == "__main__": main()
