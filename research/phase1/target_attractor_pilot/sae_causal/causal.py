from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .contracts import AXES


@dataclass(frozen=True)
class CausalGate:
    feature: int
    axis: str
    passed: bool
    target_effect: float
    max_other_axis_ratio: float
    usefulness_drop: float
    validity_drop: float
    monotonic: bool
    better_than_random: bool


def steering_hook(direction: Any, coefficient: float, token_start: int = 0, token_stop: int = 32):
    """Steer residuals that predict generated tokens in a cached generation loop."""
    state = {"step": 0, "initialized": False}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not state["initialized"]:
            state["initialized"] = True
            step = 0
            target = -1
        elif hidden.shape[1] == 1:
            state["step"] += 1
            step = state["step"]
            target = 0
        else:
            return output
        if token_start <= step < token_stop:
            update = direction.to(device=hidden.device, dtype=hidden.dtype) * coefficient
            changed = hidden.clone()
            changed[:, target, :] += update.view(1, -1)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return output

    return hook


def build_intervention_schedule(candidates: dict[str, list[Any]], random_controls: dict[int, list[int]], strengths: list[float]) -> list[dict[str, Any]]:
    schedule = []
    for axis, rows in candidates.items():
        for row in rows:
            for feature, control in [(row.feature, False), *[(value, True) for value in random_controls[row.feature]]]:
                for sign in (-1, 1):
                    for strength in strengths:
                        schedule.append({"axis": axis, "candidate_feature": row.feature, "feature": feature,
                                         "random_control": control, "sign": sign, "strength": strength})
    return schedule


def evaluate_causal_gate(
    rows: list[dict[str, Any]], axis: str, feature: int, minimum_target_effect: float = 0.2,
    maximum_other_axis_effect_ratio: float = 0.5, maximum_quality_drop: float = 0.25,
) -> CausalGate:
    selected = [row for row in rows if int(row["candidate_feature"]) == feature]
    candidate = [row for row in selected if not bool(row["random_control"])]
    strengths = np.asarray([float(row["signed_strength"]) for row in candidate])
    changes = np.asarray([float(row[f"delta_{axis}"]) for row in candidate])
    monotonic_rho = float(spearmanr(strengths, changes).statistic) if len(candidate) >= 4 else 0.0
    monotonic_rho = monotonic_rho if np.isfinite(monotonic_rho) else 0.0
    target_effect = float(np.polyfit(strengths, changes, 1)[0]) if len(candidate) >= 2 else 0.0
    monotonic = monotonic_rho >= 0.8 and target_effect >= minimum_target_effect
    other = [abs(float(row[f"delta_{name}"])) for row in candidate for name in AXES if name != axis]
    target_magnitude = max(float(np.mean(np.abs(changes))) if len(changes) else 0.0, 1e-8)
    other_ratio = (float(np.mean(other)) / target_magnitude) if other else 0.0
    usefulness_drop = max(0.0, -float(np.mean([float(row["delta_usefulness"]) for row in candidate])))
    validity_drop = max(0.0, -float(np.mean([float(row["delta_validity"]) for row in candidate])))
    random_effects = []
    for control_feature in {int(row["feature"]) for row in selected if bool(row["random_control"])}:
        control = [row for row in selected if int(row["feature"]) == control_feature]
        control_strengths = np.asarray([float(row["signed_strength"]) for row in control])
        control_changes = np.asarray([float(row[f"delta_{axis}"]) for row in control])
        slope = float(np.polyfit(control_strengths, control_changes, 1)[0]) if len(control) >= 2 else 0.0
        random_effects.append(abs(slope))
    better_than_random = bool(random_effects) and target_effect > max(random_effects)
    passed = (monotonic and other_ratio <= maximum_other_axis_effect_ratio
              and usefulness_drop <= maximum_quality_drop and validity_drop <= maximum_quality_drop
              and better_than_random)
    return CausalGate(feature, axis, passed, target_effect, other_ratio, usefulness_drop,
                      validity_drop, monotonic, better_than_random)


def write_gate_manifest(path: Path, gates: list[CausalGate]) -> dict[str, Any]:
    passed = [gate.__dict__ for gate in gates if gate.passed]
    manifest = {
        "type": "sae_causal_feature_gate_v1", "status": "passed" if passed else "blocked_no_causal_features",
        "causal_features": passed, "all_results": [gate.__dict__ for gate in gates],
        "warning": "Association-ranked features are omitted unless they pass bidirectional causal screening.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_scored_deltas(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
