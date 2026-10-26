from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AXES


WINDOWS = (4, 8, 16)


def _feature_trajectory(indices: np.ndarray, values: np.ndarray, feature: int) -> np.ndarray:
    return np.where(indices == feature, values, 0.0).sum(axis=-1)


def _window_statistics(
    trajectory: np.ndarray,
    record_index: dict[str, int],
    pairs: list[dict[str, Any]],
    window: int,
) -> dict[str, float]:
    differences = []
    for pair in pairs:
        low = trajectory[record_index[str(pair["low_record_id"])], :window].mean()
        high = trajectory[record_index[str(pair["high_record_id"])], :window].mean()
        differences.append(float(high - low))
    values = np.asarray(differences, dtype=np.float64)
    active = values != 0
    active_values = values[active]
    if not active_values.size:
        return {
            "pairs": len(values), "active_pairs": 0, "mean_delta": 0.0,
            "median_delta": 0.0, "sign_consistency": 0.0,
        }
    median = float(np.median(active_values))
    direction = 1 if median >= 0 else -1
    return {
        "pairs": len(values),
        "active_pairs": int(active.sum()),
        "mean_delta": float(active_values.mean()),
        "median_delta": median,
        "sign_consistency": float(np.mean(np.sign(active_values) == direction)),
    }


def analyze(discovery: dict[str, Any], codes_path: Path) -> dict[str, Any]:
    with np.load(codes_path) as archive:
        record_ids = archive["record_ids"].tolist()
        indices = archive["indices"].copy()
        values = archive["values"].copy()
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    axes: dict[str, Any] = {}
    for axis in AXES:
        candidates = discovery["axes"][axis]["candidate_sets"]["union_for_temporal_attribution"]
        train_pairs = [pair for pair in discovery["pairs"][axis] if pair["split"] == "train"]
        development_pairs = [
            pair for pair in discovery["pairs"][axis] if pair["split"] == "development"
        ]
        rows = []
        for feature in candidates:
            trajectory = _feature_trajectory(indices, values, int(feature))
            train = {
                str(window): _window_statistics(
                    trajectory, record_index, train_pairs, window
                )
                for window in WINDOWS
            }
            development = {
                str(window): _window_statistics(
                    trajectory, record_index, development_pairs, window
                )
                for window in WINDOWS
            }
            nonzero_directions = [
                np.sign(train[str(window)]["mean_delta"])
                for window in WINDOWS
                if train[str(window)]["mean_delta"] != 0
            ]
            direction_stable = bool(
                len(nonzero_directions) == len(WINDOWS)
                and len(set(nonzero_directions)) == 1
            )
            rows.append({
                "feature": int(feature),
                "train_windows": train,
                "development_windows_report_only": development,
                "train_early_direction_stable": direction_stable,
            })
        eligible = [
            row for row in rows
            if row["train_early_direction_stable"]
            and row["train_windows"]["4"]["active_pairs"] >= 5
            and row["train_windows"]["16"]["sign_consistency"] >= 0.6
        ]
        eligible.sort(key=lambda row: (
            row["train_windows"]["4"]["sign_consistency"],
            row["train_windows"]["4"]["active_pairs"],
            abs(row["train_windows"]["4"]["mean_delta"]),
        ), reverse=True)
        axes[axis] = {
            "candidates_examined": len(rows),
            "early_available_candidates": [row["feature"] for row in eligible],
            "features": rows,
        }
    return {
        "type": "sae_temporal_candidate_analysis_v3",
        "status": "ready_for_interchange_patching",
        "selection_split": "train",
        "development_use": "report_only",
        "windows": list(WINDOWS),
        "axes": axes,
        "evidence_boundary": (
            "Early availability is not temporal causality. Interchange patching is required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    result = analyze(discovery, args.codes)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        axis: {
            "examined": result["axes"][axis]["candidates_examined"],
            "early_available": len(result["axes"][axis]["early_available_candidates"]),
            "top": result["axes"][axis]["early_available_candidates"][:8],
        }
        for axis in AXES
    }, indent=2))


if __name__ == "__main__":
    main()
