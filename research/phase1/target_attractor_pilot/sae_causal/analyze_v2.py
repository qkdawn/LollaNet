from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .analyze import load_codes
from .candidate_selection_v2 import rank_candidates_train_only
from .candidates import matched_random_features
from .confounds import residualize_axes
from .contracts import assert_discovery_rows, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-axis", type=int, default=24)
    parser.add_argument("--minimum-active-train", type=int, default=12)
    parser.add_argument("--random-controls", type=int, default=3)
    args = parser.parse_args()

    trajectories, labels = read_jsonl(args.trajectories), read_jsonl(args.labels)
    assert_discovery_rows(trajectories, expected=600)
    assert_discovery_rows(labels, expected=600)
    residuals = residualize_axes(trajectories, labels)
    codes = load_codes(args.codes)
    candidates = rank_candidates_train_only(
        codes, residuals, args.per_axis, args.minimum_active_train,
    )
    controls = {
        row.feature: matched_random_features(codes, row.feature, args.random_controls)
        for rows in candidates.values() for row in rows
    }
    result = {
        "type": "sae_association_candidates_v2",
        "status": "association_only_not_causal",
        "selection_split": "train",
        "development_role": "untouched_report_only",
        "supersedes": "association_candidates.json",
        "candidates": {axis: [asdict(row) for row in rows] for axis, rows in candidates.items()},
        "matched_random_controls": {str(key): value for key, value in controls.items()},
        "warning": "No candidate is causal until bidirectional intervention gates pass.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
