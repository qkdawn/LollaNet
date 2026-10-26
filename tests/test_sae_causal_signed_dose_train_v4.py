from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.contracts import AXES
from research.phase1.target_attractor_pilot.sae_causal.signed_dose_train_v4 import (
    METHODS, audit_inputs, build_plan,
)


def test_train_only_plan_has_expected_count_and_static_audit(tmp_path) -> None:
    discovery = {"pairs": {}}
    trajectories = []
    for axis in AXES:
        pairs = []
        for index in range(2):
            low, high = f"{axis}-train-{index}-low", f"{axis}-train-{index}-high"
            pairs.append({
                "pair_id": f"{axis}-train-{index}", "split": "train",
                "low_record_id": low, "high_record_id": high,
            })
            for record_id in (low, high):
                trajectories.append({"record_id": record_id, "split": "train"})
                np.save(tmp_path / f"{record_id}.npy", np.ones((16, 4), dtype=np.float32))
        pairs.append({
            "pair_id": f"{axis}-development-0", "split": "development",
            "low_record_id": "forbidden-low", "high_record_id": "forbidden-high",
        })
        discovery["pairs"][axis] = pairs
    fit = {
        "selection_split": "train", "development_accessed_during_fit": False,
        "axes": {axis: {"rank": 1} for axis in AXES},
    }
    bases = {
        f"{axis}__{method}": np.asarray([[1.0], [0.0], [0.0], [0.0]])
        for axis in AXES for method in METHODS
    }
    plan = build_plan(discovery)
    audit = audit_inputs(trajectories, discovery, fit, bases, tmp_path, 2)
    assert len(plan) == 300
    assert len({row["record_id"] for row in plan}) == 300
    assert sum(row["condition"] == "baseline" for row in plan) == 12
    assert all("development" not in row["pair_id"] for row in plan)
    assert audit["expected_records"] == 300
    assert audit["selected_record_count"] == 12
    assert audit["development_pair_signs_accessed"] is False
