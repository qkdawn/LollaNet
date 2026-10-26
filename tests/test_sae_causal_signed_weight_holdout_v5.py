from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.signed_weight_holdout_v5 import (
    build_plan,
    select_holdout_pairs,
)


def _fit() -> dict:
    return {
        "development_accessed_during_fit": False,
        "axes": {
            "non_template": {
                "fit": {"passes_train_only_gate": True},
                "internal_holdout_pair_ids": ["p1", "p2"],
                "fit_case_ids": ["fit-case"],
            }
        },
    }


def test_holdout_plan_has_40_unique_records_and_no_development() -> None:
    discovery = {
        "pairs": {
            "non_template": [
                {"pair_id": "p1", "case_id": "h1", "split": "train"},
                {"pair_id": "p2", "case_id": "h2", "split": "train"},
                {"pair_id": "p3", "case_id": "d", "split": "development"},
            ]
        }
    }
    pairs = select_holdout_pairs(discovery, _fit(), 2)
    plan = build_plan(pairs)
    assert len(plan) == 40
    assert len({row["record_id"] for row in plan}) == 40
    assert {row["pair_id"] for row in plan} == {"p1", "p2"}
