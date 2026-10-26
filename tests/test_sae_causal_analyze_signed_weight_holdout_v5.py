from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.analyze_signed_weight_holdout_v5 import (
    analyze,
)


def test_analysis_orients_both_directions_and_requires_both_controls() -> None:
    rows = []
    scores = {}
    for direction in ("high-to-low", "low-to-high"):
        for pair in ("p1", "p2"):
            baseline_id = f"{pair}-{direction}-base"
            rows.append({
                "record_id": baseline_id, "pair_id": pair,
                "condition": "baseline", "direction": direction,
            })
            scores[baseline_id] = {
                "specificity": 2.0, "independence": 2.0, "non_template": 2.0,
                "usefulness": 3.0, "validity": 3.0,
            }
            for alpha in (0.5, 1.0, 2.0):
                for method, delta in (
                    ("signed_weighted", 1.0),
                    ("equal_weight", 0.25),
                    ("random_weighted", 0.0),
                ):
                    record_id = f"{pair}-{direction}-{alpha}-{method}"
                    rows.append({
                        "record_id": record_id, "pair_id": pair,
                        "condition": "signed_weight_holdout", "direction": direction,
                        "method": method, "alpha": alpha,
                    })
                    oriented = delta if direction == "high-to-low" else -delta
                    scores[record_id] = {
                        "specificity": 2.0,
                        "independence": 2.0,
                        "non_template": 2.0 + oriented,
                        "usefulness": 3.0,
                        "validity": 3.0,
                    }
    result = analyze(rows, scores)
    assert len(result["gate_passes"]) == 6
    for direction in ("high-to-low", "low-to-high"):
        assert result["directions"][direction]["doses"]["1.0"]["passes_discovery_gate"]
