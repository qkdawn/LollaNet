from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.analyze_interchange_v3 import analyze


def _scores(specificity: float, independence: float, non_template: float) -> dict:
    return {
        "specificity": specificity,
        "independence": independence,
        "non_template": non_template,
        "usefulness": 3.0,
        "validity": 3.0,
    }


def test_interchange_effect_orients_both_swap_directions_toward_high() -> None:
    rows = []
    scores = {}
    for axis in ("specificity", "independence", "non_template"):
        pair = f"pair-{axis}"
        low_baseline = f"{pair}-low-base"
        high_baseline = f"{pair}-high-base"
        rows.extend([
            {
                "record_id": low_baseline, "pair_id": pair, "axis": axis,
                "condition": "baseline", "target_class": "low",
            },
            {
                "record_id": high_baseline, "pair_id": pair, "axis": axis,
                "condition": "baseline", "target_class": "high",
            },
        ])
        scores[low_baseline] = _scores(1.0, 1.0, 1.0)
        scores[high_baseline] = _scores(3.0, 3.0, 3.0)
        for random_control, feature in ((False, 10), (True, 20)):
            for source, target in (("high", "low"), ("low", "high")):
                record_id = f"{pair}-{feature}-{source}-{target}"
                rows.append({
                    "record_id": record_id,
                    "pair_id": pair,
                    "axis": axis,
                    "condition": "interchange",
                    "candidate_feature": 10,
                    "feature": feature,
                    "random_control": random_control,
                    "source_class": source,
                    "target_class": target,
                    "residual_update_l2": 2.0,
                })
                if random_control:
                    scores[record_id] = scores[low_baseline if target == "low" else high_baseline]
                else:
                    scores[record_id] = _scores(2.0, 2.0, 2.0)

    result = analyze(rows, scores)

    for axis in result["axes"]:
        checks = result["axes"][axis]["engineering_checks"]
        assert checks["positive_oriented_target_effect"] is True
        assert checks["better_than_random"] is True
