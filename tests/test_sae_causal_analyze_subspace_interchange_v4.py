from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.analyze_subspace_interchange_v4 import (
    analyze,
)


AXES = ("specificity", "independence", "non_template")


def _score(axis_values: dict[str, float], usefulness: float = 3.0) -> dict[str, float]:
    return {
        **axis_values,
        "usefulness": usefulness,
        "validity": 3.0,
    }


def test_four_methods_orient_both_directions_and_apply_engineering_thresholds() -> None:
    rows = []
    scores: dict[str, dict[str, float]] = {}
    for axis_index, axis in enumerate(AXES):
        pair_id = f"pair-{axis}"
        low_id = f"{pair_id}-baseline-low"
        high_id = f"{pair_id}-baseline-high"
        low_values = {name: 1.0 for name in AXES}
        high_values = {name: 3.0 for name in AXES}
        rows.extend([
            {
                "record_id": low_id, "pair_id": pair_id, "axis": axis,
                "condition": "baseline", "target_class": "low",
            },
            {
                "record_id": high_id, "pair_id": pair_id, "axis": axis,
                "condition": "baseline", "target_class": "high",
            },
        ])
        scores[low_id] = _score(low_values)
        scores[high_id] = _score(high_values)

        other_axis = AXES[(axis_index + 1) % len(AXES)]
        for method in ("sae", "sae_random", "das", "das_random"):
            for source_class, target_class in (("high", "low"), ("low", "high")):
                record_id = f"{pair_id}-{method}-{source_class}-to-{target_class}"
                norm = 2.0 if method.startswith("sae") else 3.0
                rows.append({
                    "record_id": record_id,
                    "pair_id": pair_id,
                    "axis": axis,
                    "condition": "subspace_interchange",
                    "method": method,
                    "source_class": source_class,
                    "target_class": target_class,
                    "residual_update_l2": norm,
                })
                patched = dict(low_values if target_class == "low" else high_values)
                usefulness = 3.0
                direction = 1.0 if source_class == "high" else -1.0
                if method == "sae":
                    patched[axis] += direction
                    usefulness = 2.8
                elif method == "das":
                    patched[axis] += 0.5 * direction
                    patched[other_axis] += 0.5 * direction
                    usefulness = 2.5
                scores[record_id] = _score(patched, usefulness)

    result = analyze(rows, scores)

    for axis in AXES:
        axis_result = result["axes"][axis]
        assert set(axis_result["conditions"]) == {
            "sae", "sae_random", "das", "das_random"
        }
        assert all(summary["records"] == 2 for summary in axis_result["conditions"].values())

        sae = axis_result["comparisons"]["sae"]
        assert sae["target_effect"] == 1.0
        assert sae["matched_random_target_effect"] == 0.0
        assert sae["difference_vs_matched_random"] == 1.0
        assert sae["positive_oriented_target_effect"] is True
        assert sae["better_than_matched_random"] is True
        assert sae["cross_axis_ratio_at_most_0_5"] is True
        assert sae["quality_drop_at_most_0_25"] is True
        assert sae["residual_update_norm_matched"] is True
        assert sae["gate_pass"] is True

        das = axis_result["comparisons"]["das"]
        assert das["target_effect"] == 0.5
        assert das["matched_random_target_effect"] == 0.0
        assert das["difference_vs_matched_random"] == 0.5
        assert das["positive_oriented_target_effect"] is True
        assert das["better_than_matched_random"] is True
        assert das["cross_axis_ratio"] == 1.0
        assert das["cross_axis_ratio_at_most_0_5"] is False
        assert das["quality_drop_at_most_0_25"] is False
        assert das["residual_update_norm_matched"] is True
        assert das["gate_pass"] is False
