from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.analyze_signed_dose_train_v4 import analyze
from research.phase1.target_attractor_pilot.sae_causal.contracts import AXES
from research.phase1.target_attractor_pilot.sae_causal.signed_dose_train_v4 import NONZERO_ALPHAS


def _score(values: dict[str, float]) -> dict[str, float]:
    return {**values, "usefulness": 3.0, "validity": 3.0}


def test_signed_dose_analysis_keeps_directions_separate_and_recovers_slope() -> None:
    rows, scores = [], {}
    for axis in AXES:
        pair = f"pair-{axis}"
        for direction, base_value, sign in (("high-to-low", 1.0, 1.0), ("low-to-high", 3.0, -1.0)):
            base_id = f"{pair}-base-{direction}"
            base = {name: base_value for name in AXES}
            rows.append({
                "record_id": base_id, "pair_id": pair, "axis": axis,
                "condition": "baseline", "direction": direction,
            })
            scores[base_id] = _score(base)
            for alpha in NONZERO_ALPHAS:
                for method in ("sae", "sae_random", "das", "das_random"):
                    record_id = f"{pair}-{direction}-{method}-{alpha}"
                    patched = dict(base)
                    if method in ("sae", "das"):
                        patched[axis] += sign * alpha
                    rows.append({
                        "record_id": record_id, "pair_id": pair, "axis": axis,
                        "condition": "signed_dose", "direction": direction,
                        "method": method, "alpha": alpha,
                    })
                    scores[record_id] = _score(patched)
    result = analyze(rows, scores)
    for axis in AXES:
        for direction in ("high-to-low", "low-to-high"):
            for method in ("sae", "das"):
                comparison = result["axes"][axis]["directions"][direction][method]
                assert comparison["candidate_signed_dose_slope"] == 1.0
                assert comparison["matched_random_signed_dose_slope"] == 0.0
                assert comparison["slope_difference"] == 1.0
                assert comparison["doses"]["2.0"]["target_effect"] == 2.0
                assert comparison["doses"]["-2.0"]["target_effect"] == -2.0
