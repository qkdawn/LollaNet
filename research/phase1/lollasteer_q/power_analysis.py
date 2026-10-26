from __future__ import annotations

import argparse
import json
from statistics import NormalDist
from pathlib import Path

import numpy as np

from .common import atomic_write_json


CELL_ORDER = tuple((r, s, c) for r in (0, 1) for s in (0, 1) for c in (0, 1))
CELL_SIGNS = np.asarray([(-1) ** (3 - r - s - c) for r, s, c in CELL_ORDER], dtype=np.float64)


def simulate_interaction_power(
    item_counts: list[int],
    baseline_probability: float,
    interaction_effect: float,
    item_sd: float = 0.0,
    simulations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260724,
) -> dict[str, object]:
    """Simulate item-clustered binary outcomes for a pre-specified 2^3 interaction.

    The supplied interaction_effect is the expected third-order alternating sum on
    the risk-difference scale. No model or pilot outputs are read by this function.
    """
    if not item_counts or any(int(value) < 2 for value in item_counts):
        raise ValueError("item_counts must contain values >= 2")
    if not 0 < baseline_probability < 1:
        raise ValueError("baseline_probability must be in (0, 1)")
    if simulations < 100 or not 0 < alpha < 1:
        raise ValueError("simulations must be >= 100 and alpha must be in (0, 1)")
    if item_sd < 0:
        raise ValueError("item_sd must be non-negative")
    if abs(interaction_effect) > 1:
        raise ValueError("interaction_effect must be within [-1, 1]")

    rng = np.random.default_rng(seed)
    critical = NormalDist().inv_cdf(1 - alpha / 2)
    results: list[dict[str, float | int]] = []
    for raw_count in item_counts:
        count = int(raw_count)
        item_shift = rng.normal(0.0, item_sd, size=(simulations, count, 1)) if item_sd else np.zeros((simulations, count, 1))
        probabilities = baseline_probability + item_shift + (interaction_effect / 8.0) * CELL_SIGNS.reshape(1, 1, 8)
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("assumptions produce probabilities outside (0, 1); lower item_sd or effect")
        outcomes = rng.binomial(1, probabilities)
        contrasts = np.sum(outcomes * CELL_SIGNS.reshape(1, 1, 8), axis=2)
        estimates = np.mean(contrasts, axis=1)
        standard_errors = np.std(contrasts, axis=1, ddof=1) / np.sqrt(count)
        detectable = np.abs(estimates) > critical * standard_errors
        results.append({
            "items": count,
            "simulations": int(simulations),
            "power": float(np.mean(detectable)),
            "mean_estimate": float(np.mean(estimates)),
            "estimate_sd": float(np.std(estimates, ddof=1)),
            "median_standard_error": float(np.median(standard_errors)),
            "target_interaction_effect": float(interaction_effect),
        })
    return {
        "type": "lollasteer_q_interaction_power_simulation",
        "status": "design_simulation_not_empirical_evidence",
        "assumptions": {
            "baseline_probability": float(baseline_probability),
            "interaction_effect": float(interaction_effect),
            "item_sd": float(item_sd),
            "alpha": float(alpha),
            "seed": int(seed),
            "cell_order": [list(cell) for cell in CELL_ORDER],
            "estimand": "third-order alternating sum of item-clustered binary risk differences",
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate LollaSteer-Q 2^3 interaction power under explicit assumptions.")
    parser.add_argument("--items", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-probability", type=float, required=True)
    parser.add_argument("--interaction-effect", type=float, required=True)
    parser.add_argument("--item-sd", type=float, default=0.0)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = simulate_interaction_power(
        args.items, args.baseline_probability, args.interaction_effect,
        args.item_sd, args.simulations, args.alpha, args.seed,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
