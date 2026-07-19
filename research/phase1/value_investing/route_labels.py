"""Deterministic, versioned route labels derived from pre-registered financial rules."""

from __future__ import annotations

from typing import Any

import yaml


def load_rules(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _changed_cash_flow_sign(features: dict[str, Any]) -> bool:
    return bool(features.get("operating_cash_flow_sign_changed", False))


def derive_route(features: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    active = list(rules["always_active"])
    edges = [list(edge) for edge in rules["base_edges"]]
    rule_ids: list[str] = []

    checks = {
        "evidence_revision": abs(features["earnings_surprise_zscore"]) >= 1.0 or _changed_cash_flow_sign(features),
        "durable_compounding": (
            features["free_cash_flow_positive_years"] >= 3
            and features["roic"] >= 0.12
            and features["ttm_revenue_growth"] >= 0.05
        ),
        "balance_sheet_margin": (
            features["net_debt_to_ebitda"] > 3.0
            or features["interest_coverage"] < 3.0
            or features["current_ratio"] < 1.0
        ),
        "operating_constraint": (
            features["gross_margin_change"] <= -0.03
            and max(
                features["receivables_growth_minus_revenue_growth"],
                features["inventory_growth_minus_revenue_growth"],
            ) >= 0.10
        ),
        "fragile_estimate": (
            0.20 <= features["valuation_percentile"] <= 0.80
            and features["earnings_volatility"] >= 0.25
        ),
    }
    for rule in rules["rules"]:
        if not checks[rule["id"]]:
            continue
        rule_ids.append(rule["id"])
        for model_id in rule["activate"]:
            if model_id not in active:
                active.append(model_id)
            for predecessor in rule.get("order_after", []):
                edges.append([predecessor, model_id])
            for successor in rule.get("order_before", []):
                edges.append([model_id, successor])
    active_set = set(active)
    ordered_edges = [edge for edge in edges if edge[0] in active_set and edge[1] in active_set]
    predecessors = {model_id: set() for model_id in active_set}
    for source, target in ordered_edges:
        predecessors[target].add(source)

    # Stable topological ordering uses the declared model order only to break ties.
    order: list[str] = []
    remaining = set(active_set)
    while remaining:
        ready = [model_id for model_id in rules["allowed_model_ids"] if model_id in remaining and not predecessors[model_id]]
        if not ready:
            raise ValueError("route rules contain a cycle")
        current = ready[0]
        order.append(current)
        remaining.remove(current)
        for source, target in ordered_edges:
            if source == current:
                predecessors[target].discard(current)
    return {
        "rule_version": rules["rule_version"],
        "triggered_rules": rule_ids,
        "activated_models": order,
        "ordered_models": order,
        "conditional_edges": ordered_edges,
    }


def derive_route_with_coverage(features: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Apply only rules whose declared inputs are observed, preserving coverage gaps.

    This is for SEC-only candidate records, not for validated final Episodes. A
    final Episode must use :func:`derive_route`, which rejects missing inputs by
    construction.
    """
    available_rules = []
    unavailable = []
    for rule in rules["rules"]:
        missing = [name for name in rule["inputs"] if features.get(name) is None]
        if missing:
            unavailable.append({"rule_id": rule["id"], "missing_inputs": missing})
        else:
            available_rules.append(rule)
    partial_rules = {**rules, "rules": available_rules}
    labels = derive_route(features | {name: 0.0 for item in unavailable for name in item["missing_inputs"]}, partial_rules)
    labels["label_status"] = "complete" if not unavailable else "partial"
    labels["unavailable_rules"] = unavailable
    return labels
