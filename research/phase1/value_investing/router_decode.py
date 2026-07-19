"""Decode predicted rule probabilities into the explicit Phase 1 model graph."""

from __future__ import annotations

from typing import Any

try:
    from .router_data import ROUTE_RULES
except ImportError:  # Supports direct execution from the repository root.
    from router_data import ROUTE_RULES


def decode_route(rule_probabilities: dict[str, float], thresholds: dict[str, float], rules: dict[str, Any]) -> dict[str, Any]:
    triggered = [rule for rule in ROUTE_RULES if rule_probabilities[rule] >= thresholds[rule]]
    active = list(rules["always_active"])
    edges = [list(edge) for edge in rules["base_edges"]]
    definitions = {rule["id"]: rule for rule in rules["rules"]}
    for rule_id in triggered:
        rule = definitions[rule_id]
        for model in rule["activate"]:
            if model not in active:
                active.append(model)
            for predecessor in rule.get("order_after", []):
                edges.append([predecessor, model])
            for successor in rule.get("order_before", []):
                edges.append([model, successor])
    active_set = set(active)
    edges = [edge for edge in edges if edge[0] in active_set and edge[1] in active_set]
    predecessor = {model: set() for model in active_set}
    for source, target in edges:
        predecessor[target].add(source)
    ordered = []
    remaining = set(active_set)
    while remaining:
        ready = [model for model in rules["allowed_model_ids"] if model in remaining and not predecessor[model]]
        if not ready:
            raise ValueError("route graph has a cycle")
        current = ready[0]
        ordered.append(current)
        remaining.remove(current)
        for source, target in edges:
            if source == current:
                predecessor[target].discard(current)
    model_probabilities = {model: 1.0 if model in rules["always_active"] else 0.0 for model in rules["allowed_model_ids"]}
    for rule in rules["rules"]:
        if rule["id"] in rule_probabilities:
            for model in rule["activate"]:
                model_probabilities[model] = rule_probabilities[rule["id"]]
    return {"triggered_rules": triggered, "activated_models": ordered, "ordered_models": ordered, "conditional_edges": edges, "model_probabilities": model_probabilities}
