from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .common import stable_hash
from .taxonomy import MODEL_FAMILIES, model_family


@dataclass(frozen=True)
class Group:
    model_id: str
    rows: tuple[dict[str, Any], ...]
    family_counts: Counter[str]
    strata_counts: Counter[tuple[str, str]]


def _strata(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    meta = row["meta"]
    return (
        ("query_type", str(meta["query_type"])),
        ("graph_id", str(meta["graph_id"])),
        ("rung", str(meta["rung"])),
        ("answer", str(row["answer"]).lower()),
    )


def build_groups(rows: list[dict[str, Any]], excluded_model_ids: set[str] | None = None) -> list[Group]:
    excluded = excluded_model_ids or set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_id = str(row["meta"]["model_id"])
        if model_id not in excluded:
            grouped[model_id].append(row)
    groups: list[Group] = []
    for model_id, members in sorted(grouped.items(), key=lambda item: item[0]):
        family_counts = Counter(model_family(str(row["meta"]["query_type"])) for row in members)
        strata_counts: Counter[tuple[str, str]] = Counter()
        for row in members:
            strata_counts.update(_strata(row))
        groups.append(Group(model_id, tuple(members), family_counts, strata_counts))
    return groups


def allocate_groups(
    groups: list[Group], split_family_targets: dict[str, int], seed: int, time_limit_seconds: int = 180,
) -> tuple[dict[str, list[Group]], dict[str, Any]]:
    if not groups or not split_family_targets:
        raise ValueError("group allocation requires non-empty groups and split targets")
    split_names = list(split_family_targets)
    if any(int(value) < 1 for value in split_family_targets.values()):
        raise ValueError("each split family target must be positive")
    strata = sorted({key for group in groups for key in group.strata_counts})
    total_rows = sum(len(group.rows) for group in groups)
    global_strata = Counter()
    for group in groups:
        global_strata.update(group.strata_counts)

    group_count = len(groups)
    split_count = len(split_names)
    x_count = group_count * split_count
    deviation_count = split_count * len(strata)
    variable_count = x_count + deviation_count
    objective = np.zeros(variable_count, dtype=np.float64)
    for group_index, group in enumerate(groups):
        for split_index, split_name in enumerate(split_names):
            tie = int(stable_hash([seed, split_name, group.model_id])[:8], 16) / 0xFFFFFFFF
            objective[group_index * split_count + split_index] = tie * 1e-7
    for split_index, split_name in enumerate(split_names):
        split_size = int(split_family_targets[split_name]) * len(MODEL_FAMILIES)
        for stratum_index, stratum in enumerate(strata):
            target = split_size * global_strata[stratum] / total_rows
            weight = {"query_type": 2.0, "graph_id": 1.0, "rung": 1.0, "answer": 2.0}[stratum[0]]
            objective[x_count + split_index * len(strata) + stratum_index] = weight / max(target, 1.0)

    row_count = group_count + split_count * len(MODEL_FAMILIES) + 2 * split_count * len(strata)
    matrix = lil_matrix((row_count, variable_count), dtype=np.float64)
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.full(row_count, np.inf, dtype=np.float64)
    constraint_row = 0
    for group_index in range(group_count):
        for split_index in range(split_count):
            matrix[constraint_row, group_index * split_count + split_index] = 1.0
        upper[constraint_row] = 1.0
        constraint_row += 1
    for split_index, split_name in enumerate(split_names):
        target = float(split_family_targets[split_name])
        for family in MODEL_FAMILIES:
            for group_index, group in enumerate(groups):
                count = group.family_counts[family]
                if count:
                    matrix[constraint_row, group_index * split_count + split_index] = count
            lower[constraint_row] = target
            upper[constraint_row] = target
            constraint_row += 1
    for split_index, split_name in enumerate(split_names):
        split_size = int(split_family_targets[split_name]) * len(MODEL_FAMILIES)
        for stratum_index, stratum in enumerate(strata):
            for group_index, group in enumerate(groups):
                count = group.strata_counts[stratum]
                if count:
                    matrix[constraint_row, group_index * split_count + split_index] = count
            deviation_index = x_count + split_index * len(strata) + stratum_index
            matrix[constraint_row, deviation_index] = -1.0
            target = split_size * global_strata[stratum] / total_rows
            lower[constraint_row] = -np.inf
            upper[constraint_row] = target
            constraint_row += 1
            for group_index, group in enumerate(groups):
                count = group.strata_counts[stratum]
                if count:
                    matrix[constraint_row, group_index * split_count + split_index] = -count
            matrix[constraint_row, deviation_index] = -1.0
            lower[constraint_row] = -np.inf
            upper[constraint_row] = -target
            constraint_row += 1

    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[:x_count] = 1
    bounds = Bounds(
        np.zeros(variable_count, dtype=np.float64),
        np.concatenate([np.ones(x_count, dtype=np.float64), np.full(deviation_count, np.inf)]),
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": int(time_limit_seconds), "mip_rel_gap": 0.001, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"grouped split MILP failed: status={result.status}, message={result.message}")
    allocation = {name: [] for name in split_names}
    binary_values = result.x[:x_count]
    if np.max(np.abs(binary_values - np.rint(binary_values))) > 1e-6:
        raise RuntimeError("grouped split MILP returned no integral incumbent")
    decisions = np.rint(binary_values).astype(int).reshape(group_count, split_count)
    for group_index, group in enumerate(groups):
        chosen = np.flatnonzero(decisions[group_index])
        if len(chosen) > 1:
            raise RuntimeError(f"MILP split the model_id group {group.model_id}")
        if len(chosen) == 1:
            allocation[split_names[int(chosen[0])]].append(group)
    for split_name, selected in allocation.items():
        counts = Counter()
        for group in selected:
            counts.update(group.family_counts)
        expected = int(split_family_targets[split_name])
        if any(counts[family] != expected for family in MODEL_FAMILIES):
            raise RuntimeError(f"MILP returned incorrect family counts for {split_name}: {counts}")
    diagnostics = {
        "optimal": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else None,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        "time_limit_seconds": int(time_limit_seconds),
    }
    return allocation, diagnostics
