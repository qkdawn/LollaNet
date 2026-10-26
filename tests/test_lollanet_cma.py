from __future__ import annotations

from collections import Counter

from research.phase1.lollanet_cma.splitter import allocate_groups, build_groups
from research.phase1.lollanet_cma.taxonomy import MODEL_FAMILIES, QUERY_TO_FAMILY, model_family


def _row(question_id: int, model_id: int, query_type: str, graph_id: str, answer: str) -> dict:
    return {
        "question_id": question_id,
        "answer": answer,
        "meta": {"model_id": model_id, "query_type": query_type, "graph_id": graph_id, "rung": 1},
    }


def test_taxonomy_covers_the_ten_frozen_cladder_query_types():
    assert set(QUERY_TO_FAMILY) == {
        "marginal", "correlation", "exp_away", "ate", "backadj", "collider_bias",
        "ett", "det-counterfactual", "nde", "nie",
    }
    assert set(QUERY_TO_FAMILY.values()) == set(MODEL_FAMILIES)


def test_grouped_milp_keeps_model_ids_whole_and_hits_family_targets():
    representative_query = {
        "cm-associational": "marginal",
        "cm-interventional": "ate",
        "cm-counterfactual": "ett",
        "cm-mediation": "nde",
    }
    rows = []
    question_id = 0
    for family, query_type in representative_query.items():
        for index in range(8):
            rows.append(_row(question_id, question_id, query_type, f"graph-{index % 2}", "yes" if index % 2 else "no"))
            question_id += 1
    allocation, diagnostics = allocate_groups(
        build_groups(rows), {"qualification": 2, "development": 2}, seed=7, time_limit_seconds=30,
    )
    assert diagnostics["optimal"] is True
    seen: set[str] = set()
    for split, groups in allocation.items():
        ids = {group.model_id for group in groups}
        assert not seen & ids
        seen |= ids
        counts = Counter(model_family(row["meta"]["query_type"]) for group in groups for row in group.rows)
        assert counts == Counter({family: 2 for family in MODEL_FAMILIES})
