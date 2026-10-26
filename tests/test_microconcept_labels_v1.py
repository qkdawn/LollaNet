from research.phase1.microconcepts.microconcept_labels_v1 import (
    Card,
    Episode,
    build_task_labels,
    decide_state,
    project_episode,
    Signal,
)


def card() -> Card:
    return Card("capacity_headroom", "capacity", ("current_ratio",), ("liquidity_outlook",), "0.1.0")


def episode() -> Episode:
    return Episode(
        episode_id="episode-1", accession="0001", available_at="2021-01-01T00:00:00.000Z",
        form="10-K", sic_division="SIC-10", factors={"current_ratio": None},
        factor_sources={"current_ratio": "financial_features"}, missing_reasons={"current_ratio": "not_observed"},
    )


def test_projection_allowlist_ignores_route_and_outcome_values() -> None:
    raw = {
        "episode_id": "episode-1", "sic": "1000",
        "filing": {"accession": "0001", "form": "10-K"},
        "availability": {"public_available_at": "2021-01-01T00:00:00.000Z"},
        "financial_features": {"current_ratio": 2.0}, "market_state": {}, "missing_reasons": {},
        "route_labels": {"activated_models": ["forbidden"]},
        "exploratory_outcome": {"raw_price_return": 999.0}, "issuer_name": "forbidden issuer",
    }
    projected = project_episode(raw)
    assert projected.episode_id == "episode-1"
    assert projected.factors == {"current_ratio": 2.0}
    assert not hasattr(projected, "route_labels")
    assert not hasattr(projected, "exploratory_outcome")


def test_missing_evidence_is_unknown_not_contradicted() -> None:
    label, reasons = decide_state(episode(), card(), [], [])
    assert label["status"] == "unknown"
    assert label["intensity"] == 0
    assert label["label_tier"] == "Weak"
    assert "all_card_evidence_missing" in reasons


def test_conflicting_evidence_remains_unknown_and_is_queued() -> None:
    ref = {"kind": "factor", "episode_id": "episode-1", "factor_id": "current_ratio", "source_field": "financial_features", "value": 1.0, "available_at": "2021-01-01T00:00:00.000Z", "rule_id": "test", "signal": "test"}
    label, reasons = decide_state(episode(), card(), [Signal("support", ref), Signal("oppose", ref)], [])
    assert label["status"] == "unknown"
    assert label["direction"] == "mixed"
    assert "evidence_conflict" in reasons


def test_paired_tasks_share_state_and_change_only_task_relevance() -> None:
    state = {
        "label_id": "state-1", "episode_id": "episode-1", "microconcept_id": "capacity_headroom",
        "available_at": "2021-01-01T00:00:00.000Z", "status": "supported",
        "review_status": "not_queued", "dispute_status": "none",
    }
    tasks = build_task_labels([state], {"capacity_headroom": card()})
    assert {task["state_label_id"] for task in tasks} == {"state-1"}
    assert {(task["applicability"], task["relevance"]) for task in tasks} == {("applicable", 3), ("not_applicable", 0)}
