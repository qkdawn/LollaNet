from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from research.phase1.mechanism_questions_v1 import (
    build_public_corpus_v1, build_question_episodes_v1, prepare_gpt_teacher_packets_v1, question_set_v1,
)
from research.phase1.mechanism_questions_v1.public_corpus_v1 import _variants
from research.phase1.mechanism_questions_v1.question_set_v1 import (
    PROFILE_BY_QUESTION_KIND, ValidationError, build_episodes, load_semantic_assets, model_selection_contracts,
    read_jsonl, router_view, split_case_packs, teacher_packet, validate_fact_pack, validate_teacher_labels,
    write_jsonl,
)


def states() -> list[dict]:
    active = {
        "constraint_pressure": ("supported", 3, "increasing", 3, 3),
        "capacity_headroom": ("supported", 2, "decreasing", 2, 2),
        "evidence_quality": ("supported", 2, "improving", 2, 2),
    }
    result = []
    for concept_id in sorted(load_semantic_assets()["concept_ids"]):
        status, intensity, direction, persistence, relevance = active.get(concept_id, ("unknown", 0, "unknown", 0, 0))
        result.append({
            "microconcept_id": concept_id, "status": status, "intensity": intensity, "direction": direction,
            "temporal_persistence": persistence, "task_relevance": relevance,
            "evidence_refs": ["fact-1"] if status != "unknown" else [],
        })
    return result


def fact_pack(case_id: str = "fictional-public-example-001") -> dict:
    pack = {
        "case_id": case_id, "domain": "software_engineering", "as_of": "2024-01-10T00:00:00Z",
        "public_available_at": "2024-01-10T00:00:00Z", "example": True, "fictional": True,
        "source_refs": [{"source_id": "src-1", "title": "Fictional published incident record", "url": "https://example.test/incident", "published_at": "2024-01-09T00:00:00Z"}],
        "facts": [{"fact_id": "fact-1", "statement": "A shared service approached its tested concurrent-request limit while end-to-end completion time rose.", "evidence_refs": ["src-1"], "available_at": "2024-01-09T00:00:00Z", "tested_concurrent_request_limit": 100}],
        "factual_uncertainties": [], "structured_evidence": [], "microconcept_states": states(),
        "question_variants": [
            {"question_kind": "mechanism_diagnosis", "question": "Which mechanism best explains the current degradation?", "goal": "Explain the most likely driver.", "constraints": "Use only the supplied evidence state.", "cross_domain_transfer": False, "transfer_rationale": None, "counterfactual_delta": None},
            {"question_kind": "decision_tradeoff", "question": "What should be prioritized under limited engineering time?", "goal": "Choose an action that improves service completion.", "constraints": "Avoid expanding the team this week.", "cross_domain_transfer": False, "transfer_rationale": None, "counterfactual_delta": None},
            {"question_kind": "evidence_discrimination", "question": "What observation would best separate the competing explanations?", "goal": "Request the most discriminating evidence or abstain.", "constraints": "Do not assume unobserved failures.", "cross_domain_transfer": True, "transfer_rationale": {"source_domain": "science_public_systems", "structural_mapping": "Both systems are limited by a shared throughput boundary.", "evidence_refs": ["fact-1"]}, "counterfactual_delta": None},
            {"question_kind": "conditional_or_counterfactual", "question": "If the explicit capacity threshold were raised while demand stayed fixed, what would change?", "goal": "Give a conditional prediction and next check.", "constraints": "Change only the declared variable.", "cross_domain_transfer": True, "transfer_rationale": {"source_domain": "product_organization", "structural_mapping": "Both systems expose a capacity limit before the desired output improves.", "evidence_refs": ["fact-1"]}, "counterfactual_delta": {"kind": "controlled_counterfactual", "changes": [{"fact_id": "fact-1", "field": "tested_concurrent_request_limit", "before": 100, "after": 140}], "microconcept_state_changes": [{"microconcept_id": "capacity_headroom", "axis": "task_relevance", "before": 2, "after": 3}, {"microconcept_id": "capacity_headroom", "axis": "direction", "before": "decreasing", "after": "counterfactual_headroom_expanding"}]}},
        ],
    }
    for variant in pack["question_variants"]:
        variant["task_relevance_profile"] = PROFILE_BY_QUESTION_KIND[variant["question_kind"]]
        variant["task_focus_microconcept_id"] = "capacity_headroom"
    # The controlled counterfactual precondition is checked after conditional task conditioning.
    delta = pack["question_variants"][-1]["counterfactual_delta"]
    delta["microconcept_state_changes"][0].update({"before": 3, "after": 1})
    return pack


def label(episode: dict, label_pass: str, status: str, primary: str | None = "mm-p1-bottleneck") -> dict:
    return {
        "label_id": f"{episode['episode_id']}::{label_pass}", "episode_id": episode["episode_id"], "label_pass": label_pass,
        "teacher_model": "gpt-5.6-sol", "teacher_version": "2026-07-20", "prompt_version": "v1",
        "label_timestamp": "2026-07-20T00:00:00Z", "label_tier": "Weak", "confidence": 0.8,
        "abstain": False, "primary_model": primary, "auxiliary_models": [],
        "near_miss_models": ["mm-p1-rate-control"], "near_miss_exclusions": {"mm-p1-rate-control": "The evidence does not establish a control-rate mechanism."},
        "challenge_assessment": "No observed challenge condition overturns the route.", "addressed_challenge_microconcept_ids": [], "cited_microconcept_ids": ["constraint_pressure"],
        "cited_evidence_refs": ["fact-1"], "route_rationale": "The cited constraint state makes this candidate eligible.",
        "abstain_reason": "", "next_verification": "Measure completion change after relieving the identified limit.",
        "review_status": status,
    }


def test_fixed_semantic_assets_have_v1_cardinality() -> None:
    assets = load_semantic_assets()
    assert len(assets["concept_ids"]) == 11
    assert len(assets["models"]) == 20


def test_fictional_fixture_builds_four_variants_with_case_level_split() -> None:
    episodes = build_episodes([fact_pack()])
    assert len(episodes) == 4
    assert {item["question_kind"] for item in episodes} == {"mechanism_diagnosis", "decision_tradeoff", "evidence_discrimination", "conditional_or_counterfactual"}
    assert len({item["split"] for item in episodes}) == 1
    assert {"mm-p1-bottleneck", "mm-p1-rate-control"}.issubset(set(episodes[0]["candidate_subgraph"]["candidate_model_ids"]))


def test_question_conditioning_produces_four_distinct_vectors_and_applies_counterfactual() -> None:
    episodes = build_episodes([fact_pack()])
    assert len({tuple(row["task_relevance"] for row in episode["microconcept_states"]) for episode in episodes}) == 4
    assert len({str(episode["fact_microconcept_states"]) for episode in episodes}) == 1
    by_kind = {episode["question_kind"]: episode for episode in episodes}
    counterfactual = by_kind["conditional_or_counterfactual"]
    factual_capacity = next(row for row in counterfactual["fact_microconcept_states"] if row["microconcept_id"] == "capacity_headroom")
    conditioned_capacity = next(row for row in counterfactual["microconcept_states"] if row["microconcept_id"] == "capacity_headroom")
    assert factual_capacity["task_relevance"] == 2
    assert conditioned_capacity["task_relevance"] == 1
    assert counterfactual["counterfactual_delta"]["microconcept_state_changes"][0]["before"] == 3


def test_counterfactual_before_must_match_question_conditioned_state() -> None:
    invalid = fact_pack()
    invalid["question_variants"][-1]["counterfactual_delta"]["microconcept_state_changes"][0]["before"] = 2
    with pytest.raises(ValidationError, match="question-conditioned"):
        validate_fact_pack(invalid)


def test_router_hides_task_text_but_teacher_packet_keeps_it() -> None:
    episode = build_episodes([fact_pack()])[0]
    view = router_view(episode)
    assert "facts" not in view and "source_refs" not in view and "case_id" not in view
    assert set(view) == {"episode_id", "microconcept_states", "candidate_subgraph"}
    assert not {"question", "goal", "constraints"} & set(view)
    packet = teacher_packet(episode, "initial")
    assert {"question", "goal", "constraints"}.issubset(packet["episode"])
    assert "facts" not in packet["episode"] and "source_refs" not in packet["episode"]


def test_model_leak_and_route_field_fail_fact_validation() -> None:
    leaked = fact_pack()
    leaked["question_variants"][0]["question"] = "Should we choose mm-p1-bottleneck?"
    with pytest.raises(ValidationError, match="leaks"):
        validate_fact_pack(leaked)
    routed = fact_pack()
    routed["route_labels"] = {"primary_model": "mm-p1-bottleneck"}
    with pytest.raises(ValidationError, match="forbidden"):
        validate_fact_pack(routed)


def test_availability_and_counterfactual_boundaries_fail() -> None:
    late = fact_pack()
    late["facts"][0]["available_at"] = "2024-01-11T00:00:00Z"
    with pytest.raises(ValidationError, match="after"):
        validate_fact_pack(late)
    invalid_cf = fact_pack()
    invalid_cf["question_variants"][-1]["counterfactual_delta"]["changes"].append({"fact_id": "fact-1", "field": "a", "before": 1, "after": 2})
    invalid_cf["question_variants"][-1]["counterfactual_delta"]["changes"].append({"fact_id": "fact-1", "field": "b", "before": 1, "after": 2})
    with pytest.raises(ValidationError, match="one or two"):
        validate_fact_pack(invalid_cf)


def test_counterfactual_field_and_before_value_must_match_referenced_fact() -> None:
    unknown_field = fact_pack()
    unknown_field["question_variants"][-1]["counterfactual_delta"]["changes"][0]["field"] = "invented_pressure"
    with pytest.raises(ValidationError, match="field must exist"):
        validate_fact_pack(unknown_field)
    mismatched_before = fact_pack()
    mismatched_before["question_variants"][-1]["counterfactual_delta"]["changes"][0]["before"] = 99
    with pytest.raises(ValidationError, match="before value"):
        validate_fact_pack(mismatched_before)


def test_public_variant_generator_is_diverse_and_uses_real_fact_fields() -> None:
    facts = [{
        "fact_id": "fact-1", "statement": "An anonymous system has a bounded remediation window.",
        "evidence_refs": ["src-1"], "available_at": "2024-01-09T00:00:00Z", "deadline_days": 21,
    }]
    first = _variants(0, "software_engineering", facts, states())
    second = _variants(1, "software_engineering", facts, states())
    assert len(first) == len(second) == 4
    assert {row["question_kind"] for row in first} == {"mechanism_diagnosis", "decision_tradeoff", "evidence_discrimination", "conditional_or_counterfactual"}
    assert {row["question"] for row in first} != {row["question"] for row in second}
    delta = first[-1]["counterfactual_delta"]
    assert delta is not None
    change = delta["changes"][0]
    source_fact = {fact["fact_id"]: fact for fact in facts}[change["fact_id"]]
    assert change["field"] in source_fact
    assert change["before"] == source_fact[change["field"]]

def test_illegal_primary_edge_fails_teacher_label_validation() -> None:
    episode = build_episodes([fact_pack()])[0]
    initial = label(episode, "initial", "proposed", primary="mm-p1-compounding")
    review = label(episode, "review", "review_required", primary="mm-p1-compounding")
    with pytest.raises(ValidationError, match="primary"):
        validate_teacher_labels([initial, review], [episode])


def test_missing_primary_microconcept_and_review_mismatch_fail() -> None:
    episode = build_episodes([fact_pack()])[0]
    missing = label(episode, "initial", "proposed")
    missing["cited_microconcept_ids"] = ["capacity_headroom"]
    review = label(episode, "review", "accepted")
    with pytest.raises(ValidationError, match="legally connected"):
        validate_teacher_labels([missing, review], [episode])
    initial = label(episode, "initial", "proposed")
    changed = label(episode, "review", "accepted", primary="mm-p1-rate-control")
    changed["near_miss_models"] = ["mm-p1-bottleneck"]
    changed["near_miss_exclusions"] = {"mm-p1-bottleneck": "It is retained only as the excluded adjacent candidate."}
    with pytest.raises(ValidationError, match="exact initial route"):
        validate_teacher_labels([initial, changed], [episode])


def test_full_v1_size_is_explicit_and_case_split_is_deterministic() -> None:
    with pytest.raises(ValidationError, match="160"):
        build_episodes([fact_pack()], require_v1_size=True)
    ids = ["case-c", "case-a", "case-b", "case-d"]
    assert split_case_packs(iter(ids)) == split_case_packs(ids)


def test_global_requires_edge_cannot_be_bypassed_through_another_connection() -> None:
    episode = build_episodes([fact_pack()])[0]
    initial = label(episode, "initial", "proposed", primary="mm-p1-buffer-capacity")
    initial["cited_microconcept_ids"] = ["capacity_headroom"]
    review = label(episode, "review", "review_required", primary="mm-p1-buffer-capacity")
    review["cited_microconcept_ids"] = ["capacity_headroom"]
    with pytest.raises(ValidationError, match="required microconcept"):
        validate_teacher_labels([initial, review], [episode])


def test_derived_selection_contract_preserves_global_requires_edges() -> None:
    contract = {item["model_id"]: item for item in model_selection_contracts(["mm-p1-buffer-capacity"])}
    assert contract["mm-p1-buffer-capacity"]["required_microconcepts"] == ["regime_proximity"]
    assert "capacity_headroom" in contract["mm-p1-buffer-capacity"]["supporting_microconcepts"]



def three_fact_packs() -> list[dict]:
    return [fact_pack(f"fictional-public-example-{index:03d}") for index in range(3)]


def _patch_public_builder(monkeypatch: pytest.MonkeyPatch, packs: list[dict]) -> list[dict]:
    episodes = build_episodes(packs)
    monkeypatch.setattr(build_public_corpus_v1, "build_all_fact_packs", lambda _root: deepcopy(packs))
    monkeypatch.setattr(build_public_corpus_v1, "build_episodes", lambda _packs, require_v1_size=False: deepcopy(episodes))
    monkeypatch.setattr(build_public_corpus_v1, "write_public_cases", lambda _packs: [])
    monkeypatch.setattr(build_public_corpus_v1, "source_cache_manifest", lambda _root: [])
    return episodes


def test_validate_episodes_compares_task_relevance_vectors_not_entire_states(monkeypatch: pytest.MonkeyPatch) -> None:
    pack = fact_pack()
    delta = pack["question_variants"][-1]["counterfactual_delta"]
    delta["microconcept_state_changes"] = [{
        "microconcept_id": "capacity_headroom", "axis": "status", "before": "supported", "after": "candidate",
    }]
    mechanism_profile = PROFILE_BY_QUESTION_KIND["mechanism_diagnosis"]
    conditional_profile = PROFILE_BY_QUESTION_KIND["conditional_or_counterfactual"]
    monkeypatch.setitem(
        question_set_v1.TASK_RELEVANCE_PROFILES,
        conditional_profile,
        deepcopy(question_set_v1.TASK_RELEVANCE_PROFILES[mechanism_profile]),
    )
    with pytest.raises(ValidationError, match="four distinct task_relevance vectors"):
        build_episodes([pack])


def test_public_builder_default_excludes_all_sealed_artifacts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    packs = three_fact_packs()
    episodes = _patch_public_builder(monkeypatch, packs)
    sealed_case_ids = {row["case_id"] for row in episodes if row["split"] == "sealed_test"}
    output = tmp_path / "open"
    source_runtime = tmp_path / "source-runtime"
    monkeypatch.setattr(sys, "argv", [
        "build_public_corpus_v1", "--output", str(output), "--source-runtime-root", str(source_runtime), "--skip-public-cases",
    ])

    assert build_public_corpus_v1.main() == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["source_runtime_root"]).resolve() == source_runtime.resolve()
    assert not (output / "source-cache").exists()
    assert not sealed_case_ids & {row["case_id"] for row in read_jsonl(output / "fact_packs.jsonl")}
    assert not sealed_case_ids & {row["case_id"] for row in read_jsonl(output / "source_ledger.jsonl")}
    assert all(row["split"] != "sealed_test" for row in read_jsonl(output / "build" / "episodes.jsonl"))
    assert all("sealed_test" not in path.parts for path in output.rglob("*"))
    assert not (output / "sealed").exists()


def test_public_builder_rejects_source_cache_inside_training_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_builder(monkeypatch, three_fact_packs())
    output = tmp_path / "open"
    monkeypatch.setattr(sys, "argv", [
        "build_public_corpus_v1", "--output", str(output),
        "--source-runtime-root", str(output / "source-runtime"), "--skip-public-cases",
    ])
    with pytest.raises(ValidationError, match="source runtime/cache must be outside"):
        build_public_corpus_v1.main()


def test_public_builder_requires_double_confirmation_and_isolates_sealed_materialization(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    packs = three_fact_packs()
    episodes = _patch_public_builder(monkeypatch, packs)
    output = tmp_path / "open"
    monkeypatch.setattr(sys, "argv", ["build_public_corpus_v1", "--output", str(output), "--include-sealed"])
    with pytest.raises(ValidationError, match="both --include-sealed and --confirm-sealed"):
        build_public_corpus_v1.main()

    sealed_output = tmp_path / "isolated-sealed"
    monkeypatch.setattr(sys, "argv", [
        "build_public_corpus_v1", "--output", str(output), "--skip-public-cases",
        "--include-sealed", "--confirm-sealed", "--sealed-output", str(sealed_output),
    ])
    assert build_public_corpus_v1.main() == 0
    sealed_case_ids = {row["case_id"] for row in episodes if row["split"] == "sealed_test"}
    assert {row["case_id"] for row in read_jsonl(sealed_output / "fact_packs.jsonl")} == sealed_case_ids
    assert {row["case_id"] for row in read_jsonl(sealed_output / "source_ledger.jsonl")} == sealed_case_ids
    assert all(row["split"] == "sealed_test" for row in read_jsonl(sealed_output / "episodes" / "sealed_test.jsonl"))
    assert len(read_jsonl(sealed_output / "teacher_packets" / "initial.jsonl")) == 4 * len(sealed_case_ids)
    assert not sealed_case_ids & {row["case_id"] for row in read_jsonl(output / "fact_packs.jsonl")}


def test_episode_builder_partitions_sealed_output_and_fails_on_partial_confirmation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    packs_path = tmp_path / "fact_packs.jsonl"
    write_jsonl(packs_path, three_fact_packs())
    output = tmp_path / "episodes-open"
    monkeypatch.setattr(sys, "argv", [
        "build_question_episodes_v1", "--fact-packs", str(packs_path), "--output", str(output),
    ])
    assert build_question_episodes_v1.main() == 0
    assert all(row["split"] != "sealed_test" for row in read_jsonl(output / "episodes.jsonl"))
    assert not (output / "sealed").exists()

    partial = tmp_path / "partial"
    monkeypatch.setattr(sys, "argv", [
        "build_question_episodes_v1", "--fact-packs", str(packs_path), "--output", str(partial), "--include-sealed",
    ])
    with pytest.raises(ValidationError, match="both --include-sealed and --confirm-sealed"):
        build_question_episodes_v1.main()

    confirmed = tmp_path / "confirmed"
    monkeypatch.setattr(sys, "argv", [
        "build_question_episodes_v1", "--fact-packs", str(packs_path), "--output", str(confirmed),
        "--include-sealed", "--confirm-sealed",
    ])
    assert build_question_episodes_v1.main() == 0
    assert all(row["split"] != "sealed_test" for row in read_jsonl(confirmed / "episodes.jsonl"))
    assert all(row["split"] == "sealed_test" for row in read_jsonl(confirmed / "sealed" / "episodes.jsonl"))


def test_teacher_packet_prepare_fails_closed_on_sealed_and_keeps_open_packets_separate(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    episodes = build_episodes(three_fact_packs())
    write_jsonl(episodes_path, episodes)
    output = tmp_path / "packets" / "open.jsonl"

    monkeypatch.setattr(sys, "argv", [
        "prepare_gpt_teacher_packets_v1", "--episodes", str(episodes_path), "--output", str(output), "--mode", "initial",
    ])
    with pytest.raises(ValidationError, match="fail closed"):
        prepare_gpt_teacher_packets_v1.main()
    assert not output.exists()

    monkeypatch.setattr(sys, "argv", [
        "prepare_gpt_teacher_packets_v1", "--episodes", str(episodes_path), "--output", str(output), "--mode", "initial",
        "--splits", "train", "development",
    ])
    assert prepare_gpt_teacher_packets_v1.main() == 0
    open_packets = read_jsonl(output)
    assert open_packets
    assert {"question", "goal", "constraints"}.issubset(open_packets[0]["episode"])

    sealed_output = tmp_path / "packets" / "isolated" / "sealed.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "prepare_gpt_teacher_packets_v1", "--episodes", str(episodes_path), "--output", str(output), "--mode", "initial",
        "--splits", "sealed_test", "--include-sealed", "--confirm-sealed", "--sealed-output", str(sealed_output),
    ])
    assert prepare_gpt_teacher_packets_v1.main() == 0
    assert len(read_jsonl(sealed_output)) == sum(row["split"] == "sealed_test" for row in episodes)
    assert read_jsonl(output) == []
