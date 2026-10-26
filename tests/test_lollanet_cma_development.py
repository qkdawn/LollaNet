from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from research.phase1.lollanet_cma.access_views.views import create_access_view
from research.phase1.lollanet_cma.common import atomic_write_json, atomic_write_jsonl, sha256_file, stable_hash
from research.phase1.lollanet_cma.development.hooks import steer_residual
from research.phase1.lollanet_cma.development.runner import (
    _condition_id,
    _load_condition_prefix,
    _random_vectors,
    _vector_reference,
)
from research.phase1.lollanet_cma.development.selection import (
    choose_grid_setting,
    specificity_result,
    summarize_condition,
)
from research.phase1.lollanet_cma.development.preflight import audit_development_preflight


def test_grid_selection_uses_macro_accuracy_then_frozen_ties():
    rows = [
        {"layer": 20, "alpha": 1.0, "macro_family_accuracy": 0.7},
        {"layer": 10, "alpha": 0.5, "macro_family_accuracy": 0.7},
        {"layer": 5, "alpha": 0.5, "macro_family_accuracy": 0.7},
        {"layer": 15, "alpha": 2.0, "macro_family_accuracy": 0.6},
    ]
    assert choose_grid_setting(rows) == rows[2]


def test_specificity_gate_is_strict_and_does_not_accept_a_tie():
    tied = specificity_result(0.5, 0.6, 0.6, 0.55)
    assert tied["passed"] is False
    passed = specificity_result(0.5, 0.61, 0.6, 0.55)
    assert passed["passed"] is True


def test_specificity_failure_does_not_fall_back_to_the_second_grid_setting():
    summaries = [
        {"condition_id": "winner", "layer": 10, "alpha": 1.0, "macro_family_accuracy": 0.7},
        {"condition_id": "runner-up", "layer": 5, "alpha": 0.5, "macro_family_accuracy": 0.69},
    ]
    selected = choose_grid_setting(summaries)
    specificity = specificity_result(0.5, selected["macro_family_accuracy"], 0.7, 0.55)
    assert specificity["passed"] is False
    assert selected["condition_id"] == "winner"


def test_condition_id_is_stable_for_the_frozen_grid():
    assert _condition_id("base") == "base"
    assert _condition_id("oracle_correct", 9, 0.25) == "oracle_correct__layer_9__alpha_0p25"
    assert _condition_id("oracle_wrong", 17, 2.0) == "oracle_wrong__layer_17__alpha_2"


def test_condition_resume_requires_an_exact_hashed_prefix(tmp_path: Path):
    path = tmp_path / "condition.jsonl"
    rows = []
    for record_id in ("a", "b"):
        row = {"condition_id": "base", "record_id": record_id, "correct": True}
        row["record_content_sha256"] = stable_hash(row)
        rows.append(row)
    atomic_write_jsonl(path, rows)
    assert _load_condition_prefix(path, ["a", "b", "c"], "base") == rows

    tampered = [dict(rows[0]), dict(rows[1])]
    tampered[1]["correct"] = False
    atomic_write_jsonl(path, tampered)
    with pytest.raises(ValueError, match="condition row changed"):
        _load_condition_prefix(path, ["a", "b", "c"], "base")

    reordered = [dict(rows[1]), dict(rows[0])]
    atomic_write_jsonl(path, reordered)
    with pytest.raises(ValueError, match="not an exact ordered prefix"):
        _load_condition_prefix(path, ["a", "b", "c"], "base")


def test_random_vectors_are_deterministic_unit_norm_and_seed_sensitive():
    first = _random_vectors(["a", "b"], [3, 7], 16, 42)
    second = _random_vectors(["a", "b"], [3, 7], 16, 42)
    different = _random_vectors(["a", "b"], [3, 7], 16, 43)
    assert set(first) == {"a__layer_3", "b__layer_3", "a__layer_7", "b__layer_7"}
    for key, value in first.items():
        np.testing.assert_array_equal(value, second[key])
        assert np.linalg.norm(value.astype(np.float64)) == pytest.approx(1.0, abs=1e-6)
    assert any(not np.array_equal(first[key], different[key]) for key in first)


def test_vector_references_map_each_control_to_the_frozen_tensor_source():
    wrong_mapping = {"a": "d", "b": "a", "c": "b", "d": "c"}
    assert _vector_reference("oracle_correct", "b", 7, wrong_mapping) == (
        "b__layer_7", "b__layer_7",
    )
    assert _vector_reference("oracle_wrong", "b", 7, wrong_mapping) == (
        "a__layer_7", "a__layer_7",
    )
    assert _vector_reference("random_norm_matched", "b", 7, wrong_mapping) == (
        "random::b__layer_7", "b__layer_7",
    )
    with pytest.raises(ValueError, match="unsupported development steering condition"):
        _vector_reference("unknown", "b", 7, wrong_mapping)


def test_condition_summary_is_macro_averaged_across_families():
    rows = [
        {"record_id": "a-1", "model_family": "a", "correct": True, "parse_ok": True},
        {"record_id": "a-2", "model_family": "a", "correct": True, "parse_ok": True},
        {"record_id": "b-1", "model_family": "b", "correct": False, "parse_ok": False},
    ]
    summary = summarize_condition(rows, ["a", "b"])
    assert summary["macro_family_accuracy"] == 0.5
    assert summary["parse_failure_rate"] == pytest.approx(1 / 3)


def test_steering_hook_changes_only_the_current_final_token():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))

    model = Model()
    hidden = torch.zeros((1, 3, 2))
    with steer_residual(model, 0, torch.tensor([1.0, -1.0]), 0.5) as state:
        output = model.model.layers[0](hidden)
    torch.testing.assert_close(output[0, :-1], torch.zeros((2, 2)))
    torch.testing.assert_close(output[0, -1], torch.tensor([0.5, -0.5]))
    assert state == {"calls": 1, "finite": True, "hidden_size": 2}


def test_development_preflight_stays_blocked_without_a_direction_manifest(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    families = ["a", "b", "c", "d"]
    prompts = [
        {"record_id": family, "split": "development", "prompt": family, "source_variant": "main"}
        for family in families
    ]
    controllers = [
        {"record_id": family, "model_family": family, "query_type": family} for family in families
    ]
    evaluators = [{"record_id": family, "answer": "yes"} for family in families]
    for name, rows in (("prompts", prompts), ("controller", controllers), ("evaluator", evaluators)):
        atomic_write_jsonl(bundle / f"{name}.jsonl", rows)
    source_manifest = {
        "artifacts": {
            name: {"path": str(bundle / f"{name}.jsonl"), "sha256": sha256_file(bundle / f"{name}.jsonl")}
            for name in ("prompts", "controller", "evaluator")
        },
    }
    source_manifest["bundle_content_sha256"] = stable_hash(source_manifest)
    atomic_write_json(bundle / "manifest.json", source_manifest)
    view = tmp_path / "development-view"
    create_access_view(bundle, "development", view)
    charter = {
        "data": {
            "allowed_split": "development",
            "access_view": str(view),
            "expected_records": 4,
            "expected_records_per_family": 1,
        },
        "families": families,
        "prerequisites": {
            "direction_manifest_type": "lollanet_cma_direction_fit_v1",
            "direction_status": "fitted",
        },
    }
    charter_path = tmp_path / "development.yaml"
    charter_path.write_text(yaml.safe_dump(charter), encoding="utf-8")
    result = audit_development_preflight(
        charter_path, view, tmp_path / "missing-direction.json", tmp_path / "preflight.json",
    )
    assert result["development_gate_status"] == "blocked_missing_directions"
    assert result["errors"] == ["direction_manifest_missing"]


def test_development_charter_requires_pooled_general_direction_provenance():
    direction_charter = yaml.safe_load(Path(
        "research/phase1/lollanet_cma/directions/direction_charter.yaml",
    ).read_text(encoding="utf-8"))
    assert direction_charter["direction"]["general_direction"] == (
        "normalized_pooled_mean_of_all_selected_record_contrasts"
    )
