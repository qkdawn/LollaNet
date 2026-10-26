from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from research.phase1.lollanet_cma.access_views.views import create_access_view
from research.phase1.lollanet_cma.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from research.phase1.lollanet_cma.directions.preflight import (
    _prompt_length_rows,
    _vector_train_view,
    _verify_selection,
    _verify_checkpoint_files,
)
from research.phase1.lollanet_cma.directions.fitting import (
    balanced_record_ids,
    build_normalized_directions,
    is_eligible,
    layer_indices,
)
from research.phase1.lollanet_cma.directions.runner import (
    _load_prefix as load_direction_prefix,
    _mean_completion_log_probability,
)


def _stage0_fixture(tmp_path: Path) -> tuple[Path, dict]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    families = ["a", "b", "c", "d"]
    prompts = []
    controllers = []
    evaluators = []
    for family in families:
        for index, label in enumerate(("yes", "no")):
            record_id = f"{family}-{index}"
            prompts.append({"record_id": record_id, "split": "vector_train", "prompt": "question", "source_variant": "main"})
            controllers.append({"record_id": record_id, "model_family": family, "query_type": family})
            evaluators.append({"record_id": record_id, "answer": label})
    prompts.append({"record_id": "sealed", "split": "sealed_test", "prompt": "secret", "source_variant": "main"})
    controllers.append({"record_id": "sealed", "model_family": "sealed-family", "query_type": "sealed"})
    evaluators.append({"record_id": "sealed", "answer": "yes"})
    atomic_write_jsonl(bundle / "prompts.jsonl", prompts)
    atomic_write_jsonl(bundle / "controller.jsonl", controllers)
    atomic_write_jsonl(bundle / "evaluator.jsonl", evaluators)
    manifest = {
        "type": "lollanet_cma_stage0_bundle_v1",
        "artifacts": {
            name: {"path": str(bundle / f"{name}.jsonl"), "sha256": sha256_file(bundle / f"{name}.jsonl")}
            for name in ("prompts", "controller", "evaluator")
        },
    }
    manifest["bundle_content_sha256"] = stable_hash(manifest)
    atomic_write_json(bundle / "manifest.json", manifest)
    charter = {
        "prerequisites": {"stage0_type": "lollanet_cma_stage0_bundle_v1"},
        "data": {"allowed_split": "vector_train", "expected_records": 8, "expected_records_per_family": 2},
        "families": families,
        "filtering": {"labels": ["yes", "no"]},
    }
    view = tmp_path / "vector-view"
    create_access_view(bundle, "vector_train", view)
    charter["data"]["access_view"] = str(view)
    return view, charter


def test_direction_preflight_records_label_balance_without_reading_other_splits(tmp_path: Path):
    view, charter = _stage0_fixture(tmp_path)
    result = _vector_train_view(view, charter)
    assert result["record_count"] == 8
    assert result["label_counts"]["a"] == {"no": 1, "yes": 1}
    assert "sealed" not in {row["record_id"] for row in read_jsonl(view / "prompts.jsonl")}


def test_direction_preflight_rejects_duplicate_fit_record(tmp_path: Path):
    view, charter = _stage0_fixture(tmp_path)
    prompts = read_jsonl(view / "prompts.jsonl")
    atomic_write_jsonl(view / "prompts.jsonl", [*prompts, dict(prompts[0])])
    manifest = read_json(view / "manifest.json")
    manifest["artifacts"]["prompts"]["sha256"] = sha256_file(view / "prompts.jsonl")
    manifest["record_count"] += 1
    manifest["record_ids_sha256"] = stable_hash([row["record_id"] for row in [*prompts, prompts[0]]])
    manifest.pop("access_view_content_sha256")
    manifest["access_view_content_sha256"] = stable_hash(manifest)
    atomic_write_json(view / "manifest.json", manifest)
    with pytest.raises(ValueError, match="ordered record identity|uniqueness"):
        _vector_train_view(view, charter)


def test_direction_charter_balances_after_filtering_and_seals_test_split():
    path = Path("research/phase1/lollanet_cma/directions/direction_charter.yaml")
    charter = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert charter["filtering"]["balance_after_filtering"] is True
    assert charter["filtering"]["balance_key"] == "gold_answer"
    assert charter["filtering"]["labels"] == ["yes", "no"]
    assert charter["development_selection"]["forbidden_split"] == "sealed_test"
    assert set(charter["contrast_mapping"]) == set(charter["contrast_mapping"].values())


def test_prompt_length_contract_rejects_unmatched_contrast():
    charter = {
        "prompt_contract": {
            "instructions": {"a": "one two", "b": "one two three four"},
            "answer_protocol": "answer",
            "maximum_token_length_difference_fraction": 0.10,
        },
        "contrast_mapping": {"a": "b", "b": "a"},
    }
    with pytest.raises(ValueError, match="token lengths"):
        _prompt_length_rows(charter, str.split)


def test_selection_verification_rejects_changed_source_manifest(tmp_path: Path):
    qualification_charter = tmp_path / "qualification.yaml"
    candidate = {"id": "candidate", "model_id": "org/model", "revision": "a" * 40}
    qualification_charter.write_text(yaml.safe_dump({"candidates": [candidate]}), encoding="utf-8")
    artifact_paths = {}
    for name in ("run_intent", "access_view_manifest", "hook_smoke", "predictions"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        artifact_paths[name] = {"path": str(path), "sha256": sha256_file(path)}
    run = {
        "candidate_id": "candidate",
        "model_id": "org/model",
        "revision": "a" * 40,
        "hard_gate_pass": True,
        "hard_gates": {"complete": True},
        "artifacts": artifact_paths,
    }
    run["run_content_sha256"] = stable_hash(run)
    run_path = tmp_path / "run_manifest.json"
    atomic_write_json(run_path, run)
    selection = {
        "type": "lollanet_cma_base_model_qualification_v1",
        "status": "frozen_before_vector_training",
        "selected_candidate_id": "candidate",
        "selected_model_id": "org/model",
        "selected_revision": "a" * 40,
        "charter_sha256": sha256_file(qualification_charter),
        "sources": [{"manifest_path": str(run_path), "manifest_sha256": sha256_file(run_path)}],
    }
    selection["qualification_content_sha256"] = stable_hash(selection)
    selection_path = tmp_path / "selection.json"
    atomic_write_json(selection_path, selection)
    assert _verify_selection(selection_path, qualification_charter)["selected_candidate_id"] == "candidate"
    run_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        _verify_selection(selection_path, qualification_charter)


def test_checkpoint_verification_rehashes_every_recorded_file(tmp_path: Path):
    payload = tmp_path / "weights.safetensors"
    payload.write_bytes(b"weights")
    manifest = {
        "files": [{"path": payload.name, "size": payload.stat().st_size, "sha256": sha256_file(payload)}],
        "file_count": 1,
        "total_bytes": payload.stat().st_size,
    }
    manifest["checkpoint_content_sha256"] = stable_hash(manifest)
    atomic_write_json(tmp_path / "checkpoint_manifest.json", manifest)
    assert _verify_checkpoint_files(tmp_path)["file_count"] == 1
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        _verify_checkpoint_files(tmp_path)


def test_layer_indices_are_frozen_nearest_positions_and_must_be_distinct():
    assert layer_indices(32, [0.25, 0.5, 0.75]) == [8, 16, 23]
    with pytest.raises(ValueError, match="duplicate"):
        layer_indices(2, [0.25, 0.5, 0.75])


def test_direction_filter_requires_strict_margin_improvement():
    row = {
        "positive_parse_ok": True,
        "positive_correct": True,
        "positive_gold_margin": 0.5,
        "negative_gold_margin": 0.5,
        "hidden_finite": True,
    }
    assert is_eligible(row) is False
    row["positive_gold_margin"] = 0.5001
    assert is_eligible(row) is True


def test_balancing_is_deterministic_and_equal_by_gold_label():
    rows = []
    for label, count in (("yes", 3), ("no", 2)):
        for index in range(count):
            rows.append({
                "record_id": f"{label}-{index}",
                "model_family": "family",
                "gold_answer": label,
                "positive_parse_ok": True,
                "positive_correct": True,
                "positive_gold_margin": 1.0,
                "negative_gold_margin": 0.0,
                "hidden_finite": True,
            })
    selected, diagnostics = balanced_record_ids(rows, ["family"], ["yes", "no"], seed=7)
    repeated, _ = balanced_record_ids(list(reversed(rows)), ["family"], ["yes", "no"], seed=7)
    assert selected == repeated
    assert len(selected["family"]) == 4
    assert diagnostics["family"]["balanced_per_label"] == 2


def test_direction_builder_uses_pooled_contrasts_for_the_general_vector():
    import numpy as np

    selected = {"family-a": ["a-1", "a-2"], "family-b": ["b-1", "b-2"]}
    values = {
        "a-1": np.array([1.0, 0.0]),
        "a-2": np.array([3.0, 0.0]),
        "b-1": np.array([0.0, 2.0]),
        "b-2": np.array([0.0, 4.0]),
    }
    vectors, metadata = build_normalized_directions(selected, [5], lambda record_id, _layer: values[record_id])
    np.testing.assert_allclose(vectors["family-a__layer_5"], [1.0, 0.0])
    np.testing.assert_allclose(vectors["family-b__layer_5"], [0.0, 1.0])
    np.testing.assert_allclose(
        vectors["general__layer_5"],
        np.array([1.0, 1.5]) / np.linalg.norm([1.0, 1.5]),
        rtol=1e-6,
    )
    assert metadata["families"]["family-a"]["5"]["record_count"] == 2
    assert metadata["families"]["family-a"]["5"]["dtype"] == "float32"
    assert len(metadata["families"]["family-a"]["5"]["tensor_sha256"]) == 64
    assert metadata["general"]["5"]["record_count"] == 4
    assert metadata["general"]["5"]["family_record_counts"] == {"family-a": 2, "family-b": 2}
    equal_all = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
    np.testing.assert_allclose(vectors["equal_all__layer_5"], equal_all, rtol=1e-6)
    assert not np.allclose(vectors["general__layer_5"], vectors["equal_all__layer_5"])
    assert metadata["equal_all"]["5"]["source_family_keys"] == [
        "family-a__layer_5", "family-b__layer_5",
    ]


def test_direction_resume_rejects_changed_record_content(tmp_path: Path):
    contrast = tmp_path / "contrast.safetensors"
    contrast.write_bytes(b"contrast")
    row = {
        "record_id": "a",
        "contrast_path": str(contrast),
        "contrast_sha256": sha256_file(contrast),
        "positive_gold_margin": 1.0,
    }
    row["record_content_sha256"] = stable_hash(row)
    path = tmp_path / "records.jsonl"
    atomic_write_jsonl(path, [row])
    assert load_direction_prefix(path, ["a"], "test")[0]["record_id"] == "a"
    row["positive_gold_margin"] = -1.0
    atomic_write_jsonl(path, [row])
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_direction_prefix(path, ["a"], "test")


def test_teacher_forced_score_captures_the_token_before_the_answer():
    from types import SimpleNamespace

    import torch

    class Tokenizer:
        def __call__(self, text, return_tensors, add_special_tokens):
            del return_tensors, add_special_tokens
            assert text == "Final answer: yes"
            return {"input_ids": torch.tensor([[4, 5]])}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)
            self.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            hidden = self.model.layers[0](self.embedding(input_ids))
            return SimpleNamespace(logits=torch.zeros((*hidden.shape[:2], 10)))

    model = Model()
    prefix = {"input_ids": torch.tensor([[1, 2, 3]])}
    score, captured = _mean_completion_log_probability(
        model, Tokenizer(), prefix, "Final answer: yes", [0],
    )
    assert score == pytest.approx(-math.log(10))
    torch.testing.assert_close(captured[0], model.embedding(prefix["input_ids"])[0, -1].float())
