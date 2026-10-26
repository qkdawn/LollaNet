from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from research.phase1.lollasteer_q.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    runtime_code_identity,
    sha256_file,
    stable_hash,
)
from research.phase1.lollasteer_q.annotate_lures import parse_ranking, run_annotation_session
from research.phase1.lollasteer_q.annotation import (
    adjudicate_lures,
    collect_primary_annotation_delivery,
    migrate_unstarted_annotation_contract,
    prepare_primary_annotation_delivery,
    prepare_annotation_packets,
    prepare_tiebreak_packet,
    validate_annotation_packets,
    validate_primary_annotation_delivery,
)
from research.phase1.lollasteer_q.data import OPTION_ORDER_POLICY, build_pilot, load_lures, permute_options
from research.phase1.lollasteer_q.metrics import (
    brier_score,
    expected_calibration_error,
    gold_cross_entropy,
    gold_lure_log_odds,
    multiclass_brier,
    risk_coverage_curve,
    third_order_alternating_sum,
)
from research.phase1.lollasteer_q.model_runner import (
    _runtime_environment,
    conditioned_prompt,
    oracle_factor,
    parse_generation,
    relative_layer_indices,
    residual_hooks,
    score_answer_prefixes,
)
from research.phase1.lollasteer_q.power_analysis import simulate_interaction_power
from research.phase1.lollasteer_q.prompts import TEMPLATES, condition_specs, prompt_only_instruction
from research.phase1.lollasteer_q.report import generate_report
from research.phase1.lollasteer_q.runtime_audit import (
    audit_direction_manifest,
    audit_run_manifest,
    audit_runtime,
    execution_identity,
)
from research.phase1.lollasteer_q.selection import select_discovery_runs
from research.phase1.lollasteer_q.steering import Intervention, fit_directions
from research.phase1.lollasteer_q.validate import validate_bundle


CATEGORIES = ["math", "physics", "chemistry", "engineering", "computer science", "economics"]


def test_runtime_environment_handles_cuda_available_with_cpu_model():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            assert isinstance(index, int)
            return f"GPU {index}"

    class FakeTorch:
        __version__ = "fixture-torch"
        version = type("Version", (), {"cuda": "fixture-cuda"})
        cuda = FakeCuda()

    model = type("Model", (), {
        "device": "cpu",
        "hf_device_map": {"model.layers.1": 1, "model.layers.0": 0, "lm_head": "cpu"},
    })()
    transformers = type("Transformers", (), {"__version__": "fixture-transformers"})

    environment = _runtime_environment(FakeTorch(), transformers, model)

    assert environment["model_device"] == "cpu"
    assert environment["cuda_device_names"] == ["GPU 0", "GPU 1"]
    assert environment["model_device_map"] == {
        "lm_head": "cpu", "model.layers.0": "0", "model.layers.1": "1",
    }


def _config(path: Path, per_category: int = 2) -> Path:
    value = {
        "version": "fixture", "seed": 7,
        "data": {
            "dataset": "fixture", "revision": "frozen", "source_split": "test", "categories": CATEGORIES,
            "items_per_category": per_category, "discovery_per_category": per_category - 1,
            "holdout_per_category": 1, "token_budget": 80, "option_order_policy": OPTION_ORDER_POLICY,
        },
        "model": {
            "primary_checkpoint": "fixture", "fallback_checkpoint": None, "fallback_policy": "no_fallback",
            "strength_grid": [1.0], "low_rank_grid": [1],
        },
        "report": {"bootstrap_samples": 20, "confidence_level": 0.95, "minimum_harmful_effect": 0.03,
                   "maximum_preservation_loss": 0.02, "minimum_steering_improvement": 0.02},
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _source_and_lures(tmp_path: Path, per_category: int = 2) -> tuple[Path, Path]:
    rows = []
    lures = {}
    for category in CATEGORIES:
        for index in range(per_category):
            item_id = f"{category}-{index}"
            rows.append({
                "question_id": item_id, "question": f"Question {item_id}?", "options": ["gold", "lure", "other"],
                "answer_index": 0, "answer": "A", "cot_content": "secret rationale", "category": category, "src": "fixture",
            })
            lures[item_id] = 1
    source = tmp_path / "source.jsonl"
    atomic_write_jsonl(source, rows)
    lure_path = tmp_path / "lures.yaml"
    lure_path.write_text(yaml.safe_dump({
        "selector": {"id": "independent-fixture", "frozen_at": "2026-07-23T00:00:00Z", "method": "fixture", "uses_target_model_outputs": False},
        "items": lures,
    }), encoding="utf-8")
    return source, lure_path


def test_formal_config_declares_unequal_rank_search_and_family_adjusted_gates():
    config_path = Path("research/phase1/lollasteer_q/config_pilot.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["comparison_budget"]["same_hyperparameter_trials"] is False
    assert config["comparison_budget"]["unequal_trial_reason"]
    assert int(config["report"]["bootstrap_samples"]) >= 10_000
    assert float(config["report"]["gate_family_alpha"]) == 0.05
    assert config["report"]["required_internal_factors"] == ["reward", "social", "certainty"]
    assert config["report"]["required_prompt_only_factors"] == ["reward", "social", "certainty"]


def test_runtime_audit_quarantines_legacy_manifests(tmp_path):
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    atomic_write_json(manifest_path, {
        "type": "lollasteer_q_model_run", "method": "baseline", "record_count": 1,
    })
    audited = audit_run_manifest(manifest_path)
    assert audited["status"] == "legacy_ineligible"
    assert "execution_identity_sha256" in audited["missing_current_fields"]
    (tmp_path / "array.json").write_text("[]\n", encoding="utf-8")
    report = audit_runtime(tmp_path)
    assert report["summary"]["run_status_counts"] == {"legacy_ineligible": 1}
    assert report["summary"]["formal_report_eligible_runs"] == 0


def test_runtime_audit_rejects_direction_without_environment_identity(tmp_path):
    array_path = tmp_path / "directions.npz"
    array_path.write_bytes(b"fixture")
    metadata_path = tmp_path / "directions.json"
    atomic_write_json(metadata_path, {
        "type": "lollasteer_q_direction_bundle",
        "npz_path": str(array_path), "npz_sha256": sha256_file(array_path),
        "checkpoint_actual": "fixture", "checkpoint_revision": "frozen",
        "chat_template_sha256": "chat", "dtype": "float32",
        "prompt_bundle_sha256": "prompts", "layer_indices": [2],
        "factors": {
            factor: {"2": {"available_rank": 1, "hidden_size": 4}}
            for factor in ("reward", "social", "certainty")
        },
    })

    audited = audit_direction_manifest(metadata_path)

    assert audited["status"] == "legacy_ineligible"
    assert audited["missing_current_fields"] == [
        "dependency_lock_sha256", "runtime_code_sha256", "runtime_environment_sha256",
        "source_activations", "source_run", "source_run_manifest",
    ]


def _write_run_fixture(
    path: Path, rows: list[dict], prompts_path: Path, *, requested_split: str = "all",
    direction_sha256: str | None = None, activation_path: Path | None = None,
) -> Path:
    method = str(rows[0]["method"])
    is_linear = method in {"caa", "low_rank"}
    rows = [{
        **row,
        "steering_layer": 2 if is_linear else None,
        "steering_strength": 1.0 if is_linear else None,
        "steering_rank": 1 if is_linear else None,
    } for row in rows]
    atomic_write_jsonl(path, rows)
    run_key = str(rows[0]["run_key"])
    factor = rows[0].get("steering_factor")
    direction_source = None
    lock_path = path.parent / "fixture-uv.lock"
    if not lock_path.is_file():
        lock_path.write_text("fixture-lock\n", encoding="utf-8")
    runtime_environment = {"python": "fixture", "platform": "fixture", "model_device": "cpu"}
    runtime_environment_sha256 = stable_hash(runtime_environment)
    code_root = path.parent / "fixture-code"
    code_root.mkdir(exist_ok=True)
    code_file = code_root / "runner.py"
    if not code_file.is_file():
        code_file.write_text("# fixture runtime\n", encoding="utf-8")
    runtime_code = runtime_code_identity(code_root)
    if activation_path is None:
        activation_path = path.with_suffix(".activations.npz")
        activation_path.write_bytes(b"fixture-activations")
    if is_linear:
        direction_path = path.parent / "direction-fixture.npz"
        metadata_path = path.parent / "direction-fixture.json"
        source_outputs_path = path.parent / "direction-source-outputs.jsonl"
        source_manifest_path = source_outputs_path.with_suffix(".manifest.json")
        if not source_manifest_path.is_file():
            _write_run_fixture(
                source_outputs_path,
                [
                    {"record_id": prompt["record_id"], "method": "baseline", "run_key": "baseline"}
                    for prompt in read_jsonl(prompts_path)
                ],
                prompts_path,
            )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_activations_path = Path(source_manifest["artifacts"]["activations"]["path"])
        if not direction_path.is_file():
            direction_path.write_bytes(str(direction_sha256 or "direction-fixture").encode("utf-8"))
        if not metadata_path.is_file():
            atomic_write_json(metadata_path, {
                "type": "lollasteer_q_direction_bundle", "npz_sha256": sha256_file(direction_path),
                "npz_path": str(direction_path),
                "checkpoint_actual": "fixture", "checkpoint_revision": "frozen",
                "chat_template_sha256": "template", "dtype": "float32", "layer_indices": [2],
                "dependency_lock_sha256": sha256_file(lock_path),
                "runtime_code_sha256": runtime_code["sha256"],
                "runtime_environment_sha256": runtime_environment_sha256,
                "prompt_bundle_sha256": sha256_file(prompts_path),
                "source_run": source_manifest["run_id"],
                "source_run_manifest": {
                    "path": str(source_manifest_path), "sha256": sha256_file(source_manifest_path),
                },
                "source_activations": {
                    "path": str(source_activations_path), "sha256": sha256_file(source_activations_path),
                },
                "factors": {
                    name: {"2": {"available_rank": 1, "hidden_size": 3}}
                    for name in ("reward", "social", "certainty")
                },
            })
        direction_source = {
            "path": str(direction_path), "sha256": sha256_file(direction_path),
            "metadata_path": str(metadata_path), "metadata_sha256": sha256_file(metadata_path),
        }
    manifest = {
        "type": "lollasteer_q_model_run",
        "checkpoint_actual": "fixture", "checkpoint_revision": "frozen",
        "chat_template_sha256": "template", "dtype": "float32", "layer_indices": [2],
        "method": method, "run_key": run_key, "factor": factor,
        "layer": 2 if is_linear else None, "strength": 1.0 if is_linear else None,
        "rank": 1 if is_linear else None, "requested_split": requested_split,
        "record_ids_sha256": stable_hash(sorted(str(row["record_id"]) for row in rows)),
        "direction_bundle": direction_source,
        "prompt_only_instruction_sha256": None,
        "oracle_gate_policy": "single_active_factor_plus_matching_valid_control",
        "do_sample": False, "batch_size": 1, "max_new_tokens": 16,
        "activation_capture_policy": "last_prompt_token_pre_intervention",
        "intervention_position_policy": "last_sequence_token_each_forward",
        "option_scoring_policy": "teacher_forced_answer_prefix_token_by_token",
        "dependency_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "runtime_code": runtime_code,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256,
        "inputs": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
        "artifacts": {
            "outputs": {"path": str(path), "sha256": sha256_file(path)},
            "activations": {"path": str(activation_path), "sha256": sha256_file(activation_path)},
        },
        "record_count": len(rows), "prompt_record_count": len(read_jsonl(prompts_path)),
    }
    manifest["execution_identity_sha256"] = stable_hash(execution_identity(manifest))
    manifest["run_id"] = manifest["execution_identity_sha256"][:24]
    manifest_path = path.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def test_runtime_audit_rejects_code_changed_after_run(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    atomic_write_jsonl(prompts_path, [{"record_id": "item-1", "split": "all"}])
    manifest_path = _write_run_fixture(
        outputs_path,
        [{"record_id": "item-1", "method": "baseline", "run_key": "baseline"}],
        prompts_path,
    )
    assert audit_run_manifest(manifest_path)["status"] == "current_complete"

    (tmp_path / "fixture-code" / "runner.py").write_text("# changed runtime\n", encoding="utf-8")
    audited = audit_run_manifest(manifest_path)

    assert audited["status"] == "invalid"
    assert "runtime_code_missing_or_changed" in audited["errors"]


def _write_selection_fixture(path: Path, config_path: Path, evaluator_path: Path, prompt_sha256: str, run_keys: list[str]) -> Path:
    direction_path = path.parent / "direction-fixture.npz"
    metadata_path = path.parent / "direction-fixture.json"
    if not direction_path.is_file():
        direction_path.write_bytes(b"direction-fixture")
    if not metadata_path.is_file():
        lock_path = path.parent / "fixture-uv.lock"
        lock_path.write_text("fixture-lock\n", encoding="utf-8")
        code_root = path.parent / "fixture-code"
        code_root.mkdir(exist_ok=True)
        (code_root / "runner.py").write_text("# fixture runtime\n", encoding="utf-8")
        runtime_code = runtime_code_identity(code_root)
        prompts_path = evaluator_path.parent / "prompts.jsonl"
        source_outputs_path = path.parent / "direction-source-outputs.jsonl"
        source_manifest_path = _write_run_fixture(
            source_outputs_path,
            [
                {"record_id": prompt["record_id"], "method": "baseline", "run_key": "baseline"}
                for prompt in read_jsonl(prompts_path)
            ],
            prompts_path,
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_activations_path = Path(source_manifest["artifacts"]["activations"]["path"])
        atomic_write_json(metadata_path, {
            "type": "lollasteer_q_direction_bundle", "npz_sha256": sha256_file(direction_path),
            "npz_path": str(direction_path),
            "checkpoint_actual": "fixture", "checkpoint_revision": "frozen",
            "chat_template_sha256": "template", "dtype": "float32", "layer_indices": [2],
            "dependency_lock_sha256": sha256_file(lock_path),
            "runtime_code_sha256": runtime_code["sha256"],
            "runtime_environment_sha256": stable_hash({"python": "fixture", "platform": "fixture", "model_device": "cpu"}),
            "prompt_bundle_sha256": prompt_sha256,
            "source_run": source_manifest["run_id"],
            "source_run_manifest": {
                "path": str(source_manifest_path), "sha256": sha256_file(source_manifest_path),
            },
            "source_activations": {
                "path": str(source_activations_path), "sha256": sha256_file(source_activations_path),
            },
            "factors": {
                name: {"2": {"available_rank": 1, "hidden_size": 3}}
                for name in ("reward", "social", "certainty")
            },
        })
    selected = {"caa": {}, "low_rank": {}}
    for run_key in run_keys:
        method, factor = run_key.split("__", 2)[:2]
        selected[method][factor] = {
            "run_key": run_key, "layer": 2, "strength": 1.0, "rank": 1,
            "direction_bundle_sha256": sha256_file(direction_path),
            "direction_metadata_sha256": sha256_file(metadata_path), "eligible": True,
        }
    value = {
        "type": "lollasteer_q_discovery_selection", "status": "frozen_before_holdout",
        "config_sha256": sha256_file(config_path), "evaluator_sha256": sha256_file(evaluator_path),
        "prompt_bundle_sha256": prompt_sha256, "selected": selected,
    }
    value["selection_content_sha256"] = stable_hash(value)
    atomic_write_json(path, value)
    return path


def test_condition_specs_have_eight_factorial_cells_and_three_controls():
    specs = condition_specs()
    factorial = [row for row in specs if row["control_type"] == "factorial"]
    assert len(factorial) == 8
    assert len({(r["reward_lure"], r["social_lure"], r["certainty_pressure"]) for r in factorial}) == 8
    assert {row["control_type"] for row in specs} == {"factorial", "valid_reward", "valid_social", "calibration"}
    assert TEMPLATES["A"]["certainty_keyword_free"] is False
    assert TEMPLATES["B"]["certainty_keyword_free"] is True
    assert all(word not in TEMPLATES["B"]["certainty"].lower() for word in ("certain", "confidence", "firm", "doubt", "hesitat"))


def test_prompt_only_instructions_are_factor_specific_and_label_agnostic():
    instructions = {factor: prompt_only_instruction(factor) for factor in ("reward", "social", "certainty")}
    assert len(set(instructions.values())) == 3
    for text in instructions.values():
        assert "gold" not in text.lower()
        assert "lure" not in text.lower()
        assert all(f"{label}." not in text for label in "ABCDEFGHIJ")
    try:
        prompt_only_instruction("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown prompt-only factor was accepted")
    base = "Question and options"
    assert conditioned_prompt(base, "baseline", None, False) == base
    assert conditioned_prompt(base, "prompt_only", "reward", False) == base
    conditioned = conditioned_prompt(base, "prompt_only", "reward", True)
    assert conditioned.startswith(base)
    assert prompt_only_instruction("reward") in conditioned
    assert oracle_factor({"control_type": "valid_reward"}) == "reward"
    assert oracle_factor({"control_type": "valid_social"}) == "social"
    assert oracle_factor({"control_type": "calibration"}) == "certainty"
    assert oracle_factor({"control_type": "factorial", "reward_lure": 1, "social_lure": 0, "certainty_pressure": 0}) == "reward"
    assert oracle_factor({"control_type": "factorial", "reward_lure": 1, "social_lure": 1, "certainty_pressure": 0}) is None


def test_build_and_validate_separate_prompts_from_evaluator(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    output = tmp_path / "bundle"
    manifest = build_pilot(_config(tmp_path / "config.yaml"), lures, output, source)
    report = validate_bundle(output)
    assert report["passed"]
    assert report["base_items"] == 12
    assert report["records"] == 132
    assert report["split_counts"] == {"discovery": 6, "pilot_holdout": 6}
    prompt_text = (output / "prompts.jsonl").read_text(encoding="utf-8")
    assert "gold_index" not in prompt_text
    assert "lure_index" not in prompt_text
    assert "cot_content" not in prompt_text
    assert manifest["counts"]["prompt_records"] == 132
    prompt_rows = [json.loads(line) for line in prompt_text.splitlines()]
    assert {row["certainty_keyword_free"] for row in prompt_rows if row["split"] == "discovery"} == {False}
    assert {row["certainty_keyword_free"] for row in prompt_rows if row["split"] == "pilot_holdout"} == {True}
    keyword_free_pressure = next(
        row for row in prompt_rows
        if row["split"] == "pilot_holdout" and row["certainty_pressure"] and not row["reward_lure"] and not row["social_lure"]
    )
    assert "Use a concise declarative response format; avoid modal qualifiers." in keyword_free_pressure["prompt"]
    assert "firmly" not in keyword_free_pressure["prompt"].lower()


def test_builder_rejects_target_model_selected_lures(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    payload = yaml.safe_load(lures.read_text(encoding="utf-8"))
    payload["selector"]["uses_target_model_outputs"] = True
    lures.write_text(yaml.safe_dump(payload), encoding="utf-8")
    try:
        build_pilot(_config(tmp_path / "config.yaml"), lures, tmp_path / "bundle", source)
    except ValueError as exc:
        assert "target-model" in str(exc)
    else:
        raise AssertionError("target-model-selected lures were accepted")


def test_option_permutation_is_seeded_and_remaps_gold_and_lure():
    options = ["gold", "lure", "other", "fourth"]
    first = permute_options(options, 0, 1, "q-1", 7)
    second = permute_options(options, 0, 1, "q-1", 7)
    changed_seed = permute_options(options, 0, 1, "q-1", 8)
    assert first == second
    assert first != changed_seed
    permuted, gold_index, lure_index = first
    assert permuted[gold_index] == "gold"
    assert permuted[lure_index] == "lure"
    assert sorted(permuted) == sorted(options)


def test_lure_loader_rejects_placeholder_and_naive_freeze_time(tmp_path):
    lure_path = tmp_path / "lures.yaml"
    payload = {
        "selector": {
            "id": "replace-with-independent-selector-id",
            "frozen_at": "YYYY-MM-DDTHH:MM:SSZ",
            "method": "fixture",
            "uses_target_model_outputs": False,
        },
        "items": {"item-1": 1},
    }
    lure_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    try:
        load_lures(lure_path)
    except ValueError as exc:
        assert "non-placeholder" in str(exc)
    else:
        raise AssertionError("placeholder lure selector was accepted")
    payload["selector"]["id"] = "independent-fixture"
    payload["selector"]["frozen_at"] = "2026-07-24T12:00:00"
    lure_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    try:
        load_lures(lure_path)
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:
        raise AssertionError("timezone-free lure freeze time was accepted")


def test_validator_detects_split_overlap_and_hash_tampering(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    output = tmp_path / "bundle"
    build_pilot(_config(tmp_path / "config.yaml"), lures, output, source)
    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[-1]["base_item_id"] = rows[0]["base_item_id"]
    atomic_write_jsonl(output / "prompts.jsonl", rows)
    report = validate_bundle(output)
    assert not report["passed"]
    assert any("SHA-256 mismatch" in error for error in report["errors"])
    assert any("overlap" in error for error in report["errors"])


def test_generation_parser_and_layer_fractions_are_deterministic():
    parsed = parse_generation("Answer: C\nExplanation: bounded.\nConfidence: 72")
    assert parsed == {"parsed_answer": "C", "self_reported_confidence": 0.72, "parse_ok": True}
    legacy = parse_generation("Answer: <option letter>C>\nExplanation: bounded.\nConfidence: 72")
    assert legacy["parsed_answer"] == "C"
    assert not parse_generation("Answer: C")["parse_ok"]
    assert relative_layer_indices(8, [0.25, 0.5, 0.75]) == [2, 4, 5]


def test_residual_intervention_changes_only_final_sequence_position():
    import torch

    class FixtureModel:
        def __init__(self):
            self.model = type("Inner", (), {})()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class FixtureIntervention:
        layer_index = 0

        @staticmethod
        def delta(_activation):
            return np.asarray([1.0, 2.0], dtype=np.float32)

    model = FixtureModel()
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    with residual_hooks(model, [0], FixtureIntervention()):
        modified = model.model.layers[0](hidden)
    assert torch.equal(modified[0, :2], hidden[0, :2])
    assert torch.equal(modified[0, -1], torch.tensor([1.0, 2.0]))

    class InvalidIntervention:
        layer_index = 0

        @staticmethod
        def delta(_activation):
            return np.asarray([np.nan, 0.0], dtype=np.float32)

    try:
        with residual_hooks(model, [0], InvalidIntervention()):
            model.model.layers[0](hidden)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite intervention delta was accepted")


def test_answer_prefix_scoring_steers_only_each_current_prediction_position():
    import torch

    class FixtureTokenizer:
        def __call__(self, text, **_kwargs):
            if text == "PROMPT":
                values = [0, 0]
            elif text.endswith("A"):
                values = [1, 2]
            elif text.endswith("B"):
                values = [1, 3]
            else:
                raise AssertionError(text)
            return {"input_ids": torch.tensor([values], dtype=torch.long)}

    class FixtureModel:
        device = torch.device("cpu")

        def __init__(self):
            self.model = type("Inner", (), {})()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.hidden_after_hook = []

        def __call__(self, input_ids, **_kwargs):
            hidden = torch.zeros((1, input_ids.shape[1], 2), dtype=torch.float32)
            hidden = self.model.layers[0](hidden)
            self.hidden_after_hook.append(hidden.detach().clone())
            logits = torch.zeros((1, input_ids.shape[1], 5), dtype=torch.float32)
            logits[:, :, 2] = hidden[:, :, 0]
            logits[:, :, 3] = -hidden[:, :, 0]
            return type("Output", (), {"logits": logits})()

    class FixtureIntervention:
        layer_index = 0

        @staticmethod
        def delta(_activation):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    model = FixtureModel()
    scores = score_answer_prefixes(model, FixtureTokenizer(), "PROMPT", 2, FixtureIntervention())
    assert scores.shape == (2,)
    assert len(model.hidden_after_hook) == 4
    assert [value.shape[1] for value in model.hidden_after_hook] == [2, 3, 2, 3]
    for value in model.hidden_after_hook:
        assert torch.equal(value[0, :-1], torch.zeros_like(value[0, :-1]))
        assert torch.equal(value[0, -1], torch.tensor([1.0, 0.0]))


def test_power_simulation_is_explicit_and_reproducible():
    null = simulate_interaction_power([20], 0.2, 0.0, simulations=200, seed=3)
    alternative = simulate_interaction_power([20], 0.2, 0.4, simulations=200, seed=3)
    assert null["status"] == "design_simulation_not_empirical_evidence"
    assert null == simulate_interaction_power([20], 0.2, 0.0, simulations=200, seed=3)
    assert alternative["results"][0]["power"] > null["results"][0]["power"]
    assert alternative["assumptions"]["estimand"].startswith("third-order alternating")


def test_metrics_use_probability_scale_and_correct_third_order_sign():
    assert np.isclose(brier_score(0.8, True), 0.04)
    assert np.isclose(multiclass_brier([0.8, 0.1, 0.1], 0), 0.06)
    assert np.isclose(gold_cross_entropy([0.8, 0.1, 0.1], 0), -np.log(0.8))
    assert np.isclose(gold_lure_log_odds([0.6, 0.3, 0.1], 0, 1), np.log(2))
    assert np.isclose(expected_calibration_error([0.8, 0.6], [True, False], bins=2), 0.2)
    curve = risk_coverage_curve([0.9, 0.8, 0.2, 0.1], [True, False, True, False])
    assert curve[0] == {"coverage": 0.25, "count": 1, "risk": 0.0}
    assert curve[-1] == {"coverage": 1.0, "count": 4, "risk": 0.5}
    cells = {(r, s, c): float(r * s * c) for r in (0, 1) for s in (0, 1) for c in (0, 1)}
    assert third_order_alternating_sum(cells) == 1.0


def test_fit_directions_uses_harmful_minus_neutral_and_respects_rank(tmp_path):
    prompts = []
    arrays = {}
    for item_index in range(4):
        item_id = f"item-{item_index}"
        for key, factors, vector in (
            ("neutral", (0, 0, 0), np.array([0.0, 0.0, 0.0])),
            ("reward", (1, 0, 0), np.array([2.0, 0.0, 0.0])),
            ("social", (0, 1, 0), np.array([0.0, 3.0, 0.0])),
            ("certainty", (0, 0, 1), np.array([0.0, 0.0, 4.0])),
        ):
            record_id = f"{item_id}-{key}"
            prompts.append({"record_id": record_id, "base_item_id": item_id, "split": "discovery", "control_type": "factorial",
                            "reward_lure": factors[0], "social_lure": factors[1], "certainty_pressure": factors[2]})
            variations = {
                "neutral": [np.zeros(3)] * 4,
                "reward": [np.array([0, 0.1, 0]), np.array([0, -0.1, 0]), np.array([0, 0, 0.1]), np.array([0, 0, -0.1])],
                "social": [np.array([0.1, 0, 0]), np.array([-0.1, 0, 0]), np.array([0, 0, 0.1]), np.array([0, 0, -0.1])],
                "certainty": [np.array([0.1, 0, 0]), np.array([-0.1, 0, 0]), np.array([0, 0.1, 0]), np.array([0, -0.1, 0])],
            }
            arrays[f"{record_id}__layer_2"] = vector + variations[key][item_index] + item_index * 0.01
    prompts_path = tmp_path / "prompts.jsonl"
    activations_path = tmp_path / "activations.npz"
    outputs_path = tmp_path / "outputs.jsonl"
    output_path = tmp_path / "directions.npz"
    atomic_write_jsonl(prompts_path, prompts)
    np.savez_compressed(activations_path, **arrays)
    manifest_path = _write_run_fixture(
        outputs_path,
        [{"record_id": row["record_id"], "method": "baseline", "run_key": "baseline"} for row in prompts],
        prompts_path,
        activation_path=activations_path,
    )
    metadata = fit_directions(prompts_path, activations_path, manifest_path, output_path, (1, 2, 3))
    archive = np.load(output_path)
    assert np.allclose(archive["reward__layer_2__caa"], [1, 0, 0])
    assert archive["reward__layer_2__basis"].shape == (3, 3)
    assert metadata["factors"]["reward"]["2"]["available_rank"] == 3
    assert metadata["factors"]["reward"]["2"]["hidden_size"] == 3
    assert audit_direction_manifest(output_path.with_suffix(".json"))["status"] == "current_eligible"
    intervention = Intervention(output_path, "caa", "reward", 2, 0.5)
    assert np.allclose(intervention.delta(np.ones(3)), [-0.5, 0, 0])
    assert np.isclose(np.linalg.norm(intervention.delta(np.ones(3))), 0.5)
    low_rank = Intervention(output_path, "low_rank", "reward", 2, 0.5, rank=3)
    assert np.isclose(np.linalg.norm(low_rank.delta(np.ones(3))), 0.5)
    try:
        intervention.delta(np.ones(4))
    except ValueError as exc:
        assert "hidden size" in str(exc)
    else:
        raise AssertionError("mismatched activation dimension was accepted")


def test_stable_hash_is_order_invariant_for_mappings():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_report_keeps_run_keys_separate_and_requires_selective_steering(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    config_path = _config(tmp_path / "config.yaml")
    bundle = tmp_path / "bundle"
    build_pilot(config_path, lures, bundle, source)
    prompts = [json.loads(line) for line in (bundle / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluator_by_id = {row["record_id"]: row for row in read_jsonl(bundle / "evaluator.jsonl")}

    def outputs(run_key: str, method: str, improve_reward: bool) -> list[dict]:
        rows = []
        for prompt in prompts:
            harmful_reward = prompt["control_type"] == "factorial" and prompt["reward_lure"] and not prompt["social_lure"] and not prompt["certainty_pressure"]
            lure_probability = 0.7 if harmful_reward and not improve_reward else 0.1
            gold_probability = 0.2 if harmful_reward and not improve_reward else 0.8
            label = evaluator_by_id[prompt["record_id"]]
            probabilities = [0.1, 0.1, 0.1]
            probabilities[int(label["gold_index"])] = gold_probability
            probabilities[int(label["lure_index"])] = lure_probability
            probabilities = (np.asarray(probabilities) / sum(probabilities)).tolist()
            rows.append({
                **{key: prompt[key] for key in ("record_id", "base_item_id", "split", "category", "template_family", "reward_lure", "social_lure", "certainty_pressure", "control_type")},
                "method": method, "run_key": run_key, "steering_factor": "reward" if method != "baseline" else None,
                "option_probabilities": probabilities, "generated_text": "Answer: A", "parsed_answer": "A",
                "self_reported_confidence": 0.8, "parse_ok": True,
                "oracle_intervention_applied": bool(
                    method != "baseline"
                    and (harmful_reward or prompt["control_type"] == "valid_reward")
                ),
            })
        return rows

    baseline_path = tmp_path / "baseline.jsonl"
    steered_path = tmp_path / "steered.jsonl"
    atomic_write_jsonl(baseline_path, outputs("baseline", "baseline", False))
    atomic_write_jsonl(steered_path, outputs("caa__reward__layer_2__strength_1__rank_1", "caa", True))
    _write_run_fixture(baseline_path, outputs("baseline", "baseline", False), bundle / "prompts.jsonl")
    _write_run_fixture(
        steered_path, outputs("caa__reward__layer_2__strength_1__rank_1", "caa", True),
        bundle / "prompts.jsonl", direction_sha256="direction-fixture",
    )
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    selection_path = _write_selection_fixture(
        tmp_path / "selection.json", config_path, bundle / "evaluator.jsonl",
        bundle_manifest["artifacts"]["prompts"]["sha256"], ["caa__reward__layer_2__strength_1__rank_1"],
    )
    report = generate_report(
        config_path, bundle / "evaluator.jsonl", [baseline_path, steered_path], tmp_path / "report", selection_path,
    )
    assert set(report["behavior"]) == {"baseline", "caa__reward__layer_2__strength_1__rank_1"}
    gate = report["steering_gates"]["caa__reward__layer_2__strength_1__rank_1"]
    assert gate["passes"], gate
    assert not report["go_no_go"]["formal_three_factor_allowed"]
    assert report["decision"] == "do_not_expand_three_factor"

    invalid_key = "caa__reward__layer_2__strength_1__rank_1"
    invalid_rows = outputs(invalid_key, "caa", True)
    for row in invalid_rows:
        if row["control_type"] == "valid_reward":
            row["oracle_intervention_applied"] = False
    invalid_path = tmp_path / "invalid-steered.jsonl"
    _write_run_fixture(invalid_path, invalid_rows, bundle / "prompts.jsonl", direction_sha256="direction-fixture")
    invalid_selection = _write_selection_fixture(
        tmp_path / "invalid-selection.json", config_path, bundle / "evaluator.jsonl",
        bundle_manifest["artifacts"]["prompts"]["sha256"], [invalid_key],
    )
    invalid_report = generate_report(
        config_path, bundle / "evaluator.jsonl", [baseline_path, invalid_path], tmp_path / "invalid-report", invalid_selection,
    )
    invalid_gate = invalid_report["steering_gates"][invalid_key]
    assert invalid_gate["valid_control_intervention_verified"] is False
    assert invalid_gate["passes"] is False

    # A positive point estimate is insufficient when the paired interval touches zero.
    weak_rows = outputs("caa__reward__layer_2__strength_1__rank_1", "caa", False)
    strong_rows = outputs("caa__reward__layer_2__strength_1__rank_1", "caa", True)
    improved_item = next(row["base_item_id"] for row in weak_rows if row["split"] == "pilot_holdout")
    strong_by_id = {row["record_id"]: row for row in strong_rows}
    weak_rows = [strong_by_id[row["record_id"]] if row["base_item_id"] == improved_item else row for row in weak_rows]
    weak_path = tmp_path / "weak-steered.jsonl"
    _write_run_fixture(weak_path, weak_rows, bundle / "prompts.jsonl", direction_sha256="direction-fixture")
    weak_report = generate_report(
        config_path, bundle / "evaluator.jsonl", [baseline_path, weak_path], tmp_path / "weak-report", selection_path,
    )
    weak_gate = weak_report["steering_gates"]["caa__reward__layer_2__strength_1__rank_1"]
    assert weak_gate["paired_family_adjusted"]["harmful_effect_improvement"]["estimate"] > 0
    assert weak_gate["paired_family_adjusted"]["harmful_effect_improvement"]["lower"] <= 0
    assert weak_gate["passes"] is False

    # Prompt-only can be an effective comparator, but cannot satisfy the internal-steering gate.
    prompt_key = "prompt_only__reward"
    prompt_path = tmp_path / "prompt-only.jsonl"
    _write_run_fixture(prompt_path, outputs(prompt_key, "prompt_only", True), bundle / "prompts.jsonl")
    prompt_report = generate_report(
        config_path, bundle / "evaluator.jsonl", [baseline_path, prompt_path], tmp_path / "prompt-report",
    )
    assert prompt_report["steering_gates"][prompt_key]["passes"] is True
    assert prompt_report["go_no_go"]["selective_steering"] is False


def test_report_rejects_truncated_and_unselected_linear_runs(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    config_path = _config(tmp_path / "config.yaml")
    bundle = tmp_path / "bundle"
    build_pilot(config_path, lures, bundle, source)
    prompts = read_jsonl(bundle / "prompts.jsonl")

    def rows(run_key: str, method: str, source_rows: list[dict]) -> list[dict]:
        return [{
            **{key: row[key] for key in ("record_id", "base_item_id", "split", "category", "template_family", "reward_lure", "social_lure", "certainty_pressure", "control_type")},
            "method": method, "run_key": run_key, "steering_factor": "reward" if method != "baseline" else None,
            "option_probabilities": [0.8, 0.1, 0.1], "parsed_answer": "A", "parse_ok": True,
            "oracle_intervention_applied": False,
        } for row in source_rows]

    baseline_path = tmp_path / "baseline.jsonl"
    truncated_path = tmp_path / "truncated.jsonl"
    _write_run_fixture(baseline_path, rows("baseline", "baseline", prompts), bundle / "prompts.jsonl")
    _write_run_fixture(truncated_path, rows("caa__reward__layer_2__strength_1__rank_1", "caa", prompts[:-1]), bundle / "prompts.jsonl", direction_sha256="direction-fixture")
    try:
        generate_report(config_path, bundle / "evaluator.jsonl", [baseline_path, truncated_path], tmp_path / "report")
    except ValueError as exc:
        assert "current complete audited run" in str(exc) or "exactly cover" in str(exc)
    else:
        raise AssertionError("truncated run was accepted")


def test_discovery_selection_rejects_holdout_and_freezes_complete_grid(tmp_path):
    source, lures = _source_and_lures(tmp_path)
    config_path = _config(tmp_path / "config.yaml")
    bundle = tmp_path / "bundle"
    build_pilot(config_path, lures, bundle, source)
    prompts = read_jsonl(bundle / "prompts.jsonl")
    discovery = [row for row in prompts if row["split"] == "discovery"]

    def output_rows(run_key: str, method: str, factor: str | None, source_rows: list[dict]) -> list[dict]:
        values = []
        for row in source_rows:
            apply = factor is not None and (
                row["control_type"] == {"reward": "valid_reward", "social": "valid_social", "certainty": "calibration"}.get(factor)
                or (row["control_type"] == "factorial" and sum((row["reward_lure"], row["social_lure"], row["certainty_pressure"])) == 1
                    and {"reward": row["reward_lure"], "social": row["social_lure"], "certainty": row["certainty_pressure"]}[factor])
            )
            values.append({
                **{key: row[key] for key in ("record_id", "base_item_id", "split", "category", "template_family", "reward_lure", "social_lure", "certainty_pressure", "control_type")},
                "method": method, "run_key": run_key, "steering_factor": factor,
                "option_probabilities": [0.8, 0.1, 0.1], "parsed_answer": "A", "parse_ok": True,
                "oracle_intervention_applied": bool(apply),
            })
        return values

    baseline_path = tmp_path / "baseline-discovery.jsonl"
    _write_run_fixture(
        baseline_path, output_rows("baseline", "baseline", None, discovery), bundle / "prompts.jsonl",
        requested_split="discovery",
    )
    candidates = []
    for method in ("caa", "low_rank"):
        for factor in ("reward", "social", "certainty"):
            run_key = f"{method}__{factor}__layer_2__strength_1__rank_1"
            path = tmp_path / f"{method}-{factor}.jsonl"
            _write_run_fixture(
                path, output_rows(run_key, method, factor, discovery), bundle / "prompts.jsonl",
                requested_split="discovery", direction_sha256="direction-fixture",
            )
            candidates.append(path)
    selection = select_discovery_runs(
        config_path, bundle / "evaluator.jsonl", baseline_path, candidates, tmp_path / "selection.json",
    )
    assert selection["status"] == "frozen_before_holdout"
    assert selection["selected"]["caa"]["reward"]["run_key"].startswith("caa__reward")

    leaked_path = tmp_path / "leaked.jsonl"
    leaked_rows = output_rows("caa__reward__layer_2__strength_1__rank_1", "caa", "reward", prompts)
    _write_run_fixture(leaked_path, leaked_rows, bundle / "prompts.jsonl", requested_split="all", direction_sha256="direction-fixture")
    try:
        select_discovery_runs(config_path, bundle / "evaluator.jsonl", baseline_path, [leaked_path, *candidates[1:]], tmp_path / "bad.json")
    except ValueError as exc:
        assert "audited discovery run" in str(exc) or "non-discovery" in str(exc)
    else:
        raise AssertionError("holdout-bearing selection input was accepted")

    metadata_path = tmp_path / "direction-fixture.json"
    metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        select_discovery_runs(
            config_path, bundle / "evaluator.jsonl", baseline_path, candidates, tmp_path / "tampered.json",
        )
    except ValueError as exc:
        assert "metadata is missing or changed" in str(exc)
    else:
        raise AssertionError("tampered direction metadata was accepted")


def test_annotation_packets_hide_gold_and_adjudication_maps_independent_display_orders(tmp_path):
    source, _ = _source_and_lures(tmp_path, per_category=2)
    config_path = _config(tmp_path / "config.yaml", per_category=2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["annotation_candidates_per_category"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    packet_dir = tmp_path / "packets"
    manifest = prepare_annotation_packets(config_path, packet_dir, ("annotator_a", "annotator_b", "annotator_c"), source)
    assert manifest["candidate_count"] == 12
    assert manifest["config"]["annotation_contract_sha256"]
    assert validate_annotation_packets(packet_dir)["config_file_matches_provenance"] is True

    # Model/report changes do not alter the frozen annotation task.
    config["model"]["documentation_only_marker"] = "changed-after-packet-freeze"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    validation = validate_annotation_packets(packet_dir)
    assert validation["status"] == "valid"
    assert validation["config_file_matches_provenance"] is False

    # Candidate selection and display order remain fail-closed.
    config["seed"] = 8
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        validate_annotation_packets(packet_dir)
    except ValueError as exc:
        assert "annotation-relevant config changed" in str(exc)
    else:
        raise AssertionError("annotation-relevant config drift was accepted")
    config["seed"] = 7
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for annotator in ("annotator_a", "annotator_b"):
        text = (packet_dir / f"packet-{annotator}.jsonl").read_text(encoding="utf-8")
        assert "answer_index" not in text
        assert "cot_content" not in text
        assert "gold_index" not in text

    mappings = json.loads((packet_dir / "mapping-withheld.json").read_text(encoding="utf-8"))
    by_annotator = {}
    for row in mappings:
        by_annotator.setdefault(row["annotator_id"], {})[row["question_id"]] = row
    annotations = {}
    for annotator in ("annotator_a", "annotator_b"):
        rows = []
        for question_id, mapping in sorted(by_annotator[annotator].items()):
            display_to_source = mapping["display_to_source_index"]
            source_rank = [1, 2, 0]
            ranked = [label for source_index in source_rank for label, value in display_to_source.items() if value == source_index]
            rows.append({"annotation_id": mapping["annotation_id"], "question_id": question_id,
                         "ranked_display_labels": ranked, "exclude": False, "exclusion_reason": ""})
        path = packet_dir / f"annotations-{annotator}.jsonl"
        atomic_write_jsonl(path, rows)
        annotations[annotator] = path
    lure_path = tmp_path / "adjudicated-lures.yaml"
    result = adjudicate_lures(packet_dir, packet_dir / "mapping-withheld.json", lure_path, final_count_per_category=2, frozen_at="2026-07-24T00:00:00+08:00")
    assert result["status"] == "ready"
    payload = yaml.safe_load(lure_path.read_text(encoding="utf-8"))
    assert len(payload["items"]) == 12
    assert payload["selector"]["uses_target_model_outputs"] is False
    assert payload["selector"]["agreement_status"] == "adjudicated"
    assert len(payload["selector"]["primary_annotation_sha256"]) == 2
    assert result["provenance"]["mapping_sha256"] == payload["selector"]["mapping_sha256"]
    assert Path(result["report_path"]).is_file()
    loaded_lures, audit = load_lures(lure_path)
    assert len(loaded_lures) == 12
    assert audit["selector"]["annotation_manifest_sha256"]


def test_legacy_annotation_manifest_migrates_only_before_annotation_starts(tmp_path):
    source, _ = _source_and_lures(tmp_path, per_category=2)
    config_path = _config(tmp_path / "config.yaml", per_category=2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["annotation_candidates_per_category"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    packet_dir = tmp_path / "packets"
    prepare_annotation_packets(config_path, packet_dir, ("annotator_a", "annotator_b"), source)
    manifest_path = packet_dir / "annotation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"].pop("annotation_contract_version")
    manifest["config"].pop("annotation_contract_sha256")
    manifest.pop("manifest_content_sha256")
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    atomic_write_json(manifest_path, manifest)

    result = migrate_unstarted_annotation_contract(packet_dir, config_path)
    assert result["status"] == "migrated"
    assert validate_annotation_packets(packet_dir)["annotation_contract_sha256"]

    atomic_write_jsonl(packet_dir / "annotations-annotator_a.jsonl", [{"started": True}])
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current["config"].pop("annotation_contract_version")
    current["config"].pop("annotation_contract_sha256")
    current.pop("manifest_content_sha256")
    current["manifest_content_sha256"] = stable_hash(current)
    atomic_write_json(manifest_path, current)
    try:
        migrate_unstarted_annotation_contract(packet_dir, config_path)
    except ValueError as exc:
        assert "before human annotation starts" in str(exc)
    else:
        raise AssertionError("started annotation bundle was migrated")


def test_interactive_annotation_session_is_blinded_validated_and_resumable(tmp_path):
    packet_path = tmp_path / "packet-annotator_a.jsonl"
    output_path = tmp_path / "annotations-annotator_a.jsonl"
    packets = [
        {
            "annotation_id": "a-1", "question_id": "q-1", "question": "First?",
            "display_options": [
                {"display_label": "A", "text": "one"},
                {"display_label": "B", "text": "two"},
                {"display_label": "C", "text": "three"},
            ],
            "instructions": "blind",
        },
        {
            "annotation_id": "a-2", "question_id": "q-2", "question": "Second?",
            "display_options": [
                {"display_label": "A", "text": "alpha"},
                {"display_label": "B", "text": "beta"},
            ],
            "instructions": "blind",
        },
    ]
    atomic_write_jsonl(packet_path, packets)
    messages = iter(["A A C", "C, A, B"])
    first = run_annotation_session(
        packet_path, output_path, input_fn=lambda _prompt: next(messages), output_fn=lambda _value: None,
        max_items=1,
    )
    assert first["status"] == "needs_author_input"
    assert first["completed"] == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["ranked_display_labels"] == ["C", "A", "B"]
    assert "gold" not in output_path.read_text(encoding="utf-8")

    second = run_annotation_session(
        packet_path, output_path, input_fn=lambda _prompt: "exclude ambiguous wording",
        output_fn=lambda _value: None,
    )
    assert second["status"] == "complete"
    assert second["completed"] == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["exclude"] is True
    assert rows[1]["exclusion_reason"] == "ambiguous wording"
    try:
        parse_ranking("A A", ["A", "B"])
    except ValueError as exc:
        assert "every label exactly once" in str(exc)
    else:
        raise AssertionError("duplicate ranking was accepted")


def test_primary_annotation_delivery_excludes_withheld_and_model_artifacts(tmp_path):
    source, _ = _source_and_lures(tmp_path, per_category=2)
    config_path = _config(tmp_path / "config.yaml", per_category=2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["annotation_candidates_per_category"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    packet_dir = tmp_path / "packets"
    prepare_annotation_packets(
        config_path, packet_dir, ("annotator_a", "annotator_b", "annotator_c"), source,
    )
    delivery_dir = tmp_path / "delivery"
    result = prepare_primary_annotation_delivery(packet_dir, delivery_dir)
    assert result["status"] == "ready_for_independent_humans"
    validation = validate_primary_annotation_delivery(delivery_dir)
    assert validation["status"] == "valid"
    assert validation["contains_withheld_mapping"] is False
    assert {path.name for path in (delivery_dir / "annotator_a").iterdir()} == {"packet.jsonl", "INSTRUCTIONS.md"}
    assert not any("mapping" in path.name.lower() for path in delivery_dir.rglob("*"))

    (delivery_dir / "annotator_a" / "mapping-withheld.json").write_text("{}", encoding="utf-8")
    try:
        validate_primary_annotation_delivery(delivery_dir)
    except ValueError as exc:
        assert "forbidden artifacts" in str(exc)
    else:
        raise AssertionError("delivery containing a withheld mapping was accepted")
    (delivery_dir / "annotator_a" / "mapping-withheld.json").unlink()

    for annotator_id in ("annotator_a", "annotator_b"):
        packets = read_jsonl(delivery_dir / annotator_id / "packet.jsonl")
        annotations = [{
            "annotation_id": packet["annotation_id"], "question_id": packet["question_id"],
            "ranked_display_labels": [option["display_label"] for option in packet["display_options"]],
            "exclude": False, "exclusion_reason": "",
        } for packet in packets]
        atomic_write_jsonl(delivery_dir / annotator_id / "annotations.jsonl", annotations)
    collected = collect_primary_annotation_delivery(delivery_dir, packet_dir)
    assert collected["status"] == "ready_for_tiebreak"
    assert (packet_dir / "annotations-annotator_a.jsonl").is_file()
    assert (packet_dir / "annotations-annotator_b.jsonl").is_file()


def test_tiebreak_packet_contains_only_primary_top_wrong_choice_disagreements(tmp_path):
    source, _ = _source_and_lures(tmp_path, per_category=2)
    config_path = _config(tmp_path / "config.yaml", per_category=2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["annotation_candidates_per_category"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    packet_dir = tmp_path / "packets"
    prepare_annotation_packets(
        config_path, packet_dir, ("annotator_a", "annotator_b", "annotator_c"), source,
    )
    mappings = json.loads((packet_dir / "mapping-withheld.json").read_text(encoding="utf-8"))
    by_annotator = {}
    for row in mappings:
        by_annotator.setdefault(row["annotator_id"], {})[row["question_id"]] = row
    question_ids = sorted(by_annotator["annotator_a"])
    disagreement_id = question_ids[0]
    for annotator in ("annotator_a", "annotator_b"):
        rows = []
        for question_id in question_ids:
            mapping = by_annotator[annotator][question_id]
            # Both primaries rank gold first. The reduced packet must still detect
            # disagreement about the highest-ranked wrong option.
            source_rank = [0, 1, 2]
            if question_id == disagreement_id and annotator == "annotator_b":
                source_rank = [0, 2, 1]
            ranked = [
                label
                for source_index in source_rank
                for label, value in mapping["display_to_source_index"].items()
                if value == source_index
            ]
            rows.append({
                "annotation_id": mapping["annotation_id"], "question_id": question_id,
                "ranked_display_labels": ranked, "exclude": False, "exclusion_reason": "",
            })
        atomic_write_jsonl(packet_dir / f"annotations-{annotator}.jsonl", rows)
    output = packet_dir / "packet-annotator_c-disagreements.jsonl"
    report = prepare_tiebreak_packet(packet_dir, output)
    assert report["status"] == "ready"
    assert report["counts"]["top_wrong_choice_disagreements"] == 1
    assert report["counts"]["primary_agreements"] == len(question_ids) - 1
    packet_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in packet_rows] == [disagreement_id]
    serialized = output.read_text(encoding="utf-8")
    assert all(marker not in serialized for marker in ("gold_index", "answer_index", "lure_index", "cot_content"))
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact"]["sha256"]
    assert manifest["target_model_outputs_used"] is False

    third_mapping = by_annotator["annotator_c"][disagreement_id]
    third_rank = [0, 2, 1]
    third_labels = [
        label
        for source_index in third_rank
        for label, value in third_mapping["display_to_source_index"].items()
        if value == source_index
    ]
    third_annotations = packet_dir / "annotations-annotator_c.jsonl"
    atomic_write_jsonl(third_annotations, [{
        "annotation_id": third_mapping["annotation_id"], "question_id": disagreement_id,
        "ranked_display_labels": third_labels, "exclude": False, "exclusion_reason": "",
    }])
    adjudicated = adjudicate_lures(
        packet_dir, packet_dir / "mapping-withheld.json", tmp_path / "lures.yaml",
        final_count_per_category=2, third_annotation_path=third_annotations,
        frozen_at="2026-07-24T01:00:00+08:00",
    )
    assert adjudicated["status"] == "ready"
    disagreement_decision = next(row for row in adjudicated["decisions"] if row["question_id"] == disagreement_id)
    assert disagreement_decision["lure_index"] == 2
    assert disagreement_decision["adjudication"] == "third_annotator"

    exclusion_id = question_ids[1]
    b_path = packet_dir / "annotations-annotator_b.jsonl"
    b_rows = [json.loads(line) for line in b_path.read_text(encoding="utf-8").splitlines()]
    for row in b_rows:
        if row["question_id"] == exclusion_id:
            row["exclude"] = True
            row["exclusion_reason"] = "ambiguous wording"
    atomic_write_jsonl(b_path, b_rows)
    unresolved_output = packet_dir / "packet-annotator_c-with-exclusion-disagreement.jsonl"
    unresolved = prepare_tiebreak_packet(packet_dir, unresolved_output)
    assert unresolved["status"] == "needs_author_input"
    assert unresolved["counts"]["exclusion_disagreements"] == 1
    assert unresolved["exclusion_disagreement_question_ids"] == [exclusion_id]
