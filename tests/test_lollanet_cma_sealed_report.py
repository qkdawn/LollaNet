from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research.phase1.lollanet_cma.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_hash,
)
from research.phase1.lollanet_cma.sealed_eval import report as report_module
from research.phase1.lollanet_cma.sealed_eval.report import (
    assemble_report,
    common_parse_success_delta,
    query_type_leave_one_out,
)


def _row(record_id: str, family: str, query: str, correct: bool, parse_ok: bool = True) -> dict:
    return {
        "record_id": record_id,
        "model_family": family,
        "query_type": query,
        "correct": correct,
        "parse_ok": parse_ok,
    }


def test_query_type_leave_one_out_requires_positive_distributed_gain():
    base = [
        _row("a1", "a", "q1", False), _row("a2", "a", "q2", False),
        _row("b1", "b", "q1", False), _row("b2", "b", "q2", False),
    ]
    oracle = [
        _row("a1", "a", "q1", True), _row("a2", "a", "q2", True),
        _row("b1", "b", "q1", True), _row("b2", "b", "q2", True),
    ]
    result = query_type_leave_one_out(oracle, base, ["a", "b"])
    assert result == {"q1": 1.0, "q2": 1.0}


def test_common_parse_success_excludes_format_only_changes_but_keeps_pairing():
    base = [
        _row("a1", "a", "q1", False), _row("a2", "a", "q2", False, False),
        _row("b1", "b", "q1", False), _row("b2", "b", "q2", False),
    ]
    oracle = [
        _row("a1", "a", "q1", True), _row("a2", "a", "q2", True),
        _row("b1", "b", "q1", True), _row("b2", "b", "q2", True),
    ]
    result = common_parse_success_delta(oracle, base, ["a", "b"])
    assert result["record_count"] == 3
    assert result["macro_family_accuracy_delta"] == 1.0


def test_leave_one_out_fails_when_omission_removes_a_family():
    left = [_row("a", "a", "qa", True), _row("b", "b", "qb", True)]
    right = [_row("a", "a", "qa", False), _row("b", "b", "qb", False)]
    with pytest.raises(ValueError, match="lacks family coverage"):
        query_type_leave_one_out(left, right, ["a", "b"])


def _write_hashed_json(path: Path, value: dict, hash_field: str) -> dict:
    value[hash_field] = stable_hash(value)
    atomic_write_json(path, value)
    return value


def _metric_row(
    condition: str, record_id: str, family: str, query: str, correct: bool,
) -> dict:
    row = {
        "condition_id": condition,
        "record_id": record_id,
        "model_family": family,
        "query_type": query,
        "rung": 1,
        "gold_answer": "yes",
        "correct": correct,
        "parse_ok": True,
        "parsed_answer": "yes" if correct else "no",
        "parse_rule": "final_answer_marker",
        "gold_answer_margin": 1.0 if correct else -1.0,
        "brier_score": 0.1 if correct else 0.9,
        "predicted_answer_probability": 0.9,
        "output_tokens": 2,
        "latency_seconds": 0.1,
        "peak_gpu_memory_bytes": 10,
        "token_level_kl": 0.1 if condition not in {"base", "causal_cot"} else None,
        "activation_sha256": None,
    }
    row["record_content_sha256"] = stable_hash(row)
    return row


def _control_row(condition: str, record_id: str, correct: bool) -> dict:
    row = {
        "condition_id": condition,
        "record_id": record_id,
        "category": "math",
        "gold_answer": "A",
        "correct": correct,
        "parse_ok": True,
        "parsed_answer": "A" if correct else "B",
        "parse_rule": "last_nonempty_line",
    }
    row["record_content_sha256"] = stable_hash(row)
    return row


def test_report_assembles_three_splits_derived_routes_and_all_gates(tmp_path: Path, monkeypatch):
    code = {"sha256": "code"}
    lock = {"sha256": "lock"}
    environment = {"sha256": "environment"}
    monkeypatch.setattr(report_module, "runtime_code_identity", lambda _path: code)
    monkeypatch.setattr(report_module, "dependency_lock", lambda: lock)
    monkeypatch.setattr(report_module, "runtime_environment", lambda: environment)
    monkeypatch.setattr(report_module, "validate_released_view", lambda _path, split: {"split": split})
    monkeypatch.setattr(report_module, "_probe_results", lambda *_args, **_kwargs: {
        "conditions": {"base": {"accuracy": 0.5}, "oracle_correct": {"accuracy": 0.75}},
        "interpretation": "diagnostic_only",
    })

    families = ["a", "b", "c", "d"]
    family_conditions = [f"family_direction__{family}" for family in families]
    model_conditions = [
        "base", "causal_cot", "general_caa", *family_conditions, "equal_all",
        "random_norm_matched__draw_1", "random_norm_matched__draw_2",
        "random_norm_matched__draw_3",
    ]
    derived_conditions = ["oracle_correct", "oracle_wrong", "shuffled_mapping"]
    mapping_wrong = {"a": "d", "b": "a", "c": "b", "d": "c"}
    mapping_shuffle = {"a": "c", "b": "d", "c": "a", "d": "b"}
    records = [
        (f"{family}-{query}", family, query)
        for family in families for query in ("q1", "q2")
    ]
    release_views = {}
    raw_splits = {}
    for split in ("sealed_test", "ood_anticommonsense", "ood_nonsense"):
        view = tmp_path / f"view-{split}"
        view.mkdir()
        prompts_path = view / "prompts.jsonl"
        controller_path = view / "controller.jsonl"
        evaluator_path = view / "evaluator.jsonl"
        atomic_write_jsonl(prompts_path, [
            {"record_id": record_id, "prompt": record_id} for record_id, _, _ in records
        ])
        atomic_write_jsonl(controller_path, [
            {"record_id": record_id, "model_family": family, "query_type": query}
            for record_id, family, query in records
        ])
        atomic_write_jsonl(evaluator_path, [
            {"record_id": record_id, "answer": "yes"} for record_id, _, _ in records
        ])
        view_manifest = {
            "artifacts": {
                "prompts": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
                "controller": {"path": str(controller_path), "sha256": sha256_file(controller_path)},
                "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
            },
        }
        atomic_write_json(view / "manifest.json", view_manifest)
        release_views[split] = {
            "path": str(view), "manifest_sha256": sha256_file(view / "manifest.json"),
        }
        condition_summaries = {}
        condition_dir = tmp_path / "raw" / split
        condition_dir.mkdir(parents=True)
        for condition in model_conditions:
            if condition.startswith("family_direction__"):
                direction_family = condition.removeprefix("family_direction__")
                values = [
                    _metric_row(condition, record_id, family, query, family == direction_family)
                    for record_id, family, query in records
                ]
            elif condition == "base" or condition == "causal_cot" or condition.startswith("random"):
                values = [
                    _metric_row(condition, record_id, family, query, False)
                    for record_id, family, query in records
                ]
            else:
                values = [
                    _metric_row(condition, record_id, family, query, False)
                    for record_id, family, query in records
                ]
            path = condition_dir / f"{condition}.jsonl"
            atomic_write_jsonl(path, values)
            condition_summaries[condition] = {
                "artifacts": {"rows": {"path": str(path), "sha256": sha256_file(path)}},
            }
        raw_splits[split] = {
            "view_manifest_sha256": sha256_file(view / "manifest.json"),
            "conditions": condition_summaries,
        }

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control_prompts = control_dir / "prompts.jsonl"
    control_controller = control_dir / "controller.jsonl"
    control_evaluator = control_dir / "evaluator.jsonl"
    control_ids = ["control-1", "control-2", "control-3", "control-4"]
    atomic_write_jsonl(control_prompts, [{"record_id": value, "prompt": value} for value in control_ids])
    atomic_write_jsonl(control_controller, [
        {"record_id": value, "category": "math"} for value in control_ids
    ])
    atomic_write_jsonl(control_evaluator, [{"record_id": value, "answer": "A"} for value in control_ids])
    control_manifest_path = control_dir / "manifest.json"
    atomic_write_json(control_manifest_path, {
        "artifacts": {
            "prompts": {"path": str(control_prompts), "sha256": sha256_file(control_prompts)},
            "controller": {"path": str(control_controller), "sha256": sha256_file(control_controller)},
            "evaluator": {"path": str(control_evaluator), "sha256": sha256_file(control_evaluator)},
        },
    })
    control_conditions = ["base", *family_conditions]
    raw_control = {}
    for condition in control_conditions:
        path = tmp_path / "raw" / "control" / f"{condition}.jsonl"
        atomic_write_jsonl(path, [_control_row(condition, record_id, True) for record_id in control_ids])
        raw_control[condition] = {
            "artifact": {"path": str(path), "sha256": sha256_file(path)},
        }

    release_path = tmp_path / "release.json"
    _write_hashed_json(release_path, {
        "views": release_views,
    }, "evaluation_release_content_sha256")
    probe_path = tmp_path / "probe.json"
    _write_hashed_json(probe_path, {"type": "probe"}, "probe_content_sha256")
    preflight_path = tmp_path / "evaluation-preflight.json"
    _write_hashed_json(preflight_path, {
        "probe": {"path": str(probe_path)},
        "evaluation_release": {"path": str(release_path)},
    }, "sealed_evaluation_preflight_content_sha256")
    charter_path = tmp_path / "sealed.yaml"
    control_charter_path = tmp_path / "control.yaml"
    charter = {
        "prerequisites": {"general_equal_all_max_cosine_similarity": 0.999999},
        "data": {"views": {
            split: {"path": value["path"], "expected_records": 8, "expected_records_per_family": 2}
            for split, value in release_views.items()
        }},
        "families": families,
        "conditions": {
            "deterministic_model_runs": model_conditions[:8],
            "derived_from_family_direction_runs": derived_conditions,
            "random_primary": "random_norm_matched__draw_1",
            "random_sensitivity": ["random_norm_matched__draw_2", "random_norm_matched__draw_3"],
            "wrong_mapping": mapping_wrong,
            "shuffled_mapping_seed": 1,
            "shuffled_mapping": mapping_shuffle,
            "random_direction_seeds": {
                "random_norm_matched__draw_1": 1,
                "random_norm_matched__draw_2": 2,
                "random_norm_matched__draw_3": 3,
            },
        },
        "metrics": {"ece_bins": 2},
        "statistics": {
            "paired_bootstrap_samples": 100,
            "bootstrap_seed": 3,
            "faithfulness_bootstrap_seed_offset": 100,
            "confidence_level": 0.95,
        },
        "success_gates": {
            "registered_comparisons": [
                ["oracle_correct", "base"], ["oracle_correct", "causal_cot"],
                ["oracle_correct", "oracle_wrong"], ["oracle_correct", "shuffled_mapping"],
                ["oracle_correct", "random_norm_matched__draw_1"],
            ],
            "minimum_significant_diagonal_families": 3,
            "maximum_oracle_parse_failure_rate_increase": 0.01,
        },
        "readiness_blockers": {"unrelated_control_dataset": {
            "artifact": str(control_manifest_path),
            "artifact_sha256": sha256_file(control_manifest_path),
        }},
    }
    # The contract validates that the shuffled mapping is generated from its seed.
    from research.phase1.lollanet_cma.sealed_eval.conditions import frozen_derangement
    charter["conditions"]["shuffled_mapping"] = frozen_derangement(families, 1)
    if charter["conditions"]["shuffled_mapping"] == mapping_wrong:
        charter["conditions"]["shuffled_mapping_seed"] = 2
        charter["conditions"]["shuffled_mapping"] = frozen_derangement(families, 2)
    charter_path.write_text(yaml.safe_dump(charter), encoding="utf-8")
    control_charter = {
        "evaluation": {
            "conditions": control_conditions,
            "maximum_allowed_accuracy_loss": 0.02,
        },
    }
    control_charter_path.write_text(yaml.safe_dump(control_charter), encoding="utf-8")
    intent_path = tmp_path / "raw" / "run_intent.json"
    _write_hashed_json(intent_path, {
        "charter_sha256": sha256_file(charter_path),
        "control_charter_sha256": sha256_file(control_charter_path),
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "evaluation_release": {"path": str(release_path), "sha256": sha256_file(release_path)},
        "probe_manifest": {"path": str(probe_path), "sha256": sha256_file(probe_path)},
        "code": code, "dependency_lock": lock, "environment": environment,
    }, "intent_content_sha256")
    raw_manifest_path = tmp_path / "raw" / "run_manifest.json"
    _write_hashed_json(raw_manifest_path, {
        "type": "lollanet_cma_sealed_evaluation_raw_run_v1",
        "status": "complete_raw_conditions",
        "intent": {"path": str(intent_path), "sha256": sha256_file(intent_path)},
        "splits": raw_splits,
        "unrelated_control": {
            "manifest_sha256": sha256_file(control_manifest_path),
            "conditions": raw_control,
        },
        "code": code, "dependency_lock": lock, "environment": environment,
    }, "raw_run_content_sha256")
    result = assemble_report(
        charter_path, control_charter_path, raw_manifest_path, tmp_path / "report",
    )
    assert result["status"] == "phase2_eligible"
    assert result["all_success_gates_passed"] is True
    assert set(result["derived_artifacts"]["sealed_test"]) == set(derived_conditions)
