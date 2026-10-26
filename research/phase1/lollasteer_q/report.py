from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .common import atomic_write_json, read_jsonl, sha256_file, stable_hash
from .metrics import (
    cluster_bootstrap,
    enrich_outputs,
    expected_calibration_error,
    factorial_interaction,
    paired_factor_effect,
    risk_coverage_curve,
)
from .runtime_audit import audit_run_manifest


def _effect(
    rows: list[dict[str, Any]], factor: str, outcome: str, config: dict[str, Any], seed_offset: int,
    confidence_level: float | None = None,
) -> dict[str, Any]:
    return cluster_bootstrap(
        rows, lambda sample: paired_factor_effect(sample, factor, outcome),
        int(config["report"]["bootstrap_samples"]), int(config["seed"]) + seed_offset,
        float(config["report"]["confidence_level"] if confidence_level is None else confidence_level),
    )


def _paired_rows(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row["record_id"]): row for row in baseline}
    candidate_by_id = {str(row["record_id"]): row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("paired comparison requires identical record IDs")
    result = []
    for record_id in sorted(baseline_by_id):
        baseline_row = baseline_by_id[record_id]
        candidate_row = candidate_by_id[record_id]
        if baseline_row["base_item_id"] != candidate_row["base_item_id"]:
            raise ValueError(f"paired comparison base item mismatch: {record_id}")
        result.append({
            "base_item_id": baseline_row["base_item_id"],
            "baseline": baseline_row, "candidate": candidate_row,
        })
    return result


def _paired_run_comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], factor: str,
    config: dict[str, Any], seed_offset: int, family_tests: int,
) -> dict[str, Any]:
    pairs = _paired_rows(baseline, candidate)
    outcome = "selected_lure" if factor != "certainty" else "logit_brier"
    control_type = {"reward": "valid_reward", "social": "valid_social", "certainty": "calibration"}[factor]
    family_alpha = float(config["report"].get("gate_family_alpha", 0.05))
    confidence_level = 1.0 - family_alpha / max(1, int(family_tests))

    def sides(sample: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [{**row["baseline"], "base_item_id": row["base_item_id"]} for row in sample],
            [{**row["candidate"], "base_item_id": row["base_item_id"]} for row in sample],
        )

    def improvement(sample: list[dict[str, Any]]) -> float:
        left, right = sides(sample)
        return paired_factor_effect(left, factor, outcome) - paired_factor_effect(right, factor, outcome)

    def accuracy_loss(sample: list[dict[str, Any]], kind: str) -> float:
        left, right = sides(sample)
        if kind == "control":
            left = [row for row in left if row["control_type"] == control_type]
            right = [row for row in right if row["control_type"] == control_type]
        else:
            left = [row for row in left if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
            right = [row for row in right if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
        if not left or len(left) != len(right):
            raise ValueError(f"paired comparison lacks {kind} rows")
        return float(np.mean([row["correct"] for row in left]) - np.mean([row["correct"] for row in right]))

    samples = int(config["report"]["bootstrap_samples"])
    seed = int(config["seed"]) + seed_offset
    return {
        "confidence_level_family_adjusted": confidence_level,
        "harmful_effect_improvement": cluster_bootstrap(pairs, improvement, samples, seed, confidence_level),
        "valid_control_accuracy_loss": cluster_bootstrap(
            pairs, lambda sample: accuracy_loss(sample, "control"), samples, seed + 1, confidence_level,
        ),
        "neutral_accuracy_loss": cluster_bootstrap(
            pairs, lambda sample: accuracy_loss(sample, "neutral"), samples, seed + 2, confidence_level,
        ),
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [float(row["max_option_probability"]) for row in rows]
    correctness = [bool(row["correct"]) for row in rows]
    return {
        "n": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
        "lure_rate": float(np.mean([row["selected_lure"] for row in rows])) if rows else None,
        "mean_gold_lure_log_odds": float(np.mean([row["gold_lure_log_odds"] for row in rows])) if rows else None,
        "mean_logit_brier": float(np.mean([row["logit_brier"] for row in rows])) if rows else None,
        "mean_answer_cross_entropy": float(np.mean([row["answer_cross_entropy"] for row in rows])) if rows else None,
        "calibration_gap": float(np.mean(confidences) - np.mean(correctness)) if rows else None,
        "expected_calibration_error_10_bin": expected_calibration_error(confidences, correctness, bins=10) if rows else None,
        "high_confidence_error_rate_0_8": float(np.mean([row["high_confidence_error"] for row in rows])) if rows else None,
        "risk_coverage": risk_coverage_curve(confidences, correctness),
        "parse_rate": float(np.mean([row["parse_ok"] for row in rows])) if rows else None,
    }


def _run_manifest_path(output_path: Path) -> Path:
    canonical = output_path.parent / "run_manifest.json"
    sidecar = output_path.with_suffix(".manifest.json")
    if output_path.name == "outputs.jsonl" and canonical.is_file():
        return canonical
    if sidecar.is_file():
        return sidecar
    raise ValueError(f"missing run manifest for {output_path}")


def _load_verified_runs(
    evaluator_path: Path, run_outputs: list[Path], selection_path: Path | None, config_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import json

    bundle_manifest_path = evaluator_path.parent / "manifest.json"
    if not bundle_manifest_path.is_file():
        raise ValueError("evaluator must belong to a bundle with manifest.json")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if bundle_manifest.get("config", {}).get("sha256") != sha256_file(config_path):
        raise ValueError("report config does not match the frozen pilot bundle")
    artifacts = bundle_manifest.get("artifacts", {})
    if artifacts.get("evaluator", {}).get("sha256") != sha256_file(evaluator_path):
        raise ValueError("bundle manifest evaluator hash mismatch")
    prompt_sha256 = artifacts.get("prompts", {}).get("sha256")
    if not prompt_sha256:
        raise ValueError("bundle manifest lacks prompt artifact hash")
    evaluator = read_jsonl(evaluator_path)
    evaluator_ids = [str(row["record_id"]) for row in evaluator]
    if len(evaluator_ids) != len(set(evaluator_ids)):
        raise ValueError("evaluator contains duplicate record_id")
    expected_ids = set(evaluator_ids)
    all_rows: list[dict[str, Any]] = []
    run_sources: list[dict[str, Any]] = []
    seen_run_keys: set[str] = set()
    compatibility: tuple[Any, ...] | None = None
    selection = None
    selected_linear: dict[str, dict[str, Any]] = {}
    if selection_path is not None:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        content_hash = selection.pop("selection_content_sha256", None)
        if content_hash != stable_hash(selection):
            raise ValueError("selection manifest content hash mismatch")
        selection["selection_content_sha256"] = content_hash
        if selection.get("status") != "frozen_before_holdout":
            raise ValueError("selection manifest is not frozen before holdout")
        if selection.get("config_sha256") != sha256_file(config_path):
            raise ValueError("selection manifest used a different config")
        if selection.get("evaluator_sha256") != sha256_file(evaluator_path):
            raise ValueError("selection manifest used a different evaluator")
        if selection.get("prompt_bundle_sha256") != prompt_sha256:
            raise ValueError("selection manifest used a different prompt bundle")
        for method, factors in selection.get("selected", {}).items():
            for factor, selected in factors.items():
                selected_linear[str(selected["run_key"])] = {"method": method, "factor": factor, **selected}
    for output_path in run_outputs:
        manifest_path = _run_manifest_path(output_path)
        run_audit = audit_run_manifest(manifest_path)
        if run_audit.get("status") != "current_complete" or not run_audit.get("formal_report_eligible"):
            raise ValueError(f"formal report requires a current complete audited run: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = read_jsonl(output_path)
        if manifest.get("requested_split") != "all":
            raise ValueError(f"formal report requires complete all-split runs: {manifest_path}")
        ids = [str(row.get("record_id")) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"run output contains duplicate record_id: {output_path}")
        if set(ids) != expected_ids or len(ids) != len(evaluator_ids):
            raise ValueError(f"run output does not exactly cover evaluator records: {output_path}")
        if int(manifest.get("record_count", -1)) != len(rows):
            raise ValueError(f"run manifest record_count mismatch: {manifest_path}")
        if manifest.get("inputs", {}).get("sha256") != prompt_sha256:
            raise ValueError(f"run used a different prompt bundle: {manifest_path}")
        if manifest.get("artifacts", {}).get("outputs", {}).get("sha256") != sha256_file(output_path):
            raise ValueError(f"run output hash mismatch: {manifest_path}")
        row_run_keys = {str(row.get("run_key")) for row in rows}
        row_methods = {str(row.get("method")) for row in rows}
        if len(row_run_keys) != 1 or len(row_methods) != 1:
            raise ValueError(f"run output mixes run keys or methods: {output_path}")
        run_key = next(iter(row_run_keys))
        method = next(iter(row_methods))
        if run_key != str(manifest.get("run_key")) or method != str(manifest.get("method")):
            raise ValueError(f"run rows disagree with manifest identity: {output_path}")
        if run_key in seen_run_keys:
            raise ValueError(f"duplicate run_key supplied to report: {run_key}")
        seen_run_keys.add(run_key)
        if method in {"caa", "low_rank"}:
            if selection is None:
                raise ValueError("linear steering report requires a frozen discovery selection manifest")
            selected = selected_linear.get(run_key)
            if selected is None:
                raise ValueError(f"linear run was not selected on discovery: {run_key}")
            actual = {
                "method": method, "factor": manifest.get("factor"), "layer": manifest.get("layer"),
                "strength": manifest.get("strength"), "rank": manifest.get("rank"),
                "direction_bundle_sha256": manifest.get("direction_bundle", {}).get("sha256"),
                "direction_metadata_sha256": manifest.get("direction_bundle", {}).get("metadata_sha256"),
            }
            expected = {key: selected[key] for key in actual}
            if actual != expected:
                raise ValueError(f"linear run does not match frozen discovery selection: {run_key}")
            direction_source = manifest.get("direction_bundle", {})
            direction_path = Path(str(direction_source.get("path", "")))
            metadata_path = Path(str(direction_source.get("metadata_path", "")))
            if not direction_path.is_file() or sha256_file(direction_path) != direction_source.get("sha256"):
                raise ValueError(f"linear run direction artifact is missing or changed: {run_key}")
            if not metadata_path.is_file() or sha256_file(metadata_path) != direction_source.get("metadata_sha256"):
                raise ValueError(f"linear run direction metadata is missing or changed: {run_key}")
            canonical_run_key = (
                f"{method}__{manifest.get('factor')}__layer_{manifest.get('layer')}"
                f"__strength_{float(manifest.get('strength')):g}__rank_{manifest.get('rank')}"
            )
            if run_key != canonical_run_key:
                raise ValueError(f"linear run_key does not encode its manifest parameters: {run_key}")
            if any(
                row.get("steering_factor") != manifest.get("factor")
                or row.get("steering_layer") != manifest.get("layer")
                or float(row.get("steering_strength")) != float(manifest.get("strength"))
                or int(row.get("steering_rank")) != int(manifest.get("rank"))
                for row in rows
            ):
                raise ValueError(f"linear run rows disagree with manifest parameters: {run_key}")
        elif method == "prompt_only":
            factor = str(manifest.get("factor"))
            if factor not in {"reward", "social", "certainty"} or run_key != f"prompt_only__{factor}":
                raise ValueError(f"invalid prompt-only run identity: {run_key}")
            if any(
                row.get("steering_factor") != factor
                or row.get("steering_layer") is not None
                or row.get("steering_strength") is not None
                or row.get("steering_rank") is not None
                for row in rows
            ):
                raise ValueError(f"prompt-only rows disagree with manifest parameters: {run_key}")
        current_compatibility = (
            manifest.get("checkpoint_actual"), manifest.get("checkpoint_revision"),
            manifest.get("chat_template_sha256"), manifest.get("dtype"),
            manifest.get("dependency_lock", {}).get("sha256"),
            manifest.get("runtime_code", {}).get("sha256"),
            manifest.get("runtime_environment_sha256"),
            tuple(manifest.get("layer_indices", [])),
        )
        if compatibility is None:
            compatibility = current_compatibility
        elif current_compatibility != compatibility:
            raise ValueError(f"incompatible model/runtime manifest: {manifest_path}")
        all_rows.extend(enrich_outputs(rows, evaluator))
        run_sources.append({
            "output_path": str(output_path), "output_sha256": sha256_file(output_path),
            "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "run_key": run_key,
        })
    if "baseline" not in seen_run_keys:
        raise ValueError("report requires exactly one complete baseline run")
    missing_selected = sorted(set(selected_linear) - seen_run_keys)
    if missing_selected:
        raise ValueError(f"report omits frozen selected linear runs: {missing_selected}")
    sources = {
        "bundle_manifest_path": str(bundle_manifest_path),
        "bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "run_outputs": run_sources,
    }
    if selection_path is not None:
        sources["selection_manifest_path"] = str(selection_path)
        sources["selection_manifest_sha256"] = sha256_file(selection_path)
    return evaluator, all_rows, sources


def generate_report(
    config_path: Path, evaluator_path: Path, run_outputs: list[Path], output_dir: Path,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluator, all_rows, verified_sources = _load_verified_runs(evaluator_path, run_outputs, selection_path, config_path)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_method[str(row.get("run_key", row["method"]))].append(row)
    behavior: dict[str, Any] = {}
    for method, method_rows in sorted(by_method.items()):
        behavior[method] = {"overall": _method_summary(method_rows), "splits": {}}
        for split in ("discovery", "pilot_holdout"):
            subset = [row for row in method_rows if row["split"] == split]
            if not subset:
                continue
            behavior[method]["splits"][split] = {
                "summary": _method_summary(subset),
                "reward_lure_rate_effect": _effect(subset, "reward", "selected_lure", config, 1),
                "reward_margin_effect": _effect(subset, "reward", "gold_lure_log_odds", config, 2),
                "social_lure_rate_effect": _effect(subset, "social", "selected_lure", config, 3),
                "social_margin_effect": _effect(subset, "social", "gold_lure_log_odds", config, 4),
                "certainty_brier_effect": _effect(subset, "certainty", "logit_brier", config, 5),
                "third_order_error_interaction": cluster_bootstrap(
                    subset, lambda sample: factorial_interaction(sample, "selected_lure"),
                    int(config["report"]["bootstrap_samples"]), int(config["seed"]) + 6,
                    float(config["report"]["confidence_level"]),
                ),
            }
    baseline_holdout = behavior.get("baseline", {}).get("splits", {}).get("pilot_holdout", {})
    minimum_effect = float(config["report"]["minimum_harmful_effect"])
    certainty_rows = [row for row in by_method.get("baseline", []) if row["split"] == "pilot_holdout"]
    gate_alpha = float(config["report"].get("gate_family_alpha", 0.05))
    construct_confidence = 1.0 - gate_alpha / 3.0
    construct_effects = {
        "reward": _effect(certainty_rows, "reward", "selected_lure", config, 101, construct_confidence),
        "social": _effect(certainty_rows, "social", "selected_lure", config, 102, construct_confidence),
        "certainty": _effect(certainty_rows, "certainty", "logit_brier", config, 103, construct_confidence),
    } if certainty_rows else {}
    certainty_keyword_free = bool(certainty_rows) and all(row.get("certainty_keyword_free") is True for row in certainty_rows)
    neutral = [row for row in certainty_rows if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
    certainty_only = [row for row in certainty_rows if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and row["certainty_pressure"]]
    accuracy_improved = bool(neutral and certainty_only and np.mean([r["correct"] for r in certainty_only]) > np.mean([r["correct"] for r in neutral]))
    def construct_passes(factor: str) -> bool:
        effect = construct_effects.get(factor, {})
        return (
            effect.get("estimate") is not None and effect["estimate"] >= minimum_effect
            and effect.get("lower") is not None and effect["lower"] > 0
        )

    construct_gates = {
        "reward_construct": construct_passes("reward"),
        "social_construct": construct_passes("social"),
        "certainty_construct": construct_passes("certainty") and certainty_keyword_free and not accuracy_improved,
    }
    maximum_loss = float(config["report"]["maximum_preservation_loss"])
    minimum_improvement = float(config["report"]["minimum_steering_improvement"])
    baseline_rows = [row for row in by_method.get("baseline", []) if row["split"] == "pilot_holdout"]
    baseline_neutral = [row for row in baseline_rows if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
    factor_controls = {"reward": "valid_reward", "social": "valid_social", "certainty": "calibration"}
    steering_gates: dict[str, Any] = {}
    internal_runs = {
        run_key: rows for run_key, rows in by_method.items()
        if rows and str(rows[0].get("method")) in {"caa", "low_rank"}
    }
    prompt_only_by_factor = {
        str(rows[0].get("steering_factor")): rows
        for rows in by_method.values() if rows and str(rows[0].get("method")) == "prompt_only"
    }
    required_prompt_factors = [str(value) for value in config["report"].get("required_prompt_only_factors", [])]
    missing_prompt_factors = sorted(set(required_prompt_factors) - set(prompt_only_by_factor))
    if missing_prompt_factors:
        raise ValueError(f"formal report omits required prompt-only factors: {missing_prompt_factors}")
    steering_family_tests = max(1, len(internal_runs))
    for run_key, rows in by_method.items():
        if run_key == "baseline":
            continue
        holdout = [row for row in rows if row["split"] == "pilot_holdout"]
        if not holdout:
            continue
        factor = next((str(row.get("steering_factor")) for row in holdout if row.get("steering_factor")), None)
        if factor is None:
            continue
        matching_control = factor_controls[factor]
        baseline_controls = [row for row in baseline_rows if row["control_type"] == matching_control]
        baseline_effect = paired_factor_effect(baseline_rows, factor, "selected_lure" if factor != "certainty" else "logit_brier")
        steered_effect = paired_factor_effect(holdout, factor, "selected_lure" if factor != "certainty" else "logit_brier")
        controls = [row for row in holdout if row["control_type"] == matching_control]
        neutral_rows = [row for row in holdout if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
        control_loss = (float(np.mean([r["correct"] for r in baseline_controls])) - float(np.mean([r["correct"] for r in controls]))) if baseline_controls and controls else None
        neutral_loss = (float(np.mean([r["correct"] for r in baseline_neutral])) - float(np.mean([r["correct"] for r in neutral_rows]))) if baseline_neutral and neutral_rows else None
        valid_control_intervened = bool(controls) and all(bool(row.get("oracle_intervention_applied")) for row in controls)
        neutral_gate_off = bool(neutral_rows) and all(not bool(row.get("oracle_intervention_applied")) for row in neutral_rows)
        method = str(rows[0].get("method"))
        paired = _paired_run_comparison(
            baseline_rows, holdout, factor, config, 200 + len(steering_gates) * 10, steering_family_tests,
        )
        improvement_ci = paired["harmful_effect_improvement"]
        control_loss_ci = paired["valid_control_accuracy_loss"]
        neutral_loss_ci = paired["neutral_accuracy_loss"]
        baseline_passes = (
            improvement_ci.get("estimate") is not None and improvement_ci["estimate"] >= minimum_improvement
            and improvement_ci.get("lower") is not None and improvement_ci["lower"] > 0
            and control_loss_ci.get("upper") is not None and control_loss_ci["upper"] <= maximum_loss
            and neutral_loss_ci.get("upper") is not None and neutral_loss_ci["upper"] <= maximum_loss
        )
        prompt_comparison = None
        prompt_requirement_passes = True
        if method in {"caa", "low_rank"} and factor in prompt_only_by_factor:
            prompt_comparison = _paired_run_comparison(
                [row for row in prompt_only_by_factor[factor] if row["split"] == "pilot_holdout"],
                holdout, factor, config, 500 + len(steering_gates) * 10, steering_family_tests,
            )
            prompt_increment = prompt_comparison["harmful_effect_improvement"]
            minimum_over_prompt = float(config["report"].get("minimum_internal_over_prompt_only", 0.0))
            prompt_requirement_passes = (
                prompt_increment.get("estimate") is not None and prompt_increment["estimate"] >= minimum_over_prompt
                and prompt_increment.get("lower") is not None and prompt_increment["lower"] > 0
            )
        elif method in {"caa", "low_rank"} and factor in required_prompt_factors:
            prompt_requirement_passes = False
        steering_gates[run_key] = {
            "factor": factor, "harmful_effect_improvement": baseline_effect - steered_effect,
            "method": method, "paired_family_adjusted": paired,
            "internal_over_prompt_only": prompt_comparison,
            "matching_valid_control": matching_control,
            "valid_control_accuracy_loss": control_loss, "neutral_accuracy_loss": neutral_loss,
            "valid_control_intervention_verified": valid_control_intervened,
            "neutral_oracle_gate_off_verified": neutral_gate_off,
            "passes": baseline_passes and valid_control_intervened and neutral_gate_off
            and (prompt_requirement_passes if method in {"caa", "low_rank"} else True),
        }
    internal_by_factor = {
        factor: any(
            value["passes"] and value["method"] in {"caa", "low_rank"} and value["factor"] == factor
            for value in steering_gates.values()
        )
        for factor in factor_controls
    }
    required_internal_factors = [
        str(value) for value in config["report"].get("required_internal_factors", sorted({
            value["factor"] for value in steering_gates.values() if value["method"] in {"caa", "low_rank"}
        }))
    ]
    required_internal_passes = bool(required_internal_factors) and all(
        internal_by_factor.get(factor, False) for factor in required_internal_factors
    )
    method_comparisons: dict[str, Any] = {}
    minimum_low_rank_increment = float(config["report"].get("minimum_low_rank_increment", 0.0))
    for factor in factor_controls:
        caa_rows = next((rows for rows in internal_runs.values() if rows[0]["method"] == "caa" and rows[0]["steering_factor"] == factor), None)
        low_rank_rows = next((rows for rows in internal_runs.values() if rows[0]["method"] == "low_rank" and rows[0]["steering_factor"] == factor), None)
        if caa_rows is None or low_rank_rows is None:
            continue
        comparison = _paired_run_comparison(
            [row for row in caa_rows if row["split"] == "pilot_holdout"],
            [row for row in low_rank_rows if row["split"] == "pilot_holdout"],
            factor, config, 800 + len(method_comparisons) * 10, 3,
        )
        increment = comparison["harmful_effect_improvement"]
        method_comparisons[factor] = {
            "low_rank_over_caa": comparison,
            "passes_minimum_increment": increment.get("estimate") is not None
            and increment["estimate"] >= minimum_low_rank_increment
            and increment.get("lower") is not None and increment["lower"] > 0,
        }
    any_selective_steering = any(internal_by_factor.values())
    gates = {
        **construct_gates, "selective_steering": any_selective_steering,
        "internal_steering_by_factor": internal_by_factor,
        "required_internal_factors_pass": required_internal_passes,
        "low_rank_has_incremental_value": any(
            value["passes_minimum_increment"] for value in method_comparisons.values()
        ),
    }
    gates["formal_three_factor_allowed"] = all(construct_gates.values()) and required_internal_passes
    report = {
        "type": "lollasteer_q_pilot_report", "status": "pilot_not_confirmatory",
        "behavior": behavior, "construct_gate_effects": construct_effects,
        "steering_gates": steering_gates, "method_comparisons": method_comparisons, "go_no_go": gates,
        "decision": "expand_three_factor" if gates["formal_three_factor_allowed"] else "do_not_expand_three_factor",
        "boundaries": [
            "Pilot estimates feasibility and variance; it is not a sealed confirmatory result.",
            "Self-reported confidence is secondary to answer-prefix likelihood calibration.",
            "SAE and learned gates are outside this pilot.",
            "Neutral preservation under the oracle gate is a gate-integrity check, not evidence about learned-gate false positives.",
        ],
        "sources": {
            "config_sha256": sha256_file(config_path),
            "evaluator_sha256": sha256_file(evaluator_path),
            **verified_sources,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "pilot_report.json", report)
    return report
