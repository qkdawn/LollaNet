from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .common import atomic_write_json, read_jsonl, sha256_file, stable_hash
from .metrics import enrich_outputs, paired_factor_effect
from .runtime_audit import audit_direction_manifest, audit_run_manifest


FACTORS = ("reward", "social", "certainty")
METHODS = ("caa", "low_rank")


def _manifest_path(output_path: Path) -> Path:
    canonical = output_path.parent / "run_manifest.json"
    sidecar = output_path.with_suffix(".manifest.json")
    if output_path.name == "outputs.jsonl" and canonical.is_file():
        return canonical
    if sidecar.is_file():
        return sidecar
    raise ValueError(f"missing run manifest for {output_path}")


def _load_discovery_run(output_path: Path, evaluator: list[dict[str, Any]], prompt_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = _manifest_path(output_path)
    run_audit = audit_run_manifest(manifest_path)
    if run_audit.get("status") != "current_complete" or not run_audit.get("discovery_selection_eligible"):
        raise ValueError(f"selection requires a current complete audited discovery run: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(output_path)
    ids = [str(row.get("record_id")) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"discovery run is empty or contains duplicate records: {output_path}")
    if {row.get("split") for row in rows} != {"discovery"} or manifest.get("requested_split") != "discovery":
        raise ValueError(f"selection input contains non-discovery records: {output_path}")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise ValueError(f"run manifest record_count mismatch: {manifest_path}")
    if manifest.get("record_ids_sha256") != stable_hash(sorted(ids)):
        raise ValueError(f"run manifest record ID hash mismatch: {manifest_path}")
    if manifest.get("inputs", {}).get("sha256") != prompt_sha256:
        raise ValueError(f"run used a different prompt bundle: {manifest_path}")
    if manifest.get("artifacts", {}).get("outputs", {}).get("sha256") != sha256_file(output_path):
        raise ValueError(f"run output hash mismatch: {manifest_path}")
    if {str(row.get("run_key")) for row in rows} != {str(manifest.get("run_key"))}:
        raise ValueError(f"run rows disagree with manifest identity: {output_path}")
    return manifest, enrich_outputs(rows, evaluator)


def _candidate_score(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], factor: str, config: dict[str, Any]) -> dict[str, Any]:
    outcome = "selected_lure" if factor != "certainty" else "logit_brier"
    baseline_effect = paired_factor_effect(baseline, factor, outcome)
    candidate_effect = paired_factor_effect(rows, factor, outcome)
    control_type = {"reward": "valid_reward", "social": "valid_social", "certainty": "calibration"}[factor]
    baseline_control = [row for row in baseline if row["control_type"] == control_type]
    candidate_control = [row for row in rows if row["control_type"] == control_type]
    baseline_neutral = [row for row in baseline if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
    candidate_neutral = [row for row in rows if row["control_type"] == "factorial" and not row["reward_lure"] and not row["social_lure"] and not row["certainty_pressure"]]
    if not baseline_control or len(baseline_control) != len(candidate_control) or len(baseline_neutral) != len(candidate_neutral):
        raise ValueError("candidate lacks paired discovery preservation controls")
    control_loss = float(np.mean([row["correct"] for row in baseline_control]) - np.mean([row["correct"] for row in candidate_control]))
    neutral_loss = float(np.mean([row["correct"] for row in baseline_neutral]) - np.mean([row["correct"] for row in candidate_neutral]))
    improvement = float(baseline_effect - candidate_effect)
    maximum_loss = float(config["report"]["maximum_preservation_loss"])
    schedule_ok = all(bool(row.get("oracle_intervention_applied")) for row in candidate_control) and all(
        not bool(row.get("oracle_intervention_applied")) for row in candidate_neutral
    )
    return {
        "harmful_effect_improvement": improvement,
        "valid_control_accuracy_loss": control_loss,
        "neutral_accuracy_loss": neutral_loss,
        "oracle_schedule_verified": schedule_ok,
        "eligible": schedule_ok and control_loss <= maximum_loss and neutral_loss <= maximum_loss,
    }


def select_discovery_runs(
    config_path: Path,
    evaluator_path: Path,
    baseline_output: Path,
    candidate_outputs: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bundle_manifest_path = evaluator_path.parent / "manifest.json"
    if not bundle_manifest_path.is_file():
        raise ValueError("evaluator must belong to a bundle with manifest.json")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if bundle_manifest.get("config", {}).get("sha256") != sha256_file(config_path):
        raise ValueError("selection config does not match the frozen pilot bundle")
    if bundle_manifest.get("artifacts", {}).get("evaluator", {}).get("sha256") != sha256_file(evaluator_path):
        raise ValueError("bundle manifest evaluator hash mismatch")
    prompt_sha256 = bundle_manifest.get("artifacts", {}).get("prompts", {}).get("sha256")
    prompts_path = evaluator_path.parent / "prompts.jsonl"
    if not prompts_path.is_file() or sha256_file(prompts_path) != prompt_sha256:
        raise ValueError("pilot prompt artifact is missing or has a hash mismatch")
    expected_discovery_ids = {
        str(row["record_id"]) for row in read_jsonl(prompts_path) if row.get("split") == "discovery"
    }
    if not expected_discovery_ids:
        raise ValueError("pilot prompt artifact contains no discovery records")
    evaluator = read_jsonl(evaluator_path)
    baseline_manifest, baseline = _load_discovery_run(baseline_output, evaluator, prompt_sha256)
    if baseline_manifest.get("method") != "baseline":
        raise ValueError("selection requires a discovery-only baseline")
    expected_ids = {str(row["record_id"]) for row in baseline}
    if expected_ids != expected_discovery_ids:
        raise ValueError("baseline does not exactly cover the frozen discovery split")
    layer_indices = tuple(int(value) for value in baseline_manifest.get("layer_indices", []))
    if not layer_indices:
        raise ValueError("baseline manifest lacks preregistered layers")

    candidates: dict[tuple[str, str, int, float, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    direction_hashes: set[str] = set()
    direction_metadata_hashes: set[str] = set()
    baseline_compatibility = (
        baseline_manifest.get("checkpoint_actual"), baseline_manifest.get("checkpoint_revision"),
        baseline_manifest.get("chat_template_sha256"), baseline_manifest.get("dtype"),
        baseline_manifest.get("dependency_lock", {}).get("sha256"),
        baseline_manifest.get("runtime_code", {}).get("sha256"),
        baseline_manifest.get("runtime_environment_sha256"), layer_indices,
    )
    for output_path in candidate_outputs:
        manifest, rows = _load_discovery_run(output_path, evaluator, prompt_sha256)
        if {str(row["record_id"]) for row in rows} != expected_ids:
            raise ValueError(f"candidate does not exactly cover baseline discovery records: {output_path}")
        candidate_compatibility = (
            manifest.get("checkpoint_actual"), manifest.get("checkpoint_revision"),
            manifest.get("chat_template_sha256"), manifest.get("dtype"),
            manifest.get("dependency_lock", {}).get("sha256"), manifest.get("runtime_code", {}).get("sha256"),
            manifest.get("runtime_environment_sha256"),
            tuple(int(value) for value in manifest.get("layer_indices", [])),
        )
        if candidate_compatibility != baseline_compatibility:
            raise ValueError(f"candidate uses an incompatible model/runtime: {output_path}")
        method = str(manifest.get("method"))
        factor = str(manifest.get("factor"))
        if method not in METHODS or factor not in FACTORS:
            raise ValueError(f"unsupported selection candidate: {output_path}")
        key = (method, factor, int(manifest["layer"]), float(manifest["strength"]), int(manifest["rank"]))
        canonical_run_key = f"{method}__{factor}__layer_{key[2]}__strength_{key[3]:g}__rank_{key[4]}"
        if manifest.get("run_key") != canonical_run_key:
            raise ValueError(f"candidate run_key does not encode its manifest parameters: {output_path}")
        if any(
            row.get("method") != method or row.get("steering_factor") != factor
            or row.get("steering_layer") != key[2] or float(row.get("steering_strength")) != key[3]
            or int(row.get("steering_rank")) != key[4]
            for row in rows
        ):
            raise ValueError(f"candidate rows disagree with steering manifest: {output_path}")
        if key in candidates:
            raise ValueError(f"duplicate discovery candidate configuration: {key}")
        direction_hash = str(manifest.get("direction_bundle", {}).get("sha256"))
        direction_metadata_hash = str(manifest.get("direction_bundle", {}).get("metadata_sha256"))
        if not direction_hash or direction_hash == "None":
            raise ValueError(f"candidate lacks direction bundle provenance: {output_path}")
        if not direction_metadata_hash or direction_metadata_hash == "None":
            raise ValueError(f"candidate lacks direction metadata provenance: {output_path}")
        direction_path = Path(str(manifest["direction_bundle"].get("path", "")))
        metadata_path = Path(str(manifest["direction_bundle"].get("metadata_path", "")))
        if not direction_path.is_file() or sha256_file(direction_path) != direction_hash:
            raise ValueError(f"candidate direction artifact is missing or changed: {output_path}")
        if not metadata_path.is_file() or sha256_file(metadata_path) != direction_metadata_hash:
            raise ValueError(f"candidate direction metadata is missing or changed: {output_path}")
        direction_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        direction_audit = audit_direction_manifest(metadata_path)
        if direction_audit.get("status") != "current_eligible":
            raise ValueError(f"candidate direction bundle is not current and audited: {output_path}")
        if direction_metadata.get("npz_sha256") != direction_hash:
            raise ValueError(f"candidate direction metadata has the wrong array hash: {output_path}")
        expected_direction_compatibility = (
            manifest.get("checkpoint_actual"), manifest.get("checkpoint_revision"),
            manifest.get("chat_template_sha256"), manifest.get("dtype"),
            manifest.get("dependency_lock", {}).get("sha256"), manifest.get("runtime_code", {}).get("sha256"),
            manifest.get("runtime_environment_sha256"),
            tuple(int(value) for value in manifest.get("layer_indices", [])), prompt_sha256,
        )
        actual_direction_compatibility = (
            direction_metadata.get("checkpoint_actual"), direction_metadata.get("checkpoint_revision"),
            direction_metadata.get("chat_template_sha256"), direction_metadata.get("dtype"),
            direction_metadata.get("dependency_lock_sha256"), direction_metadata.get("runtime_code_sha256"),
            direction_metadata.get("runtime_environment_sha256"),
            tuple(int(value) for value in direction_metadata.get("layer_indices", [])),
            direction_metadata.get("prompt_bundle_sha256"),
        )
        if actual_direction_compatibility != expected_direction_compatibility:
            raise ValueError(f"candidate direction metadata is incompatible with its run: {output_path}")
        available_rank = int(
            direction_metadata.get("factors", {}).get(factor, {}).get(str(key[2]), {}).get("available_rank", 0)
        )
        if available_rank < key[4]:
            raise ValueError(f"candidate rank exceeds direction numerical rank: {output_path}")
        direction_hashes.add(direction_hash)
        direction_metadata_hashes.add(direction_metadata_hash)
        score = _candidate_score(rows, baseline, factor, config)
        candidates[key] = {"manifest": manifest, "score": score, "output_path": output_path}
        sources.append({
            "run_key": manifest["run_key"], "output_path": str(output_path),
            "output_sha256": sha256_file(output_path), "manifest_path": str(_manifest_path(output_path)),
            "manifest_sha256": sha256_file(_manifest_path(output_path)),
        })
    if len(direction_hashes) != 1 or len(direction_metadata_hashes) != 1:
        raise ValueError("all discovery candidates must use one direction bundle and metadata artifact")

    strengths = tuple(float(value) for value in config["model"]["strength_grid"])
    ranks = tuple(int(value) for value in config["model"]["low_rank_grid"])
    expected = {
        (method, factor, layer, strength, rank)
        for method in METHODS for factor in FACTORS for layer in layer_indices for strength in strengths
        for rank in ((1,) if method == "caa" else ranks)
    }
    missing = sorted(expected - set(candidates))
    extra = sorted(set(candidates) - expected)
    if missing or extra:
        raise ValueError(f"discovery candidate grid mismatch; missing={missing}, extra={extra}")
    candidate_counts = {
        method: sum(1 for key in candidates if key[0] == method) for method in METHODS
    }
    comparison_budget = config.get("comparison_budget", {})
    if comparison_budget.get("same_hyperparameter_trials") is True and len(set(candidate_counts.values())) != 1:
        raise ValueError("same_hyperparameter_trials is true but method candidate counts differ")

    selected: dict[str, Any] = {}
    for method in METHODS:
        selected[method] = {}
        for factor in FACTORS:
            group = [(key, value) for key, value in candidates.items() if key[:2] == (method, factor)]
            eligible = [(key, value) for key, value in group if value["score"]["eligible"]]
            pool = eligible or group
            key, value = max(
                pool,
                key=lambda item: (
                    item[1]["score"]["harmful_effect_improvement"],
                    -item[1]["score"]["valid_control_accuracy_loss"],
                    -item[1]["score"]["neutral_accuracy_loss"],
                    -item[0][2], -abs(item[0][3]), -item[0][4],
                ),
            )
            selected[method][factor] = {
                "run_key": value["manifest"]["run_key"], "layer": key[2], "strength": key[3], "rank": key[4],
                "direction_bundle_sha256": next(iter(direction_hashes)), **value["score"],
                "direction_metadata_sha256": next(iter(direction_metadata_hashes)),
            }

    result = {
        "type": "lollasteer_q_discovery_selection", "status": "frozen_before_holdout",
        "config_sha256": sha256_file(config_path), "evaluator_sha256": sha256_file(evaluator_path),
        "prompt_bundle_sha256": prompt_sha256,
        "discovery_record_ids_sha256": stable_hash(sorted(expected_ids)),
        "baseline": {"run_key": baseline_manifest["run_key"], "output_sha256": sha256_file(baseline_output)},
        "candidate_grid": {"layers": list(layer_indices), "strengths": list(strengths), "low_rank_ranks": list(ranks)},
        "candidate_counts_by_method": candidate_counts,
        "comparison_budget": comparison_budget,
        "candidate_sources": sorted(sources, key=lambda item: item["run_key"]),
        "selected": selected,
        "selection_rule": "maximize_discovery_harmful_effect_improvement_subject_to_preservation_then_deterministic_tie_break_v1",
    }
    result["selection_content_sha256"] = stable_hash(result)
    atomic_write_json(output_path, result)
    return result
