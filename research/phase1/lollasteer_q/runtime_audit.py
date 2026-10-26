from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_jsonl, runtime_code_identity, sha256_file, stable_hash


RUN_REQUIRED_FIELDS = {
    "execution_identity_sha256", "requested_split", "record_ids_sha256", "prompt_record_count",
    "activation_capture_policy", "intervention_position_policy", "option_scoring_policy",
    "prompt_only_instruction_sha256", "oracle_gate_policy",
    "dependency_lock", "runtime_code", "runtime_environment", "runtime_environment_sha256",
}
DIRECTION_REQUIRED_FIELDS = {
    "npz_sha256", "checkpoint_actual", "checkpoint_revision", "chat_template_sha256", "dtype",
    "dependency_lock_sha256", "runtime_code_sha256", "runtime_environment_sha256",
    "prompt_bundle_sha256", "layer_indices", "factors", "source_run", "source_run_manifest", "source_activations",
}


def execution_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    direction = manifest.get("direction_bundle") or {}
    return {
        "prompt_bundle_sha256": manifest.get("inputs", {}).get("sha256"),
        "record_ids_sha256": manifest.get("record_ids_sha256"),
        "requested_split": manifest.get("requested_split"),
        "checkpoint_actual": manifest.get("checkpoint_actual"),
        "checkpoint_revision": manifest.get("checkpoint_revision"),
        "dtype": manifest.get("dtype"), "chat_template_sha256": manifest.get("chat_template_sha256"),
        "method": manifest.get("method"), "factor": manifest.get("factor"),
        "layer": manifest.get("layer"), "strength": manifest.get("strength"), "rank": manifest.get("rank"),
        "direction_bundle_sha256": direction.get("sha256"),
        "prompt_only_instruction_sha256": manifest.get("prompt_only_instruction_sha256"),
        "max_new_tokens": manifest.get("max_new_tokens"), "do_sample": manifest.get("do_sample"),
        "batch_size": manifest.get("batch_size"),
        "oracle_gate_policy": manifest.get("oracle_gate_policy"),
        "dependency_lock_sha256": manifest.get("dependency_lock", {}).get("sha256"),
        "runtime_code_sha256": manifest.get("runtime_code", {}).get("sha256"),
        "runtime_environment_sha256": manifest.get("runtime_environment_sha256"),
        "activation_capture_policy": manifest.get("activation_capture_policy"),
        "intervention_position_policy": manifest.get("intervention_position_policy"),
        "option_scoring_policy": manifest.get("option_scoring_policy"),
    }


def audit_run_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = sorted(RUN_REQUIRED_FIELDS - set(manifest))
    result: dict[str, Any] = {
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path),
        "method": manifest.get("method"), "run_key": manifest.get("run_key"),
        "record_count": manifest.get("record_count"), "missing_current_fields": missing,
        "formal_report_eligible": False, "discovery_selection_eligible": False,
        "direction_fit_eligible": False,
    }
    if missing:
        result["status"] = "legacy_ineligible"
        return result
    errors: list[str] = []
    expected_identity_hash = stable_hash(execution_identity(manifest))
    if manifest.get("execution_identity_sha256") != expected_identity_hash:
        errors.append("execution_identity_hash_mismatch")
    if manifest.get("run_id") != expected_identity_hash[:24]:
        errors.append("run_id_mismatch")
    if stable_hash(manifest.get("runtime_environment")) != manifest.get("runtime_environment_sha256"):
        errors.append("runtime_environment_hash_mismatch")
    lock_path = Path(str(manifest.get("dependency_lock", {}).get("path", "")))
    if not lock_path.is_file() or sha256_file(lock_path) != manifest.get("dependency_lock", {}).get("sha256"):
        errors.append("dependency_lock_missing_or_changed")
    runtime_code = manifest.get("runtime_code", {})
    runtime_code_root = Path(str(runtime_code.get("root", "")))
    if not runtime_code_root.is_dir() or runtime_code_identity(runtime_code_root) != runtime_code:
        errors.append("runtime_code_missing_or_changed")
    prompt_path = Path(str(manifest.get("inputs", {}).get("path", "")))
    output_path = Path(str(manifest.get("artifacts", {}).get("outputs", {}).get("path", "")))
    activation_path = Path(str(manifest.get("artifacts", {}).get("activations", {}).get("path", "")))
    if not prompt_path.is_file() or sha256_file(prompt_path) != manifest.get("inputs", {}).get("sha256"):
        errors.append("prompt_artifact_missing_or_changed")
    if not output_path.is_file() or sha256_file(output_path) != manifest.get("artifacts", {}).get("outputs", {}).get("sha256"):
        errors.append("output_artifact_missing_or_changed")
    if not activation_path.is_file() or sha256_file(activation_path) != manifest.get("artifacts", {}).get("activations", {}).get("sha256"):
        errors.append("activation_artifact_missing_or_changed")
    complete = False
    if not errors:
        prompts = read_jsonl(prompt_path)
        outputs = read_jsonl(output_path)
        if int(manifest.get("prompt_record_count", -1)) != len(prompts):
            errors.append("prompt_record_count_mismatch")
        split = str(manifest.get("requested_split"))
        expected = prompts if split == "all" else [row for row in prompts if row.get("split") == split]
        expected_ids = {str(row["record_id"]) for row in expected}
        output_ids = [str(row.get("record_id")) for row in outputs]
        if len(output_ids) != len(set(output_ids)):
            errors.append("duplicate_output_record_id")
        if int(manifest.get("record_count", -1)) != len(outputs):
            errors.append("record_count_mismatch")
        if manifest.get("record_ids_sha256") != stable_hash(sorted(output_ids)):
            errors.append("record_id_hash_mismatch")
        complete = len(output_ids) == len(expected_ids) and set(output_ids) == expected_ids
    if errors:
        result["status"] = "invalid"
        result["errors"] = errors
        return result
    result["coverage"] = "complete" if complete else "limited_smoke"
    result["status"] = "current_complete" if complete else "current_smoke_only"
    split = str(manifest.get("requested_split"))
    method = str(manifest.get("method"))
    result["formal_report_eligible"] = complete and split == "all"
    result["discovery_selection_eligible"] = complete and split == "discovery"
    result["direction_fit_eligible"] = complete and split == "all" and method == "baseline"
    return result


def audit_direction_manifest(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = sorted(DIRECTION_REQUIRED_FIELDS - set(metadata))
    result: dict[str, Any] = {
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": sha256_file(metadata_path),
        "missing_current_fields": missing, "formal_candidate_eligible": False,
    }
    if missing:
        result["status"] = "legacy_ineligible"
        return result
    errors: list[str] = []
    npz_path = Path(str(metadata.get("npz_path", metadata_path.with_suffix(".npz"))))
    if not npz_path.is_file() or sha256_file(npz_path) != metadata.get("npz_sha256"):
        errors.append("direction_array_missing_or_changed")
    source_manifest = metadata.get("source_run_manifest", {})
    source_manifest_path = Path(str(source_manifest.get("path", "")))
    if not source_manifest_path.is_file() or sha256_file(source_manifest_path) != source_manifest.get("sha256"):
        errors.append("source_run_manifest_missing_or_changed")
        source_manifest_value: dict[str, Any] = {}
    else:
        source_manifest_value = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_run_audit = audit_run_manifest(source_manifest_path)
        if source_run_audit.get("status") != "current_complete" or not source_run_audit.get("direction_fit_eligible"):
            errors.append("source_run_not_current_direction_fit_eligible")
        if metadata.get("source_run") != source_manifest_value.get("run_id"):
            errors.append("source_run_identity_mismatch")
    source_activations = metadata.get("source_activations", {})
    source_activations_path = Path(str(source_activations.get("path", "")))
    if not source_activations_path.is_file() or sha256_file(source_activations_path) != source_activations.get("sha256"):
        errors.append("source_activations_missing_or_changed")
    elif source_manifest_value:
        expected_activations = source_manifest_value.get("artifacts", {}).get("activations", {})
        if (
            Path(str(expected_activations.get("path", ""))).resolve() != source_activations_path.resolve()
            or expected_activations.get("sha256") != source_activations.get("sha256")
        ):
            errors.append("source_activations_manifest_mismatch")
    for factor in ("reward", "social", "certainty"):
        layers = metadata.get("factors", {}).get(factor, {})
        for layer in metadata.get("layer_indices", []):
            item = layers.get(str(layer), {})
            if int(item.get("available_rank", 0)) < 1 or int(item.get("hidden_size", 0)) < 1:
                errors.append(f"incomplete_factor_metadata:{factor}:layer_{layer}")
    if errors:
        result["status"] = "invalid"
        result["errors"] = errors
        return result
    result["status"] = "current_eligible"
    result["formal_candidate_eligible"] = True
    return result


def audit_runtime(runtime_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    runs = [audit_run_manifest(path) for path in sorted(runtime_root.rglob("run_manifest.json"))]
    directions: list[dict[str, Any]] = []
    for path in sorted(runtime_root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and value.get("type") == "lollasteer_q_direction_bundle":
            directions.append(audit_direction_manifest(path))
    report = {
        "type": "lollasteer_q_runtime_audit_v1", "runtime_root": str(runtime_root.resolve()),
        "runs": runs, "directions": directions,
        "summary": {
            "run_status_counts": dict(sorted(Counter(row["status"] for row in runs).items())),
            "direction_status_counts": dict(sorted(Counter(row["status"] for row in directions).items())),
            "formal_report_eligible_runs": sum(bool(row["formal_report_eligible"]) for row in runs),
            "discovery_selection_eligible_runs": sum(bool(row["discovery_selection_eligible"]) for row in runs),
            "formal_candidate_direction_bundles": sum(bool(row["formal_candidate_eligible"]) for row in directions),
        },
    }
    report["audit_content_sha256"] = stable_hash(report)
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
