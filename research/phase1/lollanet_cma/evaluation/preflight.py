from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..access_views.views import validate_access_view
from ..common import atomic_write_json, read_json, read_jsonl, sha256_file, stable_hash
from ..development.preflight import _verify_direction_run
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..validate import validate_stage0


def _verified_content(value: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    content = dict(value)
    claimed = content.pop(field, None)
    if claimed != stable_hash(content):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _verified_artifact(source: dict[str, Any], label: str) -> Path:
    path = Path(str(source.get("path", "")))
    if not path.is_file() or sha256_file(path) != source.get("sha256"):
        raise ValueError(f"{label} is missing or changed")
    return path


def _verify_condition_rows(path: Path, condition_id: str, expected_ids: list[str]) -> int:
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids or len(observed) != len(set(observed)):
        raise ValueError(f"development condition coverage differs: {condition_id}")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content) or row.get("condition_id") != condition_id:
            raise ValueError(f"development condition row changed: {condition_id} {row.get('record_id')}")
    return len(rows)


def verify_development_run(
    manifest_path: Path,
    expected_type: str,
    expected_status: str,
) -> dict[str, Any]:
    manifest = _verified_content(
        read_json(manifest_path), "development_content_sha256", "development run",
    )
    if manifest.get("type") != expected_type or manifest.get("status") != expected_status:
        raise ValueError("development run has not frozen a successful selection")
    if not manifest.get("hard_gate_pass") or not all(manifest.get("hard_gates", {}).values()):
        raise ValueError("development run did not pass every hard gate")
    if not manifest.get("hard_gates", {}).get("direction_specificity"):
        raise ValueError("development direction-specificity gate failed")

    code_root = Path(str(manifest.get("code", {}).get("root", "")))
    if not code_root.is_dir() or manifest.get("code") != runtime_code_identity(code_root):
        raise ValueError("development execution code is missing or changed")
    if manifest.get("dependency_lock") != dependency_lock() or manifest.get("environment") != runtime_environment():
        raise ValueError("development dependency lock or runtime differs from release")

    charter_path = _verified_artifact(manifest.get("charter", {}), "development charter")
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    required = (
        "run_intent", "access_view_manifest", "direction_manifest", "preflight",
        "grid_selection", "random_vectors",
    )
    paths = {name: _verified_artifact(artifacts.get(name, {}), f"development {name}") for name in required}

    access_view = paths["access_view_manifest"].parent
    validation = validate_access_view(access_view, "development")
    access_rows = read_jsonl(access_view / "prompts.jsonl")
    expected_ids = [str(row["record_id"]) for row in access_rows]
    if len(expected_ids) != int(charter["data"]["expected_records"]):
        raise ValueError("development access view record count differs from its charter")

    intent = _verified_content(
        read_json(paths["run_intent"]), "intent_content_sha256", "development intent",
    )
    selection = _verified_content(
        read_json(paths["grid_selection"]),
        "development_selection_content_sha256",
        "development grid selection",
    )
    preflight = _verified_content(
        read_json(paths["preflight"]),
        "development_preflight_content_sha256",
        "development preflight",
    )
    direction = _verify_direction_run(paths["direction_manifest"], charter)
    if (
        intent.get("direction_manifest_sha256") != sha256_file(paths["direction_manifest"])
        or intent.get("access_view_manifest_sha256") != sha256_file(paths["access_view_manifest"])
        or intent.get("preflight_sha256") != sha256_file(paths["preflight"])
        or manifest.get("direction_manifest_sha256") != sha256_file(paths["direction_manifest"])
        or manifest.get("preflight_sha256") != sha256_file(paths["preflight"])
        or manifest.get("access_view_manifest_sha256") != sha256_file(paths["access_view_manifest"])
        or preflight.get("development_gate_status") != "ready"
        or preflight.get("errors")
        or intent.get("code") != manifest.get("code")
        or intent.get("dependency_lock") != manifest.get("dependency_lock")
        or intent.get("environment") != manifest.get("environment")
        or intent.get("random_vectors", {}).get("sha256") != sha256_file(paths["random_vectors"])
        or direction.get("manifest_sha256") != sha256_file(paths["direction_manifest"])
    ):
        raise ValueError("development provenance links are inconsistent")
    if (
        selection.get("selected_condition_id") != manifest.get("selected_correct_summary", {}).get("condition_id")
        or int(selection.get("selected_layer")) != int(manifest.get("selected_layer"))
        or float(selection.get("selected_alpha")) != float(manifest.get("selected_alpha"))
        or not selection.get("no_fallback_after_specificity_failure")
    ):
        raise ValueError("development selection differs from its final manifest")

    condition_artifacts = artifacts.get("conditions", {})
    grid_ids = [str(row["condition_id"]) for row in selection.get("grid_summaries", [])]
    expected_conditions = {
        "base",
        *grid_ids,
        str(manifest.get("wrong_summary", {}).get("condition_id")),
        str(manifest.get("random_summary", {}).get("condition_id")),
    }
    if set(condition_artifacts) != expected_conditions:
        raise ValueError("development condition artifact coverage is incomplete")
    summaries = [
        selection["base_summary"],
        *selection["grid_summaries"],
        manifest["wrong_summary"],
        manifest["random_summary"],
    ]
    for summary in summaries:
        condition_id = str(summary["condition_id"])
        if summary.get("artifact") != condition_artifacts.get(condition_id):
            raise ValueError(f"development summary artifact differs: {condition_id}")
    counts = {
        condition_id: _verify_condition_rows(
            _verified_artifact(condition_artifacts[condition_id], f"development condition {condition_id}"),
            condition_id,
            expected_ids,
        )
        for condition_id in sorted(expected_conditions)
    }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "selected_layer": int(manifest["selected_layer"]),
        "selected_alpha": float(manifest["selected_alpha"]),
        "condition_counts": counts,
        "access_view_validation": validation,
        "direction_manifest_sha256": sha256_file(paths["direction_manifest"]),
    }


def audit_sealed_release_preflight(
    charter_path: Path,
    stage0_bundle: Path,
    development_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    source_contract = charter["source_stage0"]
    source_manifest_path = stage0_bundle / "manifest.json"
    if stage0_bundle.resolve() != Path(str(source_contract["path"])).resolve():
        raise ValueError("Stage 0 bundle path differs from the sealed-release charter")
    if (
        not source_manifest_path.is_file()
        or sha256_file(source_manifest_path) != source_contract["manifest_sha256"]
    ):
        raise ValueError("Stage 0 manifest is missing or differs from the sealed-release charter")
    source_manifest = _verified_content(
        read_json(source_manifest_path), "bundle_content_sha256", "Stage 0 bundle",
    )
    if source_manifest["bundle_content_sha256"] != source_contract["bundle_content_sha256"]:
        raise ValueError("Stage 0 content differs from the sealed-release charter")

    errors: list[str] = []
    development: dict[str, Any] | None = None
    stage0_validation: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    prerequisites = charter["prerequisites"]
    if not development_manifest_path.is_file():
        errors.append("development_manifest_missing")
    else:
        try:
            development = verify_development_run(
                development_manifest_path,
                str(prerequisites["development_manifest_type"]),
                str(prerequisites["development_status"]),
            )
            stage0_validation = validate_stage0(stage0_bundle)
            expected = {
                split: int(value["expected_records"])
                for split, value in charter["release"]["splits"].items()
            }
            observed = {
                split: sum(stage0_validation["splits"].get(split, {}).values()) for split in expected
            }
            if observed != expected or stage0_validation.get("model_id_overlap") != 0:
                raise ValueError("Stage 0 sealed/OOD coverage differs from the release charter")
            environment = runtime_environment()
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"development_or_stage0_invalid:{exc}")
    if not errors:
        status = "ready"
    elif errors == ["development_manifest_missing"]:
        status = "blocked_missing_development"
    else:
        status = "blocked_invalid_prerequisite"
    result = {
        "type": "lollanet_cma_sealed_release_preflight_v1",
        "sealed_release_gate_status": status,
        "unseal_authorized": status == "ready",
        "errors": errors,
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "stage0": {
            "path": str(stage0_bundle.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
            "bundle_content_sha256": source_manifest["bundle_content_sha256"],
            "validation": stage0_validation,
        },
        "development": development,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": environment,
    }
    result["sealed_release_preflight_content_sha256"] = stable_hash(result)
    atomic_write_json(output_path, result)
    return result
