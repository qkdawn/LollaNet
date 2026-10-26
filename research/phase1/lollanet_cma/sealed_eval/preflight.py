from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..common import atomic_write_json, read_json, sha256_file, stable_hash
from ..evaluation.preflight import _verified_content
from ..evaluation.preflight import verify_development_run
from ..evaluation.release import validate_released_view
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from .conditions import validate_condition_contract, validate_general_equal_all
from .control_set import validate_control_bundle


def audit_sealed_evaluation_preflight(
    charter_path: Path,
    control_charter_path: Path,
    evaluation_release_path: Path,
    probe_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    control_charter = yaml.safe_load(control_charter_path.read_text(encoding="utf-8"))
    condition_contract = validate_condition_contract(charter)
    unresolved = [
        key for key, value in charter.get("readiness_blockers", {}).items()
        if value.get("status") != "resolved"
    ]
    errors: list[str] = []
    release: dict[str, Any] | None = None
    view_validation: dict[str, Any] | None = None
    control_validation: dict[str, Any] | None = None
    probe: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    direction_validation: dict[str, Any] | None = None
    if unresolved:
        errors.append(f"unresolved_contracts:{','.join(sorted(unresolved))}")
    if not evaluation_release_path.is_file():
        errors.append("evaluation_release_missing")
    if not errors:
        try:
            release = _verified_content(
                read_json(evaluation_release_path),
                "evaluation_release_content_sha256",
                "evaluation release",
            )
            prerequisites = charter["prerequisites"]
            if (
                release.get("type") != prerequisites["evaluation_release_type"]
                or release.get("status") != prerequisites["evaluation_release_status"]
            ):
                raise ValueError("evaluation release type or status differs")
            expected_views = charter["data"]["views"]
            if set(release.get("views", {})) != set(expected_views):
                raise ValueError("evaluation release does not contain exactly the frozen views")
            view_validation = {}
            for split, contract in expected_views.items():
                released = release["views"][split]
                view_dir = Path(str(released["path"]))
                if view_dir.resolve() != Path(str(contract["path"])).resolve():
                    raise ValueError(f"released view path differs: {split}")
                if sha256_file(view_dir / "manifest.json") != released["manifest_sha256"]:
                    raise ValueError(f"released view manifest changed: {split}")
                validated = validate_released_view(view_dir, split)
                if validated["record_count"] != int(contract["expected_records"]):
                    raise ValueError(f"released view record count differs: {split}")
                view_validation[split] = validated
            control_contract = charter["readiness_blockers"]["unrelated_control_dataset"]
            control_manifest_path = Path(str(control_contract["artifact"]))
            if sha256_file(control_manifest_path) != control_contract["artifact_sha256"]:
                raise ValueError("unrelated control manifest changed")
            control_validation = validate_control_bundle(control_manifest_path.parent, control_charter)
            if control_validation["control_bundle_content_sha256"] != control_contract["content_sha256"]:
                raise ValueError("unrelated control content differs from the sealed charter")
            if not probe_manifest_path.is_file():
                errors.append("probe_manifest_missing")
            else:
                probe = _verified_content(
                    read_json(probe_manifest_path), "probe_content_sha256", "hidden-state probe",
                )
                probe_artifact = Path(str(probe.get("artifacts", {}).get("probe", {}).get("path", "")))
                if (
                    probe.get("type") != "lollanet_cma_hidden_state_probe_v1"
                    or probe.get("status") != "frozen_before_sealed_evaluation"
                    or probe.get("development_manifest", {}).get("sha256")
                    != release["development_manifest_sha256"]
                    or not probe_artifact.is_file()
                    or sha256_file(probe_artifact) != probe.get("artifacts", {}).get("probe", {}).get("sha256")
                    or probe.get("charter", {}).get("sha256") != sha256_file(charter_path)
                    or probe.get("code") != runtime_code_identity(Path(__file__).parent)
                    or probe.get("dependency_lock") != dependency_lock()
                    or probe.get("environment") != runtime_environment()
                ):
                    raise ValueError("hidden-state probe is not frozen for the released development run")
                development_path = Path(str(probe["development_manifest"]["path"]))
                if sha256_file(development_path) != release["development_manifest_sha256"]:
                    raise ValueError("probe and evaluation release use different development runs")
                development = verify_development_run(
                    development_path,
                    str(prerequisites["development_manifest_type"]),
                    str(prerequisites["development_status"]),
                )
                development_manifest = read_json(development_path)
                direction_source = development_manifest["artifacts"]["direction_manifest"]
                direction_manifest_path = Path(str(direction_source["path"]))
                if sha256_file(direction_manifest_path) != direction_source["sha256"]:
                    raise ValueError("probe direction manifest is missing or changed")
                direction_manifest = _verified_content(
                    read_json(direction_manifest_path), "direction_content_sha256", "direction run",
                )
                tensor_source = direction_manifest["artifacts"]["directions"]
                tensor_path = Path(str(tensor_source["path"]))
                if sha256_file(tensor_path) != tensor_source["sha256"]:
                    raise ValueError("probe direction tensors are missing or changed")
                from safetensors.numpy import load_file
                tensors = load_file(tensor_path)
                direction_validation = validate_general_equal_all(
                    tensors,
                    list(charter["families"]),
                    int(development["selected_layer"]),
                    float(prerequisites["general_equal_all_max_cosine_similarity"]),
                )
            environment = runtime_environment()
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"released_prerequisite_invalid:{exc}")
    if not errors:
        status = "ready"
    elif errors == ["evaluation_release_missing"]:
        status = "blocked_missing_release"
    elif errors == ["probe_manifest_missing"]:
        status = "blocked_missing_probe"
    elif any(error.startswith("unresolved_contracts:") for error in errors):
        status = "blocked_unresolved_contract"
    else:
        status = "blocked_invalid_prerequisite"
    result = {
        "type": "lollanet_cma_sealed_evaluation_preflight_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sealed_evaluation_gate_status": status,
        "evaluation_authorized": status == "ready",
        "errors": errors,
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "control_charter": {
            "path": str(control_charter_path.resolve()),
            "sha256": sha256_file(control_charter_path),
        },
        "condition_contract": condition_contract,
        "evaluation_release": None if release is None else {
            "path": str(evaluation_release_path.resolve()),
            "sha256": sha256_file(evaluation_release_path),
            "content_sha256": release["evaluation_release_content_sha256"],
            "development_manifest_sha256": release["development_manifest_sha256"],
        },
        "view_validation": view_validation,
        "control_validation": control_validation,
        "probe": None if probe is None else {
            "path": str(probe_manifest_path.resolve()),
            "sha256": sha256_file(probe_manifest_path),
            "content_sha256": probe["probe_content_sha256"],
            "selected_layer": probe["selected_layer"],
            "selected_l2": probe["selected_l2"],
        },
        "direction_validation": direction_validation,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": environment,
    }
    result["sealed_evaluation_preflight_content_sha256"] = stable_hash(result)
    atomic_write_json(output_path, result)
    return result
