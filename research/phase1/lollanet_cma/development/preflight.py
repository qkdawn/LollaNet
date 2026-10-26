from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..access_views.views import validate_access_view
from ..common import atomic_write_json, read_json, read_jsonl, sha256_file, stable_hash
from ..directions.fitting import _array_sha256
from ..directions.preflight import _verify_checkpoint_files
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment


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


def _development_view(access_view: Path, charter: dict[str, Any]) -> dict[str, Any]:
    if access_view.resolve() != Path(str(charter["data"]["access_view"])).resolve():
        raise ValueError("development access-view path differs from the charter")
    validation = validate_access_view(access_view, str(charter["data"]["allowed_split"]))
    manifest = _verified_content(
        read_json(access_view / "manifest.json"), "access_view_content_sha256", "development access view",
    )
    controller_path = _verified_artifact(manifest["artifacts"]["controller"], "development controller")
    controllers = read_jsonl(controller_path)
    families = list(charter["families"])
    counts = Counter(str(row["model_family"]) for row in controllers)
    if len(controllers) != int(charter["data"]["expected_records"]):
        raise ValueError("development access view has an unexpected record count")
    if set(counts) != set(families) or any(
        counts[family] != int(charter["data"]["expected_records_per_family"]) for family in families
    ):
        raise ValueError("development access view has unexpected family coverage")
    return {
        "validation": validation,
        "family_counts": {family: counts[family] for family in families},
    }


def _verify_direction_run(direction_manifest_path: Path, charter: dict[str, Any]) -> dict[str, Any]:
    manifest = _verified_content(
        read_json(direction_manifest_path), "direction_content_sha256", "direction run",
    )
    prerequisites = charter["prerequisites"]
    if (
        manifest.get("type") != prerequisites["direction_manifest_type"]
        or manifest.get("status") != prerequisites["direction_status"]
        or not manifest.get("hard_gate_pass")
        or not all(manifest.get("hard_gates", {}).values())
    ):
        raise ValueError("direction run did not pass every prerequisite gate")
    direction_code_root = Path(str(manifest.get("code", {}).get("root", "")))
    if not direction_code_root.is_dir() or manifest.get("code") != runtime_code_identity(direction_code_root):
        raise ValueError("direction fitting code is missing or changed")
    if manifest.get("dependency_lock") != dependency_lock() or manifest.get("environment") != runtime_environment():
        raise ValueError("direction dependency lock or runtime differs from development")

    required = (
        "run_intent", "fit_records", "selection", "preflight", "access_view_manifest",
        "stage0_manifest", "checkpoint_manifest", "directions", "direction_index",
    )
    artifact_paths = {
        name: _verified_artifact(manifest.get("artifacts", {}).get(name, {}), f"direction {name}")
        for name in required
    }
    intent = _verified_content(read_json(artifact_paths["run_intent"]), "intent_content_sha256", "direction intent")
    if (
        intent.get("model_id") != manifest.get("model_id")
        or intent.get("revision") != manifest.get("revision")
        or intent.get("target_layers") != manifest.get("target_layers")
        or intent.get("access_view_manifest_sha256") != manifest.get("access_view_manifest_sha256")
        or intent.get("checkpoint") != manifest.get("checkpoint")
        or intent.get("checkpoint_manifest_sha256") != manifest.get("checkpoint_manifest_sha256")
        or intent.get("quantization") != manifest.get("quantization")
        or intent.get("compute_dtype") != manifest.get("compute_dtype")
        or intent.get("maximum_input_tokens") != manifest.get("maximum_input_tokens")
        or intent.get("maximum_new_tokens") != manifest.get("maximum_new_tokens")
        or intent.get("sampling") != manifest.get("sampling")
        or intent.get("chat_template_sha256") != manifest.get("chat_template_sha256")
    ):
        raise ValueError("direction run intent differs from its manifest")
    checkpoint = Path(str(intent["checkpoint"]))
    checkpoint_value = _verify_checkpoint_files(checkpoint)
    if checkpoint_value["checkpoint_content_sha256"] != read_json(artifact_paths["checkpoint_manifest"])["checkpoint_content_sha256"]:
        raise ValueError("direction checkpoint content differs from its recorded manifest")

    index = _verified_content(
        read_json(artifact_paths["direction_index"]), "direction_index_content_sha256", "direction index",
    )
    if (
        index.get("model_id") != manifest.get("model_id")
        or index.get("revision") != manifest.get("revision")
        or index.get("target_layers") != manifest.get("target_layers")
        or index.get("directions_sha256") != sha256_file(artifact_paths["directions"])
    ):
        raise ValueError("direction index differs from the run manifest")

    from safetensors.numpy import load_file

    tensors = load_file(artifact_paths["directions"])
    expected_keys = {
        f"{family}__layer_{layer}"
        for family in charter["families"] for layer in manifest["target_layers"]
    } | {
        f"{kind}__layer_{layer}"
        for kind in ("general", "equal_all") for layer in manifest["target_layers"]
    }
    if set(tensors) != expected_keys:
        raise ValueError("direction tensor bundle has unexpected key coverage")
    for family in charter["families"]:
        for layer in manifest["target_layers"]:
            vector = tensors[f"{family}__layer_{layer}"]
            metadata = index["direction_metadata"]["families"][family][str(layer)]
            if (
                metadata["tensor_sha256"] != _array_sha256(vector)
                or metadata["dtype"] != str(vector.dtype)
                or metadata["shape"] != list(vector.shape)
                or not np.isclose(float(np.linalg.norm(vector.astype(np.float64))), 1.0, atol=1e-5)
            ):
                raise ValueError(f"direction tensor metadata mismatch: {family} layer {layer}")
    for layer in manifest["target_layers"]:
        vector = tensors[f"general__layer_{layer}"]
        metadata = index["direction_metadata"]["general"][str(layer)]
        family_record_counts = {
            family: len(index["selected_record_ids"][family]) for family in charter["families"]
        }
        if (
            metadata["tensor_sha256"] != _array_sha256(vector)
            or metadata["dtype"] != str(vector.dtype)
            or metadata["shape"] != list(vector.shape)
            or metadata.get("record_count") != sum(family_record_counts.values())
            or metadata.get("family_record_counts") != family_record_counts
            or not np.isclose(float(np.linalg.norm(vector.astype(np.float64))), 1.0, atol=1e-5)
        ):
            raise ValueError(f"general direction tensor metadata mismatch: layer {layer}")
        equal_all = tensors[f"equal_all__layer_{layer}"]
        equal_metadata = index["direction_metadata"]["equal_all"][str(layer)]
        expected_sources = [
            f"{family}__layer_{layer}" for family in charter["families"]
        ]
        if (
            equal_metadata["tensor_sha256"] != _array_sha256(equal_all)
            or equal_metadata["dtype"] != str(equal_all.dtype)
            or equal_metadata["shape"] != list(equal_all.shape)
            or equal_metadata.get("source_family_keys") != expected_sources
            or not np.isclose(
                float(np.linalg.norm(equal_all.astype(np.float64))), 1.0, atol=1e-5,
            )
        ):
            raise ValueError(f"equal-all direction tensor metadata mismatch: layer {layer}")
    if (
        index.get("balance") != manifest.get("balance")
        or {family: len(ids) for family, ids in index.get("selected_record_ids", {}).items()}
        != manifest.get("selected_record_counts")
    ):
        raise ValueError("direction selected-record provenance differs from its manifest")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(direction_manifest_path),
        "directions_path": str(artifact_paths["directions"]),
        "directions_sha256": sha256_file(artifact_paths["directions"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_content_sha256": checkpoint_value["checkpoint_content_sha256"],
    }


def audit_development_preflight(
    charter_path: Path,
    access_view: Path,
    direction_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    view = _development_view(access_view, charter)
    direction: dict[str, Any] | None = None
    errors: list[str] = []
    environment: dict[str, Any] | None = None
    if not direction_manifest_path.is_file():
        errors.append("direction_manifest_missing")
    else:
        try:
            direction = _verify_direction_run(direction_manifest_path, charter)
            environment = runtime_environment()
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"direction_manifest_invalid:{exc}")
    result = {
        "type": "lollanet_cma_development_preflight_v1",
        "development_gate_status": "ready" if not errors else "blocked_missing_directions",
        "errors": errors,
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "access_view": {
            "path": str(access_view.resolve()),
            "manifest_sha256": sha256_file(access_view / "manifest.json"),
            "validator_code": runtime_code_identity(Path(__file__).parents[1] / "access_views"),
            **view,
        },
        "direction_run": None if direction is None else {
            key: value for key, value in direction.items() if key != "manifest"
        },
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": environment,
    }
    result["development_preflight_content_sha256"] = stable_hash(result)
    atomic_write_json(output_path, result)
    return result
