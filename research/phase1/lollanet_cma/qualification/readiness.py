from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatch
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from .common import (
    dependency_lock,
    normalized_local_checkpoint,
    read_json,
    runtime_code_identity,
    runtime_environment,
    sha256_file,
    stable_hash,
)
from .freeze import REQUIRED_PATTERNS
from ..access_views.views import validate_access_view


def _local_revision(path: Path) -> str | None:
    metadata_path = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata_path.is_file():
        return metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    revision_path = path / "REVISION"
    if revision_path.is_file():
        return revision_path.read_text(encoding="utf-8").strip()
    if path.parent.name == "snapshots" and len(path.name) == 40:
        try:
            int(path.name, 16)
        except ValueError:
            return None
        return path.name
    return None


def _checkpoint_artifact_errors(path: Path) -> list[str]:
    errors: list[str] = []
    if not (path / "config.json").is_file():
        errors.append("checkpoint_config_missing")
    if not (path / "tokenizer_config.json").is_file():
        errors.append("checkpoint_tokenizer_config_missing")
    weight_files = [
        *path.glob("*.safetensors"),
        *path.glob("*.bin"),
        *path.glob("*.safetensors.index.json"),
        *path.glob("*.bin.index.json"),
    ]
    if not any(item.is_file() for item in weight_files):
        errors.append("checkpoint_weights_missing")
    return errors


def _checkpoint_manifest_errors(
    path: Path, candidate: dict[str, Any], charter_path: Path,
) -> list[str]:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return ["checkpoint_manifest_missing"]
    try:
        manifest = read_json(manifest_path)
        content = dict(manifest)
        claimed = content.pop("checkpoint_content_sha256", None)
        if claimed != stable_hash(content):
            return ["checkpoint_manifest_content_hash_mismatch"]
        if (
            manifest.get("candidate_id") != candidate.get("id")
            or manifest.get("model_id") != candidate.get("model_id")
            or manifest.get("revision") != candidate.get("revision")
            or Path(str(manifest.get("checkpoint", ""))).resolve() != path.resolve()
            or manifest.get("charter_sha256") != sha256_file(charter_path)
        ):
            return ["checkpoint_manifest_identity_mismatch"]
        root = path.resolve()
        files = manifest.get("files", [])
        if not isinstance(files, list) or not files:
            return ["checkpoint_manifest_files_missing"]
        for row in files:
            file_path = (root / str(row["path"])).resolve()
            if not file_path.is_relative_to(root):
                return ["checkpoint_manifest_path_escape"]
            if (
                not file_path.is_file()
                or file_path.stat().st_size != int(row["size"])
                or sha256_file(file_path) != row["sha256"]
            ):
                return ["checkpoint_manifest_file_mismatch"]
        if len(files) != int(manifest.get("file_count", -1)):
            return ["checkpoint_manifest_file_count_mismatch"]
        if sum(int(row["size"]) for row in files) != int(manifest.get("total_bytes", -1)):
            return ["checkpoint_manifest_total_bytes_mismatch"]
        recorded_names = {str(row["path"]) for row in files}
        local_names = {
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file() and any(fnmatch(item.relative_to(root).as_posix(), pattern) for pattern in REQUIRED_PATTERNS)
        }
        if recorded_names != local_names:
            return ["checkpoint_manifest_coverage_mismatch"]
    except (KeyError, TypeError, ValueError, OSError):
        return ["checkpoint_manifest_invalid"]
    return []


def _validate_stage0_in_preparation_environment(bundle: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    executable = Path(str(manifest.get("preparation_environment", {}).get("values", {}).get("python_executable", "")))
    code_root = Path(str(manifest.get("preparation_code", {}).get("root", "")))
    if not executable.is_file():
        raise ValueError("Stage 0 preparation Python is missing")
    if not code_root.is_dir() or len(code_root.parents) < 3:
        raise ValueError("Stage 0 preparation code root is missing")
    completed = subprocess.run(
        [str(executable), "-m", "research.phase1.lollanet_cma.validate_stage0", str(bundle.resolve())],
        cwd=code_root.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Stage 0 validation failed in its frozen preparation environment: {detail}")
    return json.loads(completed.stdout)


def audit_readiness(charter_path: Path, stage0_bundle: Path) -> dict[str, Any]:
    import torch
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    bundle_manifest_path = stage0_bundle / "manifest.json"
    if not bundle_manifest_path.is_file():
        raise ValueError("Stage 0 bundle manifest is missing")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    stage0_validation = _validate_stage0_in_preparation_environment(stage0_bundle, bundle_manifest)
    access_view = Path(str(charter["qualification"]["access_view"])).resolve()
    access_view_validation = validate_access_view(access_view, "qualification")
    access_view_manifest_path = access_view / "manifest.json"
    if access_view_validation["source_stage0_manifest_sha256"] != sha256_file(bundle_manifest_path):
        raise ValueError("qualification access view was derived from a different Stage 0 bundle")
    environment_errors: list[str] = []
    if not torch.cuda.is_available():
        environment_errors.append("cuda_unavailable")
    try:
        import bitsandbytes as bnb

        if not hasattr(bnb, "nn") or not hasattr(bnb.nn, "Linear4bit"):
            environment_errors.append("bitsandbytes_linear4bit_unavailable")
    except Exception:
        environment_errors.append("bitsandbytes_import_failed")
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 7_500_000_000:
        environment_errors.append("gpu_memory_below_preregistered_8gb_class")
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        environment_errors.append("bfloat16_unsupported")
    candidates: list[dict[str, Any]] = []
    for candidate in charter["candidates"]:
        revision = candidate.get("revision")
        local_value = candidate.get("local_checkpoint")
        local_path = normalized_local_checkpoint(charter_path, local_value)
        errors = list(environment_errors)
        observed_revision = None
        if not revision:
            errors.append("revision_unresolved")
        if local_path is None or not local_path.is_dir():
            errors.append("local_checkpoint_missing")
        else:
            errors.extend(_checkpoint_artifact_errors(local_path))
            errors.extend(_checkpoint_manifest_errors(local_path, candidate, charter_path))
            observed_revision = _local_revision(local_path)
            if not observed_revision:
                errors.append("local_revision_metadata_missing")
            elif revision and observed_revision != revision:
                errors.append("local_revision_mismatch")
        status = "ready" if not errors else "not_run_missing_artifact"
        candidates.append({
            "candidate_id": candidate["id"],
            "model_id": candidate["model_id"],
            "revision_expected": revision,
            "local_checkpoint": str(local_path) if local_path else None,
            "revision_observed": observed_revision,
            "status": status,
            "errors": errors,
        })
    value = {
        "type": "lollanet_cma_qualification_readiness_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "stage0_bundle": {
            "path": str(stage0_bundle.resolve()),
            "manifest_sha256": sha256_file(bundle_manifest_path),
            "bundle_content_sha256": bundle_manifest.get("bundle_content_sha256"),
        },
        "stage0_validation": stage0_validation,
        "access_view": {
            "path": str(access_view),
            "manifest_sha256": sha256_file(access_view_manifest_path),
            "validation": access_view_validation,
            "validator_code": runtime_code_identity(Path(__file__).parents[1] / "access_views"),
        },
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
        "candidates": candidates,
        "ready_candidate_count": sum(row["status"] == "ready" for row in candidates),
        "qualification_gate_status": "ready" if all(row["status"] == "ready" for row in candidates) else "blocked_missing_candidates",
    }
    value["readiness_content_sha256"] = stable_hash(value)
    return value
