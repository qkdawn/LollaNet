from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common import (
    atomic_write_json,
    atomic_write_jsonl,
    preparation_environment,
    read_json,
    read_jsonl,
    runtime_code_identity,
    sha256_file,
    stable_hash,
)


ALLOWED_PRESEALED_SPLITS = {"qualification", "vector_train", "development"}


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


def create_access_view(stage0_bundle: Path, split: str, output_dir: Path) -> dict[str, Any]:
    if split not in ALLOWED_PRESEALED_SPLITS:
        raise ValueError(f"split is not allowed before sealed evaluation: {split}")
    source_manifest_path = stage0_bundle / "manifest.json"
    source_manifest = _verified_content(
        read_json(source_manifest_path), "bundle_content_sha256", "Stage 0 bundle",
    )
    source_artifacts = source_manifest["artifacts"]
    prompt_path = _verified_artifact(source_artifacts["prompts"], "Stage 0 prompts")
    controller_path = _verified_artifact(source_artifacts["controller"], "Stage 0 controller")
    evaluator_path = _verified_artifact(source_artifacts["evaluator"], "Stage 0 evaluator")

    prompts = [row for row in read_jsonl(prompt_path) if row.get("split") == split]
    record_ids = [str(row["record_id"]) for row in prompts]
    if not record_ids or len(record_ids) != len(set(record_ids)):
        raise ValueError("source split is empty or contains duplicate record IDs")
    allowed_ids = set(record_ids)
    controller_by_id = {
        str(row["record_id"]): row for row in read_jsonl(controller_path)
        if str(row["record_id"]) in allowed_ids
    }
    evaluator_by_id = {
        str(row["record_id"]): row for row in read_jsonl(evaluator_path)
        if str(row["record_id"]) in allowed_ids
    }
    if set(controller_by_id) != allowed_ids or set(evaluator_by_id) != allowed_ids:
        raise ValueError("source split has incomplete controller or evaluator coverage")
    controllers = [controller_by_id[record_id] for record_id in record_ids]
    evaluators = [evaluator_by_id[record_id] for record_id in record_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompts": output_dir / "prompts.jsonl",
        "controller": output_dir / "controller.jsonl",
        "evaluator": output_dir / "evaluator.jsonl",
    }
    atomic_write_jsonl(paths["prompts"], prompts)
    atomic_write_jsonl(paths["controller"], controllers)
    atomic_write_jsonl(paths["evaluator"], evaluators)
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    manifest = {
        "type": "lollanet_cma_split_access_view_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "record_count": len(record_ids),
        "record_ids_sha256": stable_hash(record_ids),
        "source_stage0": {
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
            "bundle_content_sha256": source_manifest["bundle_content_sha256"],
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "preparation_code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "preparation_environment": preparation_environment(),
    }
    manifest["access_view_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_access_view(view_dir: Path, expected_split: str | None = None) -> dict[str, Any]:
    manifest_path = view_dir / "manifest.json"
    manifest = _verified_content(
        read_json(manifest_path), "access_view_content_sha256", "access view",
    )
    if manifest.get("type") != "lollanet_cma_split_access_view_v1":
        raise ValueError("access view has an unexpected type")
    split = str(manifest.get("split", ""))
    if expected_split is not None and split != expected_split:
        raise ValueError("access view split differs from the requested split")
    if split not in ALLOWED_PRESEALED_SPLITS:
        raise ValueError("access view exposes a split not allowed before sealed evaluation")

    source = manifest["source_stage0"]
    source_path = Path(source["manifest_path"])
    if not source_path.is_file() or sha256_file(source_path) != source["manifest_sha256"]:
        raise ValueError("source Stage 0 manifest is missing or changed")
    source_manifest = _verified_content(
        read_json(source_path), "bundle_content_sha256", "source Stage 0 bundle",
    )
    if source_manifest["bundle_content_sha256"] != source["bundle_content_sha256"]:
        raise ValueError("access view source content differs from Stage 0")

    paths = {
        name: _verified_artifact(manifest["artifacts"][name], f"access view {name}")
        for name in ("prompts", "controller", "evaluator")
    }
    prompts = read_jsonl(paths["prompts"])
    controllers = read_jsonl(paths["controller"])
    evaluators = read_jsonl(paths["evaluator"])
    prompt_ids = [str(row["record_id"]) for row in prompts]
    controller_ids = [str(row["record_id"]) for row in controllers]
    evaluator_ids = [str(row["record_id"]) for row in evaluators]
    if any(row.get("split") != split for row in prompts):
        raise ValueError("access view prompts contain a different split")
    if prompt_ids != controller_ids or prompt_ids != evaluator_ids:
        raise ValueError("access view artifacts do not share one ordered record identity")
    if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_ids) != int(manifest["record_count"]):
        raise ValueError("access view record count or uniqueness is invalid")
    if stable_hash(prompt_ids) != manifest["record_ids_sha256"]:
        raise ValueError("access view ordered record identity hash mismatch")
    return {
        "status": "valid",
        "split": split,
        "record_count": len(prompt_ids),
        "record_ids_sha256": manifest["record_ids_sha256"],
        "source_stage0_manifest_sha256": source["manifest_sha256"],
    }

