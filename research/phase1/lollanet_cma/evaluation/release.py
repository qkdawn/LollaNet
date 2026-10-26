from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..validate import validate_stage0
from .preflight import _verified_content


def _verified_artifact(source: dict[str, Any], label: str) -> Path:
    path = Path(str(source.get("path", "")))
    if not path.is_file() or sha256_file(path) != source.get("sha256"):
        raise ValueError(f"{label} is missing or changed")
    return path


def validate_released_view(
    view_dir: Path,
    expected_split: str,
    expected_preflight_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = view_dir / "manifest.json"
    manifest = _verified_content(
        read_json(manifest_path), "released_view_content_sha256", "released evaluation view",
    )
    if manifest.get("type") != "lollanet_cma_released_evaluation_view_v1":
        raise ValueError("released evaluation view has an unexpected type")
    if manifest.get("split") != expected_split:
        raise ValueError("released evaluation view has a different split")
    if (
        expected_preflight_sha256 is not None
        and manifest.get("authorization", {}).get("preflight_sha256") != expected_preflight_sha256
    ):
        raise ValueError("released evaluation view used a different authorization")
    source = manifest["source_stage0"]
    source_path = Path(source["manifest_path"])
    if not source_path.is_file() or sha256_file(source_path) != source["manifest_sha256"]:
        raise ValueError("released evaluation source manifest is missing or changed")
    source_manifest = _verified_content(
        read_json(source_path), "bundle_content_sha256", "released evaluation source bundle",
    )
    if source_manifest["bundle_content_sha256"] != source["bundle_content_sha256"]:
        raise ValueError("released evaluation source content differs")
    authorization = manifest["authorization"]
    preflight_path = Path(authorization["preflight_path"])
    development_path = Path(authorization["development_manifest_path"])
    if (
        not preflight_path.is_file()
        or sha256_file(preflight_path) != authorization["preflight_sha256"]
        or not development_path.is_file()
        or sha256_file(development_path) != authorization["development_manifest_sha256"]
    ):
        raise ValueError("released evaluation authorization is missing or changed")
    preflight = _verified_content(
        read_json(preflight_path),
        "sealed_release_preflight_content_sha256",
        "released evaluation authorization",
    )
    if (
        preflight.get("sealed_release_gate_status") != "ready"
        or not preflight.get("unseal_authorized")
        or preflight.get("development", {}).get("manifest_sha256")
        != authorization["development_manifest_sha256"]
    ):
        raise ValueError("released evaluation authorization is not ready or does not match")
    paths = {
        name: _verified_artifact(manifest["artifacts"][name], f"released view {name}")
        for name in ("prompts", "controller", "evaluator")
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    ids = {name: [str(row["record_id"]) for row in value] for name, value in rows.items()}
    if ids["prompts"] != ids["controller"] or ids["prompts"] != ids["evaluator"]:
        raise ValueError("released evaluation artifacts do not share one ordered identity")
    if (
        len(ids["prompts"]) != len(set(ids["prompts"]))
        or len(ids["prompts"]) != int(manifest["record_count"])
        or stable_hash(ids["prompts"]) != manifest["record_ids_sha256"]
        or any(row.get("split") != expected_split for row in rows["prompts"])
    ):
        raise ValueError("released evaluation record coverage is invalid")
    return {
        "status": "valid",
        "split": expected_split,
        "record_count": len(ids["prompts"]),
        "record_ids_sha256": manifest["record_ids_sha256"],
        "source_stage0_manifest_sha256": manifest["source_stage0"]["manifest_sha256"],
        "development_manifest_sha256": authorization["development_manifest_sha256"],
    }


def _materialize_split(
    split: str,
    split_contract: dict[str, Any],
    families: list[str],
    source_manifest: dict[str, Any],
    preflight_path: Path,
    preflight: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    preflight_sha256 = sha256_file(preflight_path)
    if (output_dir / "manifest.json").is_file():
        validate_released_view(output_dir, split, preflight_sha256)
        return read_json(output_dir / "manifest.json")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing partial untracked released view: {output_dir}")

    source_paths = {
        name: _verified_artifact(source_manifest["artifacts"][name], f"Stage 0 {name}")
        for name in ("prompts", "controller", "evaluator")
    }
    prompts = [row for row in read_jsonl(source_paths["prompts"]) if row.get("split") == split]
    record_ids = [str(row["record_id"]) for row in prompts]
    if len(record_ids) != int(split_contract["expected_records"]) or len(record_ids) != len(set(record_ids)):
        raise ValueError(f"released split record count or uniqueness differs: {split}")
    allowed = set(record_ids)
    controller_by_id = {
        str(row["record_id"]): row for row in read_jsonl(source_paths["controller"])
        if str(row["record_id"]) in allowed
    }
    evaluator_by_id = {
        str(row["record_id"]): row for row in read_jsonl(source_paths["evaluator"])
        if str(row["record_id"]) in allowed
    }
    if set(controller_by_id) != allowed or set(evaluator_by_id) != allowed:
        raise ValueError(f"released split has incomplete labels or routing metadata: {split}")
    controllers = [controller_by_id[record_id] for record_id in record_ids]
    evaluators = [evaluator_by_id[record_id] for record_id in record_ids]
    family_counts = Counter(str(row["model_family"]) for row in controllers)
    expected_per_family = int(split_contract["expected_records_per_family"])
    if set(family_counts) != set(families) or any(
        family_counts[family] != expected_per_family for family in families
    ):
        raise ValueError(f"released split family balance differs: {split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompts": output_dir / "prompts.jsonl",
        "controller": output_dir / "controller.jsonl",
        "evaluator": output_dir / "evaluator.jsonl",
    }
    for name, rows in (("prompts", prompts), ("controller", controllers), ("evaluator", evaluators)):
        atomic_write_jsonl(paths[name], rows)
    manifest = {
        "type": "lollanet_cma_released_evaluation_view_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "record_count": len(record_ids),
        "record_ids_sha256": stable_hash(record_ids),
        "family_counts": {family: family_counts[family] for family in families},
        "source_stage0": {
            "manifest_path": str((Path(preflight["stage0"]["path"]) / "manifest.json").resolve()),
            "manifest_sha256": preflight["stage0"]["manifest_sha256"],
            "bundle_content_sha256": preflight["stage0"]["bundle_content_sha256"],
        },
        "authorization": {
            "preflight_path": str(preflight_path.resolve()),
            "preflight_sha256": preflight_sha256,
            "development_manifest_path": preflight["development"]["manifest_path"],
            "development_manifest_sha256": preflight["development"]["manifest_sha256"],
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "code": preflight["code"],
        "dependency_lock": preflight["dependency_lock"],
        "environment": preflight["environment"],
    }
    manifest["released_view_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    validate_released_view(output_dir, split, preflight_sha256)
    return manifest


def release_evaluation_views(
    charter_path: Path,
    stage0_bundle: Path,
    preflight_path: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    preflight = _verified_content(
        read_json(preflight_path),
        "sealed_release_preflight_content_sha256",
        "sealed release preflight",
    )
    if (
        preflight.get("type") != "lollanet_cma_sealed_release_preflight_v1"
        or preflight.get("sealed_release_gate_status") != "ready"
        or not preflight.get("unseal_authorized")
        or preflight.get("errors")
    ):
        raise ValueError("sealed release preflight is not ready")
    if (
        preflight.get("charter", {}).get("sha256") != sha256_file(charter_path)
        or preflight.get("code") != runtime_code_identity(Path(__file__).parent)
        or preflight.get("dependency_lock") != dependency_lock()
        or preflight.get("environment") != runtime_environment()
    ):
        raise ValueError("sealed release code, charter, dependency lock, or runtime changed after preflight")
    source_manifest_path = stage0_bundle / "manifest.json"
    if (
        preflight.get("stage0", {}).get("path") != str(stage0_bundle.resolve())
        or preflight.get("stage0", {}).get("manifest_sha256") != sha256_file(source_manifest_path)
    ):
        raise ValueError("sealed release Stage 0 bundle differs from preflight")
    source_manifest = _verified_content(
        read_json(source_manifest_path), "bundle_content_sha256", "Stage 0 bundle",
    )
    validation = validate_stage0(stage0_bundle)
    if validation != preflight["stage0"]["validation"]:
        raise ValueError("Stage 0 validation differs from the authorized preflight")

    views: dict[str, Any] = {}
    for split, contract in charter["release"]["splits"].items():
        output_dir = Path(str(contract["output"]))
        view = _materialize_split(
            split,
            contract,
            list(charter["release"]["families"]),
            source_manifest,
            preflight_path,
            preflight,
            output_dir,
        )
        views[split] = {
            "path": str(output_dir.resolve()),
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            "record_count": view["record_count"],
            "record_ids_sha256": view["record_ids_sha256"],
        }
    result = {
        "type": "lollanet_cma_evaluation_release_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "released_after_development_freeze",
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "preflight": {"path": str(preflight_path.resolve()), "sha256": sha256_file(preflight_path)},
        "development_manifest_sha256": preflight["development"]["manifest_sha256"],
        "stage0_manifest_sha256": sha256_file(source_manifest_path),
        "stage0_validation": validation,
        "views": views,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
    }
    result["evaluation_release_content_sha256"] = stable_hash(result)
    if output_manifest_path.is_file():
        existing = _verified_content(
            read_json(output_manifest_path),
            "evaluation_release_content_sha256",
            "evaluation release",
        )
        comparable = dict(existing)
        comparable.pop("created_at", None)
        expected = dict(result)
        expected.pop("created_at", None)
        expected.pop("evaluation_release_content_sha256", None)
        comparable.pop("evaluation_release_content_sha256", None)
        if comparable != expected:
            raise ValueError("existing evaluation release differs from current authorized views")
        return existing
    atomic_write_json(output_manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Release sealed LollaNet-CMA evaluation views.")
    parser.add_argument(
        "--charter", type=Path, default=Path(__file__).with_name("sealed_release_charter.yaml"),
    )
    parser.add_argument("--stage0-bundle", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = release_evaluation_views(
        args.charter, args.stage0_bundle, args.preflight, args.output_manifest,
    )
    print(result["status"])


if __name__ == "__main__":
    main()
