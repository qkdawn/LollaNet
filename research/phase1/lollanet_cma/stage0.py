from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    preparation_environment,
    read_json,
    runtime_code_identity,
    sha256_file,
    stable_hash,
)
from .splitter import Group, allocate_groups, build_groups
from .taxonomy import MODEL_FAMILIES, QUERY_TO_FAMILY, model_family


VARIANT_FILES = {
    "main": "cladder-v1-q-balanced.json",
    "ood_anticommonsense": "cladder-v1-q-anticommonsense.json",
    "ood_nonsense": "cladder-v1-q-noncommonsense.json",
}


def _load_questions(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"CLADDER question file must contain a non-empty array: {path}")
    ids: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"CLADDER question row must be an object: {path}")
        meta = row.get("meta")
        required = {"question_id", "given_info", "question", "answer", "reasoning", "meta"}
        if required - set(row) or not isinstance(meta, dict):
            raise ValueError(f"CLADDER question row violates the source schema: {path}")
        if {"query_type", "rung", "graph_id", "model_id", "groundtruth"} - set(meta):
            raise ValueError(f"CLADDER question metadata is incomplete: {path}")
        model_family(str(meta["query_type"]))
        if str(row["answer"]).lower() not in {"yes", "no"}:
            raise ValueError(f"CLADDER answer is not yes/no: {row['question_id']}")
        ids.append(str(row["question_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"CLADDER question IDs are not unique within {path}")
    return value


def _load_meta_models(path: Path) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError("CLADDER meta-model file must contain a non-empty array")
    models = {str(row["model_id"]): row for row in value}
    if len(models) != len(value) or any(not str(row.get("background", "")).strip() for row in value):
        raise ValueError("CLADDER meta-model IDs must be unique and contain background text")
    return models


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "model_ids": len({str(row["meta"]["model_id"]) for row in rows}),
        "families": dict(sorted(Counter(model_family(str(row["meta"]["query_type"])) for row in rows).items())),
        "query_types": dict(sorted(Counter(str(row["meta"]["query_type"]) for row in rows).items())),
        "rungs": dict(sorted(Counter(str(row["meta"]["rung"]) for row in rows).items())),
        "graph_ids": dict(sorted(Counter(str(row["meta"]["graph_id"]) for row in rows).items())),
        "answers": dict(sorted(Counter(str(row["answer"]).lower() for row in rows).items())),
    }


def _flatten(allocation: dict[str, list[Group]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: sorted([row for group in groups for row in group.rows], key=lambda row: str(row["question_id"]))
        for split, groups in allocation.items()
    }


def _record_id(variant: str, row: dict[str, Any]) -> str:
    return f"{variant}:{row['question_id']}"


def _prompt(background: str, row: dict[str, Any]) -> str:
    return "\n\n".join(
        text.strip() for text in (background, str(row["given_info"]), str(row["question"])) if text.strip()
    )


def _reuse_frozen_split(
    frozen_path: Path, variants: dict[str, list[dict[str, Any]]], charter: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    frozen = read_json(frozen_path)
    content_hash = frozen.pop("split_manifest_content_sha256", None)
    if content_hash != stable_hash(frozen):
        raise ValueError("frozen split manifest content hash mismatch")
    frozen["split_manifest_content_sha256"] = content_hash
    sources = {
        variant: {_record_id(variant, row): row for row in rows}
        for variant, rows in variants.items()
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    model_ids_by_split: dict[str, set[str]] = {}
    for split, details in frozen["splits"].items():
        variant = split if split.startswith("ood_") else "main"
        rows: list[dict[str, Any]] = []
        for record_id in details["record_ids"]:
            if record_id not in sources[variant]:
                raise ValueError(f"frozen split references a missing source record: {record_id}")
            rows.append(sources[variant][record_id])
        selected[split] = rows
        model_ids_by_split[split] = {str(row["meta"]["model_id"]) for row in rows}
        if sorted(model_ids_by_split[split]) != details["model_ids"]:
            raise ValueError(f"frozen split model IDs do not match source records: {split}")
        counts = Counter(model_family(str(row["meta"]["query_type"])) for row in rows)
        expected = int(charter["split"]["ood_family_target"]) if split.startswith("ood_") else int(
            charter["split"]["main_family_targets"][split]
        )
        if any(counts[family] != expected for family in MODEL_FAMILIES):
            raise ValueError(f"frozen split family counts violate the current charter: {split}")
    split_names = list(model_ids_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if model_ids_by_split[left] & model_ids_by_split[right]:
                raise ValueError(f"frozen split contains model_id leakage: {left}/{right}")
    provenance = {
        "path": str(frozen_path.resolve()),
        "sha256": sha256_file(frozen_path),
        "content_sha256": content_hash,
    }
    return selected, {"optimizer_diagnostics": frozen.get("optimizer_diagnostics", {}), "frozen_from": provenance}


def prepare_stage0(
    charter_path: Path, source_root: Path, output_dir: Path, frozen_split_manifest: Path | None = None,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    source = charter["source"]
    if str(source["upstream_commit"]) != "3d2d1169b4b939a09048a6a75956c8972a93cc38":
        raise ValueError("Stage 0 only accepts the reviewed frozen CLADDER commit")
    meta_path = source_root / str(source["meta_model_file"])
    expected_files = {**VARIANT_FILES, "meta_models": str(source["meta_model_file"])}
    for key, name in expected_files.items():
        path = source_root / name
        expected_hash = str(source["sha256"][key])
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"frozen CLADDER artifact is missing or changed: {path}")
    meta_models = _load_meta_models(meta_path)
    variants = {name: _load_questions(source_root / file_name) for name, file_name in VARIANT_FILES.items()}
    for variant, rows in variants.items():
        missing = sorted({str(row["meta"]["model_id"]) for row in rows} - set(meta_models))
        if missing:
            raise ValueError(f"{variant} references missing meta-model IDs: {missing[:10]}")

    seed = int(charter["split"]["seed"])
    frozen_provenance = None
    if frozen_split_manifest is not None:
        selected, reuse = _reuse_frozen_split(frozen_split_manifest, variants, charter)
        optimizer_diagnostics = reuse["optimizer_diagnostics"]
        frozen_provenance = reuse["frozen_from"]
    else:
        main_targets = {name: int(value) for name, value in charter["split"]["main_family_targets"].items()}
        main_allocation, main_solver = allocate_groups(build_groups(variants["main"]), main_targets, seed)
        selected_main_model_ids = {
            group.model_id for groups in main_allocation.values() for group in groups
        }
        anti_target = int(charter["split"]["ood_family_target"])
        anti_allocation, anti_solver = allocate_groups(
            build_groups(variants["ood_anticommonsense"], selected_main_model_ids),
            {"ood_anticommonsense": anti_target},
            seed + 1,
        )
        anti_ids = {group.model_id for group in anti_allocation["ood_anticommonsense"]}
        nonsense_allocation, nonsense_solver = allocate_groups(
            build_groups(variants["ood_nonsense"], selected_main_model_ids | anti_ids),
            {"ood_nonsense": anti_target},
            seed + 2,
        )
        selected = {
            **_flatten(main_allocation),
            **_flatten(anti_allocation),
            **_flatten(nonsense_allocation),
        }
        optimizer_diagnostics = {
            "main": main_solver,
            "ood_anticommonsense": anti_solver,
            "ood_nonsense": nonsense_solver,
        }

    prompts: list[dict[str, Any]] = []
    controller: list[dict[str, Any]] = []
    evaluator: list[dict[str, Any]] = []
    split_records: dict[str, dict[str, Any]] = {}
    for split, rows in selected.items():
        variant = split if split.startswith("ood_") else "main"
        record_ids: list[str] = []
        for row in rows:
            record_id = _record_id(variant, row)
            meta = row["meta"]
            model_id = str(meta["model_id"])
            record_ids.append(record_id)
            prompts.append({
                "record_id": record_id,
                "source_variant": variant,
                "split": split,
                "prompt": _prompt(str(meta_models[model_id]["background"]), row),
            })
            controller.append({
                "record_id": record_id,
                "model_family": model_family(str(meta["query_type"])),
                "query_type": str(meta["query_type"]),
            })
            evaluator.append({
                "record_id": record_id,
                "answer": str(row["answer"]).lower(),
                "rung": int(meta["rung"]),
                "graph_id": str(meta["graph_id"]),
                "model_id": model_id,
                "groundtruth": meta["groundtruth"],
            })
        split_records[split] = {
            "record_ids": record_ids,
            "record_ids_sha256": stable_hash(record_ids),
            "model_ids": sorted({str(row["meta"]["model_id"]) for row in rows}),
            "distribution": _distribution(rows),
        }
    prompts.sort(key=lambda row: row["record_id"])
    controller.sort(key=lambda row: row["record_id"])
    evaluator.sort(key=lambda row: row["record_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.jsonl"
    controller_path = output_dir / "controller.jsonl"
    evaluator_path = output_dir / "evaluator.jsonl"
    atomic_write_jsonl(prompts_path, prompts)
    atomic_write_jsonl(controller_path, controller)
    atomic_write_jsonl(evaluator_path, evaluator)

    dataset_manifest = {
        "type": "lollanet_cma_dataset_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_name": source["dataset_name"],
            "dataset_version": source["dataset_version"],
            "upstream_repository": source["upstream_repository"],
            "upstream_commit": source["upstream_commit"],
            "archive_sha256": source["archive_sha256"],
            "license": source["license"],
            "source_root": str(source_root.resolve()),
            "files": {
                key: {"path": str((source_root / name).resolve()), "sha256": sha256_file(source_root / name)}
                for key, name in expected_files.items()
            },
        },
        "source_distributions": {name: _distribution(rows) for name, rows in variants.items()},
        "query_to_family": QUERY_TO_FAMILY,
    }
    dataset_manifest["dataset_manifest_content_sha256"] = stable_hash(dataset_manifest)
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    atomic_write_json(dataset_manifest_path, dataset_manifest)
    split_manifest = {
        "type": "lollanet_cma_split_manifest_v1",
        "seed": seed,
        "group_key": "model_id",
        "optimizer": "scipy.optimize.milp_highs",
        "optimizer_diagnostics": optimizer_diagnostics,
        "frozen_from": frozen_provenance,
        "family_order": list(MODEL_FAMILIES),
        "splits": split_records,
    }
    split_manifest["split_manifest_content_sha256"] = stable_hash(split_manifest)
    split_manifest_path = output_dir / "split_manifest.json"
    atomic_write_json(split_manifest_path, split_manifest)
    manifest = {
        "type": "lollanet_cma_stage0_bundle_v1",
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "preparation_code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": {
            "path": str(Path(__file__).with_name("uv.lock").resolve()),
            "sha256": sha256_file(Path(__file__).with_name("uv.lock")),
        },
        "preparation_environment": preparation_environment(),
        "artifacts": {
            "dataset_manifest": {"path": str(dataset_manifest_path), "sha256": sha256_file(dataset_manifest_path)},
            "split_manifest": {"path": str(split_manifest_path), "sha256": sha256_file(split_manifest_path)},
            "prompts": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
            "controller": {"path": str(controller_path), "sha256": sha256_file(controller_path)},
            "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
        },
        "record_count": len(prompts),
    }
    manifest["bundle_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
