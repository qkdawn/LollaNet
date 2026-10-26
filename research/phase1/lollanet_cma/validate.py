from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .common import preparation_environment, read_json, read_jsonl, runtime_code_identity, sha256_file, stable_hash
from .stage0 import VARIANT_FILES, _distribution, _load_meta_models, _load_questions, _prompt, _record_id
from .taxonomy import MODEL_FAMILIES, QUERY_TO_FAMILY, model_family


def _content_hash(value: dict[str, Any], field: str) -> str:
    copy = dict(value)
    claimed = copy.pop(field, None)
    if claimed != stable_hash(copy):
        raise ValueError(f"{field} mismatch")
    return str(claimed)


def validate_stage0(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = read_json(manifest_path)
    _content_hash(manifest, "bundle_content_sha256")
    code = manifest.get("preparation_code", {})
    if not isinstance(code, dict) or not code.get("root") or not code.get("sha256"):
        raise ValueError("Stage 0 bundle lacks preparation code identity")
    code_root = Path(str(code.get("root", "")))
    if not code_root.is_dir() or runtime_code_identity(code_root) != code:
        raise ValueError("Stage 0 preparation code is missing or changed")
    lock = manifest.get("dependency_lock", {})
    lock_path = Path(str(lock.get("path", "")))
    if not lock.get("path") or not lock_path.is_file() or sha256_file(lock_path) != lock.get("sha256"):
        raise ValueError("Stage 0 dependency lock is missing or changed")
    if manifest.get("preparation_environment") != preparation_environment():
        raise ValueError("Stage 0 preparation environment differs from the recorded identity")
    artifacts = manifest["artifacts"]
    for name, source in artifacts.items():
        path = Path(str(source["path"]))
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"bundle artifact is missing or changed: {name}")
    charter_path = Path(str(manifest["charter"]["path"]))
    if not charter_path.is_file() or sha256_file(charter_path) != manifest["charter"]["sha256"]:
        raise ValueError("experiment charter is missing or changed")
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    dataset_manifest = read_json(Path(artifacts["dataset_manifest"]["path"]))
    split_manifest = read_json(Path(artifacts["split_manifest"]["path"]))
    _content_hash(dataset_manifest, "dataset_manifest_content_sha256")
    _content_hash(split_manifest, "split_manifest_content_sha256")
    source_root = Path(dataset_manifest["source"]["source_root"])
    if dataset_manifest.get("query_to_family") != QUERY_TO_FAMILY:
        raise ValueError("dataset manifest query-to-family mapping differs from the frozen taxonomy")
    charter_source = charter["source"]
    manifest_source = dataset_manifest["source"]
    for field in ("dataset_name", "dataset_version", "upstream_repository", "upstream_commit", "archive_sha256", "license"):
        if manifest_source.get(field) != charter_source.get(field):
            raise ValueError(f"dataset manifest source field differs from charter: {field}")
    for name, source in dataset_manifest["source"]["files"].items():
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"frozen source artifact is missing or changed: {name}")
        if source["sha256"] != charter_source["sha256"][name]:
            raise ValueError(f"dataset manifest source hash differs from charter: {name}")
    frozen_from = split_manifest.get("frozen_from")
    if frozen_from is not None:
        frozen_path = Path(str(frozen_from.get("path", "")))
        if not frozen_path.is_file() or sha256_file(frozen_path) != frozen_from.get("sha256"):
            raise ValueError("frozen split provenance is missing or changed")
        frozen_value = read_json(frozen_path)
        frozen_content_hash = frozen_value.pop("split_manifest_content_sha256", None)
        if frozen_content_hash != stable_hash(frozen_value) or frozen_content_hash != frozen_from.get("content_sha256"):
            raise ValueError("frozen split provenance content hash mismatch")

    meta_models = _load_meta_models(source_root / charter["source"]["meta_model_file"])
    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, file_name in VARIANT_FILES.items():
        rows = _load_questions(source_root / file_name)
        if dataset_manifest["source_distributions"].get(variant) != _distribution(rows):
            raise ValueError(f"dataset manifest source distribution mismatch: {variant}")
        source_rows[variant] = {_record_id(variant, row): row for row in rows}
    prompts = read_jsonl(Path(artifacts["prompts"]["path"]))
    controller = read_jsonl(Path(artifacts["controller"]["path"]))
    evaluator = read_jsonl(Path(artifacts["evaluator"]["path"]))
    prompt_ids = [str(row.get("record_id")) for row in prompts]
    controller_ids = [str(row.get("record_id")) for row in controller]
    evaluator_ids = [str(row.get("record_id")) for row in evaluator]
    if any(len(ids) != len(set(ids)) for ids in (prompt_ids, controller_ids, evaluator_ids)):
        raise ValueError("Stage 0 artifacts contain duplicate record IDs")
    if set(prompt_ids) != set(controller_ids) or set(prompt_ids) != set(evaluator_ids):
        raise ValueError("prompt/controller/evaluator record coverage differs")
    if len(prompt_ids) != int(manifest["record_count"]):
        raise ValueError("bundle record_count mismatch")
    controller_by_id = {row["record_id"]: row for row in controller}
    evaluator_by_id = {row["record_id"]: row for row in evaluator}
    split_model_ids: dict[str, set[str]] = {}
    observed_split_ids: dict[str, list[str]] = {}
    family_counts: dict[str, Counter[str]] = {}
    selected_source_rows: dict[str, list[dict[str, Any]]] = {}
    for prompt_row in prompts:
        if set(prompt_row) != {"record_id", "source_variant", "split", "prompt"}:
            raise ValueError("prompt rows expose fields outside the blinded input contract")
        record_id = str(prompt_row["record_id"])
        variant = str(prompt_row["source_variant"])
        split = str(prompt_row["split"])
        if record_id not in source_rows.get(variant, {}):
            raise ValueError(f"prompt record does not map to frozen source: {record_id}")
        source = source_rows[variant][record_id]
        selected_source_rows.setdefault(split, []).append(source)
        meta = source["meta"]
        model_id = str(meta["model_id"])
        expected_prompt = _prompt(str(meta_models[model_id]["background"]), source)
        if prompt_row["prompt"] != expected_prompt:
            raise ValueError(f"prompt differs from background + given_info + question: {record_id}")
        control = controller_by_id[record_id]
        if set(control) != {"record_id", "model_family", "query_type"}:
            raise ValueError("controller rows violate the oracle-only contract")
        query_type = str(meta["query_type"])
        if control["query_type"] != query_type or control["model_family"] != model_family(query_type):
            raise ValueError(f"controller mapping mismatch: {record_id}")
        score = evaluator_by_id[record_id]
        if set(score) != {"record_id", "answer", "rung", "graph_id", "model_id", "groundtruth"}:
            raise ValueError("evaluator rows violate the scoring-only contract")
        if (
            score["answer"] != str(source["answer"]).lower()
            or int(score["rung"]) != int(meta["rung"])
            or score["graph_id"] != str(meta["graph_id"])
            or score["model_id"] != model_id
            or score["groundtruth"] != meta["groundtruth"]
        ):
            raise ValueError(f"evaluator mismatch: {record_id}")
        observed_split_ids.setdefault(split, []).append(record_id)
        split_model_ids.setdefault(split, set()).add(model_id)
        family_counts.setdefault(split, Counter())[control["model_family"]] += 1
    split_names = list(split_model_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = split_model_ids[left] & split_model_ids[right]
            if overlap:
                raise ValueError(f"model_id leakage between {left} and {right}: {sorted(overlap)[:10]}")
    main_targets = charter["split"]["main_family_targets"]
    ood_target = int(charter["split"]["ood_family_target"])
    for split, counts in family_counts.items():
        expected = ood_target if split.startswith("ood_") else int(main_targets[split])
        if set(counts) != set(MODEL_FAMILIES) or any(counts[family] != expected for family in MODEL_FAMILIES):
            raise ValueError(f"family balance mismatch for {split}: {counts}")
        frozen = split_manifest["splits"][split]
        if frozen["record_ids"] != observed_split_ids[split] or frozen["record_ids_sha256"] != stable_hash(observed_split_ids[split]):
            raise ValueError(f"split record identity mismatch: {split}")
        if frozen["model_ids"] != sorted(split_model_ids[split]):
            raise ValueError(f"split model-group identity mismatch: {split}")
        if frozen["distribution"] != _distribution(selected_source_rows[split]):
            raise ValueError(f"split distribution mismatch: {split}")
    return {
        "status": "valid",
        "record_count": len(prompt_ids),
        "splits": {split: dict(sorted(counts.items())) for split, counts in sorted(family_counts.items())},
        "model_id_overlap": 0,
        "prompt_contract": "background_plus_given_info_plus_question_only",
        "controller_contract": "query_type_and_model_family_only",
        "evaluator_contract": "answer_and_evaluation_metadata_only",
    }
