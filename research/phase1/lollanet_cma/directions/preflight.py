from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..common import atomic_write_json, read_json, read_jsonl, sha256_file, stable_hash
from ..access_views.views import validate_access_view
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment


def _verified_content(value: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    content = dict(value)
    claimed = content.pop(field, None)
    if claimed != stable_hash(content):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _verify_artifact(source: dict[str, Any], label: str) -> Path:
    path = Path(str(source.get("path", "")))
    if not path.is_file() or sha256_file(path) != source.get("sha256"):
        raise ValueError(f"{label} artifact is missing or changed")
    return path


def _verify_checkpoint_files(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("selected checkpoint manifest is missing")
    manifest = _verified_content(
        read_json(manifest_path), "checkpoint_content_sha256", "selected checkpoint",
    )
    files = manifest.get("files", [])
    if not files or len(files) != int(manifest.get("file_count", -1)):
        raise ValueError("selected checkpoint manifest has invalid file coverage")
    observed_total = 0
    covered: set[str] = set()
    root = checkpoint.resolve()
    for row in files:
        relative = str(row.get("path", ""))
        path = (root / relative).resolve()
        if root not in path.parents or relative in covered:
            raise ValueError("selected checkpoint manifest contains a path escape or duplicate")
        covered.add(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"selected checkpoint file is missing or changed: {relative}")
        observed_total += path.stat().st_size
    if observed_total != int(manifest.get("total_bytes", -1)):
        raise ValueError("selected checkpoint total size differs from its manifest")
    return manifest


def _verify_selection(selection_path: Path, qualification_charter_path: Path) -> dict[str, Any]:
    selection = _verified_content(
        read_json(selection_path), "qualification_content_sha256", "qualification selection",
    )
    if selection.get("type") != "lollanet_cma_base_model_qualification_v1":
        raise ValueError("qualification selection has an unexpected type")
    if selection.get("status") != "frozen_before_vector_training":
        raise ValueError("qualification selection is not frozen before vector training")
    if sha256_file(qualification_charter_path) != selection.get("charter_sha256"):
        raise ValueError("qualification selection charter differs from the frozen charter")

    charter = yaml.safe_load(qualification_charter_path.read_text(encoding="utf-8"))
    expected = {row["id"]: row for row in charter["candidates"]}
    selected_id = str(selection.get("selected_candidate_id", ""))
    candidate = expected.get(selected_id)
    if candidate is None:
        raise ValueError("qualification selection names an unregistered candidate")
    if (
        selection.get("selected_model_id") != candidate["model_id"]
        or selection.get("selected_revision") != candidate["revision"]
    ):
        raise ValueError("qualification selection identity differs from the frozen candidate")

    sources = selection.get("sources", [])
    if len(sources) != len(expected):
        raise ValueError("qualification selection does not reference every candidate run")
    manifests: dict[str, dict[str, Any]] = {}
    for source in sources:
        path = Path(str(source.get("manifest_path", "")))
        if not path.is_file() or sha256_file(path) != source.get("manifest_sha256"):
            raise ValueError("qualification source manifest is missing or changed")
        manifest = _verified_content(read_json(path), "run_content_sha256", "qualification run")
        candidate_id = str(manifest.get("candidate_id", ""))
        if candidate_id in manifests or candidate_id not in expected:
            raise ValueError("qualification source manifests have duplicate or unknown candidates")
        frozen = expected[candidate_id]
        if (
            not manifest.get("hard_gate_pass")
            or not all(manifest.get("hard_gates", {}).values())
            or manifest.get("model_id") != frozen["model_id"]
            or manifest.get("revision") != frozen["revision"]
        ):
            raise ValueError("qualification source manifest is not eligible")
        for artifact_name in ("run_intent", "access_view_manifest", "hook_smoke", "predictions"):
            _verify_artifact(
                manifest.get("artifacts", {}).get(artifact_name, {}),
                f"qualification {candidate_id} {artifact_name}",
            )
        manifests[candidate_id] = manifest
    if set(manifests) != set(expected):
        raise ValueError("qualification source manifests do not cover all candidates")
    return selection


def _prompt_length_rows(charter: dict[str, Any], encode: Any) -> list[dict[str, Any]]:
    contract = charter["prompt_contract"]
    instructions = contract["instructions"]
    answer_protocol = str(contract["answer_protocol"])
    limit = float(contract["maximum_token_length_difference_fraction"])
    rows: list[dict[str, Any]] = []
    for family, negative_family in charter["contrast_mapping"].items():
        positive_tokens = len(encode(f"{instructions[family]} {answer_protocol}"))
        negative_tokens = len(encode(f"{instructions[negative_family]} {answer_protocol}"))
        difference = abs(positive_tokens - negative_tokens) / max(positive_tokens, negative_tokens)
        if difference > limit:
            raise ValueError(f"prompt token lengths exceed the frozen limit: {family}")
        rows.append({
            "model_family": family,
            "negative_family": negative_family,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "difference_fraction": difference,
        })
    return rows


def _selected_tokenizer_audit(
    charter: dict[str, Any], qualification_charter_path: Path, selected_candidate_id: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    qualification = yaml.safe_load(qualification_charter_path.read_text(encoding="utf-8"))
    candidate = next(
        (row for row in qualification["candidates"] if row["id"] == selected_candidate_id), None,
    )
    if candidate is None:
        raise ValueError("selected candidate is absent from the qualification charter")
    checkpoint = Path(str(candidate["local_checkpoint"])).resolve()
    if not checkpoint.is_dir():
        raise ValueError("selected checkpoint is missing")
    checkpoint_manifest_value = _verify_checkpoint_files(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    rows = _prompt_length_rows(
        charter, lambda text: tokenizer.encode(text, add_special_tokens=False),
    )
    checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
    if not checkpoint_manifest.is_file():
        raise ValueError("selected checkpoint manifest is missing")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "checkpoint_content_sha256": checkpoint_manifest_value["checkpoint_content_sha256"],
        "maximum_difference_fraction": max(row["difference_fraction"] for row in rows),
        "pairs": rows,
    }


def _vector_train_view(access_view: Path, charter: dict[str, Any]) -> dict[str, Any]:
    frozen_view = Path(str(charter["data"]["access_view"])).resolve()
    if access_view.resolve() != frozen_view:
        raise ValueError("vector_train access-view path differs from the direction charter")
    validation = validate_access_view(access_view, str(charter["data"]["allowed_split"]))
    manifest = _verified_content(
        read_json(access_view / "manifest.json"), "access_view_content_sha256", "vector_train access view",
    )
    artifacts = manifest.get("artifacts", {})
    prompt_path = _verify_artifact(artifacts.get("prompts", {}), "vector_train prompts")
    controller_path = _verify_artifact(artifacts.get("controller", {}), "vector_train controller")
    evaluator_path = _verify_artifact(artifacts.get("evaluator", {}), "vector_train evaluator")

    prompts = read_jsonl(prompt_path)
    controllers = read_jsonl(controller_path)
    evaluators = read_jsonl(evaluator_path)
    controller_by_id = {str(row["record_id"]): row for row in controllers}
    evaluator_by_id = {str(row["record_id"]): row for row in evaluators}
    if len(controller_by_id) != len(controllers) or len(evaluator_by_id) != len(evaluators):
        raise ValueError("Stage 0 controller or evaluator contains duplicate record IDs")

    allowed_split = str(charter["data"]["allowed_split"])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.get("split") != allowed_split:
            raise ValueError("vector_train access view contains a different split")
        record_id = str(prompt["record_id"])
        if record_id in seen or record_id not in controller_by_id or record_id not in evaluator_by_id:
            raise ValueError("vector_train coverage is duplicated or incomplete")
        seen.add(record_id)
        rows.append({
            "record_id": record_id,
            "model_family": str(controller_by_id[record_id]["model_family"]),
            "gold_answer": str(evaluator_by_id[record_id]["answer"]).lower(),
        })

    expected_families = set(charter["families"])
    family_counts = Counter(row["model_family"] for row in rows)
    label_counts = Counter((row["model_family"], row["gold_answer"]) for row in rows)
    expected_total = int(charter["data"]["expected_records"])
    expected_per_family = int(charter["data"]["expected_records_per_family"])
    labels = set(charter["filtering"]["labels"])
    if len(rows) != expected_total or set(family_counts) != expected_families:
        raise ValueError("vector_train has unexpected record or family coverage")
    if any(family_counts[family] != expected_per_family for family in expected_families):
        raise ValueError("vector_train is not balanced by model family")
    if any({label for family, label in label_counts if family == expected} != labels for expected in expected_families):
        raise ValueError("vector_train does not cover both frozen answer labels in every family")
    return {
        "validation": validation,
        "record_count": len(rows),
        "record_ids_sha256": stable_hash([row["record_id"] for row in rows]),
        "family_counts": dict(sorted(family_counts.items())),
        "label_counts": {
            family: {label: label_counts[(family, label)] for label in sorted(labels)}
            for family in sorted(expected_families)
        },
    }


def audit_direction_preflight(
    charter_path: Path,
    qualification_charter_path: Path,
    access_view: Path,
    selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    vector_train = _vector_train_view(access_view, charter)
    errors: list[str] = []
    selection: dict[str, Any] | None = None
    tokenizer_audit: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    if not selection_path.is_file():
        errors.append("qualification_selection_missing")
    else:
        try:
            selection = _verify_selection(selection_path, qualification_charter_path)
            tokenizer_audit = _selected_tokenizer_audit(
                charter, qualification_charter_path, str(selection["selected_candidate_id"]),
            )
            environment = runtime_environment()
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"qualification_selection_invalid:{exc}")

    result = {
        "type": "lollanet_cma_direction_preflight_v1",
        "direction_gate_status": "ready" if not errors else "blocked_missing_qualification",
        "errors": errors,
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": environment,
        "access_view": {
            "path": str(access_view.resolve()),
            "manifest_sha256": sha256_file(access_view / "manifest.json"),
            "validator_code": runtime_code_identity(Path(__file__).parents[1] / "access_views"),
            "vector_train": vector_train,
        },
        "qualification_selection": None if selection is None else {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "candidate_id": selection["selected_candidate_id"],
            "model_id": selection["selected_model_id"],
            "revision": selection["selected_revision"],
            "tokenizer_prompt_length_audit": tokenizer_audit,
        },
        "split_policy": {
            "fit": charter["data"]["allowed_split"],
            "development_selection": charter["development_selection"]["allowed_split"],
            "sealed_forbidden": charter["development_selection"]["forbidden_split"],
        },
        "balancing_policy": charter["filtering"],
    }
    result["preflight_content_sha256"] = stable_hash(result)
    atomic_write_json(output_path, result)
    return result
