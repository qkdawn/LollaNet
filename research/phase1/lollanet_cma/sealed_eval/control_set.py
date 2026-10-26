from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash


OPTION_LABELS = tuple("ABCDEFGHIJ")


def _source_record(row: dict[str, Any]) -> dict[str, Any]:
    required = {
        "question_id", "question", "options", "answer", "answer_index",
        "cot_content", "category", "src",
    }
    if set(row) != required:
        raise ValueError("MMLU-Pro source schema differs from the frozen contract")
    options = [str(value).strip() for value in row["options"]]
    answer_index = int(row["answer_index"])
    if not 2 <= len(options) <= len(OPTION_LABELS) or not 0 <= answer_index < len(options):
        raise ValueError("MMLU-Pro options or answer index are invalid")
    answer = str(row["answer"]).strip().upper()
    if answer != OPTION_LABELS[answer_index]:
        raise ValueError("MMLU-Pro answer letter and index disagree")
    return {
        "question_id": int(row["question_id"]),
        "question": str(row["question"]).strip(),
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "cot_content": str(row["cot_content"]),
        "category": str(row["category"]).strip(),
        "src": str(row["src"]).strip(),
    }


def select_control_records(
    rows: Iterable[dict[str, Any]],
    categories: list[str],
    records_per_category: int,
    seed: int,
) -> list[dict[str, Any]]:
    normalized = [_source_record(row) for row in rows]
    question_ids = [row["question_id"] for row in normalized]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("MMLU-Pro question_id values are not unique")
    selected: list[dict[str, Any]] = []
    for category in categories:
        candidates = [row for row in normalized if row["category"] == category]
        if len(candidates) < records_per_category:
            raise ValueError(f"MMLU-Pro category is too small: {category}")
        ranked = sorted(
            candidates,
            key=lambda row: (
                stable_hash({
                    "selection_seed": seed,
                    "source_record_sha256": stable_hash(row),
                }),
                row["question_id"],
            ),
        )
        selected.extend(ranked[:records_per_category])
    return selected


def render_prompt(row: dict[str, Any], template: str) -> str:
    options = "\n".join(
        f"{OPTION_LABELS[index]}. {value}" for index, value in enumerate(row["options"])
    )
    return template.format(category=row["category"], question=row["question"], options=options).strip()


def build_control_artifacts(
    selected: list[dict[str, Any]],
    template: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    controllers: list[dict[str, Any]] = []
    evaluators: list[dict[str, Any]] = []
    for row in selected:
        record_id = f"mmlu-pro-test-{row['question_id']}"
        source_record_sha256 = stable_hash(row)
        prompts.append({
            "record_id": record_id,
            "dataset": "TIGER-Lab/MMLU-Pro",
            "split": "test",
            "category": row["category"],
            "prompt": render_prompt(row, template),
        })
        controllers.append({
            "record_id": record_id,
            "category": row["category"],
            "source_question_id": row["question_id"],
            "source_subset": row["src"],
            "source_record_sha256": source_record_sha256,
        })
        evaluators.append({
            "record_id": record_id,
            "answer": row["answer"],
            "answer_index": row["answer_index"],
            "source_record_sha256": source_record_sha256,
        })
    return prompts, controllers, evaluators


def validate_control_bundle(bundle_dir: Path, charter: dict[str, Any]) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = read_json(manifest_path)
    content_hash = manifest.pop("control_bundle_content_sha256", None)
    if content_hash != stable_hash(manifest):
        raise ValueError("control bundle manifest content changed")
    manifest["control_bundle_content_sha256"] = content_hash
    if manifest.get("type") != "lollanet_cma_unrelated_control_bundle_v1":
        raise ValueError("control bundle has an unexpected type")
    artifacts = manifest["artifacts"]
    paths = {name: Path(artifacts[name]["path"]) for name in ("prompts", "controller", "evaluator")}
    if any(not path.is_file() or sha256_file(path) != artifacts[name]["sha256"] for name, path in paths.items()):
        raise ValueError("control bundle artifact is missing or changed")
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    identities = {name: [str(row["record_id"]) for row in values] for name, values in rows.items()}
    if len({tuple(value) for value in identities.values()}) != 1:
        raise ValueError("control artifacts do not share one ordered identity")
    record_ids = identities["prompts"]
    expected = int(charter["selection"]["records_per_category"]) * len(charter["selection"]["categories"])
    if len(record_ids) != expected or len(record_ids) != len(set(record_ids)):
        raise ValueError("control record coverage is invalid")
    counts = Counter(row["category"] for row in rows["controller"])
    if counts != Counter({
        category: int(charter["selection"]["records_per_category"])
        for category in charter["selection"]["categories"]
    }):
        raise ValueError("control category balance differs from the charter")
    forbidden = {"answer", "answer_index", "cot_content", "source_record_sha256"}
    if any(forbidden.intersection(row) for row in rows["prompts"]):
        raise ValueError("control prompt artifact contains evaluator-only data")
    if stable_hash(record_ids) != manifest["record_ids_sha256"]:
        raise ValueError("control record identity changed")
    return {
        "status": "valid",
        "record_count": len(record_ids),
        "category_counts": dict(sorted(counts.items())),
        "record_ids_sha256": manifest["record_ids_sha256"],
        "control_bundle_content_sha256": content_hash,
    }


def _load_source(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("building the control set requires the datasets package") from exc
    if path.suffix == ".parquet":
        dataset = Dataset.from_parquet(str(path))
    else:
        dataset = Dataset.from_file(str(path))
    return [dict(row) for row in dataset]


def freeze_control_set(
    charter_path: Path,
    source_path: Path,
    dataset_card_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    source = charter["source"]
    if sha256_file(source_path) != source["test_file_sha256"]:
        raise ValueError("MMLU-Pro source file differs from the frozen revision")
    if sha256_file(dataset_card_path) != source["dataset_card_sha256"]:
        raise ValueError("MMLU-Pro dataset card differs from the frozen revision")
    template_path = charter_path.parent / charter["prompt"]["template"]
    if sha256_file(template_path) != charter["prompt"]["template_sha256"]:
        raise ValueError("control prompt template changed")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        validation = validate_control_bundle(output_dir, charter)
        existing = read_json(manifest_path)
        if (
            existing.get("source", {}).get("test_file_sha256") != source["test_file_sha256"]
            or existing.get("source", {}).get("dataset_card_sha256") != source["dataset_card_sha256"]
            or existing.get("selection") != charter["selection"]
            or existing.get("prompt_template_sha256") != charter["prompt"]["template_sha256"]
            or validation["record_count"] != int(charter["selection"]["total_records"])
        ):
            raise ValueError("existing control bundle differs from the frozen charter")
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("refusing to overwrite a partial untracked control bundle")
    rows = _load_source(source_path)
    if len(rows) != int(source["test_records"]):
        raise ValueError("MMLU-Pro test record count differs from the frozen revision")
    selected = select_control_records(
        rows,
        list(charter["selection"]["categories"]),
        int(charter["selection"]["records_per_category"]),
        int(charter["selection"]["seed"]),
    )
    prompts, controllers, evaluators = build_control_artifacts(
        selected, template_path.read_text(encoding="utf-8"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompts": output_dir / "prompts.jsonl",
        "controller": output_dir / "controller.jsonl",
        "evaluator": output_dir / "evaluator.jsonl",
    }
    for name, values in (("prompts", prompts), ("controller", controllers), ("evaluator", evaluators)):
        atomic_write_jsonl(paths[name], values)
    record_ids = [row["record_id"] for row in prompts]
    manifest = {
        "type": "lollanet_cma_unrelated_control_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "split": source["split"],
            "license": source["license"],
            "test_file_path": str(source_path.resolve()),
            "test_file_sha256": sha256_file(source_path),
            "dataset_card_path": str(dataset_card_path.resolve()),
            "dataset_card_sha256": source["dataset_card_sha256"],
        },
        "selection": charter["selection"],
        "prompt_template_sha256": sha256_file(template_path),
        "record_count": len(record_ids),
        "record_ids_sha256": stable_hash(record_ids),
        "selected_source_records_sha256": stable_hash([
            stable_hash(row) for row in selected
        ]),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    manifest["control_bundle_content_sha256"] = stable_hash(manifest)
    atomic_write_json(manifest_path, manifest)
    validate_control_bundle(output_dir, charter)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the independent MMLU-Pro control set.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("control_set_charter.yaml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_control_set(args.charter, args.source, args.dataset_card, args.output)
    print(result["control_bundle_content_sha256"])


if __name__ == "__main__":
    main()
