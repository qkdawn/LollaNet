from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file, stable_hash
from .prompts import OPTION_LABELS, TEMPLATES, condition_specs, render_prompt

REQUIRED_SOURCE_FIELDS = {"question_id", "question", "options", "answer_index", "category"}
FORBIDDEN_PROMPT_FIELDS = {"gold_index", "lure_index", "gold_label", "lure_label", "answer", "answer_index", "cot_content", "src"}
OPTION_ORDER_POLICY = "sha256_seeded_per_base_item_v1"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return value


def load_source(source_jsonl: Path | None, dataset: str, revision: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_jsonl is not None:
        rows = read_jsonl(source_jsonl)
        return rows, {"kind": "jsonl", "path": str(source_jsonl.resolve()), "sha256": sha256_file(source_jsonl)}
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the pilot environment or pass --source-jsonl") from exc
    loaded = load_dataset(dataset, split=split, revision=revision)
    return [dict(row) for row in loaded], {"kind": "huggingface", "dataset": dataset, "revision": revision, "split": split}


def _normalized_category(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _validate_source_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_SOURCE_FIELDS - set(row)
    if missing:
        raise ValueError(f"source row missing {sorted(missing)}")
    if not isinstance(row["options"], list) or not 2 <= len(row["options"]) <= len(OPTION_LABELS):
        raise ValueError("source options must contain 2-10 choices")
    answer_index = int(row["answer_index"])
    if not 0 <= answer_index < len(row["options"]):
        raise ValueError("answer_index outside options")


def load_lures(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("selector"), dict):
        raise ValueError("lure file must declare an independent selector")
    selector = payload["selector"]
    if selector.get("uses_target_model_outputs") is not False:
        raise ValueError("lure selector must explicitly deny target-model output use")
    selector_id = str(selector.get("id", "")).strip()
    frozen_at = str(selector.get("frozen_at", "")).strip()
    method = str(selector.get("method", "")).strip()
    placeholders = {"replace-with-independent-selector-id", "YYYY-MM-DDTHH:MM:SSZ", "REPLACE_WITH_ADJUDICATION_TIMESTAMP"}
    if not selector_id or not frozen_at or not method or selector_id in placeholders or frozen_at in placeholders:
        raise ValueError("lure selector requires non-placeholder id, method, and frozen_at")
    try:
        parsed_frozen_at = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("lure selector frozen_at must be ISO-8601") from exc
    if parsed_frozen_at.tzinfo is None:
        raise ValueError("lure selector frozen_at must include a timezone")
    if selector_id == "human-blinded-ranking-v1":
        required_provenance = {
            "annotation_manifest_sha256", "mapping_sha256", "primary_annotation_sha256", "agreement_status",
        }
        missing_provenance = required_provenance - set(selector)
        if missing_provenance or selector.get("agreement_status") != "adjudicated":
            raise ValueError(f"human lure selector lacks adjudicated provenance: {sorted(missing_provenance)}")
        primary_hashes = selector.get("primary_annotation_sha256")
        if not isinstance(primary_hashes, dict) or len(primary_hashes) != 2 or not all(primary_hashes.values()):
            raise ValueError("human lure selector requires exactly two primary annotation hashes")
    raw_items = payload.get("items")
    if not isinstance(raw_items, dict):
        raise ValueError("lure file items must map question_id to lure_index")
    return {str(key): int(value) for key, value in raw_items.items()}, {
        "path": str(path.resolve()), "sha256": sha256_file(path), "selector": selector,
    }


def permute_options(
    options: list[str], gold_index: int, lure_index: int, question_id: str, seed: Any,
) -> tuple[list[str], int, int]:
    order = sorted(
        range(len(options)),
        key=lambda index: stable_hash([str(seed), "option-order", str(question_id), int(index)]),
    )
    remap = {source_index: display_index for display_index, source_index in enumerate(order)}
    return [str(options[index]) for index in order], remap[int(gold_index)], remap[int(lure_index)]


def select_base_items(rows: list[dict[str, Any]], config: dict[str, Any], lures: dict[str, int]) -> list[dict[str, Any]]:
    data_config = config["data"]
    if data_config.get("option_order_policy") != OPTION_ORDER_POLICY:
        raise ValueError(f"option_order_policy must be {OPTION_ORDER_POLICY!r}")
    categories = [_normalized_category(value) for value in data_config["categories"]]
    per_category = int(data_config["items_per_category"])
    discovery_count = int(data_config["discovery_per_category"])
    holdout_count = int(data_config["holdout_per_category"])
    if discovery_count + holdout_count != per_category:
        raise ValueError("discovery and holdout counts must equal items_per_category")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        _validate_source_row(raw)
        category = _normalized_category(raw["category"])
        question_id = str(raw["question_id"])
        if category in categories and question_id in lures:
            lure_index = lures[question_id]
            if lure_index == int(raw["answer_index"]) or not 0 <= lure_index < len(raw["options"]):
                raise ValueError(f"invalid lure for {question_id}")
            grouped[category].append(raw)
    selected: list[dict[str, Any]] = []
    seed = str(config["seed"])
    for category in categories:
        candidates = sorted(grouped[category], key=lambda row: stable_hash([seed, category, str(row["question_id"])]))
        if len(candidates) < per_category:
            raise ValueError(f"category {category!r} has {len(candidates)} lure-covered items; need {per_category}")
        for position, raw in enumerate(candidates[:per_category]):
            question_id = str(raw["question_id"])
            options, gold_index, lure_index = permute_options(
                [str(value) for value in raw["options"]], int(raw["answer_index"]),
                int(lures[question_id]), question_id, config["seed"],
            )
            selected.append({
                "base_item_id": question_id,
                "question": str(raw["question"]),
                "options": options,
                "gold_index": gold_index,
                "lure_index": lure_index,
                "category": category,
                "split": "discovery" if position < discovery_count else "pilot_holdout",
                "template_family": "A" if position < discovery_count else "B",
            })
    return sorted(selected, key=lambda row: (row["split"], row["category"], row["base_item_id"]))


def materialize_records(base_items: list[dict[str, Any]], token_budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    evaluator: list[dict[str, Any]] = []
    for item in base_items:
        option_hash = stable_hash(item["options"])
        for spec in condition_specs():
            condition_key = f"r{spec['reward_lure']}s{spec['social_lure']}c{spec['certainty_pressure']}-{spec['control_type']}"
            record_id = stable_hash([item["base_item_id"], item["split"], item["template_family"], condition_key])[:24]
            prompt_row = {
                "record_id": record_id, "base_item_id": item["base_item_id"], "split": item["split"],
                "category": item["category"], "template_family": item["template_family"],
                "certainty_keyword_free": bool(TEMPLATES[item["template_family"]]["certainty_keyword_free"]),
                **spec, "question": item["question"], "options": item["options"],
                "prompt": render_prompt(item["question"], item["options"], item["template_family"], item["lure_index"], item["gold_index"], spec, token_budget),
                "option_order_sha256": option_hash, "token_budget": token_budget,
            }
            if FORBIDDEN_PROMPT_FIELDS & set(prompt_row):
                raise AssertionError("evaluator-only field escaped into prompt row")
            prompts.append(prompt_row)
            evaluator.append({
                "record_id": record_id, "base_item_id": item["base_item_id"],
                "gold_index": item["gold_index"], "lure_index": item["lure_index"],
                "gold_label": OPTION_LABELS[item["gold_index"]], "lure_label": OPTION_LABELS[item["lure_index"]],
            })
    return prompts, evaluator


def build_pilot(config_path: Path, lure_path: Path, output_dir: Path, source_jsonl: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    rows, source = load_source(source_jsonl, config["data"]["dataset"], config["data"]["revision"], config["data"]["source_split"])
    lures, lure_audit = load_lures(lure_path)
    base_items = select_base_items(rows, config, lures)
    prompts, evaluator = materialize_records(base_items, int(config["data"]["token_budget"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.jsonl"
    evaluator_path = output_dir / "evaluator.jsonl"
    atomic_write_jsonl(prompts_path, prompts)
    atomic_write_jsonl(evaluator_path, evaluator)
    manifest = {
        "type": "lollasteer_q_pilot_manifest", "version": config["version"], "status": "pilot_unsealed",
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "source": source, "lures": lure_audit,
        "prompt_template_sha256": stable_hash(TEMPLATES),
        "model_preregistration": config["model"], "seed": config["seed"],
        "counts": {"base_items": len(base_items), "prompt_records": len(prompts), "evaluator_records": len(evaluator)},
        "artifacts": {},
    }
    manifest["artifacts"] = {
        "prompts": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
        "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
    }
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
