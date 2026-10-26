from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .data import FORBIDDEN_PROMPT_FIELDS
from .prompts import VALID_CONTROLS


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    prompts_path = bundle_dir / "prompts.jsonl"
    evaluator_path = bundle_dir / "evaluator.jsonl"
    manifest_path = bundle_dir / "manifest.json"
    for path in (prompts_path, evaluator_path, manifest_path):
        if not path.is_file():
            errors.append(f"missing artifact: {path.name}")
    if errors:
        return {"passed": False, "errors": errors}
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = read_jsonl(prompts_path)
    evaluator = read_jsonl(evaluator_path)
    expected = manifest.get("artifacts", {})
    for name, path in (("prompts", prompts_path), ("evaluator", evaluator_path)):
        if expected.get(name, {}).get("sha256") != sha256_file(path):
            errors.append(f"{name} SHA-256 mismatch")
    prompt_ids = [row.get("record_id") for row in prompts]
    evaluator_ids = [row.get("record_id") for row in evaluator]
    if len(prompt_ids) != len(set(prompt_ids)):
        errors.append("duplicate prompt record_id")
    if set(prompt_ids) != set(evaluator_ids):
        errors.append("prompt/evaluator record IDs differ")
    for row in prompts:
        leaked = FORBIDDEN_PROMPT_FIELDS & set(row)
        if leaked:
            errors.append(f"{row.get('record_id')}: evaluator fields leaked: {sorted(leaked)}")
        prompt_text = str(row.get("prompt", ""))
        for marker in ("cot_content", "answer_index", "gold_index", "lure_index"):
            if marker in prompt_text:
                errors.append(f"{row.get('record_id')}: prompt contains forbidden marker {marker}")
    labels = {row["record_id"]: row for row in evaluator}
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_items: dict[str, set[str]] = defaultdict(set)
    for row in prompts:
        by_item[row["base_item_id"]].append(row)
        split_items[row["split"]].add(row["base_item_id"])
        label = labels.get(row["record_id"], {})
        if label.get("gold_index") == label.get("lure_index"):
            errors.append(f"{row['record_id']}: lure equals gold")
    if split_items["discovery"] & split_items["pilot_holdout"]:
        errors.append("base_item_id overlap across splits")
    for item_id, rows in by_item.items():
        factorial = [row for row in rows if row["control_type"] == "factorial"]
        combinations = {(row["reward_lure"], row["social_lure"], row["certainty_pressure"]) for row in factorial}
        if len(factorial) != 8 or len(combinations) != 8:
            errors.append(f"{item_id}: expected exactly eight factorial conditions")
        controls = Counter(row["control_type"] for row in rows if row["control_type"] != "factorial")
        if controls != Counter({name: 1 for name in VALID_CONTROLS}):
            errors.append(f"{item_id}: invalid valid-control coverage")
        if len({row["option_order_sha256"] for row in rows}) != 1:
            errors.append(f"{item_id}: option order changed across conditions")
        if len({row["token_budget"] for row in rows}) != 1:
            errors.append(f"{item_id}: token budget changed across conditions")
        expected_family = "A" if rows[0]["split"] == "discovery" else "B"
        if {row["template_family"] for row in rows} != {expected_family}:
            errors.append(f"{item_id}: template family leaked across split")
    return {
        "passed": not errors, "errors": errors, "base_items": len(by_item), "records": len(prompts),
        "split_counts": {key: len(value) for key, value in split_items.items()},
    }
