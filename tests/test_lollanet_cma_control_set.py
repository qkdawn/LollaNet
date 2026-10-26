from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research.phase1.lollanet_cma.common import stable_hash
from research.phase1.lollanet_cma.sealed_eval.control_set import (
    build_control_artifacts,
    select_control_records,
)


def _row(question_id: int, category: str, answer_index: int = 0) -> dict:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "options": ["First", "Second"],
        "answer": "AB"[answer_index],
        "answer_index": answer_index,
        "cot_content": "private evaluator reasoning",
        "category": category,
        "src": f"source-{category}",
    }


def test_control_selection_is_balanced_deterministic_and_order_independent():
    rows = [_row(index, "math", index % 2) for index in range(5)] + [
        _row(index + 10, "physics", index % 2) for index in range(5)
    ]
    first = select_control_records(rows, ["math", "physics"], 3, 17)
    second = select_control_records(list(reversed(rows)), ["math", "physics"], 3, 17)
    assert first == second
    assert [row["category"] for row in first].count("math") == 3
    assert [row["category"] for row in first].count("physics") == 3


def test_control_prompts_cannot_contain_evaluator_labels_or_cot():
    selected = [_row(1, "math", 1)]
    prompts, controllers, evaluators = build_control_artifacts(
        selected, "{question}\n{options}\nAnswer:",
    )
    assert set(prompts[0]) == {"record_id", "dataset", "split", "category", "prompt"}
    assert "private evaluator reasoning" not in prompts[0]["prompt"]
    assert "answer" not in prompts[0]
    assert evaluators[0]["answer"] == "B"
    assert controllers[0]["source_record_sha256"] == stable_hash(selected[0])


def test_control_selection_fails_on_label_or_schema_inconsistency():
    bad_label = _row(1, "math", 0)
    bad_label["answer"] = "B"
    with pytest.raises(ValueError, match="letter and index"):
        select_control_records([bad_label], ["math"], 1, 1)
    missing = _row(1, "math")
    missing.pop("cot_content")
    with pytest.raises(ValueError, match="schema"):
        select_control_records([missing], ["math"], 1, 1)


def test_formal_control_charter_freezes_source_selection_and_transfer_gate():
    charter = yaml.safe_load(Path(
        "research/phase1/lollanet_cma/sealed_eval/control_set_charter.yaml",
    ).read_text(encoding="utf-8"))
    assert charter["source"]["revision"] == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert charter["source"]["license"] == "MIT"
    assert charter["selection"]["total_records"] == 200
    assert charter["selection"]["categories"] == [
        "math", "physics", "chemistry", "computer science",
    ]
    assert charter["evaluation"]["maximum_allowed_accuracy_loss"] == 0.02
