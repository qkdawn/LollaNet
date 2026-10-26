from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.phase1.lollanet_cma.common import atomic_write_jsonl, stable_hash
from research.phase1.lollanet_cma.sealed_eval.runner import (
    _load_prefix,
    _random_vectors,
    _vector_for_condition,
)


def test_sealed_random_vectors_are_deterministic_and_norm_matched():
    directions = {
        "a__layer_3": torch.tensor([3.0, 4.0]),
        "b__layer_3": torch.tensor([0.0, 2.0]),
    }
    first = _random_vectors(directions, ["a", "b"], 3, {"random_norm_matched__draw_1": 7})
    second = _random_vectors(directions, ["a", "b"], 3, {"random_norm_matched__draw_1": 7})
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.linalg.norm(first["random_norm_matched__draw_1::a__layer_3"]) == pytest.approx(5.0)
    assert np.linalg.norm(first["random_norm_matched__draw_1::b__layer_3"]) == pytest.approx(2.0)


def test_sealed_vector_routing_covers_fixed_family_and_random_conditions():
    directions = {
        "a__layer_3": torch.tensor([1.0]),
        "b__layer_3": torch.tensor([2.0]),
        "general__layer_3": torch.tensor([3.0]),
        "equal_all__layer_3": torch.tensor([4.0]),
    }
    random = {"random_norm_matched__draw_1::b__layer_3": torch.tensor([5.0])}
    assert _vector_for_condition(
        "family_direction__a", "b", 3, directions, random, {}, {},
    )[0] == "a__layer_3"
    assert _vector_for_condition("general_caa", "b", 3, directions, random, {}, {})[0] == (
        "general__layer_3"
    )
    key, value = _vector_for_condition(
        "random_norm_matched__draw_1", "b", 3, directions, random, {}, {},
    )
    assert key == "random_norm_matched__draw_1::b__layer_3"
    torch.testing.assert_close(value, torch.tensor([5.0]))


def test_sealed_resume_requires_an_exact_hash_verified_prefix(tmp_path: Path):
    path = tmp_path / "condition.jsonl"
    row = {"condition_id": "base", "record_id": "a", "value": 1}
    row["record_content_sha256"] = stable_hash(row)
    atomic_write_jsonl(path, [row])
    assert _load_prefix(path, ["a", "b"], "base") == [row]
    changed = dict(row)
    changed["value"] = 2
    atomic_write_jsonl(path, [changed])
    with pytest.raises(ValueError, match="row changed"):
        _load_prefix(path, ["a", "b"], "base")
