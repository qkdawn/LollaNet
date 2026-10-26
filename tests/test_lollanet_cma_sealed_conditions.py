from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from research.phase1.lollanet_cma.common import sha256_file
from research.phase1.lollanet_cma.sealed_eval.conditions import (
    causal_cot_prompt,
    derived_source_condition,
    normalized_equal_all,
    validate_condition_contract,
    vector_reference,
    steer_first_batch_row,
    validate_general_equal_all,
)


CHARTER_PATH = Path("research/phase1/lollanet_cma/sealed_eval/sealed_eval_charter.yaml")


def test_formal_sealed_condition_contract_is_complete_and_nonduplicated():
    charter = yaml.safe_load(CHARTER_PATH.read_text(encoding="utf-8"))
    result = validate_condition_contract(charter)
    assert result["deterministic_model_run_count"] == 8
    assert result["random_draw_count"] == 3
    assert result["derived_condition_ids"] == [
        "oracle_correct", "oracle_wrong", "shuffled_mapping",
    ]
    assert set(result["condition_ids"]) == {
        "base", "causal_cot", "general_caa", "equal_all",
        "family_direction__cm-associational", "family_direction__cm-interventional",
        "family_direction__cm-counterfactual", "family_direction__cm-mediation",
        "oracle_correct", "oracle_wrong", "shuffled_mapping",
        "random_norm_matched__draw_1", "random_norm_matched__draw_2",
        "random_norm_matched__draw_3",
    }
    assert charter["execution"]["deterministic_conditions_run_once"] is True
    assert charter["statistics"]["deterministic_repeats_are_not_independent_replications"] is True
    assert set(charter["readiness_blockers"]) == {
        "unrelated_control_dataset", "token_level_kl", "hidden_state_probe",
    }
    assert all(value["status"] == "resolved" for value in charter["readiness_blockers"].values())


def test_causal_cot_is_a_frozen_single_turn_nonoracle_adaptation(tmp_path: Path):
    template = tmp_path / "cot.txt"
    template.write_text("Infer the graph.\nEnd with Final answer.", encoding="utf-8")
    rendered = causal_cot_prompt("Question?", template)
    assert rendered == "Question?\n\nInfer the graph.\nEnd with Final answer."
    charter = yaml.safe_load(CHARTER_PATH.read_text(encoding="utf-8"))
    formal_template = CHARTER_PATH.parent / charter["prompt_only"]["template"]
    assert sha256_file(formal_template) == charter["prompt_only"]["template_sha256"]
    assert "without_query_type_or_variable_mapping_oracle_fields" in charter["prompt_only"]["adaptation"]


def test_equal_all_uses_unit_family_directions_not_the_pooled_general_vector():
    directions = {
        "a__layer_5": np.array([1.0, 0.0], dtype=np.float32),
        "b__layer_5": np.array([0.0, 1.0], dtype=np.float32),
    }
    equal_all = normalized_equal_all(directions, ["a", "b"], 5)
    np.testing.assert_allclose(equal_all, [2 ** -0.5, 2 ** -0.5], rtol=1e-6)
    pooled_general = np.array([1.0, 1.5]) / np.linalg.norm([1.0, 1.5])
    assert not np.allclose(equal_all, pooled_general)


def test_vector_reference_covers_every_frozen_condition():
    wrong = {"a": "b", "b": "a"}
    shuffled = {"a": "b", "b": "a"}
    assert vector_reference("base", "a", 5, wrong, shuffled) == (None, None)
    assert vector_reference("causal_cot", "a", 5, wrong, shuffled) == (None, None)
    assert vector_reference("general_caa", "a", 5, wrong, shuffled) == (
        "general__layer_5", "general__layer_5",
    )
    assert vector_reference("family_direction__a", "b", 5, wrong, shuffled) == (
        "a__layer_5", "a__layer_5",
    )
    assert vector_reference("equal_all", "a", 5, wrong, shuffled) == (
        "equal_all__layer_5", "equal_all__layer_5",
    )
    assert vector_reference("random_norm_matched__draw_2", "a", 5, wrong, shuffled) == (
        "random_norm_matched__draw_2::a__layer_5", "a__layer_5",
    )
    with pytest.raises(ValueError, match="unsupported sealed evaluation condition"):
        vector_reference("unknown", "a", 5, wrong, shuffled)


def test_derived_routing_uses_the_complete_family_direction_runs():
    wrong = {"a": "d", "b": "a", "c": "b", "d": "c"}
    shuffled = {"a": "c", "b": "d", "c": "a", "d": "b"}
    assert derived_source_condition("oracle_correct", "b", wrong, shuffled) == "family_direction__b"
    assert derived_source_condition("oracle_wrong", "b", wrong, shuffled) == "family_direction__a"
    assert derived_source_condition("shuffled_mapping", "b", wrong, shuffled) == "family_direction__d"
    with pytest.raises(ValueError, match="unsupported derived sealed condition"):
        derived_source_condition("base", "b", wrong, shuffled)


def test_paired_steering_modifies_only_the_first_batch_row():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, value):
            return self.model.layers[0](value)

    model = Model()
    value = torch.zeros((2, 3, 2))
    with steer_first_batch_row(model, 0, torch.tensor([1.0, 2.0]), 0.5) as state:
        output = model(value)
    torch.testing.assert_close(output[0, -1], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(output[1], torch.zeros((3, 2)))
    assert state["calls"] == 1
    torch.testing.assert_close(state["initial_activation"], torch.tensor([0.5, 1.0]))


def test_general_and_equal_all_must_be_frozen_and_distinct():
    directions = {
        "a__layer_5": np.array([1.0, 0.0], dtype=np.float32),
        "b__layer_5": np.array([0.0, 1.0], dtype=np.float32),
        "general__layer_5": np.array([1.0, 2.0], dtype=np.float32) / np.sqrt(5),
        "equal_all__layer_5": np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
    }
    result = validate_general_equal_all(directions, ["a", "b"], 5, 0.999999)
    assert result["cosine_similarity"] < 0.999999
    directions["general__layer_5"] = directions["equal_all__layer_5"]
    with pytest.raises(ValueError, match="not sufficiently distinct"):
        validate_general_equal_all(directions, ["a", "b"], 5, 0.999999)
