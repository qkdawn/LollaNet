from __future__ import annotations

from pathlib import Path

import torch
import yaml

from research.phase1.microconcept_router_v1.data import SupportSet, load_semantic_graph
from research.phase1.microconcept_router_v1.model import MicroconceptRouter, output_probabilities, soft_activation_loss


def _soft_batch(concept_count: int, model_count: int) -> dict[str, torch.Tensor]:
    return {
        "status": torch.zeros((2, concept_count), dtype=torch.long),
        "direction": torch.zeros((2, concept_count), dtype=torch.long),
        "numeric": torch.ones((2, concept_count, 3), dtype=torch.float32) * 0.5,
        "activation_target": torch.zeros((2, model_count), dtype=torch.float32),
        "activation_mask": torch.ones((2, model_count), dtype=torch.float32),
        "activation_weight": torch.ones((2, model_count), dtype=torch.float32),
    }


def test_soft_router_outputs_all_twenty_independent_activations() -> None:
    graph = load_semantic_graph()
    support = SupportSet(graph.model_ids, graph.model_ids, {}, {}, 0)
    router = MicroconceptRouter(graph, support, direction_vocab_size=2)
    router.eval()
    batch = _soft_batch(len(graph.concept_ids), len(graph.model_ids))
    output = router(batch)
    probability = output_probabilities(output)["all_activation"]
    assert probability.shape == (2, 20)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))
    # Sigmoid dimensions are independent and are not forced to sum to one.
    assert not torch.allclose(probability.sum(dim=-1), torch.ones(2))


def test_masked_soft_dimension_does_not_change_loss() -> None:
    torch.manual_seed(11)
    graph = load_semantic_graph()
    support = SupportSet(graph.model_ids, graph.model_ids, {}, {}, 0)
    router = MicroconceptRouter(graph, support, direction_vocab_size=2)
    router.eval()
    batch = _soft_batch(len(graph.concept_ids), len(graph.model_ids))
    batch["activation_mask"][:, 0] = 0.0
    batch["activation_weight"][:, 0] = 0.0
    output = router(batch)
    loss_a, _ = soft_activation_loss(output, batch)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["activation_target"][:, 0] = 1.0
    loss_b, _ = soft_activation_loss(output, changed)
    assert torch.allclose(loss_a, loss_b)


def test_all_masked_soft_batch_has_finite_zero_activation_loss() -> None:
    graph = load_semantic_graph()
    support = SupportSet(graph.model_ids, graph.model_ids, {}, {}, 0)
    router = MicroconceptRouter(graph, support, direction_vocab_size=2)
    batch = _soft_batch(len(graph.concept_ids), len(graph.model_ids))
    batch["activation_mask"].zero_()
    batch["activation_weight"].zero_()
    output = router(batch)
    loss, metrics = soft_activation_loss(output, batch)
    assert torch.isfinite(loss)
    assert metrics["activation"] == 0.0


def test_zero_task_relevance_preserves_fact_conditioned_path() -> None:
    torch.manual_seed(17)
    graph = load_semantic_graph()
    support = SupportSet(graph.model_ids, graph.model_ids, {}, {}, 0)
    router = MicroconceptRouter(graph, support, direction_vocab_size=2).eval()
    batch = _soft_batch(len(graph.concept_ids), len(graph.model_ids))
    batch["numeric"][..., 2].zero_()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["status"][:, 0] = 1
    output_a = router(batch)
    output_b = router(changed)
    assert torch.count_nonzero(output_a["factual_context"]).item() > 0
    assert torch.count_nonzero(output_a["task_context"]).item() == 0
    assert not torch.allclose(output_a["all_activation"], output_b["all_activation"])


def test_task_relevance_modulates_without_replacing_fact_context() -> None:
    torch.manual_seed(19)
    graph = load_semantic_graph()
    support = SupportSet(graph.model_ids, graph.model_ids, {}, {}, 0)
    router = MicroconceptRouter(graph, support, direction_vocab_size=2).eval()
    low = _soft_batch(len(graph.concept_ids), len(graph.model_ids))
    low["numeric"][..., 2].zero_()
    high = {key: value.clone() for key, value in low.items()}
    high["numeric"][:, 0, 2] = 1.0
    output_low = router(low)
    output_high = router(high)
    assert torch.count_nonzero(output_low["factual_context"]).item() > 0
    assert torch.count_nonzero(output_high["factual_context"]).item() > 0
    assert not torch.allclose(output_low["all_activation"], output_high["all_activation"])


def test_current_soft_config_disables_attention_alignment_and_value_gate() -> None:
    config_path = Path("research/phase1/microconcept_router_v1/config_soft_scores_v1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["training"]["attention_alignment_loss_weight"] == 0.0
    assert "task_relevance_value_gate_floor" not in config["architecture"]
    assert 0.0 <= config["architecture"]["task_relevance_attention_strength"] <= 1.0
    assert 0.0 <= config["architecture"]["task_relevance_residual_strength"] <= 1.0
