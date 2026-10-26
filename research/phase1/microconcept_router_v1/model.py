"""Masked bipartite microconcept-to-model router."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .data import IGNORE_INDEX, SemanticGraph, SupportSet


class MicroconceptRouter(nn.Module):
    """Learn numeric routing only on the fixed legal microconcept-to-model edges."""

    def __init__(
        self,
        graph: SemanticGraph,
        support: SupportSet,
        direction_vocab_size: int,
        *,
        hidden_dim: int = 64,
        relation_embedding_dim: int = 12,
        model_embedding_dim: int = 32,
        task_relevance_attention_strength: float = 0.15,
        task_relevance_residual_strength: float = 0.25,
    ) -> None:
        super().__init__()
        for name, value in (
            ("task_relevance_attention_strength", task_relevance_attention_strength),
            ("task_relevance_residual_strength", task_relevance_residual_strength),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        self.graph = graph
        self.support = support
        self.route_models = support.route_models
        graph_model_index = {model: index for index, model in enumerate(graph.model_ids)}
        self.route_model_indices = tuple(graph_model_index[model] for model in self.route_models)
        route_position = {model: index for index, model in enumerate(self.route_models)}
        self.primary_positions = tuple(route_position[model] for model in support.primary_models)
        self.auxiliary_positions = tuple(route_position[model] for model in support.auxiliary_models)
        edge_relation = torch.tensor(graph.edge_relation[list(self.route_model_indices)], dtype=torch.long)
        self.register_buffer("edge_relation", edge_relation, persistent=True)
        self.register_buffer("edge_mask", edge_relation >= 0, persistent=True)

        self.concept_embedding = nn.Embedding(len(graph.concept_ids), 12)
        self.status_embedding = nn.Embedding(5, 10)
        self.direction_embedding = nn.Embedding(direction_vocab_size, 14)
        # The factual node never receives task_relevance.  Question conditioning
        # is added later as a bounded attention/residual modulation, so zero
        # relevance cannot erase the fact representation.
        self.numeric_projection = nn.Sequential(nn.Linear(2, 12), nn.ReLU())
        self.node_projection = nn.Sequential(
            nn.Linear(48, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.08),
        )
        self.model_embedding = nn.Embedding(len(self.route_models), model_embedding_dim)
        self.query = nn.Linear(model_embedding_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.register_buffer(
            "task_relevance_attention_strength",
            torch.tensor(float(task_relevance_attention_strength)),
            persistent=True,
        )
        self.register_buffer(
            "task_relevance_residual_strength",
            torch.tensor(float(task_relevance_residual_strength)),
            persistent=True,
        )
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_embedding = nn.Embedding(len(graph.relation_ids), relation_embedding_dim)
        self.relation_bias = nn.Linear(relation_embedding_dim, 1, bias=False)
        self.model_head = nn.Sequential(
            nn.Linear(hidden_dim + model_embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08), nn.Linear(hidden_dim, 1),
        )
        self.abstain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, concept_count = batch["status"].shape
        if concept_count != len(self.graph.concept_ids):
            raise ValueError("router input must contain exactly the fixed microconcept node count")
        concept = torch.arange(concept_count, device=batch["status"].device).expand(batch_size, -1)
        node = torch.cat((
            self.concept_embedding(concept),
            self.status_embedding(batch["status"]),
            self.direction_embedding(batch["direction"]),
            self.numeric_projection(batch["numeric"][..., :2]),
        ), dim=-1)
        node = self.node_projection(node)
        model_positions = torch.arange(len(self.route_models), device=node.device)
        model = self.model_embedding(model_positions)
        query = self.query(model)
        key = self.key(node)
        value = self.value(node)
        scores = torch.einsum("mh,bnh->bmn", query, key) / math.sqrt(key.shape[-1])
        relation_ids = self.edge_relation.clamp_min(0)
        relation_bias = self.relation_bias(self.relation_embedding(relation_ids)).squeeze(-1)
        scores = scores + relation_bias.unsqueeze(0)
        edge_mask = self.edge_mask.unsqueeze(0)
        masked_factual_scores = scores.masked_fill(~edge_mask, torch.finfo(scores.dtype).min)
        factual_attention = torch.softmax(masked_factual_scores, dim=-1) * edge_mask

        relevance = batch["numeric"][..., 2].clamp(0.0, 1.0)
        task_scores = scores + self.task_relevance_attention_strength * relevance.unsqueeze(1)
        task_scores = task_scores.masked_fill(~edge_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(task_scores, dim=-1) * edge_mask

        # Always preserve a complete factual route whose nodes, attention, and
        # values are independent of task_relevance.  Relevance contributes only
        # a bounded focused residual.  With all relevance values at zero, the
        # fact-conditioned context remains intact.
        factual_context = torch.einsum("bmn,bnh->bmh", factual_attention, value)
        focused_value = value * relevance.unsqueeze(-1)
        task_context = torch.einsum("bmn,bnh->bmh", attention, focused_value)
        context = factual_context + self.task_relevance_residual_strength * task_context
        model_features = model.unsqueeze(0).expand(batch_size, -1, -1)
        activation_logits = self.model_head(torch.cat((context, model_features), dim=-1)).squeeze(-1)
        pooled = node.mean(dim=1)
        abstain_logits = self.abstain_head(pooled).squeeze(-1)
        return {
            "primary": activation_logits[:, self.primary_positions],
            "auxiliary": activation_logits[:, self.auxiliary_positions],
            "abstain": abstain_logits,
            "attention": attention,
            "factual_attention": factual_attention,
            "edge_mask": self.edge_mask,
            "all_activation": activation_logits,
            "factual_context": factual_context,
            "task_context": task_context,
        }


def router_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    primary_weight: float = 1.0,
    auxiliary_weight: float = 0.7,
    abstain_weight: float = 0.5,
    auxiliary_pos_weight: torch.Tensor | None = None,
    primary_class_weight: torch.Tensor | None = None,
    abstain_pos_weight: torch.Tensor | float | None = None,
    attention_alignment_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    primary_mask = batch["primary"] != IGNORE_INDEX
    primary = (
        F.cross_entropy(
            output["primary"][primary_mask],
            batch["primary"][primary_mask],
            weight=primary_class_weight,
        )
        if bool(primary_mask.any())
        else output["primary"].sum() * 0.0
    )
    auxiliary = F.binary_cross_entropy_with_logits(
        output["auxiliary"], batch["auxiliary"], pos_weight=auxiliary_pos_weight,
    )
    abstain = F.binary_cross_entropy_with_logits(
        output["abstain"], batch["abstain"], pos_weight=abstain_pos_weight,
    )

    # task_relevance is the third numeric channel. The target is normalized
    # independently for every legal model edge, so this auxiliary objective
    # teaches the router to use question-conditioned microstates without
    # forcing every thought model to attend to the same microconcept.
    attention = output["attention"]
    legal_mask = output.get("edge_mask")
    if legal_mask is None:
        legal_mask = torch.ones(attention.shape[1:], dtype=attention.dtype, device=attention.device)
    else:
        legal_mask = legal_mask.to(dtype=attention.dtype, device=attention.device)
    if legal_mask.ndim == 2:
        legal_mask = legal_mask.unsqueeze(0).expand(attention.shape[0], -1, -1)
    relevance = batch["numeric"][..., 2].unsqueeze(1).expand_as(attention).clamp_min(0.0)
    relevance = relevance * legal_mask
    legal_mass = legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    relevance_mass = relevance.sum(dim=-1, keepdim=True)
    target = torch.where(
        relevance_mass > 0,
        relevance / relevance_mass.clamp_min(torch.finfo(relevance.dtype).eps),
        legal_mask / legal_mass,
    )
    log_attention = attention.clamp_min(torch.finfo(attention.dtype).eps).log()
    alignment = F.kl_div(log_attention, target, reduction="batchmean")

    total = (
        primary_weight * primary
        + auxiliary_weight * auxiliary
        + abstain_weight * abstain
        + attention_alignment_weight * alignment
    )
    return total, {
        "total": float(total.detach()),
        "primary": float(primary.detach()),
        "auxiliary": float(auxiliary.detach()),
        "abstain": float(abstain.detach()),
        "attention_alignment": float(alignment.detach()),
    }


def soft_activation_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    activation_weight: float = 1.0,
    attention_alignment_weight: float = 0.0,
    huber_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train all thought-model activations from per-model soft targets.

    ``activation_mask`` is independent for every model dimension. A disagreement
    between teachers therefore removes only that dimension from the loss.
    ``activation_weight`` carries conservative teacher-confidence and agreement
    weighting produced by the manifest-bound aggregation chain.
    """
    predicted = torch.sigmoid(output["all_activation"])
    target = batch["activation_target"]
    mask = batch["activation_mask"].to(dtype=predicted.dtype)
    supervision_weight = batch["activation_weight"].to(dtype=predicted.dtype)
    effective_weight = mask * supervision_weight
    per_dimension = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=float(huber_beta),
    )
    denominator = effective_weight.sum()
    activation = (per_dimension * effective_weight).sum() / denominator.clamp_min(1.0)
    if not bool(denominator > 0):
        activation = output["all_activation"].sum() * 0.0

    attention = output["attention"]
    legal_mask = output.get("edge_mask")
    if legal_mask is None:
        legal_mask = torch.ones(attention.shape[1:], dtype=attention.dtype, device=attention.device)
    else:
        legal_mask = legal_mask.to(dtype=attention.dtype, device=attention.device)
    if legal_mask.ndim == 2:
        legal_mask = legal_mask.unsqueeze(0).expand(attention.shape[0], -1, -1)
    relevance = batch["numeric"][..., 2].unsqueeze(1).expand_as(attention).clamp_min(0.0)
    relevance = relevance * legal_mask
    legal_mass = legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    relevance_mass = relevance.sum(dim=-1, keepdim=True)
    attention_target = torch.where(
        relevance_mass > 0,
        relevance / relevance_mass.clamp_min(torch.finfo(relevance.dtype).eps),
        legal_mask / legal_mass,
    )
    alignment = F.kl_div(
        attention.clamp_min(torch.finfo(attention.dtype).eps).log(),
        attention_target,
        reduction="batchmean",
    )
    total = activation_weight * activation + attention_alignment_weight * alignment
    return total, {
        "total": float(total.detach()),
        "activation": float(activation.detach()),
        "attention_alignment": float(alignment.detach()),
        "supervised_weight": float(denominator.detach()),
    }


def output_probabilities(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "primary": torch.softmax(output["primary"], dim=-1),
        "auxiliary": torch.sigmoid(output["auxiliary"]),
        "abstain": torch.sigmoid(output["abstain"]),
        "all_activation": torch.sigmoid(output["all_activation"]),
        "attention": output["attention"],
    }


def legal_contributions(output: dict[str, torch.Tensor], router: MicroconceptRouter) -> list[list[dict[str, Any]]]:
    """Export only legal, learned numeric edge weights for audit output."""
    result: list[list[dict[str, Any]]] = []
    attention = output_probabilities(output)["attention"].detach().cpu()
    for batch_row in attention:
        row: list[dict[str, Any]] = []
        for model_position, model_id in enumerate(router.route_models):
            for concept_position, concept_id in enumerate(router.graph.concept_ids):
                relation_id = int(router.edge_relation[model_position, concept_position])
                if relation_id >= 0:
                    row.append({
                        "microconcept_id": concept_id,
                        "model_id": model_id,
                        "relation": router.graph.relation_ids[relation_id],
                        "weight": float(batch_row[model_position, concept_position]),
                    })
        result.append(row)
    return result
