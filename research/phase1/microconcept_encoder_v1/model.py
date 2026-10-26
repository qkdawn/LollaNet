"""Small fixed-node PyTorch model used by the v1 runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .data import CONCEPT_IDS


@dataclass(frozen=True)
class ModelVariant:
    name: str
    use_financial: bool = True
    use_claims: bool = True
    use_qg: bool = True
    separate_heads: bool = True
    shuffle_qg_train: bool = False
    evidence_only: bool = False
    rule_projection: bool = False


VARIANTS = (
    ModelVariant("financial_only", use_financial=True, use_claims=False, use_qg=True, separate_heads=True),
    ModelVariant("claims_only", use_financial=False, use_claims=True, use_qg=True, separate_heads=True),
    ModelVariant("f_plus_c", use_financial=True, use_claims=True, use_qg=False, separate_heads=True, evidence_only=True),
    ModelVariant("no_qg", use_financial=True, use_claims=True, use_qg=False, separate_heads=True),
    ModelVariant("joint_no_separation", use_financial=True, use_claims=True, use_qg=True, separate_heads=False),
    ModelVariant("full_dual_head", use_financial=True, use_claims=True, use_qg=True, separate_heads=True),
    ModelVariant("shuffled_qg", use_financial=True, use_claims=True, use_qg=True, separate_heads=True, shuffle_qg_train=True),
)


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.08), nn.Linear(hidden, output_dim))


class ClaimSetEncoder(nn.Module):
    """Structured categorical claim encoder with order-invariant masked mean pooling."""
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.claim_id = nn.Embedding(12, 24, padding_idx=0)
        self.polarity = nn.Embedding(5, 8, padding_idx=0)
        self.horizon = nn.Embedding(5, 8, padding_idx=0)
        self.strength = nn.Embedding(5, 8, padding_idx=0)
        self.projection = _mlp(24 + 8 + 8 + 8 + 1, hidden, hidden)

    def forward(self, claim_id: torch.Tensor, polarity: torch.Tensor, horizon: torch.Tensor, strength: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        pieces = [self.claim_id(claim_id), self.polarity(polarity), self.horizon(horizon), self.strength(strength), confidence.unsqueeze(-1)]
        encoded = self.projection(torch.cat(pieces, dim=-1))
        mask = (claim_id != 0).unsqueeze(-1)
        total = (encoded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return total / denom


class DualHeadMicroconceptEncoder(nn.Module):
    """Fixed eleven-node evidence and relevance heads.

    `separate_heads=True` means evidence logits do not depend on q/g.  It is an
    architectural invariance, rather than a post hoc assertion.
    """
    def __init__(self, financial_dim: int, text_dim: int = 96, hidden_dim: int = 128, variant: ModelVariant | None = None) -> None:
        super().__init__()
        self.variant = variant or ModelVariant("full_dual_head")
        self.node_count = len(CONCEPT_IDS)
        self.financial_encoder = _mlp(financial_dim, hidden_dim, hidden_dim)
        self.claim_encoder = ClaimSetEncoder(hidden_dim)
        self.evidence_fusion = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.text_projection = _mlp(text_dim, hidden_dim, hidden_dim)
        self.relevance_fusion = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.joint_fusion = _mlp(hidden_dim * 3, hidden_dim, hidden_dim)
        output = self.node_count
        self.status_head = nn.Linear(hidden_dim, output * 2)
        self.intensity_head = nn.Linear(hidden_dim, output * 4)
        self.direction_head = nn.Linear(hidden_dim, output * 4)
        self.applicability_head = nn.Linear(hidden_dim, output * 2)
        self.relevance_head = nn.Linear(hidden_dim, output * 4)
        # This is a calibrated differentiable gate.  Temperatures are selected
        # after training on development data and applied in the evaluator.
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        financial = self.financial_encoder(batch["financial"]) if self.variant.use_financial else torch.zeros((batch["financial"].shape[0], self.financial_encoder[-1].out_features), device=batch["financial"].device)
        claims = self.claim_encoder(batch["claim_id"], batch["polarity"], batch["horizon"], batch["strength"], batch["confidence"]) if self.variant.use_claims else torch.zeros_like(financial)
        evidence = self.evidence_fusion(torch.cat((financial, claims), dim=-1))
        text = self.text_projection(batch["text"]) if self.variant.use_qg else torch.zeros_like(evidence)
        if self.variant.separate_heads:
            evidence_state = evidence
            relevance_state = self.relevance_fusion(torch.cat((evidence, text), dim=-1))
        else:
            shared = self.joint_fusion(torch.cat((financial, claims, text), dim=-1))
            evidence_state = shared
            relevance_state = shared
        status = self.status_head(evidence_state).view(-1, self.node_count, 2)
        intensity = self.intensity_head(evidence_state).view(-1, self.node_count, 4)
        direction = self.direction_head(evidence_state).view(-1, self.node_count, 4)
        applicability = self.applicability_head(relevance_state).view(-1, self.node_count, 2)
        relevance = self.relevance_head(relevance_state).view(-1, self.node_count, 4)
        evidence_strength = torch.softmax(status, dim=-1).amax(dim=-1) * (torch.softmax(intensity, dim=-1) * torch.arange(4, device=status.device)).sum(dim=-1).div(3)
        task_strength = torch.softmax(applicability, dim=-1)[..., 0] * (torch.softmax(relevance, dim=-1) * torch.arange(4, device=status.device)).sum(dim=-1).div(3)
        activation = torch.sigmoid(self.gate_scale * (evidence_strength * task_strength) + self.gate_bias)
        return {"status": status, "intensity": intensity, "direction": direction, "applicability": applicability, "relevance": relevance, "activation": activation, "evidence_embedding": evidence}


def select_node_logits(outputs: dict[str, torch.Tensor], concepts: torch.Tensor) -> dict[str, torch.Tensor]:
    indices = torch.arange(concepts.shape[0], device=concepts.device)
    return {key: outputs[key][indices, concepts] for key in ("status", "intensity", "direction", "applicability", "relevance", "activation")}
