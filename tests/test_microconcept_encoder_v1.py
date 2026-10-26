from __future__ import annotations

import numpy as np
import torch

from research.phase1.microconcept_encoder_v1.data import Example, TrainOnlyNormalizer, make_splits, target_arrays
from research.phase1.microconcept_encoder_v1.evaluation import multitask_loss
from research.phase1.microconcept_encoder_v1.model import DualHeadMicroconceptEncoder, ModelVariant


def example(identifier: str, year: int, cik: str, *, status: str | None = "supported", intensity: int | None = 2, direction: str | None = "increasing") -> Example:
    return Example(
        example_id=identifier, episode_id=f"episode-{identifier}", accession=f"acc-{identifier}", available_at=f"{year}-01-02T00:00:00Z",
        year=year, cik_for_split=cik, concept_id="capacity_headroom", task_label_id=identifier,
        question="是否值得优先核验？", goal="识别研究优先级", task_variant="focused",
        factors={"ttm_revenue_growth": 0.2}, claims=(), status=status, intensity=intensity, direction=direction,
        applicability="applicable", relevance=3, evidence_label_tier="Silver", task_label_tier="Silver",
        evidence_label_source="test", task_label_source="test", evidence_review_status="pending_review", task_review_status="pending_review",
        evidence_dispute_status="none", task_dispute_status="none",
    )


def test_train_only_normalizer_does_not_use_development_outlier() -> None:
    train = [example("a", 2019, "001"), example("b", 2020, "002")]
    development = [example("c", 2021, "003")]
    development[0].factors["ttm_revenue_growth"] = 9999.0
    normalizer = TrainOnlyNormalizer.fit(train)
    assert normalizer.upper[0] < 1.0
    values, _ = normalizer.transform(development)
    assert np.isfinite(values).all()


def test_unknown_targets_are_ignored_by_masked_loss() -> None:
    torch.manual_seed(2)
    output = {
        "status": torch.randn(2, 11, 2, requires_grad=True), "intensity": torch.randn(2, 11, 4, requires_grad=True),
        "direction": torch.randn(2, 11, 4, requires_grad=True), "applicability": torch.randn(2, 11, 2, requires_grad=True),
        "relevance": torch.randn(2, 11, 4, requires_grad=True), "activation": torch.rand(2, 11, requires_grad=True),
    }
    batch = {"concept": torch.tensor([0, 0]), "status": torch.tensor([-100, -100]), "intensity": torch.tensor([-100, -100]),
             "direction": torch.tensor([-100, -100]), "applicability": torch.tensor([-100, -100]), "relevance": torch.tensor([-100, -100])}
    loss, _ = multitask_loss(output, batch, {"intensity": 1, "direction": 1, "applicability": 1, "relevance": 1})
    assert float(loss.detach()) == 0.0


def test_separated_evidence_head_is_question_goal_invariant() -> None:
    model = DualHeadMicroconceptEncoder(financial_dim=42, text_dim=96, hidden_dim=32, variant=ModelVariant("dual", separate_heads=True)).eval()
    base = {
        "financial": torch.zeros(2, 42), "claim_id": torch.zeros(2, 4, dtype=torch.long), "polarity": torch.zeros(2, 4, dtype=torch.long),
        "horizon": torch.zeros(2, 4, dtype=torch.long), "strength": torch.zeros(2, 4, dtype=torch.long), "confidence": torch.zeros(2, 4),
        "text": torch.zeros(2, 96),
    }
    altered = {key: value.clone() for key, value in base.items()}
    altered["text"][1, 7] = 1.0
    assert torch.equal(model(base)["status"], model(altered)["status"])


def test_issuer_holdout_has_no_temporal_training_overlap() -> None:
    from research.phase1.microconcept_encoder_v1.data import EncoderData
    rows = tuple([example("a", 2019, "001"), example("b", 2020, "002"), example("c", 2021, "003"), example("d", 2021, "004")])
    data = EncoderData(rows, {}, {}, {})
    split = make_splits(data, issuer_fraction=0.25, split_seed="unit-test")
    train_ciks = {row.cik_for_split for row in split.train}
    held_ciks = {row.cik_for_split for row in split.issuer_heldout}
    assert not train_ciks & held_ciks
    assert split.audit["issuer_overlap_train_heldout"] == 0
