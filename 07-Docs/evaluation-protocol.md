---
type: evaluation_protocol
title: LollaNet Evaluation Protocol
updated: 2026-07-12
status: active
---

# LollaNet Evaluation Protocol

LollaNet does not assume that more models produce better reasoning. This
protocol compares the system with an ordinary AI baseline and reports results
without claiming causality from a single answer or outcome.

This v1 document remains the record of the original 10-task exploratory blind evaluation. New system, ablation, and portfolio tests follow [LollaNet Model Test Protocol v2](model-test-protocol-v2.md); v2 does not retroactively reinterpret v1 results.

## Hypotheses

1. Relevant private context can make analysis less generic without leaking unrelated personal information.
2. Diverse, conflict-aware model retrieval can add useful variables beyond the default answer without sacrificing base rates.
3. Historical and multi-horizon checks can expose short-term commitments and second-order effects that a baseline misses.

## Blind Pairwise Procedure

1. Run each prompt once without LollaNet files or instructions and once with `$lollanet-analyze`.
2. Keep the model, task wording, available external evidence, and answer-length budget the same.
3. Randomize labels `A` and `B`; do not tell the evaluator which workflow produced either answer.
4. Score personal fit, non-obvious contribution, concision, evidence quality, action value, present-moment elegance, and factual reliability.
5. Store the completed record under `private/evaluations/`; never put real prompts or ratings in public fixtures.
6. Publish all ten results together. Do not discard losses or ties.

The initial support threshold is at least 7 preferences out of 10 for the
LollaNet answer, with factual reliability no lower than the baseline. This is
an exploratory threshold, not proof that the architecture generalizes.

## Retrieval Checks

- Every accepted item cites the narrowest file heading or external locator that actually supports its basis.
- The analysis uses 3 to 7 non-duplicate models for a complex judgment.
- Each selected model changes an assumption, prediction, risk, or action; decorative citations are failures.
- Major rejected candidates and their rejection reasons remain auditable.
- Irrelevant private facts are neither read into the retrieval packet nor repeated in the answer.
- Missing sources are labeled `待验证` rather than reconstructed from memory.

## Long-Horizon Checks

At answer time, record immediate, medium-term, and long-term effects, plus one
observable prediction or falsifier. Review saved high-value judgments after 30
and 90 days. Separate failures caused by bad facts, wrong model selection,
execution, environmental change, and randomness.

Before the follow-up window closes, report only time-horizon coverage. Do not
claim that LollaNet improved outcomes or reduced short-sightedness.

## Interpretation

A longer answer is not automatically better. A novel answer is not
automatically better. Base rates remain the default; deviation requires
specific user constraints, mechanisms, or evidence. Buffett and Munger are
inspiration for the hypothesis, not causal evidence for this system.
