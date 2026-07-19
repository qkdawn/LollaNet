---
type: research_protocol
title: Phase 1 Microconcept State Encoder Training Protocol
status: draft
version: 0.1.0
updated: 2026-07-20
---

# Phase 1 Microconcept State Encoder Training Protocol

## 1. Scope

Phase 1 trains only the following mapping:

```text
question q + goal g + financial factors F + structured MD&A claims C
    -> instance-specific states on a fixed microconcept graph G_micro(x)
```

This phase does not train dynamic microconcept edges, cognitive-model routing,
steering vectors, LLM hidden-state control, or semantic evolution. Those stages
remain gated on the encoder passing the tests defined below.

## 2. Output Contract

The graph uses the fixed candidate nodes under
`research/phase1/microconcepts/cards/`. For each node `k`, the encoder predicts:

1. evidence state `s_k = P(state_k | F, C)`;
2. task relevance `r_k = P(relevant_k | q, g, F, C)`;
3. calibrated activation `a_k = gate(s_k, r_k)`.

The evidence state must not change merely because a question is rephrased. The
question and goal may change relevance and final activation.

Each labelled node uses the following fields:

```yaml
microconcept_id: constraint_pressure
status: supported       # supported | contradicted | unknown
intensity: 3            # ordinal: 0 | 1 | 2 | 3
direction: increasing   # increasing | decreasing | stable | mixed | unknown
applicability: applicable  # applicable | not_applicable | unknown
relevance: 3            # ordinal: 0 | 1 | 2 | 3
evidence_refs:
  - factor:gross_margin_change
  - claim:working_capital_change
label_tier: gold        # gold | silver | weak
```

Annotators label ordinal values rather than pseudo-precise probabilities. A
trained model may emit a continuous expected value after calibration. Missing,
`unknown`, `contradicted`, and `not_applicable` remain distinct states.

## 3. Input Data

### Financial factors

Use the point-in-time fields in
`research/phase1/value_investing/feature_dictionary.yaml`, including an explicit
missingness mask. Input facts must be available no later than the Episode's
`public_available_at` timestamp.

### Structured MD&A claims

Claims follow
`research/phase1/microconcepts/qualitative_claim_types.yaml`. Every claim must
record its controlled type, polarity, horizon, strength, source locator, content
hash, extractor version, and label provenance. Unsupported text is represented
as `unknown` or an abstention, not as a negative claim.

### Questions and goals

Each sampled Episode receives multiple question-goal pairs, including at least
one relevant and one weakly relevant or irrelevant task. This paired design
tests whether the model uses `q` and `g` rather than learning a fixed route from
financial inputs.

## 4. Annotation Protocol

### Pilot sample

Sample 300-500 Episodes from the high-confidence MD&A subset. Stratify by year,
industry, filing type, candidate positive and negative mechanisms, missingness,
and conflicting evidence.

### Evidence-state labels

Label all fixed microconcept nodes once per Episode using only `F` and `C`.
Record status, intensity, direction, and evidence references. At least a held-out
subset must receive independent double annotation and adjudication.

### Task-relevance labels

For each question-goal pair, label applicability and relevance for every node.
The same Episode under different tasks forms a counterfactual pair: evidence
state should remain stable while relevance may change.

### Label tiers

- `gold`: independently human-reviewed and adjudicated;
- `silver`: single-reviewer confirmation or human correction of a machine label;
- `weak`: rule- or model-generated without sufficient human review.

Machine-only labels must never be reported as gold ground truth. Existing LLM
route pseudo-labels may provide weak leads but cannot define microconcept truth.

## 5. Model Architecture

The first implementation should remain small enough to isolate data and label
quality from foundation-model scale:

```text
F + missing mask -> FinancialEncoder (MLP)
C -> ClaimEncoder + order-invariant pooling
q + g -> frozen text encoder

Financial + Claims -> EvidenceFusion -> per-node evidence heads
Task + EvidenceFusion -> RelevanceFusion -> per-node relevance heads
Evidence state + relevance -> calibrated activation
```

Hidden dimensions of approximately 128-256 are sufficient for the first study.
The text encoder remains frozen. Dynamic adjacency and free node generation are
excluded from this phase.

## 6. Training Objective

Use a masked multi-task loss:

```text
L = L_status
  + lambda_1 * L_intensity
  + lambda_2 * L_direction
  + lambda_3 * L_relevance
  + lambda_4 * L_calibration
  + lambda_5 * L_sparsity
  + lambda_6 * L_consistency
```

- `status`: multiclass classification;
- `intensity` and `relevance`: ordinal regression;
- `direction`: multiclass classification;
- `calibration`: Brier/ECE-oriented calibration;
- `sparsity`: prevents indiscriminate activation;
- `consistency`: constrains paraphrase, evidence-mask, and counterfactual tests.

Unknown and unlabelled targets use a loss mask. They are never encoded as zero
or as negative examples.

## 7. Training Sequence

1. Freeze microconcept definitions and annotation guidelines for the run.
2. Generate and validate the first structured MD&A claim set.
3. Complete the 300-500 Episode gold/silver annotation pilot.
4. Audit prevalence, agreement, and concept confusion.
5. Expand weak labels only after the audit.
6. Train an evidence-only baseline without `q/g`.
7. Add the task-relevance head and paired task examples.
8. Select loss weights and calibration on development data.
9. Run the sealed temporal and issuer-held-out tests once.
10. Run evidence masking, question perturbation, and node interventions.

## 8. Splits and Leakage Controls

The default temporal split is:

- train: 2017-2020;
- development: 2021;
- sealed test: 2022-2023.

Add an issuer-held-out evaluation so the system cannot succeed by memorizing
company identity. Freeze splits before generating weak labels. Estimate all
normalization statistics, thresholds, and aggregation parameters from training
data only.

## 9. Baselines and Ablations

Compare:

1. deterministic rule projection;
2. financial factors only;
3. MD&A claims only;
4. financial factors plus MD&A claims;
5. model without `q/g`;
6. model that does not separate evidence state and relevance;
7. complete dual-head encoder;
8. shuffled question-goal pairs.

Interventions must mask decisive factors or claims, paraphrase questions, reverse
task goals, and perturb individual node targets. Expected directions are
registered before evaluating the sealed test.

## 10. Metrics

Report:

- macro-F1 and per-node precision/recall for status;
- MAE and weighted kappa for ordinal targets;
- Brier score and expected calibration error;
- evidence-span precision/recall;
- paraphrase stability and goal sensitivity;
- temporal, issuer-held-out, and low-frequency-node performance;
- intervention faithfulness under evidence removal and counterfactual changes.

## 11. Progression Gate

Dynamic edges or cognitive-model routing may begin only when:

1. retained concepts have adequate support and acceptable annotation agreement;
2. the dual-head encoder beats rules, single-modality, and no-`q/g` baselines on
   the sealed gold set;
3. probabilities are calibrated and missingness is not collapsed into negatives;
4. paraphrases remain stable while distinct goals change relevance;
5. evidence masking and counterfactuals cause mechanism-consistent changes;
6. gains persist in temporal and issuer-held-out evaluation.

If concept boundaries remain unreliable, revise the annotation contract before
adding model capacity. If deterministic rules match the complete encoder, report
an interpretable rule baseline rather than claiming a learned microconcept
network.

