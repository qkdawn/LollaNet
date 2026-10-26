---
type: research_protocol
title: Episode Microconcept Labels v1
status: candidate
updated: 2026-07-19
---

# Episode Microconcept Labels v1

This package builds the **data-only** Episode-level microconcept pilot. It does
not train an encoder, invoke thought-model routing, produce investment advice,
or use LLM steering. Runtime labels and evidence remain outside the repository
under `D:\LollaNetData\microconcept-labels-v1`.

## Inputs and hard boundary

The builder reads only the candidate Episode identifier, filing accession/form,
public availability timestamp, SIC code, approved financial/market factors,
missing-reason metadata, and structured MD&A claims. It projects source records
onto allowlisted fields immediately. It never reads `route_labels`,
`exploratory_outcome`, issuer identity, ticker identity, raw MD&A text, or any
thought-model output.

The eleven Markdown cards in `cards/` remain the sole microconcept source of
truth. A label may only reference a card factor or one of that card's controlled
claim types.

## Pilot and temporal freeze

The default build freezes a stratified 2022–2023 time-out set *before* it
selects or annotates a 400-Episode pilot from 2005–2021. The frozen set is
written separately and remains unlabelled. The pilot preferentially uses
Episodes with matched structured MD&A claims; deterministic stratified
non-claim additions make the pilot exactly 400 and preserve missing-claim
coverage. The seed, source hashes, selection policy, and test identifiers are
in the manifest.

## Two separate targets

### Evidence state (question-independent)

There is one `episode_microconcept_labels.jsonl` record per
Episode × microconcept. It includes:

- `status`: `supported`, `contradicted`, or `unknown`;
- `intensity`: ordinal 0–3;
- `direction`: evidence-implied increasing, decreasing, stable, mixed, or
  unknown;
- concrete factor and/or claim references.

`contradicted` means observed evidence supports the documented opposite state;
it is never produced merely because a value or claim is missing. `unknown`
remains masked—not a negative example. Rule-only outputs are provisional:
corroborated deterministic outputs are `Silver`, while unresolved or
single-source outputs are `Weak`.

### Task relevance (question/goal-conditioned)

Each evidence-state record receives two paired task labels: a concept-focused
analysis task and an explicitly unrelated filing-format task. The two records
reference the same `state_label_id`; factual state therefore cannot change with
question/goal, while `applicability` and `relevance` intentionally do.

## Evidence thresholds and boundary safeguards

Thresholds are fixed, documented domain guardrails in the implementation; they
are not estimated from the frozen temporal test. Absolute currency quantities
are not compared cross issuer. In particular:

- `constraint_pressure` needs corroboration and does not call the worst ratio a
  bottleneck;
- `conversion_gap` requires an explicit cash-conversion pattern, not merely
  accounting profit or growth;
- `countervailing_pressure` requires recognizable opposing evidence and does
  not subtract incompatible units as a net “force.”

Labels in these boundary-sensitive concepts, conflicting evidence, low
confidence, and all non-unknown labels based on a single signal are put in the
independent human review queue.

## Run

```powershell
python research/phase1/microconcepts/build_microconcept_labels_v1.py `
  --candidates D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --claims D:\LollaNetData\mda-claims-v1\claims.jsonl `
  --output D:\LollaNetData\microconcept-labels-v1

python research/phase1/microconcepts/validate_microconcept_labels_v1.py `
  --labels D:\LollaNetData\microconcept-labels-v1\episode_microconcept_labels.jsonl `
  --tasks D:\LollaNetData\microconcept-labels-v1\task_relevance_labels.jsonl `
  --review-queue D:\LollaNetData\microconcept-labels-v1\human_review_queue.jsonl `
  --frozen-test D:\LollaNetData\microconcept-labels-v1\frozen_time_test_episodes.jsonl `
  --manifest D:\LollaNetData\microconcept-labels-v1\manifest.json
```

The validator checks card vocabulary, schema, temporal availability, evidence
references, question-invariant state linkage, paired-task relevance variation,
review queue integrity, and the absence of Gold labels from the automatic
builder. It samples records for a deterministic return-to-evidence audit.

## Scope limitation

Until human confirmations or two-person arbitrations produce Gold labels, this
pilot is only suitable for pretraining, rule-baseline comparison, and
exploratory experiments. It cannot substantiate a final model-effectiveness
claim.
