# Microconcept Router V1

This module implements the first trainable routing stage:

```text
question-conditioned microconcept states
  -> fixed legal microconcept-to-model graph
  -> primary model, up to two auxiliary models, or abstain
```

It deliberately does **not** consume raw facts, evidence text, source references,
question text, goal text, labels, outcomes, issuer identity, or sector features.
Question and goal affect the router only through the `task_relevance` already stored
on each microconcept state.

The 11 microconcept definitions, 20 thought-model definitions, and candidate edges
are fixed read-only semantics. The model learns only numeric activation scores and
legal-edge attention weights.

## Run

```powershell
python -m research.phase1.microconcept_router_v1.run `
  --dataset-root D:\LollaNetData\mechanism-question-set-v1 `
  --output-root D:\LollaNetData\runs\microconcept-router-v1 `
  --run-id baseline-20260720
```

The default run trains on `train`, selects only thresholds/support from training rows,
and evaluates on `development`. It does not read sealed labels. `--evaluate-sealed`
requires the explicit `--confirm-sealed-evaluation` acknowledgement.

Outputs remain exploratory because the current accepted-review labels are Weak LLM
supervision, not human Gold labels.

## Identifiability preflight

Before a trainable data bundle is returned, V1 rejects a cohort in which two
Episodes have an identical Router-visible microconcept-state vector but different
primary / auxiliary / abstain targets. The Router is intentionally forbidden
from consuming raw question text as a shortcut, so question intent must instead
be represented by the `task_relevance` axis in the Episode microconcept states.

## Manifest-bound supervision gating

The teacher-chain manifest must cover every requested Episode. With the default
`review-v1` source, Router training and evaluation encode only Episodes whose
independent review status is `accepted`; `review_required` rows remain hash-bound
but excluded. With `--teacher-target-source adjudication-v1`, the loader instead
requires a verified child manifest and consumes the deterministic union of parent
accepted routes plus exact route consensus from two distinct adjudicator identities.
Adjudicator disagreements remain excluded and there is no fallback to the parent
`accepted_review.jsonl` if the child chain is absent or invalid.

For the 32-Episode Codex smoke pilot, `config_codex_pilot_v1.yaml` records the
original comparison architecture. The current `config_codex_pilot_improved_v1.yaml`
and `config_codex_pilot_adjudication_v1.yaml` use the v1.1 independent factual
path described below. The earlier zero-floor value gate and positive attention
alignment objective are retired because they let task relevance erase facts and
encouraged template matching. Class/abstain balancing remains train-only; no
`review_required` or sealed labels are admitted.

Training reports now include both head metrics and constrained decision metrics:
`primary_abstain_exact`, `auxiliary_set_exact`, `route_exact`, and
`route_exact_supported`. They also include a task-relevance ablation: the same
Episodes are re-evaluated after zeroing only the `task_relevance` channel, so a run
cannot claim question conditioning merely because four vectors were present in the
data.  Checkpoint selection uses the supported route score plus
primary/abstain, auxiliary-set, and abstain-F1 terms rather than primary top-1 alone.
Because the pilot has only 9 accepted training Episodes and 10 accepted development
Episodes, all results remain exploratory and must not be presented as generalization
proof.

## Router v1.1: independent factual path

The current Router no longer treats `task_relevance` as a value-path switch.
`status`, `intensity`, `direction`, and `temporal_persistence` are encoded and
aggregated through a factual attention path that never reads task relevance. A
second bounded path uses task relevance to shift attention and add a focused
residual. Therefore an all-zero relevance vector removes question modulation but
does not erase the Episode representation. The default attention-alignment loss is
`0.0`; relevance is learned only insofar as it improves the model-activation target.

```text
factual states -> factual nodes -> factual attention -> factual context ---------+
question-conditioned relevance -> task attention -> focused residual -----------+-> activations
```

## Dense all-model score supervision (current design)

The hard primary/auxiliary route above is retained only as a historical
baseline.  The current supervision contract treats one Episode as potentially
activating several thought models at once:

```text
teacher A -> one anchored 0..4 score for every one of the 20 models
teacher B -> one anchored 0..4 score for every one of the 20 models
                         |
                         v
for each model independently:
  abs(A - B) <= max_gap -> target = mean(A, B) / 4
                           loss mask = 1
                           confidence/agreement weight > 0
  abs(A - B) >  max_gap -> target = null
                           loss mask = 0
```

A disagreement therefore masks only one model dimension; it does not discard
the Episode.  Scores are independent sigmoid activations and are never
softmax-normalized to sum to one.  `mask=0` means unknown/disputed, never a
negative label.

Prepare the manifest-bound two-teacher packets:

```powershell
python -m research.phase1.mechanism_questions_v1.finalize_model_scores_v1 prepare `
  --dataset-root D:\LollaNetData\mechanism-question-set-v1.1-codex-pilot `
  --source-manifest D:\LollaNetData\mechanism-question-set-v1.1-codex-pilot\labels\train_development\teacher_chain_manifest.json `
  --output-dir D:\LollaNetData\mechanism-question-set-v1.1-codex-pilot\labels\train_development\model_scores `
  --max-score-gap 1
```

After the two independent score files and their audit ledger are finalized,
train the dense router:

```powershell
python -m research.phase1.microconcept_router_v1.soft_run `
  --dataset-root D:\LollaNetData\mechanism-question-set-v1.1-codex-pilot `
  --output-root D:\LollaNetData\runs\microconcept-router-v1 `
  --run-id dense-model-scores-v1
```

`soft_run.py` always retains all 20 fixed models in the output layer.  Coverage
is reported by model using supervised, disputed, useful-or-higher, and
strong-or-higher counts.  This matters because dense low scores provide useful
negative evidence but do not replace positive examples showing when a
long-tail model should activate.

## Task-relevance dependency audit

Router v1.1 has an independent factual path, so setting every `task_relevance`
value to zero is now a fair ablation of question modulation: factual microconcept
states remain available to the model.

Run the dedicated audit instead:

```powershell
python -m research.phase1.microconcept_router_v1.task_relevance_audit `
  --dataset-root D:\LollaNetData\mechanism-question-set-v1.1-codex-pilot `
  --run-dir D:\LollaNetData\runs\microconcept-router-v1\dense-model-scores-v1-gap1-final `
  --cpu
```

The audit does not read sealed test data. It reports:

- legal-profile replacement and sibling-question swaps;
- distribution-preserving permutation importance for task relevance and each
  factual-state channel;
- matched multi-seed `full`, `no_task`, `task_only`, and `prior_only` retraining;
- model-prior, question-template, and task-relevance nearest-neighbor shortcuts;
- same-fact question-delta accuracy, template concentration, and agreement-mask
  selection bias;
- the annotation-circularity boundary created when teachers see task relevance and
  its derived candidate subgraph.

Weighted MAE is the primary comparison. High-activation F1 is retained only as a
secondary, prevalence-sensitive diagnostic. The resulting report remains an audit
of Weak supervision, not evidence of Gold validity or end-to-end fact understanding.
