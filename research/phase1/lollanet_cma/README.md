# LollaNet-CMA

This module implements the CLADDER oracle-guided causal model activation study described in
`07-Docs/papers/causal-model-activation/01-cladder-oracle-model-activation-draft.md`.
It is separate from the MMLU-Pro cue-susceptibility study under `lollasteer_q`.

Stage 0 freezes the public upstream data, allocates complete `model_id` groups with a MILP,
and creates three isolated artifacts:

- `prompts.jsonl`: only `background + given_info + question` and routing-neutral record fields;
- `controller.jsonl`: only `query_type` and its frozen four-family mapping;
- `evaluator.jsonl`: answer and evaluation metadata, without source reasoning.

The upstream ZIP uses the filename `cladder-v1-q-noncommonsense.json` for the nonsense
variant. The charter records the exact real filename and hash instead of the README alias.
The solver may accept a time-limited integral incumbent only after rechecking every hard
group and family-count constraint. Its optimality status, objective and MIP gap remain in
`split_manifest.json`; a feasible split is not described as globally optimal.

```powershell
uv run --project research/phase1/lollanet_cma python -m research.phase1.lollanet_cma.prepare_stage0 `
  --source-root D:\LollaNetData\lollanet-cma\frozen-data\cladder-v1-3d2d1169 `
  --output D:\LollaNetData\lollanet-cma\stage0\bundle-v1

uv run --project research/phase1/lollanet_cma python -m research.phase1.lollanet_cma.validate_stage0 `
  D:\LollaNetData\lollanet-cma\stage0\bundle-v1
```

This stage does not produce model results. Qualification, vector fitting, development
selection, sealed evaluation, OOD evaluation, probes, and statistical claims remain gated.

## Qualification readiness

The qualification implementation is isolated under `qualification/` with its own charter,
dependency lock, parser, residual-hook smoke test, runner, selection audit, and tests. The
readiness audit must validate Stage 0 in the frozen Stage 0 preparation environment, then
record the separate qualification runtime before any candidate can run.

The current auditable artifact is:

```text
D:\LollaNetData\lollanet-cma\qualification\qualification-readiness-v6.json
```

The current artifact is `qualification-readiness-v6.json`, with content hash
`cd9ead17351f7da7ba649468e5e229334bae44ea50ad08fcb0cbe3d4f5522a3a`. Its status is
`ready`: Stage 0 validates with 3,200 records, zero cross-split `model_id` overlap, and exact
four-family balance; CUDA, bfloat16, and `bitsandbytes` NF4 support are available; and all
three preregistered checkpoints have exact revisions and file-level hashes. The revisions are:

- `Qwen/Qwen3-4B-Instruct-2507`: ModelScope commit `2de2439ea21be1dc5cb21f22f88af07e43393cbb`;
- `Qwen/Qwen2.5-3B-Instruct`: ModelScope commit `8f4992eda43eea7c770690ddc0de8f732da246f5`;
- `microsoft/Phi-3.5-mini-instruct`: ModelScope commit `15257cc2edeb31c5bd2b43acc5f5e6289cc4f4ba`.

An audit found that the first attempted 4B process had loaded the consolidated Stage 0
controller and evaluator files before filtering to qualification. Although it produced only
13 hook-smoke rows and no qualification predictions or run manifest, the process had access to
non-qualification labels. It was stopped and preserved under
`qualification/invalidated/20260724-holdout-exposure/`; its invalidation manifest marks it
ineligible for selection and empirical claims.

The remediation materialized and validated separate access views for qualification (200
records), vector fitting (800), and development (400). No sealed or OOD access view has been
materialized. Candidate runners now accept only `--access-view`; the v5 qualification view
manifest file SHA-256 is
`57a6054fb60ec57ab322bbc385444800e2be31fde896b69fc93751b0872cd18a`.

The v0.2.0 Thinking/R1 candidate set was retired after the completed 4B Thinking run exhausted
the 512-token budget on all 200 records and failed to emit a parseable final answer on 199. The
v0.3.0 charter requires `reasoning_mode: disabled` and the runner rejects Thinking and R1-Distill
model IDs. The three replacement checkpoints are frozen under `qualification/checkpoints-v3/`;
no candidate run is currently active. Until all three new manifests pass selection, no base model
winner or empirical model comparison exists.

Generate a new readiness artifact only from a runtime matching `qualification/uv.lock`:

```powershell
<qualification-python> -m research.phase1.lollanet_cma.qualification.audit_readiness `
  --stage0-bundle D:\LollaNetData\lollanet-cma\stage0\bundle-v4 `
  --output D:\LollaNetData\lollanet-cma\qualification\qualification-readiness-v6.json
```

## Direction fitting

Stage 1 is implemented under `directions/` and remains fail-closed behind the qualification
selection. Its charter permits only `vector_train` for fitting, balances gold Yes/No labels
within each family after correctness and margin filtering, freezes three candidate layers,
and records a separate normalized family direction at every layer. Each record and contrast
tensor is persisted with a content hash; every final vector records its own tensor hash,
dtype, shape, norm, model revision, prompt charter and execution identity.

The current audited preflight artifact is
`D:\LollaNetData\lollanet-cma\directions\direction-preflight-v2.json`. Its status is
`blocked_missing_qualification`, specifically because no complete qualification selection
artifact exists yet. This is the expected gate state, not a direction result. The formal
direction runner and a fresh v3 preflight are queued only after the three candidate manifests,
frozen selection and tokenizer/checkpoint provenance all validate.

## Development selection

Development selection is implemented under `development/` and remains fail-closed behind a
successful direction manifest. It can read only the 400-record `development-v1` access view.
The frozen joint grid contains three direction layers and alpha values
`0.25, 0.5, 1.0, 1.5, 2.0`; it selects by macro-family accuracy with lower-alpha and
shallower-layer tie-breaks. Wrong-family and norm-matched random controls run only at that
single frozen setting. Specificity failure does not trigger fallback to a different grid point.

`D:\LollaNetData\lollanet-cma\development\development-preflight-v1.json` currently records
`blocked_missing_directions`. A fresh preflight and the formal development run are queued after
direction fitting, and will stop without running if any upstream hard gate fails. No development
selection or specificity result currently exists.

## Sealed and OOD release

The `evaluation/` package adds a separate authorization boundary before any sealed or OOD labels
can be materialized. Its preflight revalidates the complete frozen development run, every
development condition row and hash, the selected grid artifact, direction tensors and checkpoint,
the development-only access view, code identity, dependency lock, and runtime. It authorizes
unsealing only when development has `status: frozen`, every hard gate passes, and direction
specificity passes.

The current artifact is
`D:\LollaNetData\lollanet-cma\evaluation\sealed-release-preflight-v1.json`, with content hash
`f69f4c184c5167bb15c9e1d7fa932c47a1c7c8eaf057acd2ddbb6a5124adb25b`. Its expected current
state is `blocked_missing_development`, with `unseal_authorized: false` and `stage0.validation:
null`. Thus the blocked audit does not inspect or materialize sealed/OOD records. A release is
queued after development; it will either stop at the failed prerequisite or create three isolated,
hash-bound views for `sealed_test`, `ood_anticommonsense`, and `ood_nonsense`. Evaluation runners
must receive only these released views, never the consolidated Stage 0 bundle.

```powershell
<qualification-python> -m research.phase1.lollanet_cma.evaluation.audit_release `
  --stage0-bundle D:\LollaNetData\lollanet-cma\stage0\bundle-v4 `
  --development-manifest D:\LollaNetData\lollanet-cma\development\run-v1\run_manifest.json `
  --output D:\LollaNetData\lollanet-cma\evaluation\sealed-release-preflight-v2.json

<qualification-python> -m research.phase1.lollanet_cma.evaluation.release `
  --stage0-bundle D:\LollaNetData\lollanet-cma\stage0\bundle-v4 `
  --preflight D:\LollaNetData\lollanet-cma\evaluation\sealed-release-preflight-v2.json `
  --output-manifest D:\LollaNetData\lollanet-cma\evaluation\evaluation-release-v1.json
```

## Sealed evaluation and probe

The `sealed_eval/` package freezes the raw condition contract, independent MMLU-Pro control,
token-level KL alignment, hidden-state probe, report statistics, and phase-two success gates.
Probe fitting is a separate post-development process: it may read only the existing
`vector_train` and `development` access views, and it freezes its weights before sealed
evaluation. The sealed runner accepts only an authorized evaluation preflight, the three
released views, the frozen probe, and the independent control bundle. It has no Stage 0
argument.

All generation and canonical answer scoring use batch size 1. For activation-intervention KL,
each decoding step performs sequential steered and unsteered batch-one forwards with identical
text prefixes and separate condition-consistent KV caches. The steered greedy token advances
both text trajectories. Deterministic model conditions run once; `oracle_correct`,
`oracle_wrong`, and `shuffled_mapping` are hash-linked derivations from the four complete
family-direction runs.

```powershell
<qualification-python> -m research.phase1.lollanet_cma.sealed_eval.prepare_probe `
  --development-manifest D:\LollaNetData\lollanet-cma\development\run-v1\run_manifest.json `
  --vector-train-view D:\LollaNetData\lollanet-cma\access-views\vector-train-v1 `
  --development-view D:\LollaNetData\lollanet-cma\access-views\development-v1 `
  --output D:\LollaNetData\lollanet-cma\evaluation\probe-v1

<qualification-python> -m research.phase1.lollanet_cma.sealed_eval.audit_preflight `
  --evaluation-release D:\LollaNetData\lollanet-cma\evaluation\evaluation-release-v1.json `
  --probe-manifest D:\LollaNetData\lollanet-cma\evaluation\probe-v1\run_manifest.json `
  --output D:\LollaNetData\lollanet-cma\evaluation\sealed-evaluation-preflight-v2.json

<qualification-python> -m research.phase1.lollanet_cma.sealed_eval.run_evaluation `
  --preflight D:\LollaNetData\lollanet-cma\evaluation\sealed-evaluation-preflight-v2.json `
  --output D:\LollaNetData\lollanet-cma\evaluation\raw-run-v1

<qualification-python> -m research.phase1.lollanet_cma.sealed_eval.assemble_report `
  --raw-manifest D:\LollaNetData\lollanet-cma\evaluation\raw-run-v1\run_manifest.json `
  --output D:\LollaNetData\lollanet-cma\evaluation\report-v1
```

The report recomputes all metrics from complete hash-verified rows. It does not trust runner
summaries, and it marks `phase2_eligible` only if every preregistered necessary gate passes.
Until a complete report manifest exists, no sealed/OOD, probe, negative-transfer, or phase-two
claim exists.

Once a split manifest has passed validation, reuse it rather than solving a new random
assignment after documentation or provenance changes:

```powershell
uv run --project research/phase1/lollanet_cma python -m research.phase1.lollanet_cma.prepare_stage0 `
  --source-root D:\LollaNetData\lollanet-cma\frozen-data\cladder-v1-3d2d1169 `
  --frozen-split-manifest D:\LollaNetData\lollanet-cma\stage0\bundle-v2\split_manifest.json `
  --output D:\LollaNetData\lollanet-cma\stage0\bundle-v3
```
