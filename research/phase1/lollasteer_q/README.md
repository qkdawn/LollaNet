# LollaSteer-Q Pilot

This module implements the preregistered 90-item feasibility pilot. It does not run a
sealed confirmatory experiment and does not implement SAE or a learned gate.

## Leakage boundary

- `prompts.jsonl` is the only dataset read by the model runner. It never contains gold,
  lure, `answer_index`, `cot_content`, or source metadata.
- `evaluator.jsonl` is joined only by the reporting process.
- Lures must be frozen by an independent selector that explicitly did not inspect the
  target model outputs.
- Runtime data belongs under `D:\LollaNetData\lollasteer-q`, outside this repository.

## Environment

```powershell
uv sync --project research/phase1/lollasteer_q
```

The preregistered model is `Qwen/Qwen3-1.7B` (Apache-2.0, non-gated). The lockfile
uses Python 3.11 and keeps ML dependencies out of the repository root environment.
The runner does not silently substitute another checkpoint.

The canonical model revision is frozen separately from the local load path. If the
local snapshot is missing or its Hugging Face revision metadata differs, the run stops
instead of downloading a newer snapshot.

Every run also records the `uv.lock` hash, a content hash over the runtime Python modules
and `pyproject.toml`, Python/platform identity, PyTorch,
Transformers and NumPy versions, CUDA runtime and device inventory, and the actual model
device map. These fields are part of the execution identity. Linear runs must match the
environment identity of their fitted direction bundle.

Within each base item, option order is deterministically permuted under
`sha256_seeded_per_base_item_v1`; gold and lure indices are remapped before prompt and
evaluator records are separated. Every condition for that item shares the same order.

## Commands

Prepare and validate the blinded lure-ranking packets before building the pilot:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.prepare_annotation_packets `
  --config research/phase1/lollasteer_q/config_pilot.yaml `
  --output D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation `
  --annotator annotator_a --annotator annotator_b --annotator annotator_c

uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.validate_annotation_packets `
  --annotation-dir D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation
```

The annotation manifest freezes a minimal annotation contract: source revision, seed,
categories, candidate/final counts, annotator identities, display randomization, and packet
instructions. Model grids, reporting thresholds, and documentation are recorded separately and
may evolve without invalidating unchanged human packets. Legacy unstarted packets can be migrated
without rewriting any packet or withheld mapping:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.migrate_annotation_contract `
  --annotation-dir D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation `
  --config research/phase1/lollasteer_q/config_pilot.yaml
```

Migration refuses to run after any `annotations-*.jsonl` or disagreement packet exists.

Each human annotator can complete their own randomized packet with the resumable CLI.
First export two isolated primary deliveries. Each directory contains only one packet and its
instructions; the validator rejects withheld mappings, evaluator labels, model outputs, and
activations.

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.prepare_annotation_delivery `
  --annotation-dir D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation `
  --output D:\LollaNetData\lollasteer-q\formal-pilot\annotation-delivery

uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.validate_annotation_delivery `
  --delivery-dir D:\LollaNetData\lollasteer-q\formal-pilot\annotation-delivery
```

The annotation command reads only the named delivery packet:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.annotate_lures `
  --packet D:\LollaNetData\lollasteer-q\formal-pilot\annotation-delivery\annotator_a\packet.jsonl `
  --output D:\LollaNetData\lollasteer-q\formal-pilot\annotation-delivery\annotator_a\annotations.jsonl
```

Enter a complete best-to-worst permutation such as `C A B D`, or enter
`exclude <reason>` only for ambiguity, broken options, strongly time-sensitive content or
normative judgment. Enter `quit` to stop; every completed row is atomically saved and the next
run resumes from the first unfinished item. Annotators must work independently and must not view
each other's output, `mapping-withheld.json`, target-model outputs or activations.

After both isolated results are complete, collect them into the central hash-linked annotation
directory. Collection refuses incomplete packets and never overwrites different existing results:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.collect_annotation_delivery `
  --delivery-dir D:\LollaNetData\lollasteer-q\formal-pilot\annotation-delivery `
  --annotation-dir D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation
```

`annotator_a` and `annotator_b` are the primary independent rankings. Use
`annotator_c` only when the primaries disagree on the highest-ranked wrong option after the
withheld mapping removes gold. This avoids treating shared gold-first rankings as agreement on
the lure. After both primary files are complete, create the reduced third-annotator packet:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.prepare_tiebreak_packet `
  --annotation-dir D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation `
  --output D:\LollaNetData\lollasteer-q\formal-pilot\lure-annotation\packet-annotator_c-disagreements.jsonl
```

The generator validates the frozen packet hashes and both complete primary annotation files,
then writes a provenance manifest beside the reduced packet. A disagreement about whether an
item should be excluded remains explicit author input; it is not silently delegated to the
third annotator. Adjudication writes both the
frozen lure YAML and a hash-linked JSON audit report; it requires an author-supplied
timezone-aware `--frozen-at` value.

Build and validate the pilot bundle:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.build_pilot `
  --lures D:\LollaNetData\lollasteer-q\lures.yaml `
  --output D:\LollaNetData\lollasteer-q\pilot

uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.validate_pilot `
  D:\LollaNetData\lollasteer-q\pilot
```

Run the frozen complete baseline and fit discovery directions. Direction fitting reads only
discovery activations, but requires the complete baseline manifest so a smoke or truncated run
cannot become the formal direction source:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.run_model `
  --prompts D:\LollaNetData\lollasteer-q\pilot\prompts.jsonl `
  --output D:\LollaNetData\lollasteer-q\runs\baseline

uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.fit_directions `
  --prompts D:\LollaNetData\lollasteer-q\pilot\prompts.jsonl `
  --activations D:\LollaNetData\lollasteer-q\runs\baseline\activations.npz `
  --run-manifest D:\LollaNetData\lollasteer-q\runs\baseline\run_manifest.json `
  --output D:\LollaNetData\lollasteer-q\directions\pilot.npz
```

Before any holdout evaluation, run the baseline and every preregistered linear candidate with
`--split discovery`. The selection command rejects any input containing holdout records and
requires the complete CAA/low-rank Cartesian grid from `layer_fractions`, `strength_grid`, and
`low_rank_grid`. It writes a content-hashed frozen selection manifest.

CAA and low-rank share discovery items, candidate layers, and strengths. Low-rank has the
additional preregistered rank axis, so the total candidate counts are intentionally unequal and
are recorded in the selection manifest; the protocol does not claim equal hyperparameter trials.

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.run_model `
  --prompts D:\LollaNetData\lollasteer-q\pilot\prompts.jsonl `
  --output D:\LollaNetData\lollasteer-q\selection-runs\baseline `
  --split discovery

# Repeat for every preregistered method/factor/layer/strength/rank candidate.
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.run_model `
  --prompts D:\LollaNetData\lollasteer-q\pilot\prompts.jsonl `
  --output D:\LollaNetData\lollasteer-q\selection-runs\caa-reward-l7-s1 `
  --method caa --factor reward --layer 7 --strength 1 --rank 1 `
  --directions D:\LollaNetData\lollasteer-q\directions\pilot.npz `
  --split discovery

uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.select_discovery `
  --evaluator D:\LollaNetData\lollasteer-q\pilot\evaluator.jsonl `
  --baseline-output D:\LollaNetData\lollasteer-q\selection-runs\baseline\outputs.jsonl `
  --candidate-output <repeat once for every candidate outputs.jsonl> `
  --output D:\LollaNetData\lollasteer-q\selection\frozen.json
```

Only after `frozen.json` exists, run the selected configurations over the complete prompt bundle.
Then pass the complete baseline and selected complete runs to the report command. The reporter
rejects truncated outputs, duplicate run keys, incompatible checkpoints/runtime settings,
unselected linear configurations, and direction bundles that differ from the frozen selection.
The formal configuration also requires all three prompt-only comparators and all six selected
CAA/low-rank factor runs, preventing favorable conditions from being reported alone.

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.generate_report `
  --evaluator D:\LollaNetData\lollasteer-q\pilot\evaluator.jsonl `
  --run-output D:\LollaNetData\lollasteer-q\runs\baseline\outputs.jsonl `
  --run-output D:\LollaNetData\lollasteer-q\runs\caa-reward\outputs.jsonl `
  --selection D:\LollaNetData\lollasteer-q\selection\frozen.json `
  --output D:\LollaNetData\lollasteer-q\reports\pilot
```

Use `--limit` only for model-loading smoke tests. A limited run is not a pilot result.

Audit all existing runtime artifacts after changing execution, dependency, scoring or
intervention contracts. The audit quarantines legacy or invalid artifacts and reports
formal eligibility without upgrading limited smoke runs:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.audit_runtime `
  --runtime-root D:\LollaNetData\lollasteer-q `
  --output D:\LollaNetData\lollasteer-q\reports\runtime-audit-v0.3.json
```

`prompt_only` is an oracle-gated text-only comparator for one active factor. It needs
`--factor` but no direction file or layer; its instruction hash and gate policy are
stored in the run manifest. The pilot comparison still requires the same discovery and
holdout boundaries as linear steering. For a selected factor, the oracle evaluation
schedule activates both on its harmful single-factor cell and its matching valid control
(`valid_reward`, `valid_social` or `calibration`). The report verifies that intervention
actually occurred before using that control for preservation. Neutral cells remain gate-off;
their unchanged accuracy is a schedule-integrity check, not learned-gate safety evidence.

Activations are captured at the final prompt token before intervention. Linear
interventions affect only the last sequence position on each forward pass. Option
probabilities use token-by-token teacher forcing over the complete `Answer: X` prefix;
all three policies are recorded in each run manifest.

Go/no-go decisions use 10,000 item-cluster bootstrap resamples. Construct and intervention
families use Bonferroni-adjusted paired intervals: harmful-effect improvement must clear both its
minimum effect and a lower bound above zero, while valid-control and neutral preservation losses
must have upper bounds below the preregistered maximum. Prompt-only may pass its own comparator
gate but never counts as internal steering. Expansion requires qualifying internal steering for
reward, social, and certainty; low-rank superiority is reported separately and requires its
preregistered increment over CAA.

Run a design-only interaction power sensitivity analysis by supplying assumptions
explicitly; the command never reads model outputs:

```powershell
uv run --project research/phase1/lollasteer_q python -m research.phase1.lollasteer_q.power_analysis `
  --items 90 100 300 --baseline-probability 0.20 --interaction-effect 0.05 `
  --simulations 10000 --output D:\LollaNetData\lollasteer-q\design-simulations\power.json
```

The result is labelled `design_simulation_not_empirical_evidence`. Replace the
illustrative assumptions with pilot-estimated item-cluster variance before freezing
the sealed sample-size decision.
