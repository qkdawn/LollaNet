# Microconcept State Encoder v1

This module implements only Phase-1 state encoding:

```text
(q, g, financial factors F, structured MD&A claims C) -> 11 fixed microconcept states
```

It snapshots the existing candidate semantic-relation registry for audit only; it has no
routing, graph message passing, dynamic-edge, new-concept, steering-vector, or
LLM-hidden-state component.  Runtime artifacts are deliberately written outside the repository under
`D:\LollaNetData\runs\microconcept-encoder-v1`.

## Data and leakage boundary

* Candidate records are allowlisted to filing availability, `financial_features`,
  `market_state`, form, accession, and CIK **only for group splitting**.
* `route_labels`, `exploratory_outcome`, issuer/ticker names, and raw MD&A text are
  never exposed to model inputs.
* Claim input is strictly structured metadata (`claim_id`, polarity, horizon,
  strength, confidence) and only claims whose `available_at <= episode.available_at`.
* Statistics are fitted on the temporal training Episodes only.  CIK is discarded
  before tensorisation.
* `unknown` status/direction/applicability and absent fields have a zero loss mask.
  Unknown is never converted to a negative class or intensity zero target.

## Labels and report boundary

The runner pretrains on Weak/Silver and fine-tunes only `Gold` records that pass
human/adjudication metadata checks.  Machine-only rows are never Gold.  When the
provided corpus has no genuine Gold labels or no labeled 2022–2023 test rows, the
runner emits an explicitly exploratory train/development/issuer-holdout report and
will not make a sealed-Gold effectiveness claim.

## Run

```powershell
python -m research.phase1.microconcept_encoder_v1.run `
  --candidates D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --claims D:\LollaNetData\mda-claims-v1\claims.jsonl `
  --labels D:\LollaNetData\microconcept-labels-v1 `
  --output-root D:\LollaNetData\runs\microconcept-encoder-v1 `
  --run-id exploratory-20260719
```

`--run-sealed-test` is refused unless labels for the frozen 2022–2023 Episodes exist
and the sealed ledger is still unused.  This prevents accidental repeated use of a
closed test set.
