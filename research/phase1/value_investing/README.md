---
type: research_data_contract
scope: phase1_value_investing
---

# Value-Investing Episodes

This directory defines a point-in-time, long-only value-investing research dataset.
It does not download market data, execute trades, train a model, or provide
investment advice.

Each Episode represents a non-financial US common share at a quarterly SEC filing
snapshot. Inputs are financial-statement features, market state, and references to
selected MD&A sections. The 12-month outcome is stored only after the route label
has been derived from pre-registered rules.

## Data Boundary

- SEC EDGAR provides filing identity, acceptance timestamps, XBRL provenance, and
  MD&A locators.
- The free pilot uses Stooq daily price data only for exploratory price-return
  outcomes. It does not claim complete delisting-return coverage.
- A licensed CRSP/Compustat-compatible provider remains necessary for point-in-time
  financial links, complete delisting returns, and research-grade return claims.
- Raw filings, licensed market data, and complete MD&A text stay outside this
  repository. The fixture is synthetic and uses no real security or filing.

## Commands

```powershell
python research/phase1/value_investing/validate_dataset.py
```

The command validates the anonymous fixture, chronology, outcome fields, and route
labels against `route_rules.yaml`.

## Free Pilot Access

Set `SEC_USER_AGENT` to a truthful identifier containing your contact email before
calling the SEC client. No SEC request is made until that variable is present.
Downloaded JSON and CSV files must be written outside this repository.

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@csu.edu.cn"
python research/phase1/value_investing/download_free_pilot.py `
  --cik 320193 --ticker AAPL --output D:\LollaNetData\free-pilot\apple
```

If Stooq returns an anti-automation page, create a free key at
`https://www.alphavantage.co/support/#api-key` and use:

```powershell
$env:ALPHAVANTAGE_API_KEY = "your-free-key"
python research/phase1/value_investing/download_free_pilot.py `
  --cik 320193 --ticker AAPL --price-source alpha_vantage `
  --output D:\LollaNetData\free-pilot\apple
```

The command is an acquisition aid, not a backtest. Alpha Vantage's free endpoint
supplies raw monthly closes, not split- and dividend-adjusted total-return prices;
it cannot support day-level entry timing or precise drawdown. Both free price
sources are exploratory and do not supply complete delisting-return coverage.

## Batch SEC Universe Collection

The batch collector creates a reproducible candidate sample from the SEC ticker
directory, excludes SIC 6000--6799 financial issuers, requires at least two
recent 10-K/10-Q records, and stores the acceptance/rejection reason in an
external `manifest.json`. It is resumable: accepted and rejected issuers are not
requested again; failed issuers may be retried by running the same command.

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@university.edu"
python research/phase1/value_investing/collect_sec_universe.py `
  --target 200 --output D:\LollaNetData\sec-universe-200
```

The collector limits itself to eight SEC requests per second and writes only
SEC submissions metadata and Company Facts JSON. It does not collect price data,
make a securities eligibility claim, or create return labels. The next step is a
point-in-time Episode materializer, which must explicitly mark unavailable market
features and outcomes until a suitable price and delisting-return source exists.

## Batch MD&A Download and Locator

SEC filing HTML is public and can be cached outside the repository. The downloader
uses each candidate's accession and cached SEC submissions metadata, stores raw
HTML under `D:\LollaNetData`, and writes only locator metadata plus hashes to
`mda_index.jsonl`. It extracts 10-K Item 7 and 10-Q Item 2 without copying the
text into the repository.

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@university.edu"
python research/phase1/value_investing/download_sec_mda.py `
  --input D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --raw-root D:\LollaNetData\sec-universe-200 `
  --output D:\LollaNetData\sec-mda-v1 `
  --workers 8 --delay 0.12
```

The command is resumable. Unlocated legacy filing formats remain explicit in
`download_report.json`; they are not treated as successful MD&A coverage.

## Partial-Supervision Router Experiment

The first neural-router experiment consumes the external market-valued candidate
file, predicts the four rules that have adequate observable coverage, and derives
the explicit model graph from those probabilities. It is a rule-distillation
experiment with an exploratory free-price auxiliary task; it is not an investment
strategy or a performance claim.

```powershell
python -m pip install -r research/phase1/value_investing/requirements-router.txt
python research/phase1/value_investing/train_router.py `
  --input D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --output D:\LollaNetData\runs\phase1-router-v1
```

The experiment uses 2017--2020 for training, 2021 for development selection, and
2022--2023 as a sealed test. Its manifest records the data fingerprint, train-only
preprocessing statistics, thresholds, model-selection result, and source limits.
Weights and predictions are always external to this repository.

## Blinded LLM Teacher Labels

LLM teacher labels are pseudo-labels, not human ground truth. The preparation
step forms a stratified external sample and removes issuer identity, existing
route labels, model predictions, and future outcomes before any teacher sees a
packet. All 20 Phase 1 model assets may compete, but physics and chemistry models
require a visible, non-null financial-variable mapping to their mechanism.

```powershell
python research/phase1/value_investing/prepare_llm_teacher_sample.py `
  --input D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --output D:\LollaNetData\llm-teacher-sample-v1

python research/phase1/value_investing/validate_llm_teacher_labels.py `
  --sample-root D:\LollaNetData\llm-teacher-sample-v1 `
  --labels-root D:\LollaNetData\llm-teacher-sample-v1\labels `
  --output D:\LollaNetData\llm-teacher-sample-v1\teacher_labels.jsonl
```

The validator rejects unknown or duplicated models, invalid orderings, ungrounded
cross-domain mappings, and evidence fields that were not visible to the teacher.
It then copies source values from the blinded packet into structured grounded
evidence, so training never depends on numbers repeated in free-form rationales.

Teacher labels can be trained without converting unsupported models into negative
examples. The trainer enables only heads with at least 20/5/5 positive labels in
the train/development/sealed-test windows; every other model remains present with
an explicit unsupported status and a null probability.

```powershell
python research/phase1/value_investing/train_llm_teacher_router.py `
  --candidates D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --teacher-labels D:\LollaNetData\llm-teacher-sample-v1\teacher_labels.jsonl `
  --output D:\LollaNetData\runs\llm-teacher-router-v1
```

The network jointly predicts model activation, selected-model count, and relative
execution rank. It reports model-level retrieval separately from exact route and
pairwise-order agreement because exact route matching is much stricter.
