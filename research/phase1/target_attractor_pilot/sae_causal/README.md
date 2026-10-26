# Three-axis SAE causal discovery

This package tests whether Qwen3-8B contains causal sparse features or a distributed causal
subspace for three continuous response qualities: specificity, independent judgment, and
non-template expression. It is discovery-only and cannot read a formal evaluation split.

## Stage order

1. Build a blinded independent-judgment packet and combine it with the existing specificity
   and formulaic-expression scores. `non_template = 4 - formulaic`.
2. Capture the mean layer-18 residual over the first 16 response tokens and encode it with the
   official Qwen-Scope TopK-50 SAE. Codes are stored as sparse indices and values.
3. On train only, regress each axis on case/domain, length/completion, visible formatting, and
   the other two labels. Rank SAE features against the residual label and check on development.
4. Intervene in both signs and at four frozen strengths. Compare with active-frequency-matched
   random features. A causal feature must have a monotonic target effect, small cross-axis and
   quality costs, and beat the random controls.
5. Only a passed causal gate unlocks Sparse Feature Circuits and SAE-Targeted Steering. DAS is
   the distributed-subspace comparison: DAS success with SAE failure supports a distributed
   causal subspace rather than a small set of sparse features.

## Current boundary

Model-assisted labels are useful for discovery but do not establish human construct validity.
Association candidates are not causal features. Later stages are deliberately locked until the
generation intervention results satisfy `charter.yaml`.

The official Qwen-Scope encoder is `residual @ W_enc.T + b_enc`, followed directly by TopK-50.
It does not subtract `b_dec` and does not apply ReLU after TopK. `b_dec` belongs to decoding.

On Windows, the extractor checks commit-memory headroom before loading Qwen3-8B. If the original
15.26 GB checkpoint cannot be converted to NF4 because of `os error 1455`, the charter permits a
discovery-only prequantized checkpoint at
`F:\LollaNetData\target-attractor-pilot\checkpoints\qwen3-8b-bnb-4bit`. Results from this resource
contingency require replication on the frozen original revision.

## Commands

The project uses `research/phase1/lollasteer_q/.venv/Scripts/python.exe` for CUDA dependencies.

```powershell
python -m research.phase1.target_attractor_pilot.sae_causal.extract `
  --trajectories F:\LollaNetData\template-attractor-pilot\natural-v4-600-discovery\trajectories.jsonl `
  --model F:\LollaNetData\lollanet-cma\qualification\checkpoints\qwen3-8b `
  --sae F:\LollaNetData\target-attractor-pilot\qwen-scope\topk50\layer18.sae.pt `
  --output F:\LollaNetData\target-attractor-pilot\sae-causal-v1\features
```

After reboot, smoke-test the prequantized contingency on one record first:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.target_attractor_pilot.sae_causal.extract `
  --trajectories F:\LollaNetData\template-attractor-pilot\natural-v4-600-discovery\trajectories.jsonl `
  --model F:\LollaNetData\target-attractor-pilot\checkpoints\qwen3-8b-bnb-4bit `
  --sae F:\LollaNetData\target-attractor-pilot\qwen-scope\topk50\layer18.sae.pt `
  --output F:\LollaNetData\target-attractor-pilot\sae-causal-v1\features `
  --minimum-commit-headroom-gb 4 --record-limit 1 --checkpoint-every 1
```
