# Target-attractor smoke pilot

This discovery-only pilot tests whether a short residual-stream intervention leaves a
measurable target-state trace after release and whether that trace recovers after a
subsequent negative-direction perturbation. Continuous steering alone is not called an
attractor. Persistence is measured at a separately fitted downstream monitor layer, not
at the intervention layer itself.

Run the frozen smoke condition:

```powershell
python -m research.phase1.target_attractor_pilot.runner `
  --checkpoint D:\LollaNetData\lollasteer-q\models\Qwen3-1.7B `
  --output F:\LollaNetData\target-attractor-pilot\qwen3-1.7b-smoke-v1 `
  --strength 2.0 --case-limit 2 --maximum-new-tokens 80
```

The direction-fitting set uses only synthetic content-matched training cases. Generation
uses development cases only. This package does not expose or read formal evaluation data.

## Factorized subspace causal smoke

The subspace experiment replaces the single mean direction with separately fitted,
potentially overlapping low-rank spaces for specificity, independence and non-template
expression. It first validates continuous causal control; persistence and recovery remain
locked until the behavioural gate passes.

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.target_attractor_pilot.subspace_runner `
  --checkpoint D:\LollaNetData\lollasteer-q\models\Qwen3-1.7B `
  --output F:\LollaNetData\target-attractor-pilot\qwen3-1.7b-subspaces-v2 `
  --cache-dir F:\LollaNetData\target-attractor-pilot\qwen3-1.7b-subspaces-v1\cache `
  --gain 0.02 --maximum-delta-norm 1.0 --begin-after-tokens 16 `
  --case-limit 2 --maximum-new-tokens 384
```
