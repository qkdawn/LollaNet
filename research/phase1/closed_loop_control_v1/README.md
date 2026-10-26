# Closed-loop latent control V1

This package implements the state-estimation and inference-time control contract for a train/development-only engineering pilot. It does not claim that the base language model contains an autonomous attractor.

The generation hook performs operations in a fixed order at each cached token step:

1. apply a scheduled perturbation;
2. estimate the rolling, prompt-conditioned response-quality state;
3. open or close the uncertainty-aware deviation gate;
4. apply a norm- and energy-bounded low-rank update;
5. record the complete state and control trace.

`specificity`, `independence`, and `non_template` remain separate observer outputs and control targets. Internal-state recovery is reported separately from blinded behavioral recovery.

The first model run must be a positive-control implementation check. A specificity smoke may then use train-only cases. Development selects a frozen controller; formal evaluation remains inaccessible.

Dry-run the positive control before loading a model:

```powershell
python -m research.phase1.closed_loop_control_v1.sentiment_smoke `
  --output <private-output-directory> --dry-run
```

The positive control constructs a sentiment contrast from matched review responses, applies a reverse-direction perturbation during generation, and compares disabled, always-on, closed-loop, and withdrawal policies. This validates the execution path only; text behavior still requires blind scoring.
