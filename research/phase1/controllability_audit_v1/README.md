# Open-response causal controllability audit V1

This package validates the experimental contract for comparing whether open-ended response qualities are merely readable from activations or causally controllable during generation.

The three target axes are fixed and separate: `specificity`, `independence`, and `non_template`. A positive control such as sentiment calibrates pipeline sensitivity but is not a fourth target axis.

## Required comparison

The frozen protocol must include no intervention, prompting, CAA/DiffMean, ReFT-r1 or LoReFT, an SAE subspace, and a random subspace matched on layer, rank, update norm, token window, and injection policy. LoRA may be added as a high-capacity upper baseline.

Readout and steering are different claims. Probe performance supports decodability only. Causal controllability additionally requires generation intervention, a target-axis effect, comparison with matched random subspaces, cross-axis reporting, and preservation of usefulness, validity, and task completion.

## Evaluation boundary

The independent statistical unit is `case_id`; seeds, responses, and tokens are within-case repetitions. Train and development cases must be disjoint. Evaluation remains sealed until the method, intervention policy, annotation procedure, primary comparisons, and statistical analysis are frozen.

The package intentionally contains no private paths or unpublished data. The frozen study-specific charter belongs under the project's private evidence directory.

## Validation

```powershell
python -m research.phase1.controllability_audit_v1.validate_charter `
  <private-charter-path> `
  --inventory-open-data
```

The inventory reads only sources explicitly declared as open train/development data and fails if any other split appears. It never discovers or opens an evaluation path.

