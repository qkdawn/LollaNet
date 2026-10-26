# Template Attractor Pilot

This pilot tests whether content-matched formulaic and specific responses are
separable in early hidden states, and whether a CAA direction can be applied
conditionally after an early generated prefix.

It is intentionally a discovery experiment. Synthetic pairs and rule-based
generation metrics are not evidence for the final paper without blinded human
annotation and a larger held-out task set.

Run:

```powershell
D:\Tools\Python\python.exe -m research.phase1.template_attractor_pilot.pilot `
  --checkpoint D:\LollaNetData\lollasteer-q\models\Qwen3-1.7B `
  --output D:\LollaNetData\template-attractor-pilot\smoke-v1 `
  --smoke
```

The split holds out the `balance` template family for development and the
`holistic` family for evaluation. Candidate layer, early window, and gate
threshold are selected on development only.

After the synthetic-pair diagnostic, collect neutral-prompt natural trajectories:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.template_attractor_pilot.natural_trajectories `
  --checkpoint D:\LollaNetData\lollasteer-q\models\Qwen3-1.7B `
  --output D:\LollaNetData\template-attractor-pilot\natural-v1 `
  --samples-per-case 2 --max-new-tokens 384 --load-in-4bit
```

This stage stores early activations but does not derive labels from surface
rules. It creates two independently randomized blind annotation packets. Probe
fitting remains blocked until both score sheets are completed independently.
The visible 4/8/16-token prefixes are frozen as a separate hashed artifact, so
the text baseline uses exactly the same observed prefix as the activation probe.
The annotation delivery is also blocked when more than 10% of responses exhaust
the token budget, because truncation can be mistaken for concise or natural
expression.

After both independent score sheets are complete, validate agreement and create
consensus labels (or a blinded adjudication packet for differences above one
point):

```powershell
D:\Tools\Python\python.exe -m research.phase1.template_attractor_pilot.labels `
  --annotation D:\LollaNetData\template-attractor-pilot\natural-v2\annotation `
  --trajectories D:\LollaNetData\template-attractor-pilot\natural-v2\trajectories.jsonl `
  --output D:\LollaNetData\template-attractor-pilot\natural-v2\labels
```

The binary probe contract is frozen before annotation: scores at or below 1.5
are negative, scores at or above 2.5 are positive, and intermediate cases are
excluded from the binary detector without being deleted from the dataset.

For one annotator at a time, launch the isolated local rating interface:

```powershell
D:\Tools\Python\python.exe -m research.phase1.template_attractor_pilot.annotation_app `
  --packet D:\LollaNetData\template-attractor-pilot\natural-v2\annotation\annotator_a_packet.jsonl `
  --scores D:\LollaNetData\template-attractor-pilot\natural-v2\annotation\annotator_a_scores.csv `
  --rubric D:\LollaNetData\template-attractor-pilot\natural-v2\annotation\rubric.md `
  --port 8765
```

The server binds to localhost only, reads no private mapping, and atomically
updates the selected annotator's existing CSV after every item.
Launch annotator B separately with the `annotator_b_packet.jsonl` and
`annotator_b_scores.csv` paths on a different port such as 8766. The two
annotators must not view or discuss each other's interface or scores.

Once `label_manifest.json` reports `ready_for_probe_fit`, fit on train and
development only:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.template_attractor_pilot.natural_probe fit `
  --label-manifest D:\LollaNetData\template-attractor-pilot\natural-v2\labels\label_manifest.json `
  --trajectory-manifest D:\LollaNetData\template-attractor-pilot\natural-v2\run_manifest.json `
  --output D:\LollaNetData\template-attractor-pilot\natural-v2\probe
```

Evaluation remains closed unless the selected activation probe exceeds the
same-window character baseline by at least 0.02 development AUROC and is not
worse in AUPRC. A frozen fit can then be released exactly once with the
`natural_probe evaluate` subcommand.

For discovery work when manual annotation is not available, a completed blind
score sheet from a named language model can be converted without representing
it as independent human annotation:

```powershell
D:\Tools\Python\python.exe -m research.phase1.template_attractor_pilot.labels `
  --annotation D:\LollaNetData\template-attractor-pilot\natural-v2\annotation `
  --trajectories D:\LollaNetData\template-attractor-pilot\natural-v2\trajectories.jsonl `
  --output D:\LollaNetData\template-attractor-pilot\natural-v2\labels-codex-5.6 `
  --model-scores D:\LollaNetData\template-attractor-pilot\natural-v2\annotation\codex_5_6_scores.csv `
  --model-id "OpenAI Codex 5.6" --packet annotator_a
```

Such a manifest reports `ready_for_discovery_probe_fit`. Fitting requires the
explicit `natural_probe fit --allow-model-assisted` flag, records the evidence
tier as `discovery_only_model_assisted`, and never permits evaluation release.

Before treating a passed discovery gate as motivation for intervention work,
audit development stability at the case level:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.template_attractor_pilot.natural_probe_audit `
  --fit-manifest D:\LollaNetData\template-attractor-pilot\natural-v3-200\probe-codex-5.6\fit_manifest.json `
  --output D:\LollaNetData\template-attractor-pilot\natural-v3-200\probe-codex-5.6\development_stability_audit.json
```

The audit resamples complete `case_id` clusters, reports leave-one-case-out
sensitivity, and counts the independent cases supporting each class. It never
reads evaluation records for metrics and cannot upgrade model-assisted labels
to confirmatory evidence.

If the discovery sample is expanded with additional responses per case, create
an evaluation-free train/development artifact before model-assisted annotation:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.template_attractor_pilot.discovery_subset subset `
  --run-manifest F:\LollaNetData\template-attractor-pilot\natural-v4-800\run_manifest.json `
  --output F:\LollaNetData\template-attractor-pilot\natural-v4-600-discovery
```

Exact responses already scored in an earlier blind pass can retain those
scores. Seed them into the new anonymous packet and emit a packet containing
only previously unseen responses:

```powershell
research\phase1\lollasteer_q\.venv\Scripts\python.exe `
  -m research.phase1.template_attractor_pilot.discovery_subset seed-scores `
  --target-annotation F:\LollaNetData\template-attractor-pilot\natural-v4-600-discovery\annotation `
  --prior-annotation F:\LollaNetData\template-attractor-pilot\natural-v3-200\annotation `
  --prior-scores F:\LollaNetData\template-attractor-pilot\natural-v3-200\annotation\codex_5_6_scores.csv `
  --output-scores F:\LollaNetData\template-attractor-pilot\natural-v4-600-discovery\annotation\codex_5_6_scores.csv
```

The seeded-score manifest records hashes and counts. Its remaining packet still
contains only anonymous item IDs, prompts, and responses. Evaluation responses
are absent rather than merely ignored downstream.
