---
name: lollanet-analyze
description: "Use LollaNet's typed mechanism graph for complex causal, strategic, predictive, or cross-domain questions. Build two falsifiable explanations from task facts, compare their predictions, and show the models and relations actually used. Skip simple factual or execution tasks."
---

# Analyze With LollaNet

Read `04-Prompts/LollaNet Analysis Protocol.md`.

## 1. State Facts

Write one question and a minimal list of sourced observations. Mark each as `fact`, `inference`, or `unknown`. Model cards are not evidence for the task.

## 2. Build Two Mechanism Chains

For the same question, build at least two competing programs. Each needs only:

- one falsifiable claim;
- 2-5 models, each with a short role and observation IDs;
- relations connecting every model;
- one timed prediction that distinguishes the rival;
- one falsifier.

Use a model only when removing it changes the mechanism or prediction. Use `requires`, `composes`, `competes`, `corrects`, or `specializes`; unregistered task-local relations need one short `why`.

## 3. Compile

Start from `03-Templates/context_pack_template.json`:

```text
python 06-Scripts/model_graph.py --root . compile-context --input <pack.json-or->
```

The compiler reads mechanism excerpts from model cards, validates versions and relations, and returns the actual call graph. Do not save a temporary Pack unless the user asks.

## 4. Answer

Lead with the stronger explanation, the live rival, and the next observation that separates them. Append a short trace of used model versions and relations.

Do not produce a model checklist. Do not read or save private context unless it can change the judgment and the user authorized it. Never modify formal models as a side effect of analysis.
