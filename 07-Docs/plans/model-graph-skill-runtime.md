---
type: implementation_plan
title: LollaNet V4 Mechanism Program Runtime
status: active
updated: 2026-07-28
---

# LollaNet V4 Mechanism Program Runtime

V4 replaces the lightweight model-list runtime with a mechanism-program compiler.

The permanent layer stores falsifiable model nodes and typed relations. The transient layer stores sourced task observations and competing programs. The host language model constructs and compares programs; deterministic tooling validates references and exposes registered versus unresolved composition edges.

V4 deliberately avoids both failed extremes:

- it is not a checklist-like retrieval wrapper where the graph only shows neighbors;
- it is not a rules engine that treats condition labels or graph traversal as proof that a mechanism is true.

The graph constrains the vocabulary of valid model composition and makes gaps inspectable. Evidence and reasoning determine which program survives.
