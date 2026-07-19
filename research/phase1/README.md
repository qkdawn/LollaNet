# Phase 1: Independent Model Assets

This directory contains the fixed semantic assets for the first neural-routing
experiment. It contains no training code, episodes, LLM integration, public-model
references, or numerical routing parameters.

Twenty self-contained YAML files live in `models/`. They cover mathematics,
physics, chemistry, engineering, and finance. Finance is a training domain; it does
not alter the public vault taxonomy.

## Asset Contract

Every asset has a stable ID, semantic definition, mechanism, variables, invocation
interface, boundaries, testable predictions, action affordances, and explicit
composition relations. `kind` keeps the distinction between world, reasoning, and
action models. `functional_roles` is multi-label so a model may support observation,
transition, evaluation, control, or challenge work in different routes.

## Validate and Browse

```powershell
python research/phase1/validate_models.py
python research/phase1/generate_index.py
```

The validator checks the schema, references, and combination examples. The index is
a read-only navigation artifact generated from the YAML assets.
