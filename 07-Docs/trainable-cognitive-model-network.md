---
type: architecture_spec
title: LollaNet Mechanism Graph
status: active
version: 5.0
updated: 2026-07-30
---

# LollaNet Mechanism Graph

## First Principle

Store reusable mechanisms once. For each task, combine only the mechanisms needed to create competing predictions.

```text
long-term graph              current task
model -> model               sourced observations
      \                         /
       -> two mechanism programs
          -> distinguishing prediction
          -> answer + call graph
```

## Model

```text
input -> transformation -> output
falsifier
```

## Edge

- `requires`: one mechanism needs another state.
- `composes`: one output enters another input.
- `competes`: two mechanisms explain the same change differently.
- `corrects`: one mechanism repairs a known omission.
- `specializes`: one narrows another to stated conditions.

## Task Program

A program has one claim, 2–5 connected models, one timed prediction, and one falsifier. At least two programs compete for the same question.

## Compiler

`06-Scripts/model_graph.py` verifies versions and edges, reads model mechanisms, and emits the actual call graph. It does not retrieve, score, or choose the winning explanation.

## Storage

The model graph persists. Task programs do not, unless the user explicitly asks to save them.
