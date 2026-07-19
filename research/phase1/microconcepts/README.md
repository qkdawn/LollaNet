---
type: research_readme
title: Phase 1 Microconcept Layer
status: candidate
updated: 2026-07-19
---

# Phase 1 Microconcept Layer

This directory defines the candidate semantic layer between observable evidence
and the twenty Phase 1 model assets.

The Markdown files under `cards/` are the only editable source of truth. Each
card combines quantitative factors from the value-investing feature dictionary
with controlled qualitative claims. Generated YAML and Markdown artifacts must
not be edited by hand.

## Browse

- [[MICROCONCEPT_INDEX]]
- [[EVIDENCE_CATALOG]]
- [[MODEL_MICROCONCEPT_GRAPH]]
- [[MICROCONCEPT_EVIDENCE_GRAPH]]
- [Phase 1 Microconcept State Encoder Training Protocol](../../../07-Docs/papers/lollanet-v1/03-microconcept-state-encoder-training-protocol.md)

## Validate and Generate

```powershell
python research/phase1/microconcepts/validate_microconcepts.py
python research/phase1/microconcepts/generate_index.py
```

The cards remain `candidate` version `0.1.0`. This phase does not extract real
MD&A text, train an encoder, or produce investment decisions.
