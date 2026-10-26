---
type: agent_protocol
title: LollaNet Agent Protocol
updated: 2026-10-09
---

# LollaNet Agent Protocol

LollaNet is a model library focused on summarizing, organizing, and maintaining reusable models. Follow README.md and PROJECT_SPEC.md for the current scope.

## First Use

Read README.md, PROJECT_SPEC.md, and db/README.md before working on the library. Analysis protocols, analysis skills, and private workspace initialization are not prerequisites for library maintenance.

## Task Boundary

- Focus on model definitions, mechanisms, premises, boundaries, counterexamples, sources, classifications, relationships, and version history.
- Keep personal cognition, project memory, task analysis, routing, answer generation, and training experiments outside the library's responsibility.
- Existing analysis tools and research files are retained compatibility content. Do not extend them as part of ordinary library maintenance.
- Do not read private context, initialize private assets, or create project and decision records merely to maintain the library.
- Preserve existing work and data. Scope separation does not authorize deleting or moving files, changing formal models, or writing to a database.

## Storage

PostgreSQL `lollanet` is the formal model library's source of truth. `01-Models/` and static graph/index files are legacy snapshots. Do not edit them to overwrite database state. Follow db/README.md and ADR-0001 for migration and storage invariants.

## Privacy And Evidence

Do not write private user data into public folders. This includes user profiles, goals, values, constraints, real decisions, unpublished project evidence, private conversations, and task-derived proposals.

Public real cases may remain under `09-Cases/<domain>/` only with verifiable non-empty `source_refs`, an evidence cutoff `as_of`, facts separate from inferences, and relevant era conditions. Cite and summarize sources rather than copying protected full text. Public availability does not remove the need to minimize personal data and establish a lawful, purpose-relevant basis.

Private or proprietary cases belong only in an explicitly requested private project record. Purely fictional examples belong under `08-Examples/cases/` with `example: true` and `fictional: true`. Never relabel private or deidentified real material as fiction.

## Content Governance

Formal models and existing private identity, values, principles, and goals are read-only for AI by default. Do not convert an AI inference into an accepted fact, value, principle, goal, or model.

Allowed model proposal types remain `new`, `amend`, `split`, `merge`, `contest`, and `deprecate`. Show a proposal first. Write it under `private/model-proposals/proposed/` only after explicit approval. Do not edit the target model in the same step unless the user separately approves acceptance.

## Validation

Use checks appropriate to the change:

```powershell
python 06-Scripts/model_db.py import-models --root . --dry-run
python 06-Scripts/validate_vault.py
pytest -q tests/test_model_db_unit.py
```
