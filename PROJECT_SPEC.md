---
type: project_spec
title: LollaNet Project Spec
version: 6.0
updated: 2026-10-09
---

# LollaNet Project Spec

## Purpose

LollaNet is a model library. Its responsibility is to summarize, organize, and maintain reusable models with clear mechanisms, boundaries, and traceable sources.

The unit of value is a well-supported model summary. Task execution and the quality of an external agent's answer are separate concerns.

## Core Content

Each model should explain:

- What it describes or explains.
- Its core mechanism, including inputs and outputs where meaningful.
- The premises and conditions under which it applies.
- Its limits, counterexamples, or falsifying observations.
- Its sources, distinguishing established findings from interpretation.
- Its category and meaningful relationships to other models.

Do not force every concept into a causal program or add fields solely to support an analysis runtime. Preserve existing structured claims and version history.

## Library Operations

The core operations are finding models, reading summaries and sources, comparing boundaries, inspecting relationships, and maintaining content through reviewed changes.

Formal relationship types remain `requires`, `composes`, `competes`, `corrects`, and `specializes`. A relationship describes reusable model knowledge; it does not prescribe a task workflow.

## Storage

PostgreSQL's `lollanet` schema remains the source of truth, following [ADR-0001](07-Docs/architecture-decisions/0001-postgresql-model-library.md).

`01-Models/` and static indexes are legacy import snapshots. Do not reverse-sync them into the database or fabricate missing historical versions. This scope change does not change the schema or perform database writes.

## External Responsibilities

Personal identity, values, goals, focus, project memory, decisions, task observations, Context Packs, mechanism-program compilation, routing, answer generation, and neural training are outside the model library.

External applications may consume model content and relationships. Maintaining or browsing the library requires neither private context nor an analysis runtime.

## Transition

README, Dashboard, and AGENTS define the current scope. Older analysis protocols and research plans describe retained compatibility tools or historical work; they do not expand the library's responsibilities.

Existing files, scripts, skills, private data, and experiments remain in place during this initial scope separation. Physical extraction, dependency removal, and retiring compatibility tests are separate implementation work.

## Content Governance

Keep private user data out of public model content. AI does not silently promote task-derived interpretations into accepted models. Formal model changes continue to require review and explicit acceptance; preserve sources and version history.

## Success

Judge the library by summary accuracy, source traceability, explicit applicability limits, useful classification and relationships, and reliable retrieval. Model count, relationship count, and generated answer length are not quality measures.
