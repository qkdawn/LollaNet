---
type: architecture_decision
title: PostgreSQL as the Public Model Library Source of Truth
status: accepted
updated: 2026-08-19
---

# PostgreSQL as the Public Model Library Source of Truth

## Decision

The public LollaNet model library uses one PostgreSQL schema, `lollanet`, as
its source of truth. Model identity, immutable versions, typed claims, public
evidence, claim-evidence mappings, formal relations, classifications, accepted
changes, full-text indexes, and graph queries live in PostgreSQL.

LollaNet retains the domain rules that decide whether a model is independent,
whether a version is semantically complete, what each relation means, which
status transitions are admissible, and how accepted changes are applied. The
database enforces storage invariants and commits an accepted change atomically.

## Entities And History

- `models` holds stable `mm-*` identities.
- `model_versions` is append-only. Status changes create a successor version.
- `claims` represents definition, mechanism, input, transformation, output,
  premise, boundary, falsifier, and prediction as independently queryable data.
- `evidence` and `claim_evidence` distinguish supporting, opposing, and
  limiting evidence.
- `model_relations` links exact versions. `competes` is stored once in canonical
  model-ID order; all other formal relations are directional.
- `categories` is an adjacency-list taxonomy. A model version links to one
  primary leaf category.
- `model_changes` stores only accepted, applied public evolution events.
  Private proposals are not imported.

The legacy Markdown cards contain only the current body plus prose revision
summaries. The importer therefore records each current card as a legacy
snapshot at its existing version number. It does not fabricate missing bodies
for earlier versions. Complete immutable history begins at database cutover.

## Replaced Capabilities

Markdown frontmatter parsing, static index generation, JSON graph persistence,
regex field checks, and Markdown revision tables are migration inputs, not
ongoing model-library infrastructure. PostgreSQL constraints, views,
`tsvector`/GIN, joins, recursive CTEs, and transactions replace them.

Markdown may later be exported from PostgreSQL for reading, but an export is a
derived artifact and must never overwrite database state.

## Exclusions

This decision does not migrate or change Runtime, Context Packs, routing,
answer generation, private data, project decisions, evaluation records,
research training pipelines, or neural-control experiments. It introduces no
S3, Crossref, pgvector, Neo4j, Elasticsearch, microservice, or nonessential
PostgreSQL extension.

## Search Constraint

Phase one uses PostgreSQL's built-in `simple` text-search configuration. It is
deliberately dependency-free but does not provide high-quality Chinese word
segmentation. Search recall must be measured before considering any later
change; this decision does not authorize additional search infrastructure.
