---
type: migration_note
title: PostgreSQL Model Library Cutover
updated: 2026-08-19
---

# PostgreSQL Model Library Cutover

## Domain Rules Retained By LollaNet

An independent model must expose a distinct input-to-transformation-to-output
mechanism, its own premises and boundaries, and a prediction that can be
falsified. Removing it from a mechanism program must change a prediction,
boundary, or action. A taxonomy label, checklist, synonym, case, or renamed
copy is not an independent model.

Formal relations use these review rules:

- `A requires B`: B supplies a premise or input without which A cannot operate.
- `A composes B`: A's output can enter B's input and the composition has a
  reviewable consequence beyond merely naming both models.
- `A competes B`: both address a comparable outcome and produce incompatible
  predictions under overlapping boundaries. It is symmetric and stored once.
- `A corrects B`: A identifies a documented failure mode or boundary error in
  B and changes the resulting prediction.
- `A specializes B`: A narrows B's premises or boundary while preserving its
  mechanism inside that subset.

Every formal relation must name exact endpoint versions and include a rationale
that answers the applicable rule. A relation is reviewed again when either
endpoint version changes. Registration does not prove Runtime activation.

Status admission is semantic, not a popularity score:

- `candidate`: independent-model test and all nine typed claim fields pass.
- `active`: candidate requirements plus attributable public support,
  opposing/limiting evidence, and a timed prediction that distinguishes an
  alternative explanation.
- `contested`: material opposing or limiting evidence is mapped explicitly;
  the model remains queryable with its dispute visible.
- `deprecated`: superseded, merged, persistently falsified, or no longer
  independent. History remains queryable, but normal search excludes it.

Applied evolution is atomic: `new` is 0-to-1; `amend` is 1-to-1 on the same
stable ID; `split` is one deprecated successor plus at least two new candidate
models; `merge` is a deprecated successor for every input plus one new
candidate model; `contest` creates contested successors; `deprecate` creates
deprecated successors. Exact current inputs are row-locked before validation.

## Retired Legacy Paths

- `06-Scripts/generate_model_index.py` is retired as a model-library index.
  Database queries and `current_model_versions` replace it.
- `05-Maps/model-graph.json` and generated graph Markdown are retired as model
  relationship storage. `model_relations` and recursive CTEs replace them.
- Model-card checks in `06-Scripts/validate_vault.py` remain only for auditing
  the frozen legacy import corpus. PostgreSQL constraints are authoritative.
- `06-Scripts/create_model.py` is no longer a post-cutover write path. Accepted
  public changes use `ModelRepository.apply_model_change`.
- Runtime-specific functions in `06-Scripts/model_graph.py` are unchanged and
  explicitly outside this migration. They must not be interpreted as an
  alternative model-library source of truth.

No legacy file is deleted during cutover. Removal can occur only after all
excluded consumers have independently migrated to database-backed exports.
The retired CLI write modes fail closed with exit code 2; their importable
legacy helpers remain temporarily available to validate the frozen corpus and
support excluded Runtime compatibility tests.
