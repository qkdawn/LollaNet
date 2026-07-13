---
type: template
template_for: model_change_proposal
updated: 2026-07-12
---

# {{proposal_title}}

Use this template under `private/model-proposals/proposed/`. A proposal is not
an accepted model change.

## Required Frontmatter

Replace the template frontmatter with this schema when creating a real proposal:

```yaml
---
type: model_change_proposal
id: {{proposal_id}}
proposal_type: new
status: proposed
target_model_ids: []
created: {{created}}
updated: {{updated}}
source_refs: []
privacy: private
---
```

`proposal_type` must be `new`, `amend`, `split`, `merge`, `contest`, or
`deprecate`. `target_model_ids` may be empty only for `new`.

## Proposal Metadata

- Change type: `new | amend | split | merge | contest | deprecate`
- Target model ID:
- Status: `proposed`
- Created:
- Source scope: `private | public | external`

## Trigger

What observation, reading, failed prediction, or repeated pattern caused this proposal?

## Evidence And Provenance

- Source:
- Date or historical period:
- Observed fact:
- What remains inferred or unverified:

## Proposed Change

State the exact addition, correction, split, merge, contest, or deprecation.

## Durable Mechanism

What mechanism may survive changes in technology, institutions, or fashion?

## Era-Specific Manifestation

Which details may only be true in the source context?

## Boundaries And Counterevidence

- Conditions where the proposal should hold:
- Conditions where it should fail:
- Evidence against it:

## Observable Predictions

What future observation would strengthen, weaken, or falsify the proposal?

## Privacy Review

- Does this contain real user data?
- What must be removed before any public promotion?

## Human Review

- Decision: `accept | reject | defer`
- Reviewer:
- Date:
- Reason:
