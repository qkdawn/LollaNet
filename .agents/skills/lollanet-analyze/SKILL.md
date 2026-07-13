---
name: lollanet-analyze
description: "Analyze consequential decisions, opportunities, strategies, investments, products, careers, projects, and reviews with LollaNet's auditable multi-model retrieval, relevant private context, counterevidence, and long-horizon checks. Use when the user invokes $lollanet-analyze, asks for LollaNet or multiple mental models, or implicitly presents a high-stakes, ambiguous, or durable judgment; skip this retrieval workflow for simple factual, formatting, translation, or low-risk requests."
---

# Analyze with LollaNet

Use LollaNet as an external, inspectable cognitive layer. Keep the user's judgment primary and make every retrieved claim traceable.

## Load the Authoritative Protocol

1. Locate the workspace root that contains `04-Prompts/LollaNet Analysis Protocol.md`.
2. Read that file at the start of every invocation. Treat it as the sole source of truth for the general analysis workflow, reasoning rules, and answer shape.
3. Use this skill only for Codex-specific orchestration, retrieval contracts, privacy, and write controls. Do not preserve an older protocol in this file.
4. If the protocol is unavailable, report that the LollaNet workflow cannot be grounded instead of inventing a replacement.

Use direct file and text search such as `rg` for v1. Do not create or depend on a vector database, embedding index, generated catalog, or retrieval cache.

## Gate the Retrieval Workflow

Classify the request before delegating:

- Treat direct factual lookup, formatting, translation, mechanical summarization, and reversible low-risk work as **simple**. Answer normally; skip all scouts and do not force mental models into the answer.
- Treat decisions, strategies, causal diagnosis, reviews, and choices with meaningful uncertainty, tradeoffs, persistence, or downside as **complex**. Run the protocol and the retrieval workflow below.
- When the classification is uncertain, run the complex path only if different assumptions, time horizons, or user-specific constraints could materially change the action.

Explicit invocation does not override the complexity gate: `$lollanet-analyze` on a simple task still skips scouts and briefly says why.

## Retrieve for Complex Tasks

Model the task according to the authoritative protocol, then dispatch no more than three read-only scouts. Give each scout the user's problem, a narrow search scope, and the retrieval packet contract. Instruct every scout not to edit files, create logs, or mutate external state.

Run these scouts concurrently when the host supports subagents:

- **model-scout**: Search `01-Models/`, relevant sourced cases under `09-Cases/`, and only relevant private additions under `private/models/`, if present. Inspect model triggers, mechanisms, boundaries, relationships, conflicts, status, predictions, and evidence. Return candidates whose use could change an assumption, prediction, or action; treat case facts as evidence only to the extent supported by their `source_refs` and `as_of`.
- **memory-scout**: Run only when personal or project context could change the answer. Search filenames and headings first, then read the smallest relevant sections under `private/me/` and the identified `private/projects/<project_id>/`. Do not enumerate unrelated projects or return unrelated personal details.
- **challenge-scout**: Search grounded counterexamples, contrary evidence, omitted mechanisms, model conflicts, historical disanalogies, and risks over multiple time horizons. Search sourced public cases under `09-Cases/`; search only the named project's private evidence when needed; otherwise prefer the public repository or attributable external sources.

Do not ask scouts for a final recommendation. The parent agent owns deduplication, conflict analysis, and synthesis.

If subagents or parallel execution are unavailable, perform the same model, memory-if-needed, and challenge searches sequentially in the parent agent. Preserve the same scopes, read-only constraint, and packet schema. Lack of subagent support is a serial fallback, not a reason to weaken traceability.

## Use One Retrieval Packet

Require every scout and serial fallback to return zero or more items with exactly these fields:

```yaml
- kind: model | memory | evidence | challenge
  title: concise source or finding title
  location: repository/path.md#Exact heading or https://source.example/item#locator
  source_category: public-model | private-model | private-identity | private-principle | private-project | private-evidence | public-evidence | external
  match_signal: task phrase, condition, or variable that made this relevant
  basis: concise source-grounded fact, mechanism, or claim
  boundary_or_counterevidence: limiting condition, conflict, or contrary evidence
  as_of: YYYY-MM-DD or unknown
  confidence: low | medium | high
```

Validate packets before synthesis:

- Make `location` point to the narrowest underlying heading that supports `basis`, or to an external URL and useful locator. A document-title anchor is insufficient when the claim comes from a deeper section. A scout's conclusion is never evidence by itself.
- Derive `as_of` from the source's status or update date. Use `unknown` when the source does not provide one; do not substitute today's date.
- Reject unsupported items and report them only as retrieval gaps. Do not invent a source or packet to fill a quota.
- Keep private `basis` text to the minimum abstraction needed for this task. Never paste an entire private section into a packet.

## Synthesize Without Voting

1. Merge packets that express the same mechanism, even when their labels differ.
2. Surface explicit metadata conflicts and incompatible predictions before choosing models.
3. For a complex task, select 3-7 non-redundant models only when each can change an assumption, prediction, or action. If fewer than three grounded models exist, state the retrieval shortfall and never pad the set.
4. Do not combine models by count, majority vote, fixed coefficient, or a generic score. Reconcile them through causal fit, evidence quality, boundary conditions, and the user's constraints.
5. Apply the protocol's base-rate, counterevidence, historical-context, and multiple-time-horizon checks.
6. Answer in a natural shape suited to the task. Append only a compact retrieval trace naming the selected item, its `location`, and what it changed; do not dump scout transcripts or impose a fixed report template.

Mark unsupported external facts as unverified. When using a historical analogy, cite its source and follow the protocol's comparison of mechanisms and era-specific conditions.

## Protect Private Context

- Read private context only when it could change the answer, and only at file and section granularity needed for the task.
- Do not send private content to web search, external tools, or unrelated agents. Share the minimum task-relevant abstraction with a memory scout.
- Do not expose unrelated profile, principle, project, evidence, or review content in the answer or retrieval trace.
- Never write real user data to public framework folders.

## Keep Analysis Read-Only

Treat every invocation as read-only by default. Do not update models, private memory, projects, decisions, reviews, evidence, dashboards, or task history as a side effect of analysis.

When analysis suggests a cognitive change, show a proposed `new`, `amend`, `split`, `merge`, `contest`, or `deprecate` action with its target, rationale, supporting source, counterevidence, and expected effect. An analysis request or skill invocation is not approval to save it.

Write only a private proposal, using the repository's current proposal schema, after the user explicitly approves the displayed proposal. Approval to save a proposal is not approval to accept or apply it. Never directly rewrite a formal model from this skill.
