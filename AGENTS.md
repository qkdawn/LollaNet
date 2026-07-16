---
type: agent_protocol
title: LollaNet Agent Protocol
updated: 2026-07-14
---

# LollaNet Agent Protocol

LollaNet 是公开认知框架、私人长期认知层和外部任务环境之间的协作协议。

```text
framework: public files under 00-Start through 09-Cases
identity, values, principles, goals, and focus: private/me
model proposals: private/model-proposals/<status>
optional project memory: private/projects/<project_id>
current task: its original chat, repository, application, or agent environment
```

## Core Rule

Do not write private user data into public framework folders.

Real user data includes:

- user profiles, goals, principles, and constraints;
- real decisions and reviews;
- unpublished project evidence and source material;
- private analysis outputs or model overrides;
- model-change proposals derived from real tasks.

Publicly sourced real cases are allowed only under `09-Cases/<domain>/` when their sources are verifiable and the case records non-empty `source_refs`, `as_of`, facts separately from inferences, and relevant era conditions. Cite and summarize sources; do not copy protected full text.

Public availability does not make personal data unrestricted. Minimize personal data and confirm a lawful, purpose-relevant basis before including it. Cases containing user information, unpublished projects or companies, private conversations, internal documents, or proprietary data belong only under `private/projects/<project_id>/evidence/cases/`. Purely fictional examples belong under `08-Examples/cases/` with `example: true` and `fictional: true`.

## First Use

Before operating LollaNet:

1. Read `README.md`.
2. Read `04-Prompts/LollaNet Analysis Protocol.md`.
3. For Codex orchestration, read `.agents/skills/lollanet-analyze/SKILL.md`.
4. If `private/` is absent or incomplete, run:

```bash
python 06-Scripts/init_private.py
```

Initialization must be additive and must not overwrite existing private files.

## Task Boundary

- Keep the current task, source files, chat history, and transient analysis in their original environment by default.
- Do not create a LollaNet project or decision log merely because a real task exists.
- Read only the minimum private context that can materially change the current judgment.
- Do not disclose unused private context in prompts, scout packets, answers, or logs.
- Simple factual, formatting, or low-risk execution requests should skip multi-scout retrieval.

Saving nothing is a valid complete outcome. Persist only a durable abstraction or a high-value case/decision the user has chosen to keep.

## Private Workspace

The initialized workspace includes:

```text
private/
├── me/
│   ├── dashboard.md
│   ├── User Profile.md
│   ├── Value Function.md
│   ├── Long-term Goals.md
│   ├── Principles.md
│   ├── Current Focus.md
│   ├── observations/
│   ├── reviews/
│   └── proposals/proposed/
├── models/
├── evaluations/
├── model-proposals/
│   ├── proposed/
│   ├── accepted/
│   ├── rejected/
│   └── deferred/
└── projects/
    └── <project_id>/
        ├── Project Profile.md
        ├── dashboard.md
        ├── decisions/
        ├── evidence/
        └── reviews/
```

`private/` is ignored by Git. Identity, values, principles, goals, focus, models, evidence, and proposals are distinct assets; do not silently convert an AI inference into a fact, chosen value, principle, goal, or accepted model. `Value Function.md` is a user-authored multi-objective decision layer, not a license to invent a fixed global score or infer terminal values from observed behavior.

## Analysis Rule

Follow the vendor-neutral protocol in `04-Prompts/LollaNet Analysis Protocol.md`.

- Select 3–7 non-duplicate models only when each changes an assumption, prediction, or action.
- Do not aggregate models by count, majority vote, or fixed coefficient.
- Treat scout conclusions as leads, not evidence; cite the underlying file section or external source.
- Surface material conflicts, counterevidence, uncertainty, and long-term effects.
- Keep the main answer natural to the task; append only a short retrieval note when useful.

## Change Governance

AI agents are read-only against formal identity, value function, principles, long-term goals, and models by default. They may draft a `Current Focus.md` allocation or a personal-data proposal only after the user asks, and must not promote observations or candidate values without explicit approval.

Allowed proposal types are `new`, `amend`, `split`, `merge`, `contest`, and `deprecate`. First show the proposal to the user. Write it under `private/model-proposals/proposed/` only after explicit approval; do not edit the target model in the same step unless the user separately approves accepting the proposal.

Proposal states are `proposed`, `accepted`, `rejected`, and `deferred`. Acceptance may create a `candidate` model or increment an existing model version. Deprecated models remain available as history.

## Optional Persistence

When the user asks to preserve a project, decision, or case:

```bash
python 06-Scripts/create_project.py <project_id> --name "<Project Name>"
python 06-Scripts/create_decision.py "<Decision Title>" --project <project_id>
python 06-Scripts/create_case.py "<Private Case Title>" --project <project_id>
python 06-Scripts/create_case.py "<Public Case Title>" --public-case --source "<verifiable source>"
python 06-Scripts/create_case.py "<Fictional Example Title>" --public-example
```

The three case destinations are mutually exclusive. Use `--public-case` only for a real case grounded in public, verifiable sources; pass at least one `--source`, distinguish `date_observed` from the evidence cutoff `as_of`, separate facts from inferences, and document era conditions. Use `--project` for any private or unpublished case; use `inbox` only when the user wants persistence but the durable project is genuinely unclear. `--public-example` asserts that the case is entirely fictional, not merely deidentified. Never publish user-derived or proprietary case material by relabeling it as an example.

## Validation

Validate the public framework:

```bash
python 06-Scripts/validate_vault.py
```

Validate the public framework plus local private workspace:

```bash
python 06-Scripts/validate_vault.py --include-private
```
