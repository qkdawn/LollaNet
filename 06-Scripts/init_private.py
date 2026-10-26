"""Initialize the local private LollaNet workspace."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


PRIVATE_DIRS = [
    "private/me",
    "private/me/observations",
    "private/me/reviews",
    "private/me/proposals/proposed",
    "private/models",
    "private/model-proposals/proposed",
    "private/model-proposals/accepted",
    "private/model-proposals/rejected",
    "private/model-proposals/deferred",
    "private/evaluations",
    "private/evaluations/episodes",
    "private/projects/inbox/decisions/open",
    "private/projects/inbox/decisions/reviewed",
    "private/projects/inbox/decisions/archived",
    "private/projects/inbox/evidence",
    "private/projects/inbox/reviews",
]


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True

def user_profile_template(today: str) -> str:
    return f"""---
type: user_profile
title: User Profile
updated: {today}
privacy: private
---

# User Profile

This file is private. It should stay under `private/` and never be committed.

This file contains factual identity, capabilities, resources, and constraints.
Aspirations belong in goals; normative commitments belong in the value
function or principles. AI inferences remain observations until the user
confirms them.

## Current Stage

- Name:
- Location:
- School / Organization:
- Role:
- Current stage:
- Last verified:

## Capabilities

- Programming:
- Math:
- Writing:
- English:
- Domain knowledge:

For each consequential capability claim, note the evidence and last observation
rather than using a personality label.

## Resources

- Weekly available time:
- Budget:
- Equipment:
- Mentors / collaborators:
- Geographic access:

## Constraints

- Academic or work obligations:
- Health or energy:
- Financial floor:
- Legal or geographic restrictions:
- Risks:

## Unconfirmed Self-Hypotheses

Keep uncertain patterns in `observations/` and link them here only after
explicit confirmation.
"""


def long_term_goals_template(today: str) -> str:
    return f"""---
type: long_term_goals
title: Long-term Goals
updated: {today}
privacy: private
---

# Long-term Goals

Goals are replaceable target states. Every active goal should name the values
it serves, observable success, opportunity cost, and a review or expiry
condition.

## Active Goals

### goal-example

- Status: `candidate | active | paused | completed | retired`
- Horizon:
- Target state:
- Serves values: `vf-*`
- Observable success:
- Why this route:
- Replaceable by:
- Opportunity cost:
- Review date:
- Stop or change conditions:

## Candidate Goals

-

## Completed Or Retired Goals

-

Keep the outcome, what changed, and why the goal no longer governs current
action.

## Directional Themes

-
"""


def principles_template(today: str) -> str:
    return f"""---
type: personal_principles
title: Principles
updated: {today}
privacy: private
---

# Principles

Personal principles describe how I choose to act. They are not evidence that
the world works in a particular way.

## Active Principles

For each principle, record:

### pr-principle-id — Principle name

- Status: `candidate | active | retired`
- Principle:
- Protects values: `vf-*`
- Originating moment, observation, or reading:
- Why it matters:
- Applies when:
- Does not apply when:
- Failure mode when overused:
- Tension with other principles:
- Last reviewed:

## Candidate Principles

Ideas here are not binding until I explicitly promote them.

## Retired Principles

Keep retired principles and the reason they changed. Do not erase their history.
"""


def personal_dashboard_template(today: str) -> str:
    return f"""---
type: personal_dashboard
title: Personal Dashboard
updated: {today}
privacy: private
---

# Personal Dashboard

## Formal Personal Assets

- `User Profile.md`: facts, capabilities, resources, and constraints.
- `Value Function.md`: user-chosen terminal values, hard constraints, and tensions.
- `Principles.md`: operational rules for recurring trade-offs.
- `Long-term Goals.md`: replaceable target states linked to values.
- `Current Focus.md`: the current 6-12 week allocation and experiments.

AI agents are read-only against formal personal assets by default. Current user
statements override stale files, but do not silently rewrite them.

## Evidence And Review

- `observations/`: evidence and self-hypotheses, not formal identity.
- `reviews/`: periodic checks of predictions, allocations, and outcomes.
- `proposals/proposed/`: suggested changes waiting for explicit review.

## Current Links

- Active values:
- Active goals:
- Current focus cycle:
- Open personal-data proposals:
"""


def value_function_template(today: str) -> str:
    return f"""---
type: personal_value_function
title: Value Function
status: draft
version: 1
updated: {today}
privacy: private
owner: user
---

# Value Function

This is not a fixed mathematical utility function. It records what the user
currently endorses as intrinsically valuable, what must not be traded away,
and where value conflicts remain unresolved. Global numeric weights are
forbidden unless the user explicitly authors and bounds them for a specific
context.

## Core Statement


## Terminal Values

Use stable `vf-*` IDs. New entries remain `candidate` until the user explicitly
promotes them.

### vf-example

- Status: `candidate | active | retired`
- Statement:
- Why it may be valuable in itself:
- Value carriers:
- What does not count:
- Status or proxy risk:
- Tension with other values:
- Counterevidence or unresolved question:
- Origin: `user-stated | user-authored | proposed-by-ai`
- Last reviewed:

## Hard Constraints

-

## Anti-Values And Proxy Risks

-

## Internal Tensions

| Value A | Value B | When the conflict appears | Current resolution | Unresolved |
| --- | --- | --- | --- | --- |
|  |  |  |  | yes |

## Context Overlays

Core values may express differently across contexts. Context overlays cannot
silently replace the core layer.

## Decision-Local Trade-offs

Do not keep permanent global weights here. Concrete decisions may define
temporary priorities, thresholds, or scoring in their own records.

## Review History

| Date | Change | Approved by | Evidence or reason |
| --- | --- | --- | --- |
"""


def current_focus_template(today: str) -> str:
    return f"""---
type: current_focus
title: Current Focus
status: draft
updated: {today}
privacy: private
owner: user
---

# Current Focus

This file translates active goals into a reversible 6-12 week allocation. It
is not a new value layer and should expire or be reviewed at the end of each
cycle.

## Current Cycle

- Period:
- Weekly available time:
- Energy or health constraints:
- Review date:

## Active Allocations

| Allocation | Weekly budget | Serves goals | Serves values | Why now |
| --- | ---: | --- | --- | --- |
|  |  |  |  |  |

## Experiments

### focus-experiment-id

- Hypothesis:
- Minimum action:
- Evidence to collect:
- Continue when:
- Stop or change when:
- End date:

## Deferred On Purpose

-

## Cycle Review

- What changed:
- What was learned:
- Proposed goal, principle, or value changes:
- Approved changes:
"""


def project_profile_template(project_id: str, name: str, today: str) -> str:
    return f"""---
type: project_profile
project_id: {project_id}
title: {name}
updated: {today}
privacy: private
---

# {name}

## Purpose

What is this project trying to decide, build, or learn?

## Scope

- In scope:
- Out of scope:

## Success Criteria

-

## Relevant Long-Term Context

List only the identity, principles, models, or evidence that materially changes
this project's judgments. Do not establish a fixed model checklist here.
"""


def project_dashboard_template(project_id: str, name: str, today: str) -> str:
    return f"""---
type: project_dashboard
project_id: {project_id}
title: {name} Dashboard
updated: {today}
privacy: private
---

# {name} Dashboard

## Project Files

- `Project Profile.md`

## Decisions

- `decisions/open/`
- `decisions/reviewed/`
- `decisions/archived/`

## Evidence

- `evidence/`

## Reviews

- `reviews/`
"""


def init_private(vault_root: Path, today: str | None = None) -> list[Path]:
    today = today or date.today().isoformat()
    root = Path(vault_root)
    created: list[Path] = []

    for directory in PRIVATE_DIRS:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)

    files = {
        root / "private" / "me" / "dashboard.md": personal_dashboard_template(today),
        root / "private" / "me" / "User Profile.md": user_profile_template(today),
        root / "private" / "me" / "Value Function.md": value_function_template(today),
        root / "private" / "me" / "Long-term Goals.md": long_term_goals_template(today),
        root / "private" / "me" / "Principles.md": principles_template(today),
        root / "private" / "me" / "Current Focus.md": current_focus_template(today),
        root / "private" / "projects" / "inbox" / "Project Profile.md": project_profile_template(
            "inbox", "Inbox", today
        ),
        root / "private" / "projects" / "inbox" / "dashboard.md": project_dashboard_template(
            "inbox", "Inbox", today
        ),
    }

    for path, content in files.items():
        if write_if_missing(path, content):
            created.append(path)

    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize local private LollaNet workspace.")
    parser.add_argument("--root", default=".", help="Vault root path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    created = init_private(Path(args.root))
    for path in created:
        print(path)
    if not created:
        print("Private workspace already initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
