"""Initialize the local private LollaNet workspace."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


PRIVATE_DIRS = [
    "private/me",
    "private/models",
    "private/model-proposals/proposed",
    "private/model-proposals/accepted",
    "private/model-proposals/rejected",
    "private/model-proposals/deferred",
    "private/evaluations",
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

## Basic

- Name:
- Location:
- School / Organization:
- Role:
- Current stage:

## Goals

- 3-month goal:
- 1-year goal:
- Long-term direction:

## Abilities

- Programming:
- Math:
- Writing:
- English:
- Domain knowledge:

## Constraints

- Weekly available time:
- Budget:
- Equipment:
- Mentors / collaborators:
- Risks:
"""


def long_term_goals_template(today: str) -> str:
    return f"""---
type: long_term_goals
title: Long-term Goals
updated: {today}
privacy: private
---

# Long-term Goals

## Direction

-

## What I Want to Compound

-

## What I Refuse to Optimize For

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

### Principle name

- Principle:
- Originating moment, observation, or reading:
- Why it matters:
- Applies when:
- Does not apply when:
- Tension with other principles:
- Last reviewed:

## Candidate Principles

Ideas here are not binding until I explicitly promote them.

## Retired Principles

Keep retired principles and the reason they changed. Do not erase their history.
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
        root / "private" / "me" / "User Profile.md": user_profile_template(today),
        root / "private" / "me" / "Long-term Goals.md": long_term_goals_template(today),
        root / "private" / "me" / "Principles.md": principles_template(today),
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
