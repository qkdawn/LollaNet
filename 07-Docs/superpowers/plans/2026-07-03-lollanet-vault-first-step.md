---
type: implementation_plan
status: superseded
superseded_on: 2026-07-12
superseded_by:
  - PROJECT_SPEC.md
  - 04-Prompts/LollaNet Analysis Protocol.md
---

# LollaNet Vault First Step Implementation Plan

> **Status: superseded.** This file is a historical record of the original Vault scaffold. Do not execute its remaining steps. `PROJECT_SPEC.md` is the current product and data contract; `04-Prompts/LollaNet Analysis Protocol.md` is the current runtime protocol.

> Historical note: the original plan required a task-by-task execution skill. That instruction is no longer active.

**Goal:** Turn `D:\Coding\LollaNet` into a clean first-step Obsidian Vault for LollaNet, focused on decision use rather than content collection.

**Architecture:** The Vault uses stable Obsidian folders, reusable Markdown templates, a dashboard entry point, seed model cards, and small standard-library Python scripts. Automation is intentionally minimal in the first step: create model cards, create cases, create decisions, and validate Vault health.

**Tech Stack:** Obsidian Markdown, YAML frontmatter, Python 3.11+ standard library, pytest for script verification.

---

### File Responsibilities

- `00-Dashboard/Dashboard.md`: Obsidian home page with practical entry points and Dataview blocks.
- `00-Dashboard/Decision Checklist.md`: Operational checklist for major decisions.
- `00-Dashboard/Model Index.md`: Human-readable index of the initial 30 model cards.
- `00-Dashboard/Weekly Review.md`: Weekly review workspace.
- `templates/*.md`: Reusable Obsidian note templates.
- `06-Prompts/*.md`: Codex operating prompts for repeat workflows.
- `07-Scripts/create_model.py`: Create one model card without overwriting existing notes.
- `07-Scripts/create_case.py`: Create one case card without overwriting existing notes.
- `07-Scripts/create_decision.py`: Create one dated decision log without overwriting existing notes.
- `07-Scripts/validate_vault.py`: Report missing structure, frontmatter, required sections, and broken wiki links.
- `tests/test_scripts.py`: Regression tests for the first-step scripts.
- `README.md`: Practical usage guide for opening and operating the Vault.

### Task 1: Vault Skeleton

**Files:**
- Create directories under `00-Dashboard`, `01-Models`, `02-Cases`, `03-Decisions`, `04-Reading`, `05-Maps`, `06-Prompts`, `07-Scripts`, `08-Exports`, `99-Inbox`, `templates`, `tests`.

- [x] **Step 1: Create directories**

Run PowerShell `New-Item -ItemType Directory -Force` for every directory in the spec.

- [ ] **Step 2: Verify skeleton**

Run: `Get-ChildItem -Directory`

Expected: all top-level Vault directories are present.

### Task 2: Script Tests

**Files:**
- Create: `tests/test_scripts.py`

- [ ] **Step 1: Write tests first**

Cover model creation, decision creation, non-overwrite behavior, and validation reporting.

- [ ] **Step 2: Run tests before implementation**

Run: `pytest -q`

Expected: failing tests because the scripts do not exist yet.

### Task 3: Core Scripts

**Files:**
- Create: `07-Scripts/create_model.py`
- Create: `07-Scripts/create_case.py`
- Create: `07-Scripts/create_decision.py`
- Create: `07-Scripts/validate_vault.py`

- [ ] **Step 1: Implement minimal behavior**

Use only the Python standard library, keep paths relative to the current Vault root, and avoid overwriting existing files.

- [ ] **Step 2: Run tests**

Run: `pytest -q`

Expected: all tests pass.

### Task 4: Vault Content

**Files:**
- Create: `templates/mental_model_template.md`
- Create: `templates/case_template.md`
- Create: `templates/decision_log_template.md`
- Create: `templates/reading_note_template.md`
- Create: `templates/weekly_review_template.md`
- Create: dashboard and prompt Markdown files.
- Generate: 30 initial model cards with `07-Scripts/create_model.py`.

- [ ] **Step 1: Add templates and dashboards**

Use concise sections from `PROJECT_SPEC.md`, keeping placeholders explicit and avoiding fabricated quotations.

- [ ] **Step 2: Generate seed models**

Run `python 07-Scripts/create_model.py ...` for all 30 seed model names with appropriate disciplines.

### Task 5: Completion Verification

**Files:**
- Inspect current workspace state.

- [ ] **Step 1: Run automated tests**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run Vault validator**

Run: `python 07-Scripts/validate_vault.py`

Expected: no structural errors; expected seed content warnings are acceptable only if clearly reported.

- [ ] **Step 3: Count model cards**

Run: `Get-ChildItem 01-Models -Recurse -Filter *.md | Measure-Object`

Expected: 30 model cards.

- [ ] **Step 4: Review created files**

Run: `rg --files`

Expected: required first-step artifacts exist.
