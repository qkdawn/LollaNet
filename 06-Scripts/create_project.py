"""Create an isolated private LollaNet project workspace."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import importlib.util


PROJECT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def load_init_private_module():
    path = Path(__file__).with_name("init_private.py")
    spec = importlib.util.spec_from_file_location("init_private", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def validate_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.fullmatch(project_id):
        raise ValueError("project_id must use lowercase letters, numbers, and hyphens")


def create_project(vault_root: Path, project_id: str, name: str, today: str | None = None) -> Path:
    validate_project_id(project_id)
    today = today or date.today().isoformat()
    helpers = load_init_private_module()
    root = Path(vault_root)
    project_root = root / "private" / "projects" / project_id

    for directory in [
        "decisions/open",
        "decisions/reviewed",
        "decisions/archived",
        "evidence",
        "reviews",
    ]:
        (project_root / directory).mkdir(parents=True, exist_ok=True)

    helpers.write_if_missing(
        project_root / "Project Profile.md",
        helpers.project_profile_template(project_id, name, today),
    )
    helpers.write_if_missing(
        project_root / "dashboard.md",
        helpers.project_dashboard_template(project_id, name, today),
    )
    return project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a private LollaNet project workspace.")
    parser.add_argument("project_id", help="Lowercase id, e.g. my-project")
    parser.add_argument("--name", required=True, help="Human-readable project name")
    parser.add_argument("--root", default=".", help="Vault root path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = create_project(Path(args.root), args.project_id, args.name)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
