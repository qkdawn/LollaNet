"""Migrate legacy one-level model folders into the canonical taxonomy."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from model_taxonomy import classification_for_seed, model_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_taxonomy import classification_for_seed, model_path


def update_frontmatter(text: str, name: str) -> str:
    classification = classification_for_seed(name)
    marker = "\n---\n"
    if not text.startswith("---\n") or marker not in text[4:]:
        raise ValueError(f"{name}: missing YAML frontmatter")
    block, body = text[4:].split(marker, 1)

    def replace_scalar(block_text: str, key: str, value: str) -> str:
        pattern = rf"(?m)^{re.escape(key)}:[^\n]*$"
        replacement = f"{key}: {value}"
        if re.search(pattern, block_text):
            return re.sub(pattern, replacement, block_text, count=1)
        discipline_match = re.search(r"(?m)^discipline:[^\n]*$", block_text)
        if discipline_match:
            insert_at = discipline_match.end()
            return block_text[:insert_at] + f"\n{replacement}" + block_text[insert_at:]
        return block_text + f"\n{replacement}"

    block = replace_scalar(block, "discipline", classification.discipline)
    block = replace_scalar(block, "source_category", classification.source_category)
    block = replace_scalar(block, "subdiscipline", classification.subdiscipline)
    block = replace_scalar(block, "formalism", "false")
    return f"---\n{block}{marker}{body}"


def migration_plan(vault_root: Path) -> list[tuple[Path, Path]]:
    model_root = Path(vault_root) / "01-Models"
    plan: list[tuple[Path, Path]] = []
    for source in sorted(model_root.glob("*/*.md")):
        name = source.stem
        classification = classification_for_seed(name)
        target = model_path(vault_root, classification, name)
        if target.exists() and target != source:
            raise FileExistsError(f"migration target already exists: {target}")
        plan.append((source, target))
    return plan


def migrate(vault_root: Path, apply: bool = False) -> list[tuple[Path, Path]]:
    plan = migration_plan(vault_root)
    for source, target in plan:
        print(f"{source.relative_to(vault_root)} -> {target.relative_to(vault_root)}")
        if not apply:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(update_frontmatter(source.read_text(encoding="utf-8"), source.stem), encoding="utf-8")
        source.unlink()
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate legacy model folders into the canonical taxonomy.")
    parser.add_argument("--root", default=".", help="Vault root path")
    parser.add_argument("--apply", action="store_true", help="Write the migration; default is a dry run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = migrate(Path(args.root), apply=args.apply)
    action = "migrated" if args.apply else "planned"
    print(f"{len(plan)} model(s) {action}; use --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
