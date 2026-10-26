from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg

from .legacy_import import LegacyImportError, load_legacy_library
from .migrations import apply_migrations
from .models import ModelRef
from .repository import ModelRepository, counts_as_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIGRATIONS = ROOT / "db" / "migrations"


def _database_url(value: str | None) -> str:
    result = value or os.environ.get("LOLLANET_DATABASE_URL")
    if not result:
        raise ValueError("set LOLLANET_DATABASE_URL or pass --database-url")
    return result


def _model_ref(value: str) -> ModelRef:
    try:
        model_id, version = value.rsplit("@", 1)
        return ModelRef(model_id, int(version))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("model reference must use mm-id@version") from error


def _write_json(payload: Any, target: str | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target:
        output = Path(target)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LollaNet PostgreSQL public model library")
    parser.add_argument("--database-url", help="PostgreSQL DSN; defaults to LOLLANET_DATABASE_URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate", help="Apply pending SQL migrations")
    migrate.add_argument("--migrations", type=Path, default=DEFAULT_MIGRATIONS)

    import_parser = subparsers.add_parser("import-models", help="Import legacy public model cards once")
    import_parser.add_argument("--root", type=Path, default=ROOT)
    import_parser.add_argument("--dry-run", action="store_true")
    import_parser.add_argument("--audit-out")
    import_parser.add_argument("--migrate", action="store_true", help="Apply migrations before import")

    subparsers.add_parser("audit", help="Report database model-library counts")
    search = subparsers.add_parser("search", help="Search current models, claims, and evidence")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    neighbors = subparsers.add_parser("neighbors", help="Query adjacent formal relations")
    neighbors.add_argument("model", type=_model_ref)
    traverse = subparsers.add_parser("traverse", help="Recursively traverse formal relations")
    traverse.add_argument("model", type=_model_ref)
    traverse.add_argument("--max-depth", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "import-models":
            library, audit = load_legacy_library(args.root)
            if args.dry_run:
                _write_json(audit.to_dict(), args.audit_out)
                return 0
            dsn = _database_url(args.database_url)
            with psycopg.connect(dsn) as conn:
                if args.migrate:
                    apply_migrations(conn, DEFAULT_MIGRATIONS)
                inserted = ModelRepository(conn).import_legacy_library(library)
                database = ModelRepository(conn).audit_counts()
            payload = audit.to_dict()
            payload["inserted"] = counts_as_dict(inserted)
            payload["database"] = database
            _write_json(payload, args.audit_out)
            return 0

        dsn = _database_url(args.database_url)
        with psycopg.connect(dsn) as conn:
            if args.command == "migrate":
                _write_json({"applied": apply_migrations(conn, args.migrations)})
            elif args.command == "audit":
                _write_json(ModelRepository(conn).audit_counts())
            elif args.command == "search":
                _write_json(ModelRepository(conn).search(args.query, args.limit))
            elif args.command == "neighbors":
                _write_json(ModelRepository(conn).neighbors(args.model))
            elif args.command == "traverse":
                _write_json(ModelRepository(conn).traverse(args.model, args.max_depth))
        return 0
    except (LegacyImportError, ValueError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
