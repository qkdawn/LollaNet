from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg


class MigrationError(RuntimeError):
    pass


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path) -> list[str]:
    migrations_dir = Path(migrations_dir)
    paths = sorted(migrations_dir.glob("*.sql"))
    if not paths:
        raise MigrationError(f"no SQL migrations found in {migrations_dir}")

    with conn.transaction():
        conn.execute("CREATE SCHEMA IF NOT EXISTS lollanet")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lollanet.schema_migrations (
                name text PRIMARY KEY,
                checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
            """
        )

    applied: list[str] = []
    for path in paths:
        sql_text = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
        row = conn.execute(
            "SELECT checksum FROM lollanet.schema_migrations WHERE name = %s",
            (path.name,),
        ).fetchone()
        conn.commit()
        if row:
            if row[0] != checksum:
                raise MigrationError(f"applied migration changed: {path.name}")
            continue
        with conn.transaction():
            conn.execute(sql_text)
            conn.execute(
                "INSERT INTO lollanet.schema_migrations(name, checksum) VALUES (%s, %s)",
                (path.name, checksum),
            )
        applied.append(path.name)
    return applied
