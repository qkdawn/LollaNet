from __future__ import annotations

import hashlib
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lollanet_db.legacy_import import load_legacy_library  # noqa: E402
from lollanet_db.migrations import apply_migrations  # noqa: E402
from lollanet_db.models import LegacyLibrary, ModelRef, RelationSpec  # noqa: E402
from lollanet_db.repository import ModelRepository, RepositoryError  # noqa: E402


MIGRATIONS = ROOT / "db" / "migrations"


@pytest.fixture(scope="session")
def legacy_library():
    return load_legacy_library(ROOT)[0]


@pytest.fixture(scope="session")
def database_url(legacy_library):
    admin_dsn = os.environ.get("LOLLANET_TEST_DATABASE_URL")
    if not admin_dsn:
        pytest.skip("LOLLANET_TEST_DATABASE_URL is not set")
    database_name = "lollanet_test_" + uuid.uuid4().hex[:12]
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    info = conninfo_to_dict(admin_dsn)
    info["dbname"] = database_name
    test_dsn = make_conninfo(**info)
    try:
        with psycopg.connect(test_dsn) as conn:
            apply_migrations(conn, MIGRATIONS)
            inserted = ModelRepository(conn).import_legacy_library(legacy_library)
            assert inserted.versions == 88
        yield test_dsn
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (database_name,),
            )
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database_name)))


@pytest.fixture
def db_conn(database_url):
    conn = psycopg.connect(database_url)
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _future(base, model_id: str, version_no: int, status: str, *, relations=()):
    claims = tuple(replace(item, legacy_key=None) for item in base.claims)
    evidence = tuple(replace(item, legacy_key=None) for item in base.evidence)
    digest = hashlib.sha256(f"{model_id}@{version_no}:{status}".encode()).hexdigest()
    return replace(
        base,
        ref=ModelRef(model_id, version_no),
        status=status,
        name=f"Test {model_id}",
        english_name=f"Test {model_id}",
        content_hash=digest,
        claims=claims,
        evidence=evidence,
        relations=tuple(relations),
        legacy_source_path=None,
    )


def _candidates(legacy_library):
    return [version for version in legacy_library.versions if version.status == "candidate"]


def test_database_import_counts_and_idempotency(db_conn, legacy_library):
    repository = ModelRepository(db_conn)
    assert repository.audit_counts() == {
        "models": 88,
        "versions": 88,
        "claims": 792,
        "evidence": 266,
        "claim_evidence": 266,
        "formal_relations": 2,
        "categories": 91,
        "statuses": {"candidate": 58, "active": 28, "contested": 2, "deprecated": 0},
    }
    repeated = repository.import_legacy_library(legacy_library)
    assert repeated.versions == repeated.claims == repeated.evidence == repeated.relations == 0


def test_changed_same_version_is_rejected(db_conn, legacy_library):
    first = legacy_library.versions[0]
    changed = replace(first, content_hash="f" * 64)
    library = LegacyLibrary((changed,), 0, 1)
    with pytest.raises(RepositoryError, match="without a version bump"):
        ModelRepository(db_conn).import_legacy_library(library)


def test_failed_legacy_import_rolls_back_every_insert(db_conn, legacy_library):
    first = _candidates(legacy_library)[0]
    source_path = "01-Models/test-import-rollback.md"
    version = replace(
        _future(
            first,
            "mm-test-import-rollback",
            1,
            "candidate",
            relations=(
                RelationSpec(
                    ModelRef("mm-does-not-exist", 1),
                    "requires",
                    "Deliberately invalid target for import rollback.",
                    source_path,
                ),
            ),
        ),
        legacy_source_path=source_path,
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        ModelRepository(db_conn).import_legacy_library(LegacyLibrary((version,), 0, 1))
    assert db_conn.execute(
        "SELECT 1 FROM lollanet.models WHERE model_id = %s", (version.ref.model_id,)
    ).fetchone() is None


def test_full_text_neighbors_and_recursive_queries(db_conn):
    repository = ModelRepository(db_conn)
    assert any(item["model_id"] == "mm-base-rate" for item in repository.search("基础概率"))
    neighbors = repository.neighbors(ModelRef("mm-base-rate", 3))
    assert {(item["kind"], item["model_id"]) for item in neighbors} == {
        ("competes", "mm-availability-bias"),
        ("composes", "mm-bayesian-updating"),
    }
    traversed = repository.traverse(ModelRef("mm-base-rate", 3), max_depth=3)
    assert {item["model_id"] for item in traversed} == {
        "mm-base-rate",
        "mm-availability-bias",
        "mm-bayesian-updating",
    }


def test_database_constraints_and_immutability(db_conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.transaction():
            db_conn.execute("INSERT INTO lollanet.models(model_id, created_on) VALUES ('bad', current_date)")
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.model_versions(
                    model_id, version_no, status, kind, formalism, name, english_name,
                    created_on, legacy_source_path, content_hash
                ) SELECT model_id, version_no, status, kind, formalism, name, english_name,
                         created_on, legacy_source_path, content_hash
                    FROM lollanet.model_versions WHERE model_id = 'mm-base-rate'
                """
            )
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
        with db_conn.transaction():
            db_conn.execute(
                "UPDATE lollanet.model_versions SET name = 'changed' WHERE model_id = 'mm-base-rate'"
            )
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        with db_conn.transaction():
            db_conn.execute("SELECT 'seed'::lollanet.model_status")
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.claims(model_id, version_no, kind, ordinal, statement)
                VALUES ('mm-base-rate', 3, 'definition', 2, '   ')
                """
            )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.model_categories(
                    model_id, version_no, category_id, is_primary
                ) VALUES ('mm-base-rate', 3, 9223372036854775807, false)
                """
            )
    existing_slug = db_conn.execute("SELECT slug FROM lollanet.categories LIMIT 1").fetchone()[0]
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.categories(parent_id, kind, slug, name)
                VALUES (NULL, 'source_category', %s, 'Duplicate')
                """,
                (existing_slug,),
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.model_relations(
                    source_model_id, source_version_no, target_model_id, target_version_no,
                    kind, rationale, legacy_source_path
                ) VALUES ('mm-base-rate', 3, 'mm-base-rate', 3, 'requires', 'invalid',
                          '01-Models/test.md')
                """
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.transaction():
            db_conn.execute(
                """
                INSERT INTO lollanet.model_relations(
                    source_model_id, source_version_no, target_model_id, target_version_no,
                    kind, rationale, legacy_source_path
                ) VALUES ('mm-base-rate', 3, 'mm-availability-bias', 3, 'competes', 'reversed',
                          '01-Models/test.md')
                """
            )


def test_database_rejects_incomplete_version_at_commit(db_conn):
    with pytest.raises(psycopg.errors.CheckViolation, match="missing required claims"):
        with db_conn.transaction():
            db_conn.execute(
                "INSERT INTO lollanet.models(model_id, created_on) VALUES ('mm-test-incomplete', current_date)"
            )
            db_conn.execute(
                """
                INSERT INTO lollanet.model_versions(
                    model_id, version_no, status, kind, formalism, name, english_name,
                    created_on, legacy_source_path, content_hash
                ) VALUES ('mm-test-incomplete', 1, 'candidate', 'reasoning', false,
                          'Incomplete', 'Incomplete', current_date, '01-Models/test.md', %s)
                """,
                ("a" * 64,),
            )
            db_conn.execute("SET CONSTRAINTS ALL IMMEDIATE")


@pytest.mark.parametrize("change_kind", ["new", "amend", "contest", "deprecate", "split", "merge"])
def test_apply_model_change_is_atomic_for_each_evolution_kind(
    db_conn, legacy_library, change_kind
):
    first, second = _candidates(legacy_library)[:2]
    if change_kind == "new":
        inputs = []
        outputs = [_future(first, "mm-test-new", 1, "candidate")]
    elif change_kind == "amend":
        inputs = [first.ref]
        outputs = [_future(first, first.ref.model_id, first.ref.version_no + 1, "candidate")]
    elif change_kind == "contest":
        inputs = [first.ref]
        outputs = [_future(first, first.ref.model_id, first.ref.version_no + 1, "contested")]
    elif change_kind == "deprecate":
        inputs = [first.ref]
        outputs = [_future(first, first.ref.model_id, first.ref.version_no + 1, "deprecated")]
    elif change_kind == "split":
        inputs = [first.ref]
        outputs = [
            _future(first, first.ref.model_id, first.ref.version_no + 1, "deprecated"),
            _future(first, "mm-test-split-a", 1, "candidate"),
            _future(first, "mm-test-split-b", 1, "candidate"),
        ]
    else:
        inputs = [first.ref, second.ref]
        outputs = [
            _future(first, first.ref.model_id, first.ref.version_no + 1, "deprecated"),
            _future(second, second.ref.model_id, second.ref.version_no + 1, "deprecated"),
            _future(first, "mm-test-merged", 1, "candidate"),
        ]
    change_id = ModelRepository(db_conn).apply_model_change(
        change_kind, f"test {change_kind}", inputs, outputs
    )
    assert change_id > 0
    for output in outputs:
        assert db_conn.execute(
            "SELECT 1 FROM lollanet.model_versions WHERE (model_id, version_no) = (%s, %s)",
            (output.ref.model_id, output.ref.version_no),
        ).fetchone()
        assert ModelRepository(db_conn).current_version(output.ref.model_id)["version_no"] == output.ref.version_no
    if change_kind == "deprecate":
        assert not any(
            item["model_id"] == outputs[0].ref.model_id
            for item in ModelRepository(db_conn).search(outputs[0].ref.model_id)
        )
    assert db_conn.execute(
        "SELECT count(*) FROM lollanet.model_change_inputs WHERE change_id = %s", (change_id,)
    ).fetchone()[0] == len(inputs)
    for ref in inputs:
        assert db_conn.execute(
            "SELECT 1 FROM lollanet.model_versions WHERE (model_id, version_no) = (%s, %s)",
            (ref.model_id, ref.version_no),
        ).fetchone()


def test_competes_relation_is_stored_once_in_canonical_order(db_conn, legacy_library):
    first = _candidates(legacy_library)[0]
    output = _future(
        first,
        "mm-test-z-competitor",
        1,
        "candidate",
        relations=(
            RelationSpec(
                ModelRef("mm-base-rate", 3),
                "competes",
                "Both explain the same outcome with incompatible predictions.",
            ),
        ),
    )
    ModelRepository(db_conn).apply_model_change("new", "test competes normalization", [], [output])
    stored = db_conn.execute(
        """
        SELECT source_model_id, target_model_id
          FROM lollanet.model_relations
         WHERE kind = 'competes'
           AND (source_model_id = %s OR target_model_id = %s)
        """,
        (output.ref.model_id, output.ref.model_id),
    ).fetchall()
    assert stored == [("mm-base-rate", output.ref.model_id)]


def test_failed_relation_rolls_back_entire_change(db_conn, legacy_library):
    first = _candidates(legacy_library)[0]
    output = _future(
        first,
        first.ref.model_id,
        first.ref.version_no + 1,
        "candidate",
        relations=(
            RelationSpec(
                ModelRef("mm-does-not-exist", 1),
                "requires",
                "Deliberately invalid target for rollback test.",
            ),
        ),
    )
    before = db_conn.execute("SELECT count(*) FROM lollanet.model_changes").fetchone()[0]
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        ModelRepository(db_conn).apply_model_change("amend", "must roll back", [first.ref], [output])
    assert db_conn.execute("SELECT count(*) FROM lollanet.model_changes").fetchone()[0] == before
    assert db_conn.execute(
        "SELECT 1 FROM lollanet.model_versions WHERE (model_id, version_no) = (%s, %s)",
        (output.ref.model_id, output.ref.version_no),
    ).fetchone() is None
