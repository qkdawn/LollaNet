from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .domain import DomainError, validate_change, validate_version
from .models import InsertCounts, LegacyLibrary, ModelRef, RelationSpec, VersionSpec


class RepositoryError(RuntimeError):
    pass


class ModelRepository:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def import_legacy_library(self, library: LegacyLibrary) -> InsertCounts:
        counts = InsertCounts()
        with self.conn.transaction():
            for version in library.versions:
                inserted = self._insert_version(version, origin_change_id=None, allow_existing=True, counts=counts)
                if inserted:
                    counts.models += 1
            for version in library.versions:
                for relation in version.relations:
                    if self._insert_relation(version.ref, relation, origin_change_id=None, allow_existing=True):
                        counts.relations += 1
        return counts

    def apply_model_change(
        self,
        kind: str,
        rationale: str,
        inputs: Sequence[ModelRef],
        outputs: Sequence[VersionSpec],
    ) -> int:
        if not rationale.strip():
            raise DomainError("accepted change requires a rationale")
        with self.conn.transaction():
            input_statuses = self._lock_current_inputs(inputs)
            validate_change(kind, inputs, input_statuses, outputs)
            change_id = self.conn.execute(
                "INSERT INTO lollanet.model_changes(kind, rationale) VALUES (%s, %s) RETURNING change_id",
                (kind, rationale.strip()),
            ).fetchone()[0]
            for ref in inputs:
                self.conn.execute(
                    """
                    INSERT INTO lollanet.model_change_inputs(change_id, model_id, version_no)
                    VALUES (%s, %s, %s)
                    """,
                    (change_id, ref.model_id, ref.version_no),
                )
            counts = InsertCounts()
            for output in outputs:
                self._insert_version(
                    output,
                    origin_change_id=change_id,
                    allow_existing=False,
                    counts=counts,
                )
            for output in outputs:
                for relation in output.relations:
                    self._insert_relation(
                        output.ref,
                        relation,
                        origin_change_id=change_id,
                        allow_existing=False,
                    )
            return change_id

    def current_version(self, model_id: str) -> dict[str, Any] | None:
        with self.conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT * FROM lollanet.current_model_versions WHERE model_id = %s",
                (model_id,),
            ).fetchone()

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        with self.conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                """
                WITH q AS (SELECT websearch_to_tsquery('simple'::regconfig, %s) AS value),
                hits AS (
                    SELECT v.model_id, v.version_no, 'model'::text AS entity_type,
                           v.name AS text,
                           ts_rank(v.search_vector, q.value) AS rank
                      FROM lollanet.current_model_versions v CROSS JOIN q
                     WHERE v.status <> 'deprecated' AND v.search_vector @@ q.value
                    UNION ALL
                    SELECT c.model_id, c.version_no, 'claim'::text,
                           c.statement, ts_rank(c.search_vector, q.value)
                      FROM lollanet.claims c
                      JOIN lollanet.current_model_versions v
                        ON (v.model_id, v.version_no) = (c.model_id, c.version_no)
                      CROSS JOIN q
                     WHERE v.status <> 'deprecated' AND c.search_vector @@ q.value
                    UNION ALL
                    SELECT c.model_id, c.version_no, 'evidence'::text,
                           e.summary, ts_rank(e.search_vector, q.value)
                      FROM lollanet.evidence e
                      JOIN lollanet.claim_evidence ce ON ce.evidence_id = e.evidence_id
                      JOIN lollanet.claims c ON c.claim_id = ce.claim_id
                      JOIN lollanet.current_model_versions v
                        ON (v.model_id, v.version_no) = (c.model_id, c.version_no)
                      CROSS JOIN q
                     WHERE v.status <> 'deprecated' AND e.search_vector @@ q.value
                )
                SELECT model_id, version_no, entity_type, text, max(rank) AS rank
                  FROM hits
                 GROUP BY model_id, version_no, entity_type, text
                 ORDER BY rank DESC, model_id, entity_type
                 LIMIT %s
                """,
                (query.strip(), limit),
            ).fetchall()

    def neighbors(self, ref: ModelRef) -> list[dict[str, Any]]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                """
                SELECT kind::text,
                       CASE WHEN source_model_id = %s AND source_version_no = %s
                            THEN target_model_id ELSE source_model_id END AS model_id,
                       CASE WHEN source_model_id = %s AND source_version_no = %s
                            THEN target_version_no ELSE source_version_no END AS version_no,
                       CASE
                           WHEN kind = 'competes' THEN 'peer'
                           WHEN source_model_id = %s AND source_version_no = %s THEN 'out'
                           ELSE 'in'
                       END AS direction,
                       rationale
                  FROM lollanet.model_relations
                 WHERE (source_model_id, source_version_no) = (%s, %s)
                    OR (target_model_id, target_version_no) = (%s, %s)
                 ORDER BY kind, model_id, version_no
                """,
                (
                    ref.model_id,
                    ref.version_no,
                    ref.model_id,
                    ref.version_no,
                    ref.model_id,
                    ref.version_no,
                    ref.model_id,
                    ref.version_no,
                    ref.model_id,
                    ref.version_no,
                ),
            ).fetchall()

    def traverse(self, ref: ModelRef, max_depth: int = 4) -> list[dict[str, Any]]:
        if max_depth < 0 or max_depth > 20:
            raise ValueError("max_depth must be between 0 and 20")
        with self.conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                """
                WITH RECURSIVE walk(model_id, version_no, depth, path) AS (
                    SELECT %s::text, %s::integer, 0,
                           ARRAY[%s::text || '@' || %s::text]
                    UNION ALL
                    SELECT next_ref.model_id, next_ref.version_no, walk.depth + 1,
                           walk.path || (next_ref.model_id || '@' || next_ref.version_no::text)
                      FROM walk
                      JOIN LATERAL (
                          SELECT CASE WHEN r.source_model_id = walk.model_id
                                           AND r.source_version_no = walk.version_no
                                      THEN r.target_model_id ELSE r.source_model_id END AS model_id,
                                 CASE WHEN r.source_model_id = walk.model_id
                                           AND r.source_version_no = walk.version_no
                                      THEN r.target_version_no ELSE r.source_version_no END AS version_no
                            FROM lollanet.model_relations r
                           WHERE (r.source_model_id, r.source_version_no) = (walk.model_id, walk.version_no)
                              OR (r.target_model_id, r.target_version_no) = (walk.model_id, walk.version_no)
                      ) AS next_ref ON true
                     WHERE walk.depth < %s
                       AND NOT (next_ref.model_id || '@' || next_ref.version_no::text = ANY(walk.path))
                )
                SELECT model_id, version_no, min(depth) AS depth
                  FROM walk
                 GROUP BY model_id, version_no
                 ORDER BY depth, model_id, version_no
                """,
                (
                    ref.model_id,
                    ref.version_no,
                    ref.model_id,
                    ref.version_no,
                    max_depth,
                ),
            ).fetchall()

    def audit_counts(self) -> dict[str, Any]:
        observed_statuses = dict(
            self.conn.execute(
                """
                SELECT status::text, count(*)
                  FROM lollanet.current_model_versions
                 GROUP BY status ORDER BY status
                """
            ).fetchall()
        )
        statuses = {
            status: observed_statuses.get(status, 0)
            for status in ("candidate", "active", "contested", "deprecated")
        }
        counts = self.conn.execute(
            """
            SELECT
                (SELECT count(*) FROM lollanet.models),
                (SELECT count(*) FROM lollanet.model_versions),
                (SELECT count(*) FROM lollanet.claims),
                (SELECT count(*) FROM lollanet.evidence),
                (SELECT count(*) FROM lollanet.claim_evidence),
                (SELECT count(*) FROM lollanet.model_relations),
                (SELECT count(*) FROM lollanet.categories)
            """
        ).fetchone()
        return {
            "models": counts[0],
            "versions": counts[1],
            "claims": counts[2],
            "evidence": counts[3],
            "claim_evidence": counts[4],
            "formal_relations": counts[5],
            "categories": counts[6],
            "statuses": statuses,
        }

    def _lock_current_inputs(self, inputs: Sequence[ModelRef]) -> dict[ModelRef, str]:
        statuses: dict[ModelRef, str] = {}
        for ref in sorted(inputs):
            row = self.conn.execute(
                """
                SELECT v.status::text,
                       (SELECT max(version_no) FROM lollanet.model_versions WHERE model_id = v.model_id)
                  FROM lollanet.model_versions v
                 WHERE (v.model_id, v.version_no) = (%s, %s)
                 FOR UPDATE
                """,
                (ref.model_id, ref.version_no),
            ).fetchone()
            if row is None:
                raise DomainError(f"unknown input version: {ref}")
            if row[1] != ref.version_no:
                raise DomainError(f"change input is not current: {ref}")
            statuses[ref] = row[0]
        return statuses

    def _insert_version(
        self,
        version: VersionSpec,
        *,
        origin_change_id: int | None,
        allow_existing: bool,
        counts: InsertCounts,
    ) -> bool:
        validate_version(version)
        legacy_path = version.legacy_source_path if origin_change_id is None else None
        if origin_change_id is None and not legacy_path:
            raise RepositoryError(f"legacy version {version.ref} needs source provenance")
        model_inserted = self.conn.execute(
            """
            INSERT INTO lollanet.models(model_id, created_on)
            VALUES (%s, %s)
            ON CONFLICT (model_id) DO NOTHING
            RETURNING model_id
            """,
            (version.ref.model_id, version.created_on),
        ).fetchone()
        if model_inserted is None:
            created_on = self.conn.execute(
                "SELECT created_on::text FROM lollanet.models WHERE model_id = %s",
                (version.ref.model_id,),
            ).fetchone()[0]
            if created_on != version.created_on:
                raise RepositoryError(f"stable model metadata changed: {version.ref.model_id}")

        params = (
            version.ref.model_id,
            version.ref.version_no,
            version.status,
            version.kind,
            version.formalism,
            version.name,
            version.english_name,
            version.created_on,
            origin_change_id,
            legacy_path,
            version.content_hash,
        )
        if allow_existing:
            inserted = self.conn.execute(
                """
                INSERT INTO lollanet.model_versions(
                    model_id, version_no, status, kind, formalism, name,
                    english_name, created_on, origin_change_id,
                    legacy_source_path, content_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id, version_no) DO NOTHING
                RETURNING model_id
                """,
                params,
            ).fetchone()
            if inserted is None:
                existing_hash = self.conn.execute(
                    """
                    SELECT content_hash FROM lollanet.model_versions
                     WHERE (model_id, version_no) = (%s, %s)
                    """,
                    (version.ref.model_id, version.ref.version_no),
                ).fetchone()[0]
                if existing_hash != version.content_hash:
                    raise RepositoryError(f"legacy snapshot changed without a version bump: {version.ref}")
                return False
        else:
            self.conn.execute(
                """
                INSERT INTO lollanet.model_versions(
                    model_id, version_no, status, kind, formalism, name,
                    english_name, created_on, origin_change_id,
                    legacy_source_path, content_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params,
            )
        counts.versions += 1

        claim_ids: dict[tuple[str, int], int] = {}
        for claim in version.claims:
            claim_id = self.conn.execute(
                """
                INSERT INTO lollanet.claims(
                    model_id, version_no, kind, ordinal, statement, time_window, legacy_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING claim_id
                """,
                (
                    version.ref.model_id,
                    version.ref.version_no,
                    claim.kind,
                    claim.ordinal,
                    claim.statement,
                    claim.time_window,
                    claim.legacy_key,
                ),
            ).fetchone()[0]
            claim_ids[(claim.kind, claim.ordinal)] = claim_id
            counts.claims += 1

        parent_id: int | None = None
        leaf_id: int | None = None
        for level in version.categories:
            row = self.conn.execute(
                """
                INSERT INTO lollanet.categories(parent_id, kind, slug, name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (slug) DO NOTHING
                RETURNING category_id
                """,
                (parent_id, level.kind, level.slug, level.name),
            ).fetchone()
            if row is None:
                existing = self.conn.execute(
                    "SELECT category_id, parent_id, kind::text, name FROM lollanet.categories WHERE slug = %s",
                    (level.slug,),
                ).fetchone()
                if existing[1:] != (parent_id, level.kind, level.name):
                    raise RepositoryError(f"category slug collision: {level.slug}")
                leaf_id = existing[0]
            else:
                leaf_id = row[0]
                counts.categories += 1
            parent_id = leaf_id
        self.conn.execute(
            """
            INSERT INTO lollanet.model_categories(model_id, version_no, category_id, is_primary)
            VALUES (%s, %s, %s, true)
            """,
            (version.ref.model_id, version.ref.version_no, leaf_id),
        )
        counts.mappings += 1

        for item in version.evidence:
            evidence_id = self.conn.execute(
                """
                INSERT INTO lollanet.evidence(source_ref, locator, url, summary, legacy_key)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING evidence_id
                """,
                (item.source_ref, item.locator, item.url, item.summary, item.legacy_key),
            ).fetchone()[0]
            self.conn.execute(
                """
                INSERT INTO lollanet.claim_evidence(claim_id, evidence_id, stance, rationale)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    claim_ids[(item.claim_kind, item.claim_ordinal)],
                    evidence_id,
                    item.stance,
                    item.rationale,
                ),
            )
            counts.evidence += 1
            counts.mappings += 1
        return True

    def _insert_relation(
        self,
        source: ModelRef,
        relation: RelationSpec,
        *,
        origin_change_id: int | None,
        allow_existing: bool,
    ) -> bool:
        target = relation.target
        if relation.kind == "competes" and source.model_id > target.model_id:
            source, target = target, source
        legacy_path = relation.legacy_source_path if origin_change_id is None else None
        params = (
            source.model_id,
            source.version_no,
            target.model_id,
            target.version_no,
            relation.kind,
            relation.rationale,
            origin_change_id,
            legacy_path,
        )
        if allow_existing:
            row = self.conn.execute(
                """
                INSERT INTO lollanet.model_relations(
                    source_model_id, source_version_no, target_model_id,
                    target_version_no, kind, rationale, origin_change_id,
                    legacy_source_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING RETURNING source_model_id
                """,
                params,
            ).fetchone()
            if row is not None:
                return True
            existing = self.conn.execute(
                """
                SELECT rationale FROM lollanet.model_relations
                 WHERE (source_model_id, source_version_no, target_model_id,
                        target_version_no, kind) = (%s, %s, %s, %s, %s)
                """,
                params[:5],
            ).fetchone()
            if existing is None or existing[0] != relation.rationale:
                raise RepositoryError(f"formal relation changed without a version bump: {source}")
            return False
        self.conn.execute(
            """
            INSERT INTO lollanet.model_relations(
                source_model_id, source_version_no, target_model_id,
                target_version_no, kind, rationale, origin_change_id,
                legacy_source_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            params,
        )
        return True


def counts_as_dict(counts: InsertCounts) -> dict[str, Any]:
    return asdict(counts)
