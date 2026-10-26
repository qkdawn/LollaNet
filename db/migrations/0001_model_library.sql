CREATE SCHEMA IF NOT EXISTS lollanet;

CREATE TYPE lollanet.model_status AS ENUM (
    'candidate', 'active', 'contested', 'deprecated'
);
CREATE TYPE lollanet.model_kind AS ENUM ('world', 'reasoning', 'action');
CREATE TYPE lollanet.claim_kind AS ENUM (
    'definition', 'mechanism', 'input', 'transformation', 'output',
    'premise', 'boundary', 'falsifier', 'prediction'
);
CREATE TYPE lollanet.evidence_stance AS ENUM ('supports', 'opposes', 'limits');
CREATE TYPE lollanet.relation_kind AS ENUM (
    'requires', 'composes', 'competes', 'corrects', 'specializes'
);
CREATE TYPE lollanet.change_kind AS ENUM (
    'new', 'amend', 'split', 'merge', 'contest', 'deprecate'
);
CREATE TYPE lollanet.category_kind AS ENUM (
    'source_category', 'discipline', 'subdiscipline'
);

CREATE TABLE lollanet.model_changes (
    change_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind lollanet.change_kind NOT NULL,
    rationale text NOT NULL CHECK (btrim(rationale) <> ''),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE lollanet.models (
    model_id text PRIMARY KEY
        CHECK (model_id ~ '^mm-[a-z0-9]+(?:-[a-z0-9]+)*$'),
    created_on date NOT NULL
);

CREATE TABLE lollanet.model_versions (
    model_id text NOT NULL REFERENCES lollanet.models(model_id),
    version_no integer NOT NULL CHECK (version_no > 0),
    status lollanet.model_status NOT NULL,
    kind lollanet.model_kind NOT NULL,
    formalism boolean NOT NULL,
    name text NOT NULL CHECK (btrim(name) <> ''),
    english_name text NOT NULL CHECK (btrim(english_name) <> ''),
    created_on date NOT NULL,
    origin_change_id bigint REFERENCES lollanet.model_changes(change_id),
    legacy_source_path text,
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, name || ' ' || english_name || ' ' || model_id)
    ) STORED,
    PRIMARY KEY (model_id, version_no),
    CHECK (num_nonnulls(origin_change_id, legacy_source_path) = 1),
    CHECK (legacy_source_path IS NULL OR legacy_source_path LIKE '01-Models/%')
);

CREATE TABLE lollanet.claims (
    claim_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id text NOT NULL,
    version_no integer NOT NULL,
    kind lollanet.claim_kind NOT NULL,
    ordinal smallint NOT NULL CHECK (ordinal > 0),
    statement text NOT NULL CHECK (btrim(statement) <> ''),
    time_window text,
    legacy_key text UNIQUE,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, statement)
    ) STORED,
    FOREIGN KEY (model_id, version_no)
        REFERENCES lollanet.model_versions(model_id, version_no),
    UNIQUE (model_id, version_no, kind, ordinal),
    CHECK ((kind = 'prediction') = (time_window IS NOT NULL)),
    CHECK (time_window IS NULL OR btrim(time_window) <> '')
);

CREATE TABLE lollanet.evidence (
    evidence_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_ref text NOT NULL CHECK (btrim(source_ref) <> ''),
    locator text,
    url text,
    summary text NOT NULL CHECK (btrim(summary) <> ''),
    legacy_key text UNIQUE,
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'simple'::regconfig,
            source_ref || ' ' || coalesce(locator, '') || ' ' || summary
        )
    ) STORED
);

CREATE TABLE lollanet.claim_evidence (
    claim_id bigint NOT NULL REFERENCES lollanet.claims(claim_id),
    evidence_id bigint NOT NULL REFERENCES lollanet.evidence(evidence_id),
    stance lollanet.evidence_stance NOT NULL,
    rationale text NOT NULL CHECK (btrim(rationale) <> ''),
    PRIMARY KEY (claim_id, evidence_id, stance)
);

CREATE TABLE lollanet.categories (
    category_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    parent_id bigint REFERENCES lollanet.categories(category_id),
    kind lollanet.category_kind NOT NULL,
    slug text NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'),
    name text NOT NULL CHECK (btrim(name) <> ''),
    CHECK (parent_id IS NULL OR parent_id <> category_id)
);

CREATE TABLE lollanet.model_categories (
    model_id text NOT NULL,
    version_no integer NOT NULL,
    category_id bigint NOT NULL REFERENCES lollanet.categories(category_id),
    is_primary boolean NOT NULL DEFAULT true,
    PRIMARY KEY (model_id, version_no, category_id),
    FOREIGN KEY (model_id, version_no)
        REFERENCES lollanet.model_versions(model_id, version_no)
);
CREATE UNIQUE INDEX model_categories_one_primary
    ON lollanet.model_categories(model_id, version_no)
    WHERE is_primary;

CREATE TABLE lollanet.model_relations (
    source_model_id text NOT NULL,
    source_version_no integer NOT NULL,
    target_model_id text NOT NULL,
    target_version_no integer NOT NULL,
    kind lollanet.relation_kind NOT NULL,
    rationale text NOT NULL CHECK (btrim(rationale) <> ''),
    origin_change_id bigint REFERENCES lollanet.model_changes(change_id),
    legacy_source_path text,
    FOREIGN KEY (source_model_id, source_version_no)
        REFERENCES lollanet.model_versions(model_id, version_no),
    FOREIGN KEY (target_model_id, target_version_no)
        REFERENCES lollanet.model_versions(model_id, version_no),
    PRIMARY KEY (
        source_model_id, source_version_no,
        target_model_id, target_version_no, kind
    ),
    CHECK (source_model_id <> target_model_id),
    CHECK (num_nonnulls(origin_change_id, legacy_source_path) = 1),
    CHECK (legacy_source_path IS NULL OR legacy_source_path LIKE '01-Models/%'),
    CHECK (kind <> 'competes' OR source_model_id < target_model_id)
);

CREATE TABLE lollanet.model_change_inputs (
    change_id bigint NOT NULL REFERENCES lollanet.model_changes(change_id),
    model_id text NOT NULL,
    version_no integer NOT NULL,
    PRIMARY KEY (change_id, model_id, version_no),
    FOREIGN KEY (model_id, version_no)
        REFERENCES lollanet.model_versions(model_id, version_no)
);

CREATE VIEW lollanet.current_model_versions AS
SELECT DISTINCT ON (model_id) *
FROM lollanet.model_versions
ORDER BY model_id, version_no DESC;

CREATE INDEX model_versions_search_gin
    ON lollanet.model_versions USING gin(search_vector);
CREATE INDEX claims_search_gin ON lollanet.claims USING gin(search_vector);
CREATE INDEX evidence_search_gin ON lollanet.evidence USING gin(search_vector);
CREATE INDEX claims_version_idx ON lollanet.claims(model_id, version_no);
CREATE INDEX claim_evidence_evidence_idx ON lollanet.claim_evidence(evidence_id);
CREATE INDEX categories_parent_idx ON lollanet.categories(parent_id);
CREATE INDEX model_categories_category_idx ON lollanet.model_categories(category_id);
CREATE INDEX model_relations_target_idx
    ON lollanet.model_relations(target_model_id, target_version_no);
CREATE INDEX model_change_inputs_version_idx
    ON lollanet.model_change_inputs(model_id, version_no);

CREATE FUNCTION lollanet.assert_model_version_complete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    missing lollanet.claim_kind[];
BEGIN
    SELECT array_agg(required_kind ORDER BY required_kind::text)
      INTO missing
      FROM unnest(enum_range(NULL::lollanet.claim_kind)) AS required_kind
     WHERE NOT EXISTS (
         SELECT 1
           FROM lollanet.claims c
          WHERE c.model_id = NEW.model_id
            AND c.version_no = NEW.version_no
            AND c.kind = required_kind
     );
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION 'model version % missing required claims: %',
            NEW.model_id || '@' || NEW.version_no, missing
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER model_version_complete_at_commit
AFTER INSERT ON lollanet.model_versions
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION lollanet.assert_model_version_complete();

CREATE FUNCTION lollanet.reject_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; create a successor model version', TG_TABLE_NAME
        USING ERRCODE = '55000';
END;
$$;

CREATE TRIGGER model_versions_append_only
BEFORE UPDATE OR DELETE ON lollanet.model_versions
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER claims_append_only
BEFORE UPDATE OR DELETE ON lollanet.claims
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER evidence_append_only
BEFORE UPDATE OR DELETE ON lollanet.evidence
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER claim_evidence_append_only
BEFORE UPDATE OR DELETE ON lollanet.claim_evidence
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER model_categories_append_only
BEFORE UPDATE OR DELETE ON lollanet.model_categories
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER model_relations_append_only
BEFORE UPDATE OR DELETE ON lollanet.model_relations
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER model_changes_append_only
BEFORE UPDATE OR DELETE ON lollanet.model_changes
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
CREATE TRIGGER model_change_inputs_append_only
BEFORE UPDATE OR DELETE ON lollanet.model_change_inputs
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
