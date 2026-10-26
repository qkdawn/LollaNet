# LollaNet PostgreSQL Model Library

`db/migrations/` contains ordered, immutable SQL migrations. The lightweight
migration runner records file names and SHA-256 checksums in
`lollanet.schema_migrations`; changing an applied migration is rejected.

Set a PostgreSQL DSN without committing credentials:

```powershell
$env:LOLLANET_DATABASE_URL = "postgresql://lollanet:password@localhost:5432/lollanet"
python 06-Scripts/model_db.py migrate
python 06-Scripts/model_db.py import-models --root . --dry-run
python 06-Scripts/model_db.py import-models --root .
python 06-Scripts/model_db.py audit
```

The importer reads only `01-Models/**/*.md`. A dry-run does not connect to the
database. A real import is one transaction and is idempotent for unchanged
legacy snapshots. Reusing a model ID and version after changing its source
content is rejected.

The legacy write CLIs are retired after cutover. `create_model.py` no longer
creates Markdown cards, `generate_model_index.py --apply` no longer writes a
static index, and `model_graph.py build --apply` no longer persists JSON or
Markdown graph files. In-memory graph compilation remains available only for
the excluded Runtime compatibility path.

Rollback before cutover means dropping the disposable database/schema and
continuing to use the legacy files. After cutover, do not reverse-sync a
Markdown export. Restore PostgreSQL from backup or apply a forward migration.
