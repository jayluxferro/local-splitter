# Spec: Postgres backend for local-splitter (T3 semantic cache)

Status: implemented (as-built matches; full suite green on local_splitter_test)
Date: 2026-08-29
Author: Jay

## Context

T3 (semantic cache) stores every cloud response together with its
embedding in a local SQLite file (`cache.sqlite`) using the sqlite-vec
virtual table (vec0, cosine distance, `MATCH ... k = 1` KNN).  It is
the only database in this project.  Lattice just completed the same
move (sqlite → Postgres, psycopg3 + pgvector, fail-open,
schema_migrations — see `lattice/src/lattice/db.py` and
`lattice/docs/pg-migration-spec.md`); this spec ports local-splitter's
`CacheStore` the same way: one storage, one SQL dialect, one test
matrix.

Target environment: the same local Postgres.app 17.6 instance Lattice
uses (port 5432, trust auth on local sockets), pgvector 0.8.1
available.  A dedicated `local_splitter` role + database (plus
`local_splitter_test` for the suite), mirroring `lattice` /
`lattice_test`.  The role and databases already exist (created during
environment prep) with the `vector` extension installed in both.

## Survey findings that shaped this spec

- `sem_cache.py` is the only sqlite consumer.  Call sites:
  `cli.py` (serve-http `_build_pipeline`, eval command),
  `evals/run_eval.py` (per-workload cache), tests in
  `test_pipeline_sem_cache.py` and `test_pipeline_compress.py`
  (one `_store(tmp_path)` helper each).
- **KNN is the core operation** — unlike Lattice, whose embeddings were
  a blob store with cosine computed in Python, vec0's `embedding MATCH
  ? AND k = 1` is a real nearest-neighbour query.  The PG port uses
  pgvector's cosine distance operator: `ORDER BY embedding <=> %s
  LIMIT 1`.  Exact scan is deliberate (lattice-style): the cache is
  ≤ thousands of rows with a 24 h TTL; a comment records where an
  hnsw/ivfflat index would go if that ever changes.
- `created_at` is stored as TEXT because of a vec0 bug with REAL
  auxiliary columns.  PG stores `DOUBLE PRECISION` directly and the
  string-cast workaround dies.
- `evict_expired` does SELECT rowids → executemany DELETE (vec0
  aux-column deletes).  PG replaces it with one
  `DELETE ... WHERE created_at < %s`; `rowcount` gives the count.
- Cache contents are ephemeral (TTL 24 h).  **No data migration
  tooling** — unlike Lattice's 4.6 GB recovery, old `.sqlite` files
  are simply dead data after this change.
- Embedding dimension: 768 in production (nomic-embed-text), 32 in
  tests.  A separate test database keeps them apart; within one
  database the `vector(N)` column type pins a single dimension.
- Row access in sem_cache is positional-only (`rows[0]` unpacking),
  so psycopg's default tuple rows suffice — the custom sqlite3.Row
  shim Lattice needed for its ~200 named-access sites is NOT needed
  here.

## Objectives

1. Replace SQLite with Postgres as the cache backend (no dual support).
2. Keep the `CacheStore` public API (`lookup`, `store`,
   `evict_expired`, `size`, `close`) — pipeline callers change only
   where they construct a store.
3. Port the full test suite to run against a dedicated
   `local_splitter_test` database.
4. GitHub Actions: postgres service with pgvector.
5. No data migration for cache contents (ephemeral by design).

## Architecture

- **Driver**: `psycopg[binary]>=3.2`; vectors via `pgvector>=0.3`
  (`register_vector` maps `list[float]` → `vector` on write).  The
  `sqlite-vec` dependency is removed.  One connection per `CacheStore`,
  `autocommit = True` (same reasoning as lattice's `Database`: the
  sqlite era's explicit commits become per-statement persistence).
- **Fail-open**: pgvector missing at connect → `has_vec = False` +
  warning; `lookup` returns a miss and `store` no-ops returning 0.
  Matches lattice's `_try_load_vec` and the pipeline's existing
  fail-open handling (cache errors are already swallowed as misses).
  Connection errors still raise (lattice parity — only the vector
  layer is fail-open).
- **Schema** (versioned via `schema_migrations`, lattice style):

  ```sql
  CREATE TABLE IF NOT EXISTS schema_migrations (
      version    INTEGER PRIMARY KEY,
      applied_at BIGINT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS cache_entry (
      id            BIGSERIAL PRIMARY KEY,
      namespace     TEXT NOT NULL DEFAULT 'default',
      embedding     vector(N),          -- only when has_vec (see below)
      response      TEXT NOT NULL,
      model         TEXT NOT NULL,
      finish_reason TEXT NOT NULL,
      created_at    DOUBLE PRECISION NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_cache_entry_namespace ON cache_entry(namespace);
  CREATE INDEX IF NOT EXISTS idx_cache_entry_created  ON cache_entry(created_at);
  ```

  `cache_entry` is created only when pgvector is present (the same
  skip-and-record shape lattice v1 uses for its `embedding` table);
  without it the cache is inert but the constructor still succeeds.
- **KNN lookup**:

  ```sql
  SELECT id, embedding <=> %s AS distance,
         response, model, finish_reason, created_at
    FROM cache_entry
   WHERE namespace = %s
   ORDER BY embedding <=> %s
   LIMIT 1
  ```

  Threshold and TTL checks stay in Python (same semantics as today).
- **Namespace**: per-workload eval caches (`cache_{wl_name}.sqlite`
  files) become rows with `namespace = 'cache_{wl_name}'` in the one
  database.
- **Dialect ports**: `?` → `%s`; sqlite rowid → BIGSERIAL id;
  `str(time.time())` → float column.
## API contract (pinned — parallel workstreams code against this)

`CacheEntry` (public dataclass) is unchanged:
`rowid: int, distance: float, similarity: float, response: str,
model: str, finish_reason: str, created_at: float`.

```python
class CacheStore:
    """Postgres + pgvector semantic cache.  Fail-open like lattice's Database."""

    def __init__(
        self,
        dsn: str,
        *,
        embed_dim: int = 768,
        namespace: str = "default",
    ) -> None:
        # psycopg.connect(dsn); self._conn.autocommit = True
        # _try_load_vec(): CREATE EXTENSION IF NOT EXISTS vector (best-effort,
        #   trusted extension — DB owner privilege), then pg_extension check +
        #   register_vector (lattice shape)
        # _apply_migrations(): schema_migrations + v1 (cache_entry if has_vec)

    @property
    def has_vec(self) -> bool: ...

    def lookup(
        self,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl: int = DEFAULT_TTL,
    ) -> CacheEntry | None:
        # ValueError if len(embedding) != self._embed_dim
        # KNN <=> LIMIT 1; distance > 1 - threshold -> None;
        # created_at < time.time() - ttl -> None (expired)

    def store(
        self,
        embedding: list[float],
        *,
        response: str,
        model: str,
        finish_reason: str,
    ) -> int:
        # INSERT ... RETURNING id; 0 when not has_vec (fail-open);
        # ValueError on dim mismatch

    def evict_expired(self, ttl: int = DEFAULT_TTL) -> int:
        # DELETE FROM cache_entry WHERE namespace = %s AND created_at < %s;
        # return cur.rowcount

    @property
    def size(self) -> int: ...

    def close(self) -> None: ...


def open_cache_store(
    dsn: str, *, embed_dim: int = 768, namespace: str = "default"
) -> CacheStore:
    """Convenience opener (open_database parity)."""
```

Module constants `DEFAULT_SIMILARITY_THRESHOLD`, `DEFAULT_TTL`,
`DEFAULT_CHARS_PER_TOKEN` stay.  `_serialize`/struct and the
sqlite-vec import go away.  The pipeline-facing functions (`lookup`,
`store_response`, `cache_embed_source`, `CacheLookupResult`, the
redactor-normalization helpers) are untouched — this is a storage-layer
swap only.

## Config and call sites

- **Env**: `LOCAL_SPLITTER_DB_URL` (default
  `postgresql://local_splitter@localhost:5432/local_splitter`); tests
  override with `LOCAL_SPLITTER_TEST_DB_URL` (default
  `postgresql://local_splitter@localhost:5432/local_splitter_test`).
- **cli.py**: module-level `_env(name, default)` helper (lattice cli
  style); new `--cache-db-url` typer option on `serve-http` and the
  eval command, defaulting through `_env`.  `_build_pipeline` gains a
  `cache_db_url` parameter and constructs
  `CacheStore(cache_db_url, embed_dim=768)` under the existing T3 +
  embed_model gating.  The eval command replaces
  `cache_{wl_path.stem}.sqlite` with
  `CacheStore(cache_db_url, embed_dim=768,
  namespace=f"cache_{wl_path.stem}")`.
- **evals/run_eval.py**: same namespace pattern
  (`namespace=f"cache_{wl_name}"`), DSN from `LOCAL_SPLITTER_DB_URL`
  env (no CLI — it is a script).
- **pyproject.toml**: drop `sqlite-vec>=0.1.1`; add
  `psycopg[binary]>=3.2`, `pgvector>=0.3`.  Refresh `uv.lock`
  (`uv lock`).

## Tests

- **tests/conftest.py** (new):
  - `TEST_DB_URL = os.environ.get("LOCAL_SPLITTER_TEST_DB_URL",
    "postgresql://local_splitter@localhost:5432/local_splitter_test")`
  - `KNOWN_TABLES = ("cache_entry", "schema_migrations")`
  - `drop_cache_tables()` — fresh short-lived psycopg connection,
    autocommit, `DROP TABLE IF EXISTS ... CASCADE` per table (lattice
    conftest shape).
- Test files construct stores through their `_store()` helpers, which
  become `drop_cache_tables()` + `CacheStore(TEST_DB_URL,
  embed_dim=32)`.  Drop-per-construction gives per-test isolation
  without an autouse fixture, so the non-DB tests keep running without
  Postgres.  `test_pipeline_sem_cache.py` imports
  `from conftest import TEST_DB_URL`.
- `test_pipeline_compress.py`: same swap for its one direct
  `CacheStore(...)` construction.

## CI

`.github/workflows/ci.yml` gains a service container and env:

```yaml
    services:
      postgres:
        image: pgvector/pgvector:pg17
        env:
          POSTGRES_USER: local_splitter
          POSTGRES_PASSWORD: local_splitter
          POSTGRES_DB: local_splitter_test
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U local_splitter"
          --health-interval 5s
          --health-timeout 5s
          --health-retries 10
    env:
      LOCAL_SPLITTER_TEST_DB_URL: postgresql://local_splitter:local_splitter@localhost:5432/local_splitter_test
```

The `pgvector/pgvector` image ships the extension files but does NOT
run `CREATE EXTENSION` itself — the store provisions it: `_try_load_vec`
issues `CREATE EXTENSION IF NOT EXISTS vector` before its detection
check (pgvector is a trusted extension, so the database owner —
`POSTGRES_USER` in CI, `local_splitter` locally — has the needed
privilege).  No extra CI step; fail-open covers restricted installs.
## Workstreams (agent fan-out)

| # | Workstream | Depends on | Delivers |
|---|------------|------------|----------|
| W1 | `sem_cache.py` storage port: psycopg3 `CacheStore`, `_try_load_vec` (auto-provision + register), schema_migrations + v1 DDL, KNN `<=>` lookup, namespace, `open_cache_store` | — | rewritten `src/local_splitter/pipeline/sem_cache.py` |
| W2 | Call sites + deps: cli.py (`_env`, `--cache-db-url`, `_build_pipeline`, eval command), evals/run_eval.py, pyproject.toml, uv.lock refresh | W1 API (pinned above) | those files |
| W3 | Test infra: tests/conftest.py, `_store()` helpers in test_pipeline_sem_cache.py + test_pipeline_compress.py | W1 API | tests green on `local_splitter_test` |
| W4 | CI + docs: ci.yml postgres service, ARCHITECTURE.md, TACTICS.md, README.md, .agent/memory/gotchas.md | — | those files |
| W5 | Environment + verification: role/DBs/extension (done during prep), full pytest, ruff, quality review | W1–W4 | green suite, clean lint |

W1 first (or parallel from the pinned contract); W2/W3/W4 in parallel;
W5 is the review gate: full suite + ruff + spot-check of the diff
against the quality standards (data-driven, single source of truth,
no scattered ad-hoc checks).

## Success metrics

- Full pytest suite green against `local_splitter_test` (same semantics
  as today — hit/miss/TTL/eviction/namespace isolation).
- `ruff check src/ tests/` clean.
- `ci.yml` runs the same tests against the pgvector service container
  (LOCAL_SPLITTER_TEST_DB_URL pointing at it).
- Existing `.sqlite` cache files become dead data; documented that they
  are ignored (no import tooling — cache is ephemeral).

## Decisions (stated defaults; change on approval)

1. **Backend replacement**: SQLite removed entirely (no dual support).
   Old cache files are not migrated — entries expire in 24 h anyway.
2. **DB location**: local Postgres.app instance, new `local_splitter`
   role (trust auth locally) + `local_splitter` / `local_splitter_test`
   databases, `vector` extension in both.  Already provisioned.
3. **pgvector provisioning**: `CacheStore._try_load_vec` attempts
   `CREATE EXTENSION IF NOT EXISTS vector` before its detection check.
   Deliberate deviation from lattice's detect-only `db.py` — lattice
   has no GitHub CI and pre-creates the extension by hand;
   local-splitter's CI service container needs zero-step provisioning,
   and pgvector is a trusted extension so the DB owner can create it.
4. **KNN strategy**: exact scan with `<=>`, no hnsw/ivfflat index
   (lattice-style deliberate absence; documented upgrade path in a
   comment).
5. **Test DB**: `local_splitter_test` on the same instance; per-test
   isolation via drop-before-construct in the `_store()` helpers (no
   autouse fixture, so non-DB tests stay Postgres-free).
6. **Namespace**: per-workload eval caches become namespace-tagged rows
   in one database instead of per-workload sqlite files.
