"""T3 sem_cache — semantic similarity cache.

Two backends share one ``cache_entry`` table in the local Postgres
instance (DSN from ``LOCAL_SPLITTER_DB_URL``):

- **embedding** (default, :class:`CacheStore`) — every request is
  embedded by the local embedding model; a near-duplicate (cosine
  similarity ≥ threshold via pgvector ``<=>``) is served directly.
  Needs a local model.
- **lexical** (:class:`LexicalCacheStore`) — pg_trgm trigram similarity
  over the raw cache text.  Needs no local model at all: this is what
  keeps the cache alive in the no-local mode (``configs/proxy/no-local.yaml``).
  Trigram similarity reads lower than cosine — roughly cosine 0.95 ≈
  trigram 0.65 — so pair this backend with a lower threshold.

The store instance selects the code path (see :func:`store_backend`);
on a miss the request proceeds to the cloud and the response is stored
for future hits, identically in both backends.

Fail-open: embedding or DB errors fall back to a cache miss
(ARCHITECTURE.md principle 2).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from local_splitter.chunked_embedding import embed_text_dynamic
from local_splitter.models import ChatClient, ModelBackendError, OllamaClient

from .types import StageEvent

_log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL = 86400  # 24 hours


# ---------------------------------------------------------------------------
# CacheStore — Postgres + pgvector semantic cache
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """A cached response retrieved from the store."""

    rowid: int
    distance: float
    similarity: float
    response: str
    model: str
    finish_reason: str
    created_at: float


class CacheStore:
    """Postgres + pgvector semantic cache.  Fail-open like lattice's Database.

    Each entry stores an embedding (for search), the response text,
    model name, finish_reason, and creation timestamp.  TTL eviction
    is explicit — call :meth:`evict_expired` periodically or at lookup.

    KNN is an exact scan (``ORDER BY embedding <=> %s LIMIT 1``) with no
    hnsw/ivfflat index, deliberate: the cache is at most a few thousand
    rows behind a 24 h TTL, so an index would only add write cost and
    recall tuning.  If it ever grows, the HNSW index belongs here, on
    the column the operator scans:

        CREATE INDEX idx_cache_entry_embedding
          ON cache_entry USING hnsw (embedding vector_cosine_ops);
    """

    def __init__(
        self,
        dsn: str,
        *,
        embed_dim: int = 768,
        namespace: str = "default",
    ) -> None:
        self._embed_dim = embed_dim
        self._namespace = namespace
        self._has_vec = False
        self._conn = psycopg.connect(dsn)
        # The sqlite backend persisted every write explicitly (commit()).
        # psycopg defaults to implicit transactions, so flip autocommit
        # on to restore per-statement persistence: every execute commits
        # on its own and callers never need a commit call.  Autocommit
        # only skips the *implicit* BEGIN — explicit BEGIN / COMMIT
        # statements still work if a statement group ever needs to be
        # atomic (lattice's Database does the same).
        self._conn.autocommit = True
        self._try_load_vec()
        self._apply_migrations()

    @property
    def has_vec(self) -> bool:
        """True iff pgvector is available and cache writes will persist."""
        return self._has_vec

    def _try_load_vec(self) -> None:
        """Provision + detect pgvector and register its adapters (fail-open).

        pgvector is a *trusted* extension, so the database owner can
        create it without superuser rights; we attempt ``CREATE
        EXTENSION`` first (lattice's detect-only version pre-creates by
        hand, but our CI service container ships the extension files
        without running CREATE EXTENSION, so the store provisions it as
        part of construction).  Everything here is best-effort: any
        failure leaves ``has_vec`` False and the cache inert — lookups
        miss, stores no-op — instead of failing the constructor, which
        is the pipeline's fail-open contract.

        ``register_vector`` maps ``list[float]`` → ``vector`` on write.
        Unlike lattice we skip its read-back loader overrides: the cache
        never projects the ``embedding`` column (the KNN SELECT lists
        id, distance, response, model, finish_reason, created_at), so
        the stock write adaptation is all we need — plus an explicit
        plain-``list`` dumper (pgvector ≥ 0.5 dropped it; see below).
        """
        try:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            _log.warning("could not create vector extension (%s); semantic cache disabled", exc)
            return
        row = self._conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            _log.warning("pgvector extension not installed; semantic cache disabled")
            return
        try:
            # Local import, lattice-style: keeps fail-open intact even
            # when the Python package is absent at runtime (a state in
            # which the extension check above can still pass).
            from pgvector.psycopg import register_vector
            from pgvector.psycopg.vector import VectorBinaryDumper, VectorDumper

            register_vector(self._conn)
            # pgvector >= 0.5 only registers dumpers for ``Vector`` and
            # numpy arrays, not plain ``list`` — a list[float] param
            # would reach psycopg's default list dumper and arrive as
            # ``double precision[]``, which the ``<=>`` operator cannot
            # resolve (array → vector has no *implicit* cast; INSERT
            # survives via the assignment cast, operator arguments
            # don't).  Re-register the list dumpers the way pgvector
            # < 0.5 did implicitly.
            info = psycopg.types.TypeInfo.fetch(self._conn, "vector")
            if info is not None:
                self._conn.adapters.register_dumper(
                    list, type("", (VectorDumper,), {"oid": info.oid})
                )
                self._conn.adapters.register_dumper(
                    list, type("", (VectorBinaryDumper,), {"oid": info.oid})
                )
            self._has_vec = True
        except Exception as exc:
            _log.warning("pgvector adapter registration failed (%s); semantic cache disabled", exc)

    def _apply_migrations(self) -> None:
        """Apply schema migrations not yet recorded in ``schema_migrations``.

        Mirrors lattice's runner: one version table, idempotent DDL, so
        re-opening an existing database is a no-op pass.  No explicit
        BEGIN/COMMIT around each migration — autocommit persists every
        statement on its own, and the DDL is idempotent, so a
        mid-migration failure simply leaves the version unrecorded and
        the next open retries to completion.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at BIGINT NOT NULL
            )
            """
        )
        applied = {row[0] for row in cur.execute("SELECT version FROM schema_migrations")}
        for version, migration in sorted(_MIGRATIONS.items()):
            if version in applied:
                continue
            _log.info("applying local_splitter schema migration v%d", version)
            migration(self._conn, **self._migration_kwargs())
            # Epoch millis, matching lattice's BIGINT applied_at (now_ms).
            cur.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                (version, int(time.time() * 1000)),
            )
        cur.close()

    def _migration_kwargs(self) -> dict[str, Any]:
        """Capabilities handed to every migration function.

        All migrations share one uniform signature so the runner stays
        backend-agnostic; ``LexicalCacheStore`` overrides this to add
        ``has_trgm`` (and to source ``has_vec`` from its own lighter,
        server-side-only check).
        """
        return {"embed_dim": self._embed_dim, "has_vec": self._has_vec, "has_trgm": False}

    def lookup(
        self,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl: int = DEFAULT_TTL,
    ) -> CacheEntry | None:
        """Find the nearest cached embedding.

        Returns a :class:`CacheEntry` if similarity ≥ ``threshold`` and
        the entry is within ``ttl`` seconds old.  Otherwise ``None``.

        The ``embedding IS NOT NULL`` filter mirrors the lexical store's
        ``cache_text IS NOT NULL``: both backends share one table, and a
        lexical-written row has a NULL embedding.  Without the filter,
        an all-lexical table yields a row whose ``<=>`` distance is NULL
        and the Python-side ``distance > max_distance`` raises TypeError.
        """
        if not self._has_vec:
            # Fail-open: without pgvector the table may not even exist,
            # so every lookup is a miss (lattice's set_embedding no-ops
            # the same way).
            return None
        if len(embedding) != self._embed_dim:
            raise ValueError(f"embedding length {len(embedding)} != configured {self._embed_dim}")
        max_distance = 1.0 - threshold
        cutoff = time.time() - ttl

        row = self._conn.execute(
            """
            SELECT id, embedding <=> %s AS distance,
                   response, model, finish_reason, created_at
              FROM cache_entry
             WHERE namespace = %s
               AND embedding IS NOT NULL
             ORDER BY embedding <=> %s
             LIMIT 1
            """,
            (embedding, self._namespace, embedding),
        ).fetchone()

        if row is None:
            return None

        rowid, distance, response, model, finish_reason, created_at = row
        if distance > max_distance:
            return None
        if created_at < cutoff:
            return None

        return CacheEntry(
            rowid=rowid,
            distance=distance,
            similarity=1.0 - distance,
            response=response,
            model=model,
            finish_reason=finish_reason,
            created_at=created_at,
        )

    def store(
        self,
        embedding: list[float],
        *,
        response: str,
        model: str,
        finish_reason: str,
    ) -> int:
        """Insert a new entry and return its rowid (0 when pgvector is absent)."""
        if not self._has_vec:
            return 0
        if len(embedding) != self._embed_dim:
            raise ValueError(f"embedding length {len(embedding)} != configured {self._embed_dim}")
        row = self._conn.execute(
            """
            INSERT INTO cache_entry (namespace, embedding, response, model,
                                     finish_reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (self._namespace, embedding, response, model, finish_reason, time.time()),
        ).fetchone()
        return int(row[0])

    def evict_expired(self, ttl: int = DEFAULT_TTL) -> int:
        """Delete entries older than ``ttl`` seconds.  Returns count deleted.

        The sqlite backend needed a SELECT-then-executemany-DELETE dance
        (vec0 aux-column deletes); Postgres does it in one DELETE, with
        ``rowcount`` giving the count.
        """
        cutoff = time.time() - ttl
        cur = self._conn.execute(
            "DELETE FROM cache_entry WHERE namespace = %s AND created_at < %s",
            (self._namespace, cutoff),
        )
        return cur.rowcount

    @property
    def size(self) -> int:
        """Number of entries in this store's namespace."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM cache_entry WHERE namespace = %s",
            (self._namespace,),
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover — best-effort close
            pass


class LexicalCacheStore(CacheStore):
    """pg_trgm trigram cache — T3 with no embeddings and no local model.

    Same ``cache_entry`` table and TTL/namespace semantics as the vector
    backend, but the lookup key is the raw cache text (the same
    normalization ``_cache_embed_text`` produces) compared with
    ``similarity(cache_text, %s)``.  This is what lets the splitter run
    with zero ollama dependency (the ``no-local`` preset) while keeping
    a working cache: trigram similarity reads lower than embedding
    cosine — roughly, cosine 0.95 ≈ trigram 0.65 — so the threshold
    must come down with this backend.

    Degradation mirrors ``has_vec`` fail-open: if ``pg_trgm`` cannot be
    provisioned, the store is inert (lookups miss, stores return 0).
    One honest wrinkle: the *table* itself is still created by v1,
    which needs the server-side ``vector`` type for the (unused,
    NULL) embedding column.  We therefore provision pgvector here too —
    but only server-side; unlike the base class we never import the
    pgvector Python package or register adapters, because this backend
    never binds a vector value.  A Postgres with pg_trgm but no vector
    files leaves the lexical store inert.

    ``similarity()`` returns a float4; ``distance``/``similarity`` on
    the returned :class:`CacheEntry` are derived from it (1 - sim), so
    downstream similarity logging reads like the vector backend's.
    """

    def __init__(self, dsn: str, *, namespace: str = "default") -> None:
        # Not calling CacheStore.__init__: no pgvector Python adapters,
        # and the trgm check must run before migrations so v2 knows
        # whether to add the column/index.  The 768 here only shapes the
        # (never-written, NULL) embedding column in v1's DDL — Postgres
        # rejects vector(0) — it is NOT an expected embedding size.
        self._embed_dim = 768
        self._namespace = namespace
        self._has_vec = False
        self._has_trgm = False
        self._conn = psycopg.connect(dsn)
        # Autocommit, same rationale as the base class: per-statement
        # persistence restores the sqlite-era write semantics.
        self._conn.autocommit = True
        self._has_vector_type = self._provision_vector_type()
        self._try_load_trgm()
        self._apply_migrations()
        self._ensure_trgm_index()

    def _migration_kwargs(self) -> dict[str, Any]:
        return {
            "embed_dim": self._embed_dim,
            # In this subclass the flag only gates v1's table creation;
            # no vector value is ever written through this store.
            "has_vec": self._has_vector_type,
            "has_trgm": self._has_trgm,
        }

    @property
    def has_trgm(self) -> bool:
        """True iff pg_trgm is available and cache writes will persist."""
        return self._has_trgm

    def _provision_vector_type(self) -> bool:
        """Server-side-only pgvector check so v1 can create the table."""
        try:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            _log.warning(
                "could not create vector extension (%s); lexical cache table unavailable", exc
            )
            return False
        row = self._conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            _log.warning("pgvector extension not installed; lexical cache table unavailable")
            return False
        return True

    def _try_load_trgm(self) -> None:
        """Provision pg_trgm (fail-open, mirrors ``_try_load_vec``)."""
        try:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        except Exception as exc:
            _log.warning("could not create pg_trgm extension (%s); lexical cache disabled", exc)
            return
        row = self._conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'").fetchone()
        if row is None:
            _log.warning("pg_trgm extension not installed; lexical cache disabled")
            return
        self._has_trgm = True

    def _ensure_trgm_index(self) -> None:
        """Create the GIN trgm index if v2 predated a working pg_trgm.

        v2 records its version even when pg_trgm was unavailable (v1
        parity), so a database first opened by the *vector* store gets
        the cache_text column (v2 adds it unconditionally) but no index.
        The DDL is idempotent, so simply re-run it on every lexical
        open — a catalog check once the index exists.
        """
        if not self._has_trgm or not self._has_vector_type:
            return
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_entry_trgm "
            "ON cache_entry USING gin (cache_text gin_trgm_ops)"
        )

    def lookup_text(
        self,
        cache_text: str,
        *,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl: int = DEFAULT_TTL,
        ns_prefix: str = "",
    ) -> CacheEntry | None:
        """Find the most trigram-similar cached text (exact-scan parity).

        Returns a :class:`CacheEntry` when similarity ≥ ``threshold``
        and the entry is within ``ttl`` seconds old; ``None`` otherwise
        (including the degraded no-pg_trgm state).  The TTL filter lives
        in the WHERE clause so an expired near-duplicate cannot outrank
        a fresh exact match.

        ``ns_prefix`` (the ``[ns:…]`` key prefix from
        :func:`_ns_prefix`) must match *exactly*, never fuzzily: the
        trigrams of a shared query body dominate the trigram set, so
        plain ``similarity("[ns:acme]\\nq", "[ns:other]\\nq")`` reads
        ~0.9 and tenant B would be served tenant A's cached answer at
        any sane threshold.  The empty-prefix case excludes namespaced
        rows symmetrically, so a later-added namespace can't be bypassed
        by old keys either way.  (Rows whose body *literally* starts
        with ``[ns:`` become unreachable without a namespace — a safe
        false miss, not a wrong hit.)
        """
        if not self._has_trgm:
            return None
        cutoff = time.time() - ttl
        sql = """
            SELECT id, similarity(cache_text, %s) AS sim,
                   response, model, finish_reason, created_at
              FROM cache_entry
             WHERE namespace = %s
               AND cache_text IS NOT NULL
               AND created_at >= %s
               AND similarity(cache_text, %s) >= %s
        """
        args: list[Any] = [cache_text, self._namespace, cutoff, cache_text, threshold]
        if ns_prefix:
            # left()/equality (not LIKE): ns values are user-supplied and
            # may contain % or _ pattern characters.
            sql += " AND left(cache_text, %s) = %s"
            args.extend((len(ns_prefix), ns_prefix))
        else:
            sql += " AND left(cache_text, 4) <> '[ns:'"
        sql += " ORDER BY sim DESC LIMIT 1"
        row = self._conn.execute(sql, args).fetchone()
        if row is None:
            return None

        rowid, sim, response, model, finish_reason, created_at = row
        similarity = float(sim)
        return CacheEntry(
            rowid=rowid,
            distance=1.0 - similarity,
            similarity=similarity,
            response=response,
            model=model,
            finish_reason=finish_reason,
            created_at=created_at,
        )

    def store_text(
        self,
        *,
        cache_text: str,
        response: str,
        model: str,
        finish_reason: str,
    ) -> int:
        """Insert a text-keyed entry; returns its rowid (0 when degraded)."""
        if not self._has_trgm:
            return 0
        row = self._conn.execute(
            """
            INSERT INTO cache_entry (namespace, cache_text, response, model,
                                     finish_reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (self._namespace, cache_text, response, model, finish_reason, time.time()),
        ).fetchone()
        return int(row[0])


def store_backend(store: "CacheStore | None") -> str:
    """Which code path a store instance drives: ``"lexical"`` or ``"embedding"``.

    The runtime source of truth for every backend-aware gate (pre-cloud
    activation, miss-path store, the MCP cache_lookup tool): the config
    knob only chooses the class ``_build_pipeline`` constructs, so a
    hand-wired Pipeline is always consistent with whatever store it was
    actually given.
    """
    return "lexical" if isinstance(store, LexicalCacheStore) else "embedding"


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
#
# Ported from the sqlite-era vec0 virtual table to a plain Postgres
# table: BIGSERIAL id replaces rowid, created_at TEXT (a workaround for
# a vec0 REAL aux-column bug) becomes DOUBLE PRECISION directly, and the
# sqlite-vec float[] embedding becomes a pgvector ``vector`` column.
# All DDL is idempotent (CREATE TABLE/INDEX IF NOT EXISTS), so
# re-opening an existing database is a no-op pass over applied versions.


def _migration_v1(
    conn: psycopg.Connection,
    *,
    embed_dim: int,
    has_vec: bool,
    has_trgm: bool = False,  # uniform migration signature; unused here
) -> None:
    """v1 — ``cache_entry`` (the whole schema; skipped without pgvector).

    The embedding column needs the ``vector`` type, so the table exists
    only when pgvector was detected.  The version is still recorded when
    skipped (lattice parity) — the tradeoff: if the extension is
    installed *after* v1 was recorded, the cache stays inert until a
    future migration recreates the table (lattice's v4 did exactly that
    for its embedding table).
    """
    if not has_vec:
        return
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS cache_entry (
            id            BIGSERIAL PRIMARY KEY,
            namespace     TEXT NOT NULL DEFAULT 'default',
            embedding     vector({embed_dim}),
            response      TEXT NOT NULL,
            model         TEXT NOT NULL,
            finish_reason TEXT NOT NULL,
            created_at    DOUBLE PRECISION NOT NULL
        )
        """
    )
    # Plain btree indexes for the two hot filters (namespace scoping,
    # TTL eviction).  The embedding column deliberately gets none — see
    # the CacheStore docstring on exact KNN scans.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_entry_namespace ON cache_entry(namespace)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_entry_created ON cache_entry(created_at)")
    cur.close()


def _migration_v2(
    conn: psycopg.Connection,
    *,
    embed_dim: int,  # uniform migration signature; unused here
    has_vec: bool,
    has_trgm: bool = False,
) -> None:
    """v2 — lexical backend: ``cache_text`` column (+ GIN trgm index).

    The vector backend never persisted the cache text (only its
    embedding), so the lexical backend — whose lookup key *is* the text
    — needs a column.  It is nullable and left NULL by vector-backend
    inserts, which is what lets both backends share one table: a
    ``vector``-written row has ``cache_text IS NULL`` (and never matches
    a trigram lookup), a lexical row has ``embedding IS NULL`` (v1
    created the column without NOT NULL, so no ALTER was needed).

    The column is added whenever the table exists, *regardless* of
    pg_trgm — the schema should not depend on which backend happened to
    open the database first.  Only the index needs trgm; when it's
    missing here, :meth:`LexicalCacheStore._ensure_trgm_index` creates
    it on a later lexical open (the version is recorded either way, v1
    parity — the index is a pure optimization, so that gap is safe).
    """
    if not has_vec:
        return  # no table (v1 skipped) — nothing to alter
    cur = conn.cursor()
    cur.execute("ALTER TABLE cache_entry ADD COLUMN IF NOT EXISTS cache_text TEXT")
    if has_trgm:
        # gin_trgm_ops lets the planner serve ``similarity(a, b) >= const``
        # filters from the index (with a recheck); lookups remain exact,
        # not approximate — same deliberate no-hnsw stance as the vector
        # side.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_entry_trgm "
            "ON cache_entry USING gin (cache_text gin_trgm_ops)"
        )
    cur.close()


_MIGRATIONS: dict[int, Callable[..., None]] = {1: _migration_v1, 2: _migration_v2}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def open_cache_store(dsn: str, *, embed_dim: int = 768, namespace: str = "default") -> CacheStore:
    """Convenience opener used by cli.py, evals, and tests (open_database parity)."""
    return CacheStore(dsn, embed_dim=embed_dim, namespace=namespace)


# ---------------------------------------------------------------------------
# Pipeline-facing functions
# ---------------------------------------------------------------------------

# Match llm-redactor placeholders ``⟨KIND_n·tag⟩`` and drop the per-request
# tag.  ``⟨EMAIL_1·a1b2c3d4⟩`` → ``⟨EMAIL_1⟩``.  This lets two redacted
# requests that differ only in concrete PII collapse to one cache entry.
_REDACTOR_TAG_RE = re.compile(r"(⟨[A-Z_0-9]+)·[^⟩]+(⟩)")


def _normalize_redactor_placeholders(text: str) -> str:
    return _REDACTOR_TAG_RE.sub(r"\1\2", text)


def _stringify_content(content: Any) -> str:
    """Reduce OpenAI string content or Anthropic block-list content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    # Anthropic tool_result content can be a string.
                    parts.append(part["content"])
        return "\n".join(parts)
    return ""


def _extract_cache_text(messages: list[dict[str, Any]]) -> str:
    """Build the text that will be embedded for cache lookup.

    Uses the last user message — the most query-specific part.  Handles
    both OpenAI string content and Anthropic block-list content.
    Redactor per-request tags are normalized so PII-equivalent queries
    collapse to one cache key.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _stringify_content(msg.get("content", ""))
            return _normalize_redactor_placeholders(text)
    return ""


def cache_embed_source(
    messages: list[dict[str, str]],
    params: dict[str, Any] | None,
    meta: dict[str, Any] | None,
) -> str:
    """Text embedded for T3 (matches :func:`lookup`). Exposed for store-time checks."""
    return _cache_embed_text(messages, params or {}, meta)


def _ns_prefix(params: dict[str, Any], meta: dict[str, Any] | None) -> str:
    """The ``[ns:…]`` key prefix ``_cache_embed_text`` prepends (empty if none).

    Factored out so the lexical backend can require the prefix to match
    *exactly* — see :meth:`LexicalCacheStore.lookup_text` for why fuzzy
    matching it would cross tenant boundaries.
    """
    key = params.get("cache_namespace_from_meta")
    if not key:
        return ""
    ns = (meta or {}).get(str(key), "") or ""
    return f"[ns:{ns}]\n" if ns else ""


def _cache_embed_text(
    messages: list[dict[str, str]],
    params: dict[str, Any],
    meta: dict[str, Any] | None,
) -> str:
    """User text optionally prefixed with a namespace from ``meta``."""
    base = _extract_cache_text(messages)
    prefix = _ns_prefix(params, meta)
    return f"{prefix}{base}" if prefix else base


def _never_cache_patterns(params: dict[str, Any]) -> list[re.Pattern[str]]:
    raw = params.get("never_cache_regex") or []
    if not isinstance(raw, list):
        return []
    out: list[re.Pattern[str]] = []
    for pat in raw:
        try:
            out.append(re.compile(str(pat), re.DOTALL))
        except re.error as exc:
            _log.warning("invalid never_cache_regex %r: %s", pat, exc)
    return out


def _should_skip_cache_for_privacy(
    *,
    params: dict[str, Any],
    meta: dict[str, Any] | None,
    cache_text: str,
    response_text: str | None,
) -> tuple[bool, str | None]:
    """Return (skip, reason) when lookup/store must be skipped."""
    skip_tools = params.get("skip_cache_for_tools") or []
    if isinstance(skip_tools, list) and skip_tools:
        tn = (meta or {}).get("tool_name") or (meta or {}).get("tool")
        if tn is not None and str(tn) in {str(x) for x in skip_tools}:
            return True, "skip_cache_for_tools"

    for rx in _never_cache_patterns(params):
        if rx.search(cache_text):
            return True, "never_cache_regex"
        if response_text is not None and rx.search(response_text):
            return True, "never_cache_regex"

    return False, None


@dataclass(slots=True)
class CacheLookupResult:
    """Outcome of a T3 cache lookup."""

    hit: bool
    entry: CacheEntry | None
    embedding: list[float] | None  # kept so we can store on miss
    events: list[StageEvent]


def _lookup_lexical(
    store: LexicalCacheStore,
    cache_text: str,
    *,
    threshold: float,
    ttl: int,
    ns_prefix: str = "",
) -> CacheLookupResult:
    """Lexical-backend lookup: trigram compare, no embed call anywhere."""
    t0 = time.perf_counter()
    try:
        entry = store.lookup_text(cache_text, threshold=threshold, ttl=ttl, ns_prefix=ns_prefix)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T3 cache lookup failed, treating as miss: %s", exc)
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="ERROR",
                    ms=elapsed,
                    detail={"error": str(exc)},
                )
            ],
        )

    total_ms = (time.perf_counter() - t0) * 1000
    if entry is not None:
        return CacheLookupResult(
            hit=True,
            entry=entry,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="HIT",
                    ms=total_ms,
                    detail={"similarity": round(entry.similarity, 4)},
                )
            ],
        )

    return CacheLookupResult(
        hit=False,
        entry=None,
        embedding=None,
        events=[StageEvent(stage="t3_cache_lookup", decision="MISS", ms=total_ms)],
    )


async def lookup(
    messages: list[dict[str, str]],
    *,
    local: ChatClient | None,
    store: CacheStore,
    params: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> CacheLookupResult:
    """Search the cache for the request.

    Embedding backend: embed the request (via ``local``) and run the
    vector KNN.  Lexical store: pass the raw cache text straight to the
    store — no embed call, no local client needed.  On embedding or DB
    failure, returns a miss (fail-open).
    """
    p = params or {}
    threshold = float(p.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    ttl = int(p.get("ttl", DEFAULT_TTL))

    cache_text = _cache_embed_text(messages, p, meta)
    if not cache_text:
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="SKIP",
                    ms=0.0,
                    detail={"reason": "no user text"},
                )
            ],
        )

    skip, reason = _should_skip_cache_for_privacy(
        params=p,
        meta=meta,
        cache_text=cache_text,
        response_text=None,
    )
    if skip:
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup", decision="SKIP", ms=0.0, detail={"reason": reason}
                )
            ],
        )

    # Backend dispatch by store instance (see store_backend).  The
    # lexical path shares only the cache_text normalization with the
    # embedding path; everything below the privacy gate differs.  The
    # namespace prefix rides along so the trigram filter can require it
    # exactly (see LexicalCacheStore.lookup_text) — for the embedding
    # backend it stays inside the embedded text.
    if isinstance(store, LexicalCacheStore):
        return _lookup_lexical(
            store,
            cache_text,
            threshold=threshold,
            ttl=ttl,
            ns_prefix=_ns_prefix(p, meta),
        )

    # --- embedding backend ---
    if local is None:
        # Unreachable through the pipeline (pre_cloud gates activation on
        # has_local for the embedding backend); kept as a fail-open SKIP
        # for direct callers so the old AttributeError can't resurface.
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="SKIP",
                    ms=0.0,
                    detail={"reason": "no local embedder"},
                )
            ],
        )

    # Prompts longer than the embedding model's context are no longer
    # skipped: chunked_embedding splits them into overlapping windows and
    # pools the vectors in one batched call, so T3 can cache arbitrarily
    # large agent payloads (e.g. a Palisade-transformed request).  The
    # old num_ctx-derived cap remains available as an explicit operator
    # escape hatch — set params["embed_max_chars"] to skip embedding
    # outright instead (the pre-chunking behavior).  Fail-open either
    # way: skip rather than crash the pipeline.  (Lexical mode has no
    # embedding cost, so the cap doesn't apply there.)
    embed_max = p.get("embed_max_chars")
    if embed_max is not None and len(cache_text) > int(embed_max):
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="SKIP",
                    ms=0.0,
                    detail={
                        "reason": "cache_text exceeds embed_max_chars",
                        "length": len(cache_text),
                        "max_chars": int(embed_max),
                    },
                )
            ],
        )

    t0 = time.perf_counter()
    try:
        embedding = await embed_text_dynamic(
            cache_text,
            model=getattr(local, "embed_model", None),
            embed_many=local.embed,
            # Only an Ollama backend can answer /api/show; anything else
            # (OpenAI-compatible, fake) uses the fallback context without
            # being probed.
            endpoint=local.endpoint if isinstance(local, OllamaClient) else None,
        )
    except (ModelBackendError, Exception) as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T3 embed failed, treating as cache miss: %s", exc)
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=None,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="ERROR",
                    ms=elapsed,
                    detail={"error": str(exc)},
                )
            ],
        )

    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        entry = store.lookup(embedding, threshold=threshold, ttl=ttl)
    except Exception as exc:
        elapsed = embed_ms + (time.perf_counter() - t1) * 1000
        _log.warning("T3 cache lookup failed, treating as miss: %s", exc)
        return CacheLookupResult(
            hit=False,
            entry=None,
            embedding=embedding,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="ERROR",
                    ms=elapsed,
                    detail={"error": str(exc)},
                )
            ],
        )

    total_ms = embed_ms + (time.perf_counter() - t1) * 1000

    if entry is not None:
        return CacheLookupResult(
            hit=True,
            entry=entry,
            embedding=embedding,
            events=[
                StageEvent(
                    stage="t3_cache_lookup",
                    decision="HIT",
                    ms=total_ms,
                    detail={"similarity": round(entry.similarity, 4)},
                )
            ],
        )

    return CacheLookupResult(
        hit=False,
        entry=None,
        embedding=embedding,
        events=[StageEvent(stage="t3_cache_lookup", decision="MISS", ms=total_ms)],
    )


def store_response(
    embedding: list[float] | None,
    *,
    response: str,
    model: str,
    finish_reason: str,
    cache_store: CacheStore,
    params: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    cache_text: str | None = None,
) -> StageEvent:
    """Store a cloud response in the cache after a miss.

    The store instance picks the key: a lexical store persists
    ``cache_text`` (``embedding`` is ignored/None there), the vector
    store persists ``embedding``.  Returns a stage event for the trace.
    Errors are swallowed (fail-open).
    """
    p = params or {}
    ct = cache_text or ""
    skip, reason = _should_skip_cache_for_privacy(
        params=p,
        meta=meta,
        cache_text=ct,
        response_text=response,
    )
    if skip:
        return StageEvent(
            stage="t3_cache_store",
            decision="SKIP",
            ms=0.0,
            detail={"reason": reason},
        )

    if isinstance(cache_store, LexicalCacheStore):
        if not ct:
            # No user text to key on — the pipeline gate normally filters
            # this out (empty cache_text ⇒ no lookup either); direct
            # callers get a visible SKIP instead of a NULL-keyed row.
            return StageEvent(
                stage="t3_cache_store",
                decision="SKIP",
                ms=0.0,
                detail={"reason": "no cache text"},
            )
        t0 = time.perf_counter()
        try:
            cache_store.store_text(
                cache_text=ct, response=response, model=model, finish_reason=finish_reason
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            _log.warning("T3 cache store failed: %s", exc)
            return StageEvent(
                stage="t3_cache_store",
                decision="ERROR",
                ms=elapsed,
                detail={"error": str(exc)},
            )
        elapsed = (time.perf_counter() - t0) * 1000
        return StageEvent(stage="t3_cache_store", decision="STORED", ms=elapsed)

    if embedding is None:
        # Unreachable through the pipeline (its store gate requires a
        # non-None embedding for the vector backend); direct callers get
        # a visible SKIP rather than a TypeError from psycopg.
        return StageEvent(
            stage="t3_cache_store",
            decision="SKIP",
            ms=0.0,
            detail={"reason": "no embedding"},
        )

    t0 = time.perf_counter()
    try:
        cache_store.store(embedding, response=response, model=model, finish_reason=finish_reason)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T3 cache store failed: %s", exc)
        return StageEvent(
            stage="t3_cache_store",
            decision="ERROR",
            ms=elapsed,
            detail={"error": str(exc)},
        )

    elapsed = (time.perf_counter() - t0) * 1000
    return StageEvent(stage="t3_cache_store", decision="STORED", ms=elapsed)


__all__ = [
    "CacheEntry",
    "CacheLookupResult",
    "CacheStore",
    "LexicalCacheStore",
    "cache_embed_source",
    "lookup",
    "store_backend",
    "store_response",
]
