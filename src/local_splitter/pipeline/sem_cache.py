"""T3 sem_cache — semantic similarity cache.

Every request is embedded by the local embedding model.  If a
near-duplicate exists in the cache (cosine similarity ≥ threshold), the
cached response is served directly.  On a miss the request proceeds to
the cloud and the response is stored for future hits.

Storage: SQLite + ``sqlite-vec`` for vector search.  The cache is a
single file (``cache.sqlite``) in the state directory.

Fail-open: embedding or DB errors fall back to a cache miss
(ARCHITECTURE.md principle 2).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from local_splitter.models import ChatClient, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL = 86400  # 24 hours


# ---------------------------------------------------------------------------
# Serialization helpers for sqlite-vec float[] columns
# ---------------------------------------------------------------------------

def _serialize(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# CacheStore — thin wrapper around sqlite + sqlite-vec
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
    """SQLite-backed vector cache for semantic deduplication.

    Each entry stores an embedding (for search), the response text,
    model name, finish_reason, and creation timestamp.  TTL eviction
    is explicit — call :meth:`evict_expired` periodically or at lookup.
    """

    def __init__(self, db_path: Path | str, *, embed_dim: int = 768) -> None:
        self._embed_dim = embed_dim
        self._db = sqlite3.connect(str(db_path))
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self) -> None:
        # Note: +created_at uses ``text`` not ``real`` because sqlite-vec
        # has a bug with ``real`` auxiliary columns at higher dimensions.
        # We store the epoch float as a string and cast in Python.
        self._db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_cache USING vec0(
                embedding float[{self._embed_dim}] distance_metric=cosine,
                +response text,
                +model text,
                +finish_reason text,
                +created_at text
            )
        """)
        self._db.commit()

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
        """
        max_distance = 1.0 - threshold
        now = time.time()
        cutoff = now - ttl

        rows = self._db.execute(
            """
            SELECT rowid, distance, response, model, finish_reason, created_at
            FROM vec_cache
            WHERE embedding MATCH ?
              AND k = 1
            """,
            (_serialize(embedding),),
        ).fetchall()

        if not rows:
            return None

        rowid, distance, response, model, finish_reason, created_at_str = rows[0]
        created_at = float(created_at_str)
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
        """Insert a new entry and return its rowid."""
        cur = self._db.execute(
            """
            INSERT INTO vec_cache(embedding, response, model, finish_reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_serialize(embedding), response, model, finish_reason, str(time.time())),
        )
        self._db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def evict_expired(self, ttl: int = DEFAULT_TTL) -> int:
        """Delete entries older than ``ttl`` seconds.  Returns count deleted."""
        cutoff = str(time.time() - ttl)
        # vec0 tables support DELETE by rowid.  Identify expired rows first.
        rows = self._db.execute(
            "SELECT rowid FROM vec_cache WHERE created_at < ?", (cutoff,)
        ).fetchall()
        if not rows:
            return 0
        rowids = [r[0] for r in rows]
        self._db.executemany("DELETE FROM vec_cache WHERE rowid = ?", [(r,) for r in rowids])
        self._db.commit()
        return len(rowids)

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return self._db.execute("SELECT count(*) FROM vec_cache").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# ---------------------------------------------------------------------------
# Pipeline-facing functions
# ---------------------------------------------------------------------------

def _extract_cache_text(messages: list[dict[str, str]]) -> str:
    """Build the text that will be embedded for cache lookup.

    Uses the last user message — the most query-specific part.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


@dataclass(slots=True)
class CacheLookupResult:
    """Outcome of a T3 cache lookup."""

    hit: bool
    entry: CacheEntry | None
    embedding: list[float] | None  # kept so we can store on miss
    events: list[StageEvent]


async def lookup(
    messages: list[dict[str, str]],
    *,
    local: ChatClient,
    store: CacheStore,
    params: dict[str, Any] | None = None,
) -> CacheLookupResult:
    """Embed the request and search the cache.

    On embedding failure, returns a miss (fail-open).
    """
    p = params or {}
    threshold = float(p.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    ttl = int(p.get("ttl", DEFAULT_TTL))

    cache_text = _extract_cache_text(messages)
    if not cache_text:
        return CacheLookupResult(hit=False, entry=None, embedding=None, events=[
            StageEvent(stage="t3_cache_lookup", decision="SKIP", ms=0.0,
                       detail={"reason": "no user text"})
        ])

    t0 = time.perf_counter()
    try:
        embeddings = await local.embed([cache_text])
        embedding = embeddings[0]
    except (ModelBackendError, Exception) as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T3 embed failed, treating as cache miss: %s", exc)
        return CacheLookupResult(hit=False, entry=None, embedding=None, events=[
            StageEvent(stage="t3_cache_lookup", decision="ERROR", ms=elapsed,
                       detail={"error": str(exc)})
        ])

    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        entry = store.lookup(embedding, threshold=threshold, ttl=ttl)
    except Exception as exc:
        elapsed = embed_ms + (time.perf_counter() - t1) * 1000
        _log.warning("T3 cache lookup failed, treating as miss: %s", exc)
        return CacheLookupResult(hit=False, entry=None, embedding=embedding, events=[
            StageEvent(stage="t3_cache_lookup", decision="ERROR",
                       ms=elapsed, detail={"error": str(exc)})
        ])

    total_ms = embed_ms + (time.perf_counter() - t1) * 1000

    if entry is not None:
        return CacheLookupResult(hit=True, entry=entry, embedding=embedding, events=[
            StageEvent(stage="t3_cache_lookup", decision="HIT", ms=total_ms,
                       detail={"similarity": round(entry.similarity, 4)})
        ])

    return CacheLookupResult(hit=False, entry=None, embedding=embedding, events=[
        StageEvent(stage="t3_cache_lookup", decision="MISS", ms=total_ms)
    ])


def store_response(
    embedding: list[float],
    *,
    response: str,
    model: str,
    finish_reason: str,
    cache_store: CacheStore,
) -> StageEvent:
    """Store a cloud response in the cache after a miss.

    Returns a stage event for the trace.  Errors are swallowed (fail-open).
    """
    t0 = time.perf_counter()
    try:
        cache_store.store(
            embedding, response=response, model=model, finish_reason=finish_reason
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T3 cache store failed: %s", exc)
        return StageEvent(
            stage="t3_cache_store", decision="ERROR", ms=elapsed,
            detail={"error": str(exc)},
        )

    elapsed = (time.perf_counter() - t0) * 1000
    return StageEvent(stage="t3_cache_store", decision="STORED", ms=elapsed)


__all__ = [
    "CacheEntry",
    "CacheLookupResult",
    "CacheStore",
    "lookup",
    "store_response",
]
