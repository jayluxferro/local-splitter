"""T3 semantic-cache backend comparison: pg_trgm lexical vs pgvector embedding.

Runs the *same* query set through both cache backends against the
splitter's own test database (LOCAL_SPLITTER_TEST_DB_URL, same default
as tests/conftest.py) and prints hit rate / precision / latency per
query class, plus the pass bars:

  1. lexical exact-repeat hit rate >= embedding exact-repeat hit rate
     (the no-local mode must not give up cache hits on repeats)
  2. lexical makes zero local-model calls (it never embeds)

Query set: 10 base queries; each yields an exact repeat, a handcrafted
near-paraphrase, and an unrelated probe (30 lookups per backend).  The
seed phase stores one response per base query through the same
miss→store path the pipeline uses (sem_cache.lookup + store_response),
so the numbers reflect serving behavior, not raw SQL.

Embedding backend needs a live ollama with nomic-embed-text.  Probe
order: $LOCAL_SPLITTER_BENCH_OLLAMA_ENDPOINT, then http://127.0.0.1:11435
(ollama's temporary home on this machine — never hardcoded in src/),
then the stock 11434.  Without any of them the bench falls back to a
synthetic character-trigram embedder and says so: exact repeats still
exercise the machinery, but paraphrase numbers are meaningless there.

Usage:  uv run python tools/bench_sem_cache.py
Exit:   0 if the pass bars hold (or DB unavailable → SKIP), 1 otherwise.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

TEST_DB_URL = os.environ.get(
    "LOCAL_SPLITTER_TEST_DB_URL",
    "postgresql://local_splitter@localhost:5432/local_splitter_test",
)

# Prod-ish thresholds per backend (config.yaml uses 0.95 for embedding;
# the no-local preset uses 0.65 for trigram — cosine 0.95 ≈ trigram 0.65).
EMBED_THRESHOLD = 0.92
LEXICAL_THRESHOLD = 0.65
TTL = 86400

OLLAMA_PROBE_ENDPOINTS = [
    os.environ.get("LOCAL_SPLITTER_BENCH_OLLAMA_ENDPOINT") or "",
    "http://127.0.0.1:11435",  # ollama's temporary port on this machine
    "http://127.0.0.1:11434",  # stock
]
OLLAMA_EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Query set (fixed, inline)
# ---------------------------------------------------------------------------

BASE_QUERIES: list[str] = [
    "Fix the off-by-one error in the binary search implementation",
    "Explain how Python's garbage collector handles reference cycles",
    "Write a SQL query to find the top 5 customers by total order value",
    "What is the difference between a mutex and a semaphore?",
    "Summarize the tradeoffs between REST and GraphQL APIs",
    "Refactor this function to use list comprehensions instead of loops",
    "How do I configure retry logic with exponential backoff in httpx?",
    "Explain the CAP theorem and give an example of each consistency level",
    "Why does my Docker build fail with 'no space left on device'?",
    "Convert this pandas groupby aggregation to polars",
]

PARAPHRASES: list[str] = [
    "Fix an off-by-one bug in this binary search",
    "How does Python's GC deal with circular references?",
    "Write SQL for the five customers with the highest order totals",
    "Mutex vs semaphore — what's the difference?",
    "What are the pros and cons of REST compared to GraphQL?",
    "Rewrite this function with list comprehensions rather than for loops",
    "How can I add exponential backoff retries in httpx?",
    "Explain CAP theorem with an example per consistency level",
    "Docker build says 'no space left on device' — why?",
    "Translate this pandas groupby aggregation into polars",
]

UNRELATED: list[str] = [
    "Recommend a good database for time-series sensor data",
    "What's the tallest building in the world?",
    "Explain the rules of cricket in simple terms",
    "How do I make sourdough bread at home?",
    "Who wrote the novel One Hundred Years of Solitude?",
    "What is the capital of Australia?",
    "Give me a workout plan for a beginner runner",
    "Explain photosynthesis to a ten-year-old",
    "What causes the northern lights?",
    "Suggest a name for a coffee shop that also sells books",
]


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class SyntheticEmbedder:
    """Deterministic char-trigram hashing embedder (fallback, no model).

    Exact repeats embed identically, so the machinery is exercised;
    paraphrase overlap is weak, so those numbers are disclosed as
    meaningless when this embedder is in use.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim
        self.embed_texts_seen = 0
        self.embed_model = "synthetic-trigram"

    async def embed(self, texts: Any, *, model: str | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            norm = text.lower()
            tris = [norm[i : i + 3] for i in range(max(0, len(norm) - 2))] or [norm]
            for t in tris:
                h = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "big")
                vec[h % self._dim] += 1.0
            magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / magnitude for v in vec])
        self.embed_texts_seen += len(texts)
        return out


async def _resolve_embedder() -> tuple[Any, str]:
    """Live ollama (counting wrapper) if reachable, else the synthetic fallback."""
    import httpx

    from local_splitter.models import OllamaClient

    class CountingOllamaClient(OllamaClient):
        """Counts embedded texts so the bench can prove the embedding
        backend really calls a local model (and lexical never does)."""

        embed_texts_seen = 0

        async def embed(self, texts: Any, *, model: str | None = None) -> list[list[float]]:
            self.embed_texts_seen += len(texts)
            return await super().embed(texts, model=model)

    for endpoint in OLLAMA_PROBE_ENDPOINTS:
        if not endpoint:
            continue
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{endpoint}/api/tags")
                models = [m.get("name", "") for m in resp.json().get("models", [])]
            if any(m.startswith(OLLAMA_EMBED_MODEL) for m in models):
                counter = CountingOllamaClient(
                    chat_model=OLLAMA_EMBED_MODEL,
                    embed_model=OLLAMA_EMBED_MODEL,
                    endpoint=endpoint,
                )
                return counter, f"live ollama @ {endpoint} ({OLLAMA_EMBED_MODEL})"
        except Exception:
            continue
    return (
        SyntheticEmbedder(),
        "SYNTHETIC trigram-hash embedder (no live ollama with "
        f"{OLLAMA_EMBED_MODEL} found — paraphrase numbers are not "
        "meaningful; exact repeats still exercise the machinery)",
    )


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


@dataclass
class ClassStats:
    lookups: int = 0
    hits: int = 0
    latency_ms: list[float] = field(default_factory=list)


@dataclass
class BackendRun:
    name: str
    classes: dict[str, ClassStats] = field(default_factory=dict)
    embed_calls: int = 0

    def record(self, cls: str, hit: bool, ms: float) -> None:
        st = self.classes.setdefault(cls, ClassStats())
        st.lookups += 1
        st.hits += int(hit)
        st.latency_ms.append(ms)

    def hit_rate(self, cls: str) -> float:
        st = self.classes.get(cls)
        return (st.hits / st.lookups) if st and st.lookups else 0.0


def _delete_namespace(namespace: str) -> None:
    """Bench hygiene: rows are namespaced, so cleanup is a scoped DELETE."""
    conn = psycopg.connect(TEST_DB_URL, connect_timeout=3)
    try:
        conn.autocommit = True
        conn.execute("DELETE FROM cache_entry WHERE namespace = %s", (namespace,))
    finally:
        conn.close()


async def _run_backend(
    name: str,
    *,
    namespace: str,
    local: Any,
    threshold: float,
) -> BackendRun:
    """Seed one response per base query, then probe all 30 lookups."""
    from local_splitter.pipeline.sem_cache import (
        CacheStore,
        LexicalCacheStore,
        cache_embed_source,
        lookup,
        store_response,
    )

    lexical = name == "lexical"
    store: CacheStore
    if lexical:
        store = LexicalCacheStore(TEST_DB_URL, namespace=namespace)
    else:
        store = CacheStore(TEST_DB_URL, embed_dim=768, namespace=namespace)
    run = BackendRun(name=name)
    params = {"similarity_threshold": threshold, "ttl": TTL}

    try:
        # Seed: miss → store, exactly like the pipeline's miss path.
        for base in BASE_QUERIES:
            msgs = _messages(base)
            meta: dict[str, str] = {}
            result = await lookup(msgs, local=local, store=store, params=params, meta=meta)
            assert not result.hit, f"fresh namespace lookup hit: {base!r}"
            store_response(
                result.embedding,
                response=f"answer to: {base}",
                model="bench-cloud",
                finish_reason="stop",
                cache_store=store,
                params=params,
                meta=meta,
                cache_text=cache_embed_source(msgs, params, meta),
            )

        # Probe: serving behavior only — hits are not re-stored.
        probes: list[tuple[str, str]] = []
        probes += [("exact_repeat", q) for q in BASE_QUERIES]
        probes += [("paraphrase", q) for q in PARAPHRASES]
        probes += [("unrelated", q) for q in UNRELATED]

        for cls, text in probes:
            result = await lookup(_messages(text), local=local, store=store, params=params, meta={})
            ms = result.events[0].ms if result.events else 0.0
            run.record(cls, result.hit, ms)

        run.embed_calls = getattr(local, "embed_texts_seen", 0)
    finally:
        store.close()
        _delete_namespace(namespace)
    return run


def _print_table(runs: list[BackendRun], embedder_note: str) -> bool:
    """Print the results table; True iff every pass bar holds."""
    print(f"\nembedder: {embedder_note}")
    header = f"{'backend':<10} {'class':<13} {'hits':>7} {'hit_rate':>9} {'mean_ms':>8}"
    print(header)
    print("-" * len(header))

    total_hits = 0
    true_positives = 0
    false_positives = 0
    for run in runs:
        for cls in ("exact_repeat", "paraphrase", "unrelated"):
            st = run.classes.get(cls)
            if st is None or st.lookups == 0:
                continue
            hit_rate = st.hits / st.lookups
            total_hits += st.hits
            if cls == "unrelated":
                false_positives += st.hits  # a hit here is a false positive
            else:
                true_positives += st.hits
            print(
                f"{run.name:<10} {cls:<13} {st.hits:>3}/{st.lookups:<3} "
                f"{hit_rate:>9.2f} {sum(st.latency_ms) / len(st.latency_ms):>8.2f}"
            )
        denom = true_positives + false_positives
        prec = (true_positives / denom) if denom else float("nan")
        print(f"{run.name:<10} {'ALL':<13} {total_hits:>3}/30 {'':>9}     precision={prec:.2f}")
        total_hits = true_positives = false_positives = 0  # reset per backend

    lex = next((r for r in runs if r.name == "lexical"), None)
    emb = next((r for r in runs if r.name == "embedding"), None)
    if lex is None or emb is None:
        print("\nFAIL: both backends must run for the comparison")
        return False

    ok = True
    lex_exact = lex.hit_rate("exact_repeat")
    emb_exact = emb.hit_rate("exact_repeat")
    if lex_exact >= emb_exact:
        print(f"\nPASS: lexical exact-repeat hit rate {lex_exact:.2f} >= embedding {emb_exact:.2f}")
    else:
        print(f"\nFAIL: lexical exact-repeat hit rate {lex_exact:.2f} < embedding {emb_exact:.2f}")
        ok = False

    if lex.embed_calls == 0:
        print(
            "PASS: lexical made 0 local-model embed calls "
            f"(embedding backend: {emb.embed_calls} texts embedded)"
        )
    else:
        print(f"FAIL: lexical made {lex.embed_calls} embed calls — decoupling broken")
        ok = False
    return ok


async def main_async() -> int:
    # DB gate: report skipped, exit 0 (mirrors the conftest TEST_DB_URL).
    try:
        conn = psycopg.connect(TEST_DB_URL, connect_timeout=3)
        conn.close()
    except Exception as exc:
        print(f"SKIP: test database unavailable ({exc})")
        print("Set LOCAL_SPLITTER_TEST_DB_URL or start Postgres to run the bench.")
        return 0

    embedder, note = await _resolve_embedder()
    stamp = int(time.time())
    runs = [
        await _run_backend(
            "lexical",
            namespace=f"bench_lex_{stamp}",
            local=None,  # the no-local mode: no model client at all
            threshold=LEXICAL_THRESHOLD,
        ),
        await _run_backend(
            "embedding",
            namespace=f"bench_emb_{stamp}",
            local=embedder,
            threshold=EMBED_THRESHOLD,
        ),
    ]

    ok = _print_table(runs, note)
    print()
    return 0 if ok else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
