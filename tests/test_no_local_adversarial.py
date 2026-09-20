"""Adversarial edge cases for the no-local mode (lexical T3 backend).

Complements the lexical block in test_pipeline_sem_cache.py with what it
does not cover: the lexical→embedding direction of shared-table
coexistence (a NULL-embedding row used to crash vector lookups), trigram
input edges (1-char, whitespace, 10k-char, CJK/punctuation), migration
reopen idempotency, the real configs/proxy/no-local.yaml preset, the
tactics-override interplay, and split.cache_lookup on a lexical store.

Safety direction: a lexical cache FALSE HIT serves one user's cached
cloud answer for another's query (cross-tenant poisoning when namespaces
are involved); a false MISS only costs a cloud call. Tests weight
accordingly — the hit-side partitions (namespace meta, shared-table
isolation) get the hard cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg

from local_splitter.config import Config, ModelConfig, TacticsConfig, load_config
from local_splitter.models import Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.sem_cache import (
    CacheStore,
    LexicalCacheStore,
    lookup,
    store_backend,
)

from _fakes import FakeChatClient
from conftest import TEST_DB_URL, drop_cache_tables

EMBED_DIM = 32  # must match FakeChatClient default embed_dim
_LEX_THRESHOLD = 0.65


def _lexical_store(namespace: str = "default") -> LexicalCacheStore:
    drop_cache_tables()
    return LexicalCacheStore(TEST_DB_URL, namespace=namespace)


def _lexical_config(**t3_params: Any) -> Config:
    params: dict[str, Any] = {"backend": "lexical", "similarity_threshold": _LEX_THRESHOLD}
    params.update(t3_params)
    return Config(
        cloud=ModelConfig(backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"),
        local=None,
        tactics=TacticsConfig(t3_sem_cache=True, params={"t3_sem_cache": params}),
    )


def _lexical_pipeline(
    store: LexicalCacheStore, **t3_params: Any
) -> tuple[Pipeline, FakeChatClient]:
    cloud = FakeChatClient(
        chat_model="cloud-m",
        reply_content="cloud answer",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    return (
        Pipeline(cloud=cloud, local=None, config=_lexical_config(**t3_params), cache_store=store),
        cloud,
    )


# ---------------------------------------------------------------------------
# Shared-table coexistence: the lexical→embedding direction
# ---------------------------------------------------------------------------


def test_vector_lookup_over_lexical_only_table_is_a_miss_not_a_crash():
    """A lexical-written row has embedding NULL; ``embedding <=> %s`` is
    then NULL and the old code returned that row as the LIMIT 1 hit, so
    ``distance > max_distance`` raised TypeError on None. Flipping a
    deployment from lexical to embedding (the documented shared-table
    story) made every vector lookup error out — invisible through the
    pipeline (fail-open miss, warning spam) but fatal for direct users
    of the store.

    The vector store here uses the default 768 dims on purpose: v1's
    DDL is typed by whichever store opened the DB first, and the
    lexical store hardcodes 768 (there is no embedder to ask). A
    non-768 CacheStore opened after a lexical one fails its INSERTs —
    production can't hit that (``_build_pipeline`` passes 768 on both
    sides), but it is the pinned coupling that keeps the shared table
    coherent."""
    drop_cache_tables()
    lex = LexicalCacheStore(TEST_DB_URL)
    lex.store_text(
        cache_text="what is the capital of France?",
        response="Paris",
        model="m",
        finish_reason="stop",
    )
    lex.close()

    vec = CacheStore(TEST_DB_URL)  # default embed_dim=768, as _build_pipeline
    assert vec.size == 1  # the shared table really does hold the lexical row
    assert vec.lookup([0.1] * 768) is None  # regression: TypeError here

    # ...and the vector backend still works alongside the lexical rows.
    v = [1.0] + [0.0] * 767
    assert vec.store(v, response="vector answer", model="m", finish_reason="stop") > 0
    assert vec.lookup(v, threshold=0.9) is not None
    vec.close()


def test_store_backend_dispatch():
    assert store_backend(LexicalCacheStore.__new__(LexicalCacheStore)) == "lexical"
    assert store_backend(None) == "embedding"  # no store = vector semantics = inert


# ---------------------------------------------------------------------------
# Trigram input edges
# ---------------------------------------------------------------------------


def test_lexical_short_and_whitespace_queries():
    store = _lexical_store()
    store.store_text(cache_text="q", response="one-char", model="m", finish_reason="stop")
    assert store.lookup_text("q", threshold=0.5) is not None  # exact 1-char hit
    assert store.lookup_text("x", threshold=0.5) is None  # different 1-char: no hit

    # Whitespace normalizes to an empty trigram set: similarity is 0, so
    # whitespace-only text can never hit ANY row — including its own.
    # Degenerate, but the fail-safe direction (a miss, never a wrong hit).
    store.store_text(cache_text=" ", response="ws", model="m", finish_reason="stop")
    assert store.lookup_text("\t", threshold=0.01) is None
    assert store.lookup_text(" ", threshold=0.01) is None
    store.close()


def test_lexical_10k_char_query_roundtrip():
    store = _lexical_store()
    long_text = ("capital of France " * 600)[:10000]
    assert len(long_text) == 10000
    # Miss path must complete (exact scan cost), not error or hang.
    assert store.lookup_text(long_text, threshold=0.99) is None
    store.store_text(cache_text=long_text, response="big", model="m", finish_reason="stop")
    assert store.lookup_text(long_text, threshold=0.99) is not None
    # A short query against the big row still returns quickly (miss).
    assert store.lookup_text("unrelated short query", threshold=_LEX_THRESHOLD) is None
    store.close()


def test_lexical_cjk_roundtrip_and_punctuation_insensitivity():
    store = _lexical_store()
    cjk = "缓存查询测试"
    store.store_text(cache_text=cjk, response="ok", model="m", finish_reason="stop")
    assert store.lookup_text(cjk, threshold=_LEX_THRESHOLD) is not None
    assert store.lookup_text("完全不相关的中文查询文本", threshold=0.01) is None

    # pg_trgm strips punctuation, so a query differing only by punctuation
    # reads as identical at ANY threshold. Property of trigram similarity,
    # pinned so threshold tuning guidance can assume it (cosine would not
    # behave this way).
    assert store.lookup_text(cjk + "!", threshold=0.99) is not None
    store.close()


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------


def test_lexical_reopen_cycles_are_idempotent():
    """Rapid open/close cycles (the serve --reload scenario): migrations
    must no-op on reopen and data must survive."""
    drop_cache_tables()
    s1 = LexicalCacheStore(TEST_DB_URL)
    s1.store_text(cache_text="persist me", response="r1", model="m", finish_reason="stop")
    s1.close()

    s2 = LexicalCacheStore(TEST_DB_URL)
    assert s2.has_trgm is True
    assert s2.lookup_text("persist me", threshold=0.9) is not None
    s2.store_text(cache_text="second entry", response="r2", model="m", finish_reason="stop")
    s2.close()

    s3 = LexicalCacheStore(TEST_DB_URL)
    assert s3.size == 2
    s3.close()

    conn = psycopg.connect(TEST_DB_URL)
    try:
        versions = {
            row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
    finally:
        conn.close()
    assert versions == {1, 2}  # recorded once each, no duplicates


# ---------------------------------------------------------------------------
# The real no-local.yaml preset
# ---------------------------------------------------------------------------


def test_no_local_preset_loads_and_builds_lexical_pipeline():
    from local_splitter.cli import _build_pipeline

    preset = Path(__file__).resolve().parents[1] / "configs" / "proxy" / "no-local.yaml"
    config = load_config(preset)

    assert config.local is None  # structurally no local model
    tactics = config.tactics
    assert tactics.t3_sem_cache is True
    assert not any(
        (
            tactics.t1_route,
            tactics.t2_compress,
            tactics.t4_draft,
            tactics.t5_diff,
            tactics.t6_intent,
            tactics.t7_batch,
        ),
    )
    params = config.tactics.params["t3_sem_cache"]
    assert params["backend"] == "lexical"
    assert params["similarity_threshold"] == 0.65
    assert params["ttl"] == 86400

    pipeline = _build_pipeline(config, TEST_DB_URL)
    try:
        assert pipeline.local is None
        assert isinstance(pipeline.cache_store, LexicalCacheStore)
        assert store_backend(pipeline.cache_store) == "lexical"
        assert pipeline.cache_store.has_trgm is True
        assert pipeline.cloud is not None  # cloud passthrough unaffected
    finally:
        if pipeline.cache_store is not None:
            pipeline.cache_store.close()


def test_no_local_openai_preset_loads_as_openai_compat_mirror():
    """The doublewordai-chain variant: identical no-local shape, but the cloud
    hop speaks the OpenAI-compatible wire format (openai_compat + /v1), the
    exact cloud block of the repo-root config-openai.yaml.  Loader-only — no
    DB needed to prove the preset parses and stays structurally no-local."""
    preset = Path(__file__).resolve().parents[1] / "configs" / "proxy" / "no-local-openai.yaml"
    config = load_config(preset)

    assert config.local is None  # structurally no local model, like no-local
    cloud = config.cloud
    assert cloud is not None and cloud.backend == "openai_compat"
    assert cloud.endpoint == "http://127.0.0.1:8765/v1"
    assert cloud.chat_model == "claude-sonnet-4-20250514"
    assert cloud.api_key_env is None  # auth flows via chain headers, not env

    tactics = config.tactics
    assert tactics.t3_sem_cache is True
    assert not any(
        (
            tactics.t1_route,
            tactics.t2_compress,
            tactics.t4_draft,
            tactics.t5_diff,
            tactics.t6_intent,
            tactics.t7_batch,
        ),
    )
    params = config.tactics.params["t3_sem_cache"]
    assert params["backend"] == "lexical"
    assert params["similarity_threshold"] == 0.65
    assert params["ttl"] == 86400
    assert config.tactics.tools_require_cloud is True


# ---------------------------------------------------------------------------
# Pipeline-level interplay
# ---------------------------------------------------------------------------


async def test_tactics_override_disables_t3_on_lexical_store():
    """Per-request disable_tactics=["t3_sem_cache"] must bypass the cache
    entirely — no lookup, no store — even though the store is usable."""
    store = _lexical_store()
    pipeline, cloud = _lexical_pipeline(store)
    for _ in range(2):
        resp = await pipeline.complete(
            PipelineRequest(
                messages=[{"role": "user", "content": "override me"}],
                tactics_override=frozenset({"t3_sem_cache"}),
            )
        )
        assert resp.served_by == "cloud"
        assert not any(e.stage.startswith("t3_") for e in resp.trace)
    assert len(cloud.calls) == 2
    assert store.size == 0
    store.close()


async def test_lexical_namespace_from_meta_partitions_hits():
    """cache_namespace_from_meta folds meta into the cache key: two
    tenants sending the IDENTICAL text must not share a cache entry
    (cross-tenant poisoning), while the same tenant's repeat hits.

    Regression: trigram similarity over the full prefixed key reads ~0.9
    for different tenants (the shared body dominates the trigram set),
    so tenant B used to be served tenant A's answer. The namespace
    prefix is now required to match exactly, in both directions:
    namespaced lookups never see unnamespaced rows and vice versa."""
    store = _lexical_store()
    pipeline, cloud = _lexical_pipeline(store, cache_namespace_from_meta="tenant")

    async def ask(tenant: str) -> Any:
        return await pipeline.complete(
            PipelineRequest(
                messages=[{"role": "user", "content": "tenant scoped question"}],
                meta={"tenant": tenant},
            )
        )

    resp_a1 = await ask("acme")
    resp_b = await ask("other")
    resp_a2 = await ask("acme")

    assert resp_a1.served_by == "cloud"
    assert resp_b.served_by == "cloud"  # different namespace ⇒ no shared hit
    assert resp_a2.served_by == "cache"  # same namespace repeat ⇒ hit
    assert len(cloud.calls) == 2

    # Reverse direction: with the namespace param absent (a deployment
    # that never namespaced, or removed it), namespaced rows must not
    # serve plain lookups either.
    plain_pipeline, plain_cloud = _lexical_pipeline(store)
    resp_plain = await plain_pipeline.complete(
        PipelineRequest(messages=[{"role": "user", "content": "tenant scoped question"}])
    )
    assert resp_plain.served_by == "cloud"
    assert len(plain_cloud.calls) == 1
    store.close()


async def test_direct_lookup_embedding_backend_without_local_skips():
    """Direct callers of sem_cache.lookup with an embedding store and
    local=None get a visible SKIP (the old AttributeError path)."""
    drop_cache_tables()
    vec = CacheStore(TEST_DB_URL, embed_dim=EMBED_DIM)
    result = await lookup(
        [{"role": "user", "content": "hello"}],
        local=None,
        store=vec,
    )
    assert result.hit is False
    assert result.events[0].decision == "SKIP"
    assert result.events[0].detail["reason"] == "no local embedder"
    vec.close()


# ---------------------------------------------------------------------------
# MCP tool on the lexical backend
# ---------------------------------------------------------------------------


async def _call(server: Any, name: str, args: dict) -> dict:
    """Invoke a FastMCP tool and return its structured dict result.

    Same unwrapping as test_transport_mcp._call: ``call_tool`` returns
    ``(content_list, structured_dict)`` on recent SDKs.
    """
    result = await server.call_tool(name, args)
    if isinstance(result, tuple):
        content, structured = result
    else:  # pragma: no cover — older SDKs
        content, structured = result, None
    if structured is not None:
        return structured
    assert content, f"{name} returned no content"
    text = getattr(content[0], "text", None) or str(content[0])
    return json.loads(text)


async def test_mcp_cache_lookup_lexical_no_local():
    """split.cache_lookup on a no-local lexical pipeline: real hit with a
    preview, real miss for unrelated text (the disabled-path stub test
    lives in test_transport_mcp.py)."""
    from local_splitter.transport import create_mcp_server

    store = _lexical_store()
    store.store_text(
        cache_text="lookup me", response="cached answer", model="cloud-m", finish_reason="stop"
    )
    config = _lexical_config()
    pipeline = Pipeline(cloud=FakeChatClient(), local=None, config=config, cache_store=store)
    server = create_mcp_server(pipeline, config)

    hit = await _call(
        server,
        "split.cache_lookup",
        {"messages": [{"role": "user", "content": "lookup me"}]},
    )
    assert hit["hit"] is True
    assert hit["response_preview"] == "cached answer"

    miss = await _call(
        server,
        "split.cache_lookup",
        {"messages": [{"role": "user", "content": "completely unrelated"}]},
    )
    assert miss["hit"] is False
    store.close()
