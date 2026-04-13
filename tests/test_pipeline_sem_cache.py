"""Unit tests for T3 sem_cache — semantic similarity cache.

Tests cover:
- CacheStore: insert, lookup (hit/miss/TTL expiry), eviction, size
- lookup() pipeline function: hit, miss, embed error (fail-open)
- store_response() pipeline function
- Pipeline.complete integration: miss→store, hit on repeat, T3 disabled
- T1+T3 composition: trivial bypasses cache, complex goes through cache
"""

from __future__ import annotations

from pathlib import Path

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.sem_cache import (
    CacheStore,
    lookup,
    store_response,
)

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 32  # must match FakeChatClient default embed_dim


def _store(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache.sqlite", embed_dim=EMBED_DIM)


def _config(*, t3: bool = True, t1: bool = False, **t3_params) -> Config:
    # Use a low threshold for tests since FakeChatClient produces sparse
    # deterministic embeddings that are identical for identical text.
    t3_defaults = {"similarity_threshold": 0.5}
    t3_defaults.update(t3_params)
    return Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m",
            embed_model="nomic-embed-text",
        ),
        tactics=TacticsConfig(
            t1_route=t1,
            t3_sem_cache=t3,
            params={"t3_sem_cache": t3_defaults} if t3 else {},
        ),
    )


_MSGS = [{"role": "user", "content": "what is a monad?"}]


# ---------------------------------------------------------------------------
# CacheStore unit tests
# ---------------------------------------------------------------------------

def _vec(first: float = 1.0, second: float = 0.0) -> list[float]:
    """Build a EMBED_DIM-sized vector with the first two dims set."""
    v = [0.0] * EMBED_DIM
    v[0] = first
    if EMBED_DIM > 1:
        v[1] = second
    return v


class TestCacheStore:
    def test_store_and_lookup_hit(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        vec = _vec(1.0)
        store.store(vec, response="it's a burrito", model="cloud-m", finish_reason="stop")

        entry = store.lookup(vec, threshold=0.9)
        assert entry is not None
        assert entry.response == "it's a burrito"
        assert entry.similarity > 0.99
        assert entry.model == "cloud-m"
        store.close()

    def test_lookup_miss_below_threshold(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.store(_vec(1.0, 0.0), response="a", model="m", finish_reason="stop")

        # Orthogonal vector → cosine similarity ≈ 0
        entry = store.lookup(_vec(0.0, 1.0), threshold=0.5)
        assert entry is None
        store.close()

    def test_lookup_miss_empty_cache(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.lookup(_vec(1.0))
        assert entry is None
        store.close()

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        vec = _vec(1.0)
        store.store(vec, response="old", model="m", finish_reason="stop")

        # Should hit with large TTL.
        assert store.lookup(vec, ttl=99999) is not None
        # Should miss with zero TTL (everything expired).
        assert store.lookup(vec, ttl=0) is None
        store.close()

    def test_evict_expired(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.store(_vec(1.0), response="a", model="m", finish_reason="stop")
        assert store.size == 1
        # Evict with ttl=0 → everything is "expired".
        evicted = store.evict_expired(ttl=0)
        assert evicted == 1
        assert store.size == 0
        store.close()

    def test_size(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.size == 0
        store.store(_vec(1.0, 0.0), response="a", model="m", finish_reason="stop")
        store.store(_vec(0.0, 1.0), response="b", model="m", finish_reason="stop")
        assert store.size == 2
        store.close()


# ---------------------------------------------------------------------------
# Async: lookup()
# ---------------------------------------------------------------------------

async def test_lookup_miss_on_empty_cache(tmp_path: Path) -> None:
    local = FakeChatClient(chat_model="local-m")
    store = _store(tmp_path)
    result = await lookup(_MSGS, local=local, store=store)
    assert result.hit is False
    assert result.embedding is not None
    assert result.events[0].decision == "MISS"
    store.close()


async def test_lookup_hit_after_store(tmp_path: Path) -> None:
    local = FakeChatClient(chat_model="local-m")
    store = _store(tmp_path)

    # Embed the same text that lookup() will embed, then store it.
    emb = (await local.embed(["what is a monad?"]))[0]
    store.store(emb, response="cached answer", model="cloud-m", finish_reason="stop")

    # lookup() embeds the last user message — "what is a monad?" — and
    # the FakeChatClient is deterministic, so it'll get the same vector.
    result = await lookup(_MSGS, local=local, store=store, params={"similarity_threshold": 0.5})
    assert result.hit is True
    assert result.entry is not None
    assert result.entry.response == "cached answer"
    assert result.events[0].decision == "HIT"
    store.close()


async def test_lookup_embed_error_fails_open(tmp_path: Path) -> None:
    local = FakeChatClient(
        chat_model="local-m",
        raise_on_complete=ModelBackendError("embed down"),
    )
    # Override embed to raise.
    async def bad_embed(*a, **kw):
        raise ModelBackendError("embed down")

    local.embed = bad_embed  # type: ignore[assignment]
    store = _store(tmp_path)

    result = await lookup(_MSGS, local=local, store=store)
    assert result.hit is False
    assert result.embedding is None
    assert result.events[0].decision == "ERROR"
    store.close()


async def test_lookup_no_user_text_skips() -> None:
    local = FakeChatClient(chat_model="local-m")
    # We don't need a real store since we should skip before DB access.
    result = await lookup(
        [{"role": "system", "content": "be helpful"}],
        local=local,
        store=None,  # type: ignore[arg-type]
    )
    assert result.hit is False
    assert result.events[0].decision == "SKIP"


# ---------------------------------------------------------------------------
# Sync: store_response()
# ---------------------------------------------------------------------------

def test_store_response_succeeds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = store_response(
        _vec(1.0),
        response="answer",
        model="cloud-m",
        finish_reason="stop",
        cache_store=store,
    )
    assert event.stage == "t3_cache_store"
    assert event.decision == "STORED"
    assert store.size == 1
    store.close()


# ---------------------------------------------------------------------------
# Integration: Pipeline.complete with T3 enabled
# ---------------------------------------------------------------------------

async def test_pipeline_t3_miss_then_hit(tmp_path: Path) -> None:
    """First call is a miss (goes to cloud, stores). Second call is a hit."""
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="cloud answer",
        usage=Usage(input_tokens=50, output_tokens=10),
    )
    local = FakeChatClient(
        chat_model="local-m",
        usage=Usage(input_tokens=5, output_tokens=1),
    )
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(), cache_store=store,
    )

    # First request: cache miss → cloud.
    resp1 = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp1.served_by == "cloud"
    assert resp1.content == "cloud answer"
    # Trace should have: t3_cache_lookup(MISS) + cloud_call + t3_cache_store
    stages = [e.stage for e in resp1.trace]
    assert "t3_cache_lookup" in stages
    assert "cloud_call" in stages
    assert "t3_cache_store" in stages

    # Second request (same message): cache hit.
    resp2 = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp2.served_by == "cache"
    assert resp2.content == "cloud answer"
    assert resp2.trace[0].stage == "t3_cache_lookup"
    assert resp2.trace[0].decision == "HIT"

    # Cloud was only called once.
    assert len(cloud.calls) == 1
    store.close()


async def test_pipeline_t3_disabled_skips_cache(tmp_path: Path) -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(chat_model="local-m")
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(t3=False), cache_store=store,
    )

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp.served_by == "cloud"
    # No cache events in trace.
    assert not any(e.stage.startswith("t3_") for e in resp.trace)
    store.close()


async def test_pipeline_t3_no_cache_store_skips(tmp_path: Path) -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(chat_model="local-m")
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(), cache_store=None,
    )

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp.served_by == "cloud"


async def test_pipeline_t3_explicit_hint_bypasses_cache(tmp_path: Path) -> None:
    cloud = FakeChatClient(chat_model="cloud-m", reply_content="direct")
    local = FakeChatClient(chat_model="local-m")
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(), cache_store=store,
    )

    resp = await pipeline.complete(
        PipelineRequest(messages=_MSGS, model_hint="cloud")
    )
    assert resp.served_by == "cloud"
    assert not any(e.stage.startswith("t3_") for e in resp.trace)
    store.close()


async def test_pipeline_t3_stats_count_cache_hits(tmp_path: Path) -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", usage=Usage(input_tokens=20, output_tokens=5),
    )
    local = FakeChatClient(chat_model="local-m")
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(), cache_store=store,
    )

    await pipeline.complete(PipelineRequest(messages=_MSGS))  # miss
    await pipeline.complete(PipelineRequest(messages=_MSGS))  # hit

    snap = pipeline.stats()
    assert snap.total_requests == 2
    assert snap.by_served.get("cloud", 0) == 1
    assert snap.by_served.get("cache", 0) == 1
    # Only one cloud call's worth of tokens.
    assert snap.tokens_in_cloud == 20
    store.close()


# ---------------------------------------------------------------------------
# Integration: T1 + T3 composition
# ---------------------------------------------------------------------------

async def test_t1_trivial_bypasses_t3(tmp_path: Path) -> None:
    """T1 routes trivial locally — T3 cache is never consulted."""
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "local answer"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(t3=True, t1=True),
        cache_store=store,
    )

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp.served_by == "local"
    # No cache events — T1 short-circuited before T3.
    assert not any(e.stage.startswith("t3_") for e in resp.trace)
    assert store.size == 0
    store.close()


async def test_t1_complex_then_t3_miss(tmp_path: Path) -> None:
    """T1 classifies COMPLEX → T3 cache lookup (miss) → cloud."""
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="cloud answer",
        usage=Usage(input_tokens=50, output_tokens=10),
    )
    local = FakeChatClient(
        chat_model="local-m", reply_content="COMPLEX",
        usage=Usage(input_tokens=5, output_tokens=1),
    )
    store = _store(tmp_path)
    pipeline = Pipeline(
        cloud=cloud, local=local, config=_config(t3=True, t1=True),
        cache_store=store,
    )

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))
    assert resp.served_by == "cloud"
    stages = [e.stage for e in resp.trace]
    assert "t1_classify" in stages
    assert "t3_cache_lookup" in stages
    assert "cloud_call" in stages
    assert "t3_cache_store" in stages
    assert store.size == 1
    store.close()
