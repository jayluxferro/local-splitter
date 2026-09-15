"""Dynamic chunked embeddings — model-derived context, chunk, pool.

Covers the helper directly (chunk geometry, pooling, context lookup with
its TTL cache and its fallback floor) and the sem_cache call site that
now routes through it.  The Ollama HTTP layer is mocked with
``httpx.MockTransport``, matching ``test_models_ollama.py`` — nothing
here touches the network.
"""

from __future__ import annotations

import httpx
import pytest

import local_splitter.chunked_embedding as ce
from local_splitter.chunked_embedding import (
    CHUNK_MARGIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CTX_CACHE_TTL_S,
    FALLBACK_CTX_TOKENS,
    TOKENS_PER_CHAR,
    embed_text_dynamic,
    estimate_tokens,
    l2_normalize,
    length_weighted_mean,
    model_ctx_tokens,
    split_overlapping,
)
from local_splitter.models import ModelBackendError
from local_splitter.pipeline.sem_cache import lookup

from _fakes import FakeChatClient

ENDPOINT = "http://ollama.test"
FALLBACK_BUDGET_CHARS = int((FALLBACK_CTX_TOKENS - CHUNK_MARGIN_TOKENS) / TOKENS_PER_CHAR)


@pytest.fixture(autouse=True)
def _clear_ctx_cache() -> None:
    """The context cache is module-global by design — isolate each test."""
    ce._ctx_cache.clear()


class _Clock:
    """Controllable stand-in for time.monotonic (TTL tests, no sleeping)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _show_handler(
    payload: dict | None,
    *,
    calls: list[httpx.Request] | None = None,
    status: int = 200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if status != 200 or payload is None:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(200, json=payload)

    return handler


class _Embedder:
    """Records embed_many calls; returns deterministic per-index vectors.

    ``vectors`` (optional) is cycled by chunk index, which lets a test
    attribute a pooled dimension back to a specific chunk.
    """

    def __init__(self, dim: int = 8, vectors: list[list[float]] | None = None) -> None:
        self.dim = dim
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def __call__(self, texts):  # type: ignore[no-untyped-def]
        batch = list(texts)
        self.calls.append(batch)
        if self.vectors is not None:
            return [self.vectors[i % len(self.vectors)] for i in range(len(batch))]
        return [[float(i % 7 + 1) for i in range(self.dim)] for _ in batch]


# ---------------------------------------------------------------------------
# Estimation / splitting (unit)
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_chars_over_four() -> None:
    assert estimate_tokens("a" * 400) == int(400 * TOKENS_PER_CHAR) == 100
    assert estimate_tokens("") == 0


def test_split_short_text_is_returned_unchanged() -> None:
    assert split_overlapping("hello", budget_tokens=100) == ["hello"]
    assert split_overlapping("", budget_tokens=100) == []


def test_split_covers_whole_text_within_budget_and_overlap() -> None:
    text = "".join(str(i % 10) for i in range(5000))
    budget_tokens = 250  # → 1000 chars/window, 48 tokens (192 chars) of overlap
    chunks = split_overlapping(text, budget_tokens=budget_tokens)

    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= budget_tokens for c in chunks)  # never over budget
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])

    # Adjacent chunks share exactly the configured overlap (context
    # continuity across a boundary).
    overlap_chars = int(CHUNK_OVERLAP_TOKENS / TOKENS_PER_CHAR)
    assert chunks[0][-overlap_chars:] == chunks[1][:overlap_chars]

    # Nothing is lost: each chunk's non-overlapping prefix tiles the text.
    step = int(budget_tokens / TOKENS_PER_CHAR) - overlap_chars
    assert "".join(c[:step] for c in chunks[:-1]) + chunks[-1] == text


def test_split_clamps_overlap_so_step_stays_positive() -> None:
    """An overlap ≥ budget must not spin forever (the step would be ≤ 0)."""
    chunks = split_overlapping("x" * 1000, budget_tokens=10, overlap_tokens=9999)
    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= 10 for c in chunks)
    assert chunks[-1].endswith("x")


# ---------------------------------------------------------------------------
# Pooling (unit)
# ---------------------------------------------------------------------------


def test_weighted_mean_weights_unequal_chunks_by_length() -> None:
    pooled = length_weighted_mean([[1.0, 0.0], [0.0, 1.0]], [1, 3])
    assert pooled == pytest.approx([0.25, 0.75])


def test_weighted_mean_handles_all_zero_weights() -> None:
    """All-empty chunks fall back to a plain mean instead of dividing by zero."""
    assert length_weighted_mean([[1.0, 0.0], [3.0, 0.0]], [0, 0]) == [2.0, 0.0]


def test_weighted_mean_rejects_ragged_vectors() -> None:
    with pytest.raises(ValueError):
        length_weighted_mean([[1.0, 0.0], [1.0]], [1, 1])


def test_l2_normalize_unit_norm_and_zero_vector() -> None:
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # no NaN from a zero norm


# ---------------------------------------------------------------------------
# Context lookup over a mocked Ollama
# ---------------------------------------------------------------------------


async def test_ctx_parses_llama_prefixed_key() -> None:
    transport = httpx.MockTransport(_show_handler({"model_info": {"llama.context_length": 8192}}))
    assert await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport) == 8192


async def test_ctx_parses_architecture_prefixed_key() -> None:
    """Real Ollama names the key after the architecture — ``nomic-bert``
    for nomic-embed-text, ``qwen3`` for qwen3-embedding, never ``llama``."""
    transport = httpx.MockTransport(
        _show_handler({"model_info": {"nomic-bert.context_length": 2048}})
    )
    got = await model_ctx_tokens("nomic-embed-text", endpoint=ENDPOINT, transport=transport)
    assert got == 2048


async def test_ctx_parses_legacy_top_level_key() -> None:
    transport = httpx.MockTransport(_show_handler({"context_length": 4096}))
    assert await model_ctx_tokens("old", endpoint=ENDPOINT, transport=transport) == 4096


async def test_ctx_requests_show_with_post_and_model_body() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _show_handler({"model_info": {"qwen3.context_length": 32768}}, calls=calls)
    )
    await model_ctx_tokens("qwen3-embedding:0.6b", endpoint=ENDPOINT, transport=transport)

    assert len(calls) == 1
    assert calls[0].url.path == "/api/show"
    # GET on this route 405s on Ollama 0.33.x (verified live) — POST is
    # the only shape that actually answers.
    assert calls[0].method == "POST"
    assert calls[0].read().decode() == '{"model":"qwen3-embedding:0.6b"}'


async def test_ctx_api_failure_falls_back() -> None:
    transport = httpx.MockTransport(_show_handler(None, status=500))
    got = await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert got == FALLBACK_CTX_TOKENS


async def test_ctx_unknown_model_404_falls_back() -> None:
    transport = httpx.MockTransport(_show_handler(None, status=404))
    got = await model_ctx_tokens("ghost", endpoint=ENDPOINT, transport=transport)
    assert got == FALLBACK_CTX_TOKENS


async def test_ctx_unreachable_ollama_falls_back_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("ollama down")

    transport = httpx.MockTransport(handler)
    got = await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert got == FALLBACK_CTX_TOKENS


async def test_ctx_body_without_context_length_falls_back() -> None:
    transport = httpx.MockTransport(_show_handler({"model_info": {"x.embedding_length": 768}}))
    got = await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert got == FALLBACK_CTX_TOKENS


async def test_ctx_malformed_json_falls_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    transport = httpx.MockTransport(handler)
    got = await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert got == FALLBACK_CTX_TOKENS


async def test_ctx_no_endpoint_or_model_skips_http_entirely() -> None:
    """A non-Ollama backend (or a missing model name) must not be probed."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP expected")

    transport = httpx.MockTransport(handler)
    assert await model_ctx_tokens("m", endpoint=None, transport=transport) == FALLBACK_CTX_TOKENS
    assert (
        await model_ctx_tokens(None, endpoint=ENDPOINT, transport=transport) == FALLBACK_CTX_TOKENS
    )


async def test_ctx_cached_within_ttl() -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _show_handler({"model_info": {"llama.context_length": 8192}}, calls=calls)
    )
    assert await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport) == 8192
    assert await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport) == 8192
    assert len(calls) == 1  # one query per model per TTL, not per call


async def test_ctx_cache_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _show_handler({"model_info": {"llama.context_length": 8192}}, calls=calls)
    )
    clock = _Clock()
    monkeypatch.setattr(ce, "_monotonic", clock)

    await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    clock.t += CTX_CACHE_TTL_S + 1
    await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert len(calls) == 2


async def test_ctx_failure_is_not_cached() -> None:
    """A transient /api/show failure must re-query, not pin the fallback
    for the whole TTL."""
    state = {"fail": True}
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if state["fail"]:
            return httpx.Response(503, json={"error": "loading"})
        return httpx.Response(200, json={"model_info": {"llama.context_length": 8192}})

    transport = httpx.MockTransport(handler)
    first = await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport)
    assert first == FALLBACK_CTX_TOKENS
    state["fail"] = False
    assert await model_ctx_tokens("m", endpoint=ENDPOINT, transport=transport) == 8192
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# embed_text_dynamic
# ---------------------------------------------------------------------------

_UNIT_VEC = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]  # _Embedder(dim=8) output


async def test_short_text_makes_one_call_and_returns_vector_unchanged() -> None:
    embedder = _Embedder()
    vec = await embed_text_dynamic("hi there", model=None, embed_many=embedder)
    assert embedder.calls == [["hi there"]]
    assert vec == _UNIT_VEC  # untouched: no pooling, no normalization


async def test_long_text_is_chunked_into_one_batched_call() -> None:
    embedder = _Embedder()
    text = "a" * 30_000  # fallback ctx 2048 → ~7.9k-char windows → 4 chunks

    vec = await embed_text_dynamic(text, model=None, embed_many=embedder)

    assert len(embedder.calls) == 1  # ONE round trip for all chunks
    assert len(embedder.calls[0]) > 1
    assert all(len(c) <= FALLBACK_BUDGET_CHARS for c in embedder.calls[0])
    assert "".join(embedder.calls[0]) != text  # chunked, not concatenated
    assert len(vec) == embedder.dim  # dims unchanged, KNN schema safe
    assert sum(x * x for x in vec) == pytest.approx(1.0)  # L2-normalized


async def test_dynamic_ctx_keeps_a_fitting_text_on_the_single_call_path() -> None:
    """The model's REAL context decides, not the 2048 fallback: 10k chars
    (~2.5k tokens) would be chunked under the fallback floor but fits an
    8k-context model, so it goes in one piece."""
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _show_handler({"model_info": {"llama.context_length": 8192}}, calls=calls)
    )
    embedder = _Embedder()
    text = "b" * 10_000

    vec = await embed_text_dynamic(
        text, model="m", embed_many=embedder, endpoint=ENDPOINT, transport=transport
    )

    assert len(calls) == 1  # context looked up exactly once
    assert embedder.calls == [[text]]  # and the text went in one piece
    assert len(vec) == embedder.dim


async def test_chunked_embedding_is_deterministic() -> None:
    embedder = _Embedder()
    text = "c" * 20_000
    first = await embed_text_dynamic(text, model=None, embed_many=embedder)
    second = await embed_text_dynamic(text, model=None, embed_many=embedder)
    assert first == second


async def test_chunked_embedding_pools_by_chunk_length() -> None:
    """End-to-end weighted mean: with a long first chunk and a short
    second one, the pooled vector must lean toward the first chunk's."""
    embedder = _Embedder(dim=2, vectors=[[1.0, 0.0], [0.0, 1.0]])
    text = "d" * 12_000  # → 7936-char first chunk, 4256-char second

    vec = await embed_text_dynamic(text, model=None, embed_many=embedder)

    chunks = embedder.calls[0]
    assert len(chunks) == 2
    longer, shorter = len(chunks[0]), len(chunks[1])
    assert longer > shorter
    total = float(longer + shorter)
    expected = [longer / total, shorter / total]
    norm = sum(x * x for x in expected) ** 0.5
    assert vec == pytest.approx([x / norm for x in expected])


async def test_chunk_count_mismatch_raises_model_backend_error() -> None:
    """A short vector list would silently misalign chunks with vectors —
    surface it so the caller fail-opens to a cache miss."""

    async def stingy(texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0]]

    with pytest.raises(ModelBackendError):
        await embed_text_dynamic("e" * 20_000, model=None, embed_many=stingy)


async def test_long_text_embedder_failure_propagates_for_fail_open() -> None:
    async def broken(texts):  # type: ignore[no-untyped-def]
        raise ModelBackendError("ollama down")

    with pytest.raises(ModelBackendError):
        await embed_text_dynamic("f" * 20_000, model=None, embed_many=broken)


# ---------------------------------------------------------------------------
# sem_cache call site
# ---------------------------------------------------------------------------


class _RecordingClient(FakeChatClient):
    """FakeChatClient that also records every embed batch it receives."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.embed_batches: list[list[str]] = []

    async def embed(self, texts, *, model=None):  # type: ignore[no-untyped-def]
        self.embed_batches.append(list(texts))
        return await super().embed(texts, model=model)


async def test_sem_cache_embeds_long_prompt_instead_of_skipping() -> None:
    """Regression: a prompt over the embedding model's context used to be
    skipped outright, so it could never be cached.  It must now be
    embedded in one batched call and stay a single 32-dim vector."""
    local = _RecordingClient(chat_model="local-m")

    result = await lookup(
        [{"role": "user", "content": "z" * 40_000}],
        local=local,
        store=None,  # never reached: store.lookup fails open after embedding
    )

    assert result.embedding is not None
    assert len(result.embedding) == 32  # FakeChatClient's embed_dim, unchanged
    assert len(local.embed_batches) == 1
    assert len(local.embed_batches[0]) > 1


async def test_sem_cache_short_prompt_still_single_embed_call() -> None:
    local = _RecordingClient(chat_model="local-m")
    result = await lookup(
        [{"role": "user", "content": "what is a monad?"}],
        local=local,
        store=None,
    )
    assert result.embedding is not None
    assert local.embed_batches == [["what is a monad?"]]


async def test_sem_cache_explicit_embed_max_chars_still_skips() -> None:
    """The operator escape hatch survives: an explicit cap skips instead
    of chunking."""
    local = _RecordingClient(chat_model="local-m")
    result = await lookup(
        [{"role": "user", "content": "w" * 500}],
        local=local,
        store=None,
        params={"embed_max_chars": 100},
    )
    assert result.hit is False
    assert result.embedding is None
    assert result.events[0].decision == "SKIP"
    assert result.events[0].detail["reason"] == "cache_text exceeds embed_max_chars"
    assert local.embed_batches == []
