"""Dynamic chunked embeddings — no hardcoded context caps.

Two jobs, both for local-splitter's semantic cache, whose prompts are
arbitrary agent text and can exceed nomic-embed-text's 2048-token
window:

1. **Context lookup.** Ask the model what it can actually take
   (``POST /api/show`` → ``<architecture>.context_length``), cached per
   model for 10 minutes so it is one query per model per window, never
   per request.  Unknown model or a failed call falls back to
   :data:`FALLBACK_CTX_TOKENS` — a floor for models we cannot identify,
   not a cap on the ones we can.
2. **Chunk and pool.** Before this module the choice was "fit or skip":
   a prompt over the window got no embedding at all, so T3 could never
   cache it.  Now it is split into overlapping character windows,
   embedded in ONE batched ``/api/embed`` call, and the vectors are
   combined with a length-weighted mean so a single vector comes out.

Chunk geometry therefore comes from the model's real reported context,
never from a constant, and the pooled vector keeps the model's native
dimension — single-vector consumers like the ``cache_entry`` KNN
(``ORDER BY embedding <=> %s``) are untouched.

Fail-open stays the rule: every function here either returns a usable
vector or raises, and the context lookup never raises at all — a failed
``/api/show`` means "use the fallback floor and carry on", not an error.

Mean pooling is for SIMILARITY consumers only (cache lookup, cosine
gates).  It is not a retrieval-ranking representation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx

from local_splitter.models.base import ModelBackendError

_log = logging.getLogger(__name__)

# Estimation heuristic, no tokenizer dependency (chars/4).  The
# chunk margin absorbs its error; pathological tokenization (base64
# blobs, CJK) may over- or under-chunk but can never exceed the model's
# context once the margin is subtracted.
TOKENS_PER_CHAR = 0.25
CHUNK_MARGIN_TOKENS = 64  # headroom below the model's ctx
CHUNK_OVERLAP_TOKENS = 48  # context continuity between chunks

# Conservative floor for models we cannot identify — a *fallback*, not a
# cap: any model whose context we can read gets its real number.
FALLBACK_CTX_TOKENS = 2048

# One lookup per model per 10 minutes, never per request.  Failures are
# deliberately not cached, so a transient /api/show error re-queries on
# the next call instead of pinning the fallback for the whole TTL.
CTX_CACHE_TTL_S = 600.0

# /api/show is a metadata call on the request path — keep it short so a
# wedged Ollama cannot stall T3 behind a slow lookup.
SHOW_TIMEOUT = httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=5.0)

# (endpoint, model) → (monotonic timestamp, ctx tokens)
_ctx_cache: dict[tuple[str, str], tuple[float, int]] = {}

# Indirection so tests can advance a fake clock (embedder idiomatic
# pattern in the sibling repos) instead of sleeping 600s.
_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# Estimation, splitting, pooling
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """chars → tokens, heuristic by design (no tokenizer dependency)."""
    return int(len(text) * TOKENS_PER_CHAR)


def split_overlapping(
    text: str,
    *,
    budget_tokens: int,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split *text* into character windows of at most *budget_tokens*.

    Windows are stepped by ``budget - overlap`` characters so adjacent
    chunks share their boundary context (a sentence cut in half still
    has its other half in a neighbouring chunk).  Returns ``[text]``
    unchanged when it already fits, and ``[]`` for empty input.

    Splitting is by *character*, so no multi-byte character is ever torn
    — the token estimate is what can be wrong, and the caller's margin
    covers that.  An ``overlap_tokens`` at or above the budget would
    make the step non-positive (infinite loop), so it is clamped to half
    the budget.
    """
    if not text:
        return []
    budget_chars = max(1, int(budget_tokens / TOKENS_PER_CHAR))
    if len(text) <= budget_chars:
        return [text]
    overlap_chars = max(0, min(int(overlap_tokens / TOKENS_PER_CHAR), budget_chars // 2))
    step = budget_chars - overlap_chars
    return [text[i : i + budget_chars] for i in range(0, len(text), step)]


def length_weighted_mean(
    vectors: Sequence[Sequence[float]], weights: Sequence[float]
) -> list[float]:
    """Weighted mean of *vectors*, weighting longer chunks more.

    Every chunk vector comes from the same model, so dimensions agree;
    a mismatch is a bug worth surfacing rather than averaging into a
    shorter vector.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(f"ragged embeddings: {len(vec)} != {dim}")
    total = float(sum(weights))
    if total <= 0:
        # Degenerate (all-empty chunks): plain mean keeps the result a
        # finite vector instead of dividing by zero.
        total = float(len(vectors))
        weights = [1.0] * len(vectors)
    out = [0.0] * dim
    for vec, weight in zip(vectors, weights, strict=True):
        for i, x in enumerate(vec):
            out[i] += x * weight
    return [x / total for x in out]


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize *vector*; a zero vector stays zero (no NaN)."""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return [float(x) for x in vector]
    return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# Dynamic context lookup
# ---------------------------------------------------------------------------


def _parse_context_length(payload: Mapping[str, Any]) -> int | None:
    """Pull the context length out of an ``/api/show`` body.

    The obvious key is ``model_info["llama.context_length"]``, but real
    Ollama prefixes it with the architecture — ``nomic-bert`` for
    nomic-embed-text, ``qwen3`` for qwen3-embedding — so an exact
    ``llama.`` lookup would miss every model we actually use and pin
    them all to the fallback.  Exact key first, then any
    ``*.context_length`` entry.  Older Ollama returned a bare top-level
    ``context_length``; that is the last resort before the caller's
    fallback.
    """
    model_info = payload.get("model_info")
    if isinstance(model_info, Mapping):
        exact = model_info.get("llama.context_length")
        if isinstance(exact, int) and not isinstance(exact, bool) and exact > 0:
            return int(exact)
        for key in sorted(model_info):
            if key != "context_length" and not key.endswith(".context_length"):
                continue
            value = model_info[key]
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
    legacy = payload.get("context_length")
    if isinstance(legacy, int) and not isinstance(legacy, bool) and legacy > 0:
        return int(legacy)
    return None


async def _fetch_ctx(
    model: str, *, endpoint: str, transport: httpx.AsyncBaseTransport | None
) -> int | None:
    """One ``POST /api/show``.  ``None`` means "unknown" (never raises).

    POST, not GET: Ollama 0.33.x (the build these layers run against)
    answers GET on this route with 405 — verified against a live server,
    GET with a body *and* GET with ``?model=`` both 405.  That is also
    the shape Ollama documents, and the model name travels in the body.
    """
    try:
        async with httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            timeout=SHOW_TIMEOUT,
            transport=transport,
        ) as client:
            resp = await client.post("/api/show", json={"model": model})
    except Exception as exc:  # unreachable, timeout, bad transport — all fail open
        _log.debug("ollama /api/show lookup failed for %s: %s", model, exc)
        return None
    if resp.status_code != 200:
        _log.debug("ollama /api/show returned %s for %s", resp.status_code, model)
        return None
    try:
        payload = resp.json()
    except ValueError as exc:
        _log.debug("ollama /api/show bad JSON for %s: %s", model, exc)
        return None
    if not isinstance(payload, Mapping):
        return None
    ctx = _parse_context_length(payload)
    if ctx is None:
        _log.debug("ollama /api/show had no context_length for %s", model)
    return ctx


async def model_ctx_tokens(
    model: str | None,
    *,
    endpoint: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The model's real context length in tokens (never raises).

    ``endpoint`` is ``None`` for non-Ollama backends (an OpenAI-compatible
    ``local`` model has no ``/api/show``), which short-circuits to the
    fallback without any network I/O — that keeps a fake/remote backend
    from being probed at all, and keeps the unit suite offline.

    Cached per ``(endpoint, model)`` for :data:`CTX_CACHE_TTL_S`, so the
    short-lived ``httpx`` client inside :func:`_fetch_ctx` is built at
    most once per model per window — T3 calls this on every request, and
    a client per request would churn connections for no reason.  A
    failed lookup is *not* cached, so the next call re-queries.
    """
    if not model or not endpoint:
        return FALLBACK_CTX_TOKENS
    key = (endpoint, model)
    now = _monotonic()
    cached = _ctx_cache.get(key)
    if cached is not None and now - cached[0] < CTX_CACHE_TTL_S:
        return cached[1]
    ctx = await _fetch_ctx(model, endpoint=endpoint, transport=transport)
    if ctx is None:
        return FALLBACK_CTX_TOKENS
    _ctx_cache[key] = (now, ctx)
    return ctx


# ---------------------------------------------------------------------------
# The embed call site
# ---------------------------------------------------------------------------


EmbedMany = Callable[[Sequence[str]], Awaitable[list[list[float]]]]


async def embed_text_dynamic(
    text: str,
    *,
    model: str | None,
    embed_many: EmbedMany,
    endpoint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[float]:
    """Embed *text* at the model's full context, chunking when needed.

    Fits → one ``embed_many([text])`` call, vector returned untouched
    (bit-identical to the pre-chunking behavior).  Does not fit → one
    batched call with every chunk, then a length-weighted mean, then
    L2-normalized.  Never truncates silently; never issues more than one
    embed call.

    Raises whatever the embedder raised (callers fail open to a cache
    miss) and :class:`ModelBackendError` if the backend returns a vector
    count that does not match the chunk count — a pooled vector built
    from misaligned chunks would silently poison the cache.
    """
    ctx = await model_ctx_tokens(model, endpoint=endpoint, transport=transport)
    budget = max(1, ctx - CHUNK_MARGIN_TOKENS)

    if estimate_tokens(text) <= budget:
        vectors = await embed_many([text])
        if len(vectors) != 1:
            raise ModelBackendError(f"embedder returned {len(vectors)} vectors for 1 input")
        return vectors[0]

    chunks = split_overlapping(text, budget_tokens=budget)
    vectors = await embed_many(chunks)
    if len(vectors) != len(chunks):
        raise ModelBackendError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    pooled = length_weighted_mean(vectors, [len(c) for c in chunks])
    return l2_normalize(pooled)


__all__ = [
    "CHUNK_MARGIN_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "CTX_CACHE_TTL_S",
    "FALLBACK_CTX_TOKENS",
    "TOKENS_PER_CHAR",
    "embed_text_dynamic",
    "estimate_tokens",
    "l2_normalize",
    "length_weighted_mean",
    "model_ctx_tokens",
    "split_overlapping",
]
