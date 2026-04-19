"""Shared pre-cloud stages: T1 → T3 lookup → T2 → T6 → T5 → T7.

Used by ``Pipeline.complete``, ``Pipeline.transform``, and ``Pipeline.stream``
so ordering and fail-open behaviour stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_splitter.config import TacticsConfig
from local_splitter.models import ChatClient, ChatResponse, Message

from . import batch as _batch
from . import compress as _compress
from . import diff as _diff
from . import intent as _intent
from . import route as _route
from . import sem_cache as _sem_cache
from .types import PipelineRequest, StageEvent
from .sem_cache import CacheEntry


@dataclass(slots=True)
class PreCloudResult:
    """Outcome after pre-cloud transforms (before cloud / T4 / cache store)."""

    trace: list[StageEvent]
    messages: list[Message]
    cache_embedding: list[float] | None
    t3_active: bool
    t1_local_reply: ChatResponse | None
    t3_cache_entry: CacheEntry | None


def _auto_route(request: PipelineRequest) -> bool:
    return request.model_hint == "auto"


async def run_pre_cloud(
    request: PipelineRequest,
    *,
    tactics: TacticsConfig,
    local: ChatClient | None,
    cache_store: _sem_cache.CacheStore | None,
) -> PreCloudResult:
    """Run T1, T3 miss path, T2, T6, T5, T7. Short-circuit fields set when applicable."""
    trace: list[StageEvent] = []
    auto_route = _auto_route(request)
    if not auto_route:
        return PreCloudResult(
            trace=[],
            messages=list(request.messages),
            cache_embedding=None,
            t3_active=False,
            t1_local_reply=None,
            t3_cache_entry=None,
        )

    has_local = local is not None

    t1_local: ChatResponse | None = None
    t3_entry: CacheEntry | None = None
    cache_embedding: list[float] | None = None

    # --- T1 ---
    if tactics.t1_route and has_local and auto_route:
        route_result = await _route.apply(
            request.messages,
            local=local,
            params=tactics.params.get("t1_route"),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
            extra=request.extra,
            meta=request.meta,
        )
        trace.extend(route_result.events)
        if route_result.local_reply is not None:
            t1_local = route_result.local_reply
            return PreCloudResult(
                trace=trace,
                messages=request.messages,
                cache_embedding=None,
                t3_active=False,
                t1_local_reply=t1_local,
                t3_cache_entry=None,
            )

    t3_active = (
        tactics.t3_sem_cache
        and has_local
        and cache_store is not None
        and auto_route
    )

    # --- T3 lookup ---
    if t3_active:
        cache_result = await _sem_cache.lookup(
            request.messages,
            local=local,
            store=cache_store,
            params=tactics.params.get("t3_sem_cache"),
            meta=request.meta,
        )
        trace.extend(cache_result.events)
        cache_embedding = cache_result.embedding
        if cache_result.hit and cache_result.entry is not None:
            t3_entry = cache_result.entry
            return PreCloudResult(
                trace=trace,
                messages=request.messages,
                cache_embedding=cache_embedding,
                t3_active=t3_active,
                t1_local_reply=None,
                t3_cache_entry=t3_entry,
            )

    messages_for_backend: list[Message] = list(request.messages)

    if tactics.t2_compress and has_local and auto_route:
        compress_result = await _compress.apply(
            messages_for_backend,
            local=local,
            params=tactics.params.get("t2_compress"),
        )
        trace.extend(compress_result.events)
        messages_for_backend = compress_result.messages

    if tactics.t6_intent and has_local and auto_route:
        intent_result = await _intent.apply(
            messages_for_backend,
            local=local,
            params=tactics.params.get("t6_intent"),
        )
        trace.extend(intent_result.events)
        messages_for_backend = intent_result.messages

    if tactics.t5_diff and has_local and auto_route:
        diff_result = await _diff.apply(
            messages_for_backend,
            local=local,
            params=tactics.params.get("t5_diff"),
        )
        trace.extend(diff_result.events)
        messages_for_backend = diff_result.messages

    if tactics.t7_batch and auto_route:
        batch_result = _batch.apply(
            messages_for_backend,
            params=tactics.params.get("t7_batch"),
        )
        trace.extend(batch_result.events)
        messages_for_backend = batch_result.messages

    return PreCloudResult(
        trace=trace,
        messages=messages_for_backend,
        cache_embedding=cache_embedding,
        t3_active=t3_active,
        t1_local_reply=None,
        t3_cache_entry=None,
    )


__all__ = ["PreCloudResult", "run_pre_cloud"]
