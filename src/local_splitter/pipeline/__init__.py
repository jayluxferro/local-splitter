"""Pipeline orchestrator — all seven tactics are wired in.

Pipeline order (ARCHITECTURE.md):

    T1 route → T3 sem_cache → T2 compress → T6 intent →
    T5 diff → T7 batch → T4 draft (or direct cloud call)

Each tactic is independently togglable via ``Config.tactics``.  The
orchestrator calls enabled stages in order, emits ``StageEvent``\\ s
for each, and returns a normalized ``PipelineResponse``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field

from local_splitter.config import Config
from local_splitter.models import ChatClient, ModelBackendError, Usage

from . import batch as _batch
from . import compress as _compress
from . import diff as _diff
from . import draft as _draft
from . import intent as _intent
from . import route as _route
from . import sem_cache as _sem_cache
from .types import (
    ModelHint,
    PipelineRequest,
    PipelineResponse,
    ServedBy,
    StageEvent,
    StatsSnapshot,
)

_log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot serve a request (e.g. no backend)."""


_LATENCY_WINDOW = 4096  # keep a bounded tail for p50/p99


@dataclass(slots=True)
class _Stats:
    """Mutable metrics accumulator kept inside the Pipeline instance.

    Latencies are stored in a bounded ring buffer (last ~4096 requests)
    to keep p50/p99 computation O(log n) without unbounded memory.
    """

    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    by_served: Counter[str] = field(default_factory=Counter)
    tokens_in_cloud: int = 0
    tokens_out_cloud: int = 0
    tokens_in_local: int = 0
    tokens_out_local: int = 0
    _latencies: list[float] = field(default_factory=list)
    _latency_cursor: int = 0

    def record(self, resp: PipelineResponse) -> None:
        self.total_requests += 1
        self.by_served[resp.served_by] += 1
        self.tokens_in_cloud += resp.usage_cloud.input_tokens or 0
        self.tokens_out_cloud += resp.usage_cloud.output_tokens or 0
        self.tokens_in_local += resp.usage_local.input_tokens or 0
        self.tokens_out_local += resp.usage_local.output_tokens or 0

        if len(self._latencies) < _LATENCY_WINDOW:
            self._latencies.append(resp.latency_ms)
        else:
            self._latencies[self._latency_cursor] = resp.latency_ms
            self._latency_cursor = (self._latency_cursor + 1) % _LATENCY_WINDOW

    def snapshot(self) -> StatsSnapshot:
        p50: float | None = None
        p99: float | None = None
        if self._latencies:
            sorted_l = sorted(self._latencies)
            p50 = _percentile(sorted_l, 0.50)
            p99 = _percentile(sorted_l, 0.99)

        return StatsSnapshot(
            started_at=self.started_at,
            total_requests=self.total_requests,
            by_served=dict(self.by_served),
            tokens_in_cloud=self.tokens_in_cloud,
            tokens_out_cloud=self.tokens_out_cloud,
            tokens_in_local=self.tokens_in_local,
            tokens_out_local=self.tokens_out_local,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
        )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. Not fancy, but reproducible."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    idx = min(n - 1, max(0, round(q * (n - 1))))
    return sorted_values[idx]


class Pipeline:
    """The single orchestrator both transports feed into.

    Holds references to the configured chat clients (local is optional),
    an optional semantic cache store, and runs the enabled tactics in
    order.
    """

    def __init__(
        self,
        *,
        cloud: ChatClient,
        local: ChatClient | None,
        config: Config,
        cache_store: _sem_cache.CacheStore | None = None,
    ) -> None:
        self.cloud = cloud
        self.local = local
        self.config = config
        self.cache_store = cache_store
        self._stats = _Stats()

    async def aclose(self) -> None:
        if self.local is not None:
            await self.local.aclose()
        await self.cloud.aclose()

    async def complete(self, request: PipelineRequest) -> PipelineResponse:
        """Run the pipeline end-to-end.

        Enabled stages run in ARCHITECTURE.md order:

        1. **T1 route** — classify TRIVIAL / COMPLEX (``auto`` only).
        2. **T3 sem_cache** — serve from cache on hit (``auto`` only).
        3. Cloud call — the actual backend request.
        4. **T3 store** — write the cloud response to cache on miss.

        Explicit ``model_hint`` values ("local" / "cloud") bypass T1+T3.
        """
        t_start = time.perf_counter()
        trace: list[StageEvent] = []
        cache_embedding: list[float] | None = None  # kept for T3 store-on-miss

        auto_route = request.model_hint == "auto"
        has_local = self.local is not None

        # --- T1 route (only for auto-routed requests) ---
        if self.config.tactics.t1_route and has_local and auto_route:
            route_result = await _route.apply(
                request.messages,
                local=self.local,
                params=self.config.tactics.params.get("t1_route"),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
                seed=request.seed,
                extra=request.extra,
            )
            trace.extend(route_result.events)

            if route_result.local_reply is not None:
                reply = route_result.local_reply
                resp = PipelineResponse(
                    content=reply.content,
                    finish_reason=reply.finish_reason,
                    served_by="local",
                    model=reply.model,
                    usage_local=reply.usage,
                    usage_cloud=Usage(),
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                    trace=trace,
                    raw=reply.raw,
                )
                self._stats.record(resp)
                return resp
            # COMPLEX — fall through to T3 / cloud.

        # --- T3 sem_cache lookup (auto requests only) ---
        t3_active = (
            self.config.tactics.t3_sem_cache
            and has_local
            and self.cache_store is not None
            and auto_route
        )
        if t3_active:
            cache_result = await _sem_cache.lookup(
                request.messages,
                local=self.local,  # type: ignore[arg-type]
                store=self.cache_store,  # type: ignore[arg-type]
                params=self.config.tactics.params.get("t3_sem_cache"),
            )
            trace.extend(cache_result.events)
            cache_embedding = cache_result.embedding

            if cache_result.hit and cache_result.entry is not None:
                entry = cache_result.entry
                resp = PipelineResponse(
                    content=entry.response,
                    finish_reason=entry.finish_reason,  # type: ignore[arg-type]
                    served_by="cache",
                    model=entry.model,
                    usage_local=Usage(),  # embed cost is negligible
                    usage_cloud=Usage(),
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                    trace=trace,
                    raw={},
                )
                self._stats.record(resp)
                return resp

        # --- T2 compress (auto requests, after T3 miss) ---
        messages_for_backend = request.messages
        if self.config.tactics.t2_compress and has_local and auto_route:
            compress_result = await _compress.apply(
                messages_for_backend,
                local=self.local,  # type: ignore[arg-type]
                params=self.config.tactics.params.get("t2_compress"),
            )
            trace.extend(compress_result.events)
            messages_for_backend = compress_result.messages

        # --- T6 intent (auto requests, after T2) ---
        if self.config.tactics.t6_intent and has_local and auto_route:
            intent_result = await _intent.apply(
                messages_for_backend,
                local=self.local,  # type: ignore[arg-type]
                params=self.config.tactics.params.get("t6_intent"),
            )
            trace.extend(intent_result.events)
            messages_for_backend = intent_result.messages

        # --- T5 diff (auto requests, after T6) ---
        if self.config.tactics.t5_diff and has_local and auto_route:
            diff_result = await _diff.apply(
                messages_for_backend,
                local=self.local,  # type: ignore[arg-type]
                params=self.config.tactics.params.get("t5_diff"),
            )
            trace.extend(diff_result.events)
            messages_for_backend = diff_result.messages

        # --- T7 batch / prompt-cache tagging (auto requests, last transform) ---
        if self.config.tactics.t7_batch and auto_route:
            batch_result = _batch.apply(
                messages_for_backend,
                params=self.config.tactics.params.get("t7_batch"),
            )
            trace.extend(batch_result.events)
            messages_for_backend = batch_result.messages

        # --- T4 draft-review (auto requests, replaces direct cloud call) ---
        if (
            self.config.tactics.t4_draft
            and has_local
            and auto_route
        ):
            draft_result = await _draft.apply(
                messages_for_backend,
                local=self.local,  # type: ignore[arg-type]
                cloud=self.cloud,
                params=self.config.tactics.params.get("t4_draft"),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
                extra=request.extra,
            )
            if draft_result is not None:
                trace.extend(draft_result.events)
                reply = draft_result.review

                # T3 store: cache the final answer (draft or revised).
                if t3_active and cache_embedding is not None and self.cache_store is not None:
                    store_event = _sem_cache.store_response(
                        cache_embedding,
                        response=reply.content,
                        model=reply.model,
                        finish_reason=reply.finish_reason,
                        cache_store=self.cache_store,
                    )
                    trace.append(store_event)

                # Usage: local draft + cloud review.
                draft_ev = draft_result.events[0]
                usage_local = Usage(
                    input_tokens=draft_ev.tokens_in,
                    output_tokens=draft_ev.tokens_out,
                )
                resp = PipelineResponse(
                    content=reply.content,
                    finish_reason=reply.finish_reason,
                    served_by="draft+cloud",
                    model=reply.model,
                    usage_local=usage_local,
                    usage_cloud=reply.usage,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                    trace=trace,
                    raw=reply.raw,
                )
                self._stats.record(resp)
                return resp
            # draft_result is None → local draft failed, fall through to
            # direct cloud call below.

        # --- direct backend call (cloud passthrough or explicit hint) ---
        client, served_by, stage_name = self._choose_backend(request.model_hint)

        stage_start = time.perf_counter()
        try:
            reply = await client.complete(
                messages_for_backend,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
                seed=request.seed,
                extra=request.extra,
            )
        except ModelBackendError as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            trace.append(
                StageEvent(
                    stage=stage_name,
                    decision="ERROR",
                    ms=elapsed,
                    detail={"error": str(e)},
                )
            )
            raise

        trace.append(
            StageEvent(
                stage=stage_name,
                decision="APPLIED",
                ms=(time.perf_counter() - stage_start) * 1000,
                tokens_in=reply.usage.input_tokens,
                tokens_out=reply.usage.output_tokens,
            )
        )

        # --- T3 store on miss ---
        if t3_active and cache_embedding is not None and self.cache_store is not None:
            store_event = _sem_cache.store_response(
                cache_embedding,
                response=reply.content,
                model=reply.model,
                finish_reason=reply.finish_reason,
                cache_store=self.cache_store,
            )
            trace.append(store_event)

        if served_by == "cloud":
            usage_cloud, usage_local = reply.usage, Usage()
        else:
            usage_cloud, usage_local = Usage(), reply.usage

        resp = PipelineResponse(
            content=reply.content,
            finish_reason=reply.finish_reason,
            served_by=served_by,
            model=reply.model,
            usage_local=usage_local,
            usage_cloud=usage_cloud,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            trace=trace,
            raw=reply.raw,
        )
        self._stats.record(resp)
        return resp

    def _choose_backend(
        self, hint: ModelHint
    ) -> tuple[ChatClient, ServedBy, str]:
        """Pick a backend given a model_hint.

        When T1 is enabled ``auto`` requests are routed by the T1 stage
        *before* reaching this method.  If we get here with ``auto`` it
        means T1 is disabled or classified the request as COMPLEX, so we
        fall through to the cloud.
        """
        if hint == "local":
            if self.local is None:
                raise PipelineError(
                    "model_hint=local but no local backend is configured"
                )
            return self.local, "local", "local_call"
        # auto or cloud
        return self.cloud, "cloud", "cloud_call"

    def stats(self) -> StatsSnapshot:
        return self._stats.snapshot()


__all__ = [
    "Pipeline",
    "PipelineError",
    "PipelineRequest",
    "PipelineResponse",
    "StageEvent",
    "StatsSnapshot",
]
