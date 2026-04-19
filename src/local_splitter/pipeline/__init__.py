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

from collections.abc import AsyncIterator

from local_splitter.config import AdaptiveConfig, Config, apply_tactics_override
from local_splitter.models import ChatClient, ModelBackendError, StreamChunk, Usage

from . import compress as _compress
from . import draft as _draft
from . import sem_cache as _sem_cache
from .pre_cloud import run_pre_cloud
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

    def _latency_values_chronological(self) -> list[float]:
        if len(self._latencies) < _LATENCY_WINDOW:
            return list(self._latencies)
        i = self._latency_cursor
        return self._latencies[i:] + self._latencies[:i]

    def snapshot(self, adaptive: AdaptiveConfig | None = None) -> StatsSnapshot:
        p50: float | None = None
        p99: float | None = None
        sample_size = 0
        raw = self._latency_values_chronological()
        n = len(raw)
        if n:
            if n <= 512:
                sorted_l = sorted(raw)
            else:
                step = n / 512.0
                sorted_l = sorted(raw[int(i * step) % n] for i in range(512))
            sample_size = len(sorted_l)
            p50 = _percentile(sorted_l, 0.50)
            p99 = _percentile(sorted_l, 0.99)

        hints: list[str] = []
        if (
            adaptive
            and adaptive.enabled
            and self.total_requests >= adaptive.min_requests
        ):
            localish = self.by_served.get("local", 0) + self.by_served.get("cache", 0)
            frac = localish / max(1, self.total_requests)
            if frac > adaptive.max_local_fraction:
                hints.append(adaptive.hint)

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
            latency_sample_size=sample_size,
            adaptive_hints=tuple(hints),
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
        cloud: ChatClient | None = None,
        local: ChatClient | None = None,
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
        if self.cloud is not None:
            await self.cloud.aclose()
        if self.cache_store is not None:
            self.cache_store.close()

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
        tac = apply_tactics_override(self.config.tactics, request.tactics_override)
        pc = await run_pre_cloud(
            request,
            tactics=tac,
            local=self.local,
            cache_store=self.cache_store,
        )
        trace = list(pc.trace)

        if pc.t1_local_reply is not None:
            reply = pc.t1_local_reply
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

        if pc.t3_cache_entry is not None:
            entry = pc.t3_cache_entry
            resp = PipelineResponse(
                content=entry.response,
                finish_reason=entry.finish_reason,  # type: ignore[arg-type]
                served_by="cache",
                model=entry.model,
                usage_local=Usage(),
                usage_cloud=Usage(),
                latency_ms=(time.perf_counter() - t_start) * 1000,
                trace=trace,
                raw={},
            )
            self._stats.record(resp)
            return resp

        messages_for_backend = pc.messages
        cache_embedding = pc.cache_embedding
        t3_active = pc.t3_active
        has_local = self.local is not None
        t3_params = tac.params.get("t3_sem_cache") or {}
        meta_dict = dict(request.meta)
        cache_key_text = _sem_cache.cache_embed_source(
            request.messages, t3_params, meta_dict
        )

        # --- T4 draft-review (auto requests, replaces direct cloud call) ---
        if (
            tac.t4_draft
            and has_local
            and request.model_hint == "auto"
        ):
            draft_result = await _draft.apply(
                messages_for_backend,
                local=self.local,  # type: ignore[arg-type]
                cloud=self.cloud,
                params=tac.params.get("t4_draft"),
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
                        params=t3_params,
                        meta=meta_dict,
                        cache_text=cache_key_text,
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
                upstream_headers=request.upstream_headers or None,
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
                params=t3_params,
                meta=meta_dict,
                cache_text=cache_key_text,
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

    async def transform(
        self, request: PipelineRequest
    ) -> tuple[list[dict], list[StageEvent], str | None]:
        """Run tactic transforms without calling any backend.

        Returns ``(messages, trace, local_response)``. If T1 routes
        locally or T3 hits cache, ``local_response`` contains the
        answer and the caller can use it directly. Otherwise
        ``local_response`` is ``None`` and ``messages`` contains the
        transformed prompt for the caller's own model.

        This is the core of local-only MCP mode: the agent calls
        ``split.transform``, gets back either a ready answer or a
        leaner prompt to send to its own cloud model.
        """
        tac = apply_tactics_override(self.config.tactics, request.tactics_override)
        pc = await run_pre_cloud(
            request,
            tactics=tac,
            local=self.local,
            cache_store=self.cache_store,
        )
        trace = list(pc.trace)
        if pc.t1_local_reply is not None:
            return request.messages, trace, pc.t1_local_reply.content
        if pc.t3_cache_entry is not None:
            return request.messages, trace, pc.t3_cache_entry.response
        return pc.messages, trace, None

    async def stream(
        self, request: PipelineRequest
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-cloud transforms, then stream the cloud response.

        Tactics T1 (route-local), T3 (cache hit), and T4 (draft-review)
        short-circuit and cannot stream — they fall back to a single
        non-streaming chunk.  All other tactics transform the messages
        before handing off to the cloud backend's ``stream()`` method.
        """
        # Run the synchronous pipeline path first.  If T1 routes locally
        # or T3 hits cache, yield the full answer as a single chunk.
        tac = apply_tactics_override(self.config.tactics, request.tactics_override)
        pc = await run_pre_cloud(
            request,
            tactics=tac,
            local=self.local,
            cache_store=self.cache_store,
        )
        if pc.t1_local_reply is not None:
            lr = pc.t1_local_reply
            yield StreamChunk(
                delta=lr.content,
                done=True,
                finish_reason=lr.finish_reason,
                usage=lr.usage,
            )
            return

        if pc.t3_cache_entry is not None:
            ent = pc.t3_cache_entry
            yield StreamChunk(
                delta=ent.response,
                done=True,
                finish_reason=ent.finish_reason,
            )
            return

        messages_for_backend = pc.messages

        # --- Stream from cloud (skip T4 draft in streaming mode) ---
        client = self.cloud if request.model_hint != "local" else self.local
        if client is None:
            raise PipelineError("no backend available for streaming")

        chunks = await client.stream(
            messages_for_backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
            extra=request.extra,
            upstream_headers=request.upstream_headers or None,
        )
        async for chunk in chunks:
            yield chunk

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
        if self.cloud is None:
            raise PipelineError(
                "no cloud backend configured; use split.transform for "
                "local-only mode or configure a cloud backend"
            )
        return self.cloud, "cloud", "cloud_call"

    async def compress_messages_only(
        self,
        messages: list[dict[str, str]],
        *,
        tactics_override: frozenset[str] | None = None,
    ) -> tuple[list[dict[str, str]], list[StageEvent]]:
        """Run **T2 compress** only (for tool-bearing HTTP requests).

        Returns the original messages unchanged when T2 is disabled or
        there is no local backend.
        """
        tac = apply_tactics_override(self.config.tactics, tactics_override)
        if not tac.t2_compress or self.local is None:
            return messages, []
        r = await _compress.apply(
            list(messages),
            local=self.local,
            params=tac.params.get("t2_compress"),
        )
        return r.messages, r.events

    def stats(self) -> StatsSnapshot:
        return self._stats.snapshot(self.config.adaptive)


__all__ = [
    "Pipeline",
    "PipelineError",
    "PipelineRequest",
    "PipelineResponse",
    "StageEvent",
    "StatsSnapshot",
]
