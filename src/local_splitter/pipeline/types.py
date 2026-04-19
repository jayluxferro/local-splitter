"""Data types shared between the pipeline, both transports, and eval.

These dataclasses are the internal currency of the system. The transport
layer (MCP + HTTP proxy) translates external shapes (MCP tool calls /
OpenAI chat completion JSON) into `PipelineRequest` and translates
`PipelineResponse` back out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from local_splitter.models import FinishReason, Message, Usage

ModelHint = Literal["auto", "local", "cloud"]
ServedBy = Literal["local", "cache", "cloud", "draft+cloud"]


@dataclass(slots=True)
class PipelineRequest:
    """A request entering the pipeline. Backend-agnostic."""

    messages: list[Message]
    model_hint: ModelHint = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    stop: Sequence[str] | None = None
    seed: int | None = None
    stream: bool = False
    extra: Mapping[str, Any] | None = None
    # Headers to forward to the upstream backend (e.g. auth tokens).
    upstream_headers: dict[str, str] = field(default_factory=dict)
    # Caller metadata: tool_name, session_id, tag, ...
    meta: dict[str, Any] = field(default_factory=dict)
    # Per-call: frozenset of tactic keys to *disable* for this request only
    # (e.g. frozenset({"t2_compress"})); see ``apply_tactics_override`` in config.
    tactics_override: frozenset[str] | None = None


@dataclass(slots=True)
class StageEvent:
    """One pipeline stage's observable result (the unit of measurement).

    Every tactic must emit exactly one of these so the evaluation harness
    can attribute tokens / latency to specific mechanisms.
    """

    stage: str
    decision: str
    ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stage": self.stage,
            "decision": self.decision,
            "ms": round(self.ms, 3),
        }
        if self.tokens_in is not None:
            out["tokens_in"] = self.tokens_in
        if self.tokens_out is not None:
            out["tokens_out"] = self.tokens_out
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(slots=True)
class PipelineResponse:
    """The final result after all enabled tactics have run."""

    content: str
    finish_reason: FinishReason
    served_by: ServedBy
    model: str
    usage_local: Usage
    usage_cloud: Usage
    latency_ms: float
    trace: list[StageEvent]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StatsSnapshot:
    """Aggregate metrics since process start. Shape matches /v1/splitter/stats."""

    started_at: float
    total_requests: int
    by_served: dict[str, int]
    tokens_in_cloud: int
    tokens_out_cloud: int
    tokens_in_local: int
    tokens_out_local: int
    p50_latency_ms: float | None
    p99_latency_ms: float | None
    # How many latency samples were used for p50/p99 (may be subsampled).
    latency_sample_size: int = 0
    # Optional hints when ``Config.adaptive`` is enabled (e.g. high local rate).
    adaptive_hints: tuple[str, ...] = ()


__all__ = [
    "ModelHint",
    "PipelineRequest",
    "PipelineResponse",
    "ServedBy",
    "StageEvent",
    "StatsSnapshot",
]
