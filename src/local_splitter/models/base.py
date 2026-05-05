"""Common types and the `ChatClient` protocol that backends implement.

Two backends satisfy this protocol: `OllamaClient` and `OpenAICompatClient`.
The pipeline layer depends only on this protocol — never on a concrete
backend — so tactics stay backend-agnostic.

Design notes
------------
- Async-first. All I/O is `httpx.AsyncClient`.
- `ChatResponse.raw` keeps the decoded JSON so callers can read
  backend-specific fields without the protocol having to know about them
  (see gotchas.md: "Not all OpenAI-compatible endpoints are equal").
- Token counts on `Usage` are reported as the backend reports them. The
  cloud-side counts are authoritative for billing (gotchas.md).
- `stream()` yields `StreamChunk`s ending with `done=True`; the final
  chunk carries `finish_reason` and `usage` when the backend surfaces
  them. Intermediate chunks carry content deltas only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    """One chat message. Matches the OpenAI/Ollama shape."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(slots=True, frozen=True)
class Usage:
    """Token counts for a single call.

    `input_tokens` covers prompt / context tokens the backend billed for;
    `output_tokens` covers generated tokens. Either may be `None` when the
    backend doesn't report it (some OpenAI-compat endpoints omit usage on
    streaming calls).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "unknown"]


@dataclass(slots=True)
class ChatResponse:
    """Normalized response from a non-streaming chat call."""

    content: str
    finish_reason: FinishReason
    usage: Usage
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamChunk:
    """One streamed piece of a chat response.

    `delta` is the incremental content since the previous chunk (may be
    empty). `done` is `True` on the final chunk only; that chunk carries
    `finish_reason` and `usage` when available.
    """

    delta: str
    done: bool = False
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ModelBackendError(RuntimeError):
    """Raised when a backend returns a non-2xx response or malformed body.

    Tactics that depend on the local model must catch this and fail open
    (pass the request through unchanged) per ARCHITECTURE.md principle 2.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@runtime_checkable
class ChatClient(Protocol):
    """Minimum interface every model backend must satisfy.

    Implementations hold their own `httpx.AsyncClient` and must be closed
    via `aclose()` at shutdown. Keyword-only params map 1:1 onto the
    OpenAI chat schema; unknown params go through `extra`.
    """

    chat_model: str
    embed_model: str | None

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
        upstream_headers: Mapping[str, str] | None = None,
    ) -> ChatResponse: ...

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
        upstream_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


__all__ = [
    "ChatClient",
    "ChatResponse",
    "FinishReason",
    "Message",
    "ModelBackendError",
    "StreamChunk",
    "Usage",
]
