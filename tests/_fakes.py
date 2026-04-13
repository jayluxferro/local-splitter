"""Shared fake `ChatClient` for tests that don't want real HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from local_splitter.models import ChatResponse, Message, StreamChunk, Usage


class FakeChatClient:
    """In-memory ChatClient that records calls and returns scripted replies.

    Parameters
    ----------
    reply_sequence : list[str] | None
        If given, successive ``complete()`` calls pop replies from this
        list.  When the list is exhausted, ``reply_content`` is used as
        the fallback.  This is useful for T1 tests where the local
        client receives a classifier call first and an answer call
        second.
    raise_sequence : list[Exception | None] | None
        If given, successive ``complete()`` calls pop from this list to
        decide whether to raise.  ``None`` entries mean "don't raise".
        When exhausted, ``raise_on_complete`` is used as fallback.
    """

    def __init__(
        self,
        *,
        chat_model: str = "fake-model",
        reply_content: str = "hello",
        usage: Usage = Usage(input_tokens=10, output_tokens=3),
        raise_on_complete: Exception | None = None,
        reply_sequence: list[str] | None = None,
        raise_sequence: list[Exception | None] | None = None,
        embed_dim: int = 32,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model: str | None = None
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._reply_content = reply_content
        self._usage = usage
        self._raise = raise_on_complete
        self._reply_seq: list[str] = list(reply_sequence) if reply_sequence else []
        self._raise_seq: list[Exception | None] = (
            list(raise_sequence) if raise_sequence else []
        )
        self._embed_dim = embed_dim

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
    ) -> ChatResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "seed": seed,
                "extra": extra,
            }
        )
        exc = self._raise_seq.pop(0) if self._raise_seq else self._raise
        if exc is not None:
            raise exc
        content = self._reply_seq.pop(0) if self._reply_seq else self._reply_content
        return ChatResponse(
            content=content,
            finish_reason="stop",
            usage=self._usage,
            model=self.chat_model,
            raw={"message": {"content": content}},
        )

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        async def gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(delta=self._reply_content, done=True, finish_reason="stop")

        return gen()

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        # Produce deterministic embeddings based on text content so that
        # identical texts get identical vectors (useful for cache tests).
        result: list[list[float]] = []
        for text in texts:
            h = hash(text) & 0xFFFF_FFFF
            vec = [0.0] * self._embed_dim
            # Spread the hash across the first few dims.
            for i in range(min(4, self._embed_dim)):
                vec[i] = float((h >> (i * 8)) & 0xFF) / 255.0
            result.append(vec)
        return result

    async def aclose(self) -> None:
        self.closed = True
