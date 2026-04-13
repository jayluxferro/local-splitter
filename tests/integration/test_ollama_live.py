"""Live integration test against a real Ollama server.

Skipped by default. Opt in with `LOCAL_SPLITTER_OLLAMA_LIVE=1`, and
optionally override:

    LOCAL_SPLITTER_OLLAMA_ENDPOINT   (default http://127.0.0.1:11434)
    LOCAL_SPLITTER_OLLAMA_CHAT       (default llama3.2:3b)
    LOCAL_SPLITTER_OLLAMA_EMBED      (default nomic-embed-text)

Pull the models first (per .agent/memory/gotchas.md) so the first call
isn't a multi-minute download:

    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import os

import pytest

from local_splitter.models import OllamaClient


pytestmark = pytest.mark.skipif(
    os.environ.get("LOCAL_SPLITTER_OLLAMA_LIVE") != "1",
    reason="set LOCAL_SPLITTER_OLLAMA_LIVE=1 to run live Ollama tests",
)


def _endpoint() -> str:
    return os.environ.get("LOCAL_SPLITTER_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")


def _chat_model() -> str:
    return os.environ.get("LOCAL_SPLITTER_OLLAMA_CHAT", "llama3.2:3b")


def _embed_model() -> str:
    return os.environ.get("LOCAL_SPLITTER_OLLAMA_EMBED", "nomic-embed-text")


async def test_live_chat_roundtrip() -> None:
    async with OllamaClient(
        chat_model=_chat_model(),
        endpoint=_endpoint(),
    ) as c:
        # Warmup: ignored by the assertions, exists to surface cold-start.
        await c.complete(
            [{"role": "user", "content": "say the single word: ready"}],
            temperature=0.0,
            max_tokens=8,
            seed=1,
        )
        reply = await c.complete(
            [{"role": "user", "content": "reply with the single word: pong"}],
            temperature=0.0,
            max_tokens=8,
            seed=1,
        )

    assert reply.content.strip() != ""
    assert reply.usage.input_tokens is not None and reply.usage.input_tokens > 0
    assert reply.usage.output_tokens is not None and reply.usage.output_tokens > 0
    assert reply.finish_reason in {"stop", "length"}


async def test_live_embed_roundtrip() -> None:
    async with OllamaClient(
        chat_model=_chat_model(),
        embed_model=_embed_model(),
        endpoint=_endpoint(),
    ) as c:
        vecs = await c.embed(["hello world", "a second sentence"])

    assert len(vecs) == 2
    assert len(vecs[0]) > 0
    assert len(vecs[0]) == len(vecs[1])
    assert all(isinstance(x, float) for x in vecs[0])


async def test_live_stream_roundtrip() -> None:
    async with OllamaClient(
        chat_model=_chat_model(),
        endpoint=_endpoint(),
    ) as c:
        chunks = []
        async for ch in await c.stream(
            [{"role": "user", "content": "count to three separated by commas"}],
            temperature=0.0,
            max_tokens=32,
            seed=1,
        ):
            chunks.append(ch)

    assert chunks, "expected at least one chunk"
    assert chunks[-1].done is True
    full = "".join(ch.delta for ch in chunks)
    assert full.strip() != ""
