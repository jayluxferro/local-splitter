"""Unit tests for `local_splitter.models.ollama`.

Uses `httpx.MockTransport` so nothing hits the network. We assert the
on-the-wire request shape as strongly as we assert the parsed response —
that's the contract with Ollama and the most likely thing to regress.
"""

from __future__ import annotations

import json

import httpx
import pytest

from local_splitter.models import ModelBackendError, OllamaClient
from local_splitter.models.ollama import DEFAULT_NUM_CTX


def _json_handler(
    expect_path: str,
    response_body: dict | list,
    *,
    captured: list[httpx.Request],
    status: int = 200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert request.url.path == expect_path
        return httpx.Response(status, json=response_body)

    return handler


async def test_complete_happy_path_sets_num_ctx_and_parses_usage() -> None:
    captured: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _json_handler(
            "/api/chat",
            {
                "model": "llama3.2:3b",
                "message": {"role": "assistant", "content": "hi there"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 42,
                "eval_count": 7,
            },
            captured=captured,
        )
    )

    async with OllamaClient(
        chat_model="llama3.2:3b", transport=transport
    ) as client:
        reply = await client.complete(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=16,
            seed=1,
        )

    assert reply.content == "hi there"
    assert reply.finish_reason == "stop"
    assert reply.usage.input_tokens == 42
    assert reply.usage.output_tokens == 7
    assert reply.usage.total == 49
    assert reply.model == "llama3.2:3b"

    # Verify the outbound body — this is the hard gotcha.
    assert len(captured) == 1
    body = json.loads(captured[0].content)
    assert body["model"] == "llama3.2:3b"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is False
    # num_ctx MUST always be present to avoid silent truncation.
    assert body["options"]["num_ctx"] == DEFAULT_NUM_CTX
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 16
    assert body["options"]["seed"] == 1


async def test_complete_caller_can_override_num_ctx_via_extra() -> None:
    captured: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _json_handler(
            "/api/chat",
            {
                "model": "m",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
            },
            captured=captured,
        )
    )
    async with OllamaClient(chat_model="m", transport=transport) as c:
        await c.complete(
            [{"role": "user", "content": "x"}],
            extra={"options": {"num_ctx": 32768, "mirostat": 2}},
        )
    body = json.loads(captured[0].content)
    assert body["options"]["num_ctx"] == 32768
    assert body["options"]["mirostat"] == 2


async def test_complete_raises_on_non_200() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(500, text="boom")
    )
    async with OllamaClient(chat_model="m", transport=transport) as c:
        with pytest.raises(ModelBackendError, match="500"):
            await c.complete([{"role": "user", "content": "x"}])


async def test_complete_raises_on_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async with OllamaClient(chat_model="m", transport=transport) as c:
        with pytest.raises(ModelBackendError, match="ollama chat request failed"):
            await c.complete([{"role": "user", "content": "x"}])


async def test_complete_warns_on_length_finish(caplog: pytest.LogCaptureFixture) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "truncated"},
                "done": True,
                "done_reason": "length",
            },
        )
    )
    async with OllamaClient(chat_model="m", transport=transport) as c:
        with caplog.at_level("WARNING"):
            reply = await c.complete([{"role": "user", "content": "x"}])
    assert reply.finish_reason == "length"
    assert any("finish_reason=length" in r.message for r in caplog.records)


async def test_stream_yields_deltas_then_done_with_usage() -> None:
    lines = [
        json.dumps({"message": {"content": "hel"}, "done": False}),
        json.dumps({"message": {"content": "lo "}, "done": False}),
        json.dumps(
            {
                "message": {"content": "world"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 3,
            }
        ),
    ]
    body_bytes = ("\n".join(lines) + "\n").encode()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_bytes)

    transport = httpx.MockTransport(handler)
    async with OllamaClient(chat_model="m", transport=transport) as c:
        chunks = [chunk async for chunk in await c.stream([{"role": "user", "content": "hi"}])]

    assert [c.delta for c in chunks] == ["hel", "lo ", "world"]
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens == 10
    assert chunks[-1].usage.output_tokens == 3
    assert all(not c.done for c in chunks[:-1])


async def test_embed_batches_inputs_and_returns_float_lists() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2] for _ in payload["input"]]},
        )

    transport = httpx.MockTransport(handler)
    async with OllamaClient(
        chat_model="m", embed_model="nomic-embed-text", transport=transport
    ) as c:
        vecs = await c.embed(["a", "b", "c"])

    assert len(vecs) == 3
    assert all(len(v) == 2 for v in vecs)
    body = json.loads(captured[0].content)
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["a", "b", "c"]


async def test_embed_requires_embed_model() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with OllamaClient(chat_model="m", transport=transport) as c:
        with pytest.raises(ModelBackendError, match="embed_model"):
            await c.embed(["x"])


async def test_embed_empty_input_returns_empty_without_call() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={"embeddings": []})

    transport = httpx.MockTransport(handler)
    async with OllamaClient(
        chat_model="m", embed_model="e", transport=transport
    ) as c:
        assert await c.embed([]) == []
    assert calls == []
