"""Unit tests for `local_splitter.models.openai_compat`."""

from __future__ import annotations

import json

import httpx
import pytest

from local_splitter.models import ModelBackendError, OpenAICompatClient


BASE = "https://example.invalid/v1"


def _static(response: httpx.Response, captured: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return response

    return handler


async def test_api_key_env_is_resolved_and_sent_as_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-xyz")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    async with OpenAICompatClient(
        chat_model="gpt-4o-mini",
        endpoint=BASE,
        api_key_env="TEST_API_KEY",
        transport=transport,
    ) as c:
        reply = await c.complete([{"role": "user", "content": "hi"}])

    assert captured[0].headers["authorization"] == "Bearer sk-xyz"
    assert reply.content == "hello"
    assert reply.usage.input_tokens == 3
    assert reply.usage.output_tokens == 1
    assert reply.model == "gpt-4o-mini"


async def test_missing_api_key_env_raises() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ModelBackendError, match="unset"):
        OpenAICompatClient(
            chat_model="m",
            endpoint=BASE,
            api_key_env="DEFINITELY_NOT_SET_XYZ",
            transport=transport,
        )


async def test_both_api_key_and_env_rejected() -> None:
    with pytest.raises(ValueError, match="either"):
        OpenAICompatClient(
            chat_model="m", endpoint=BASE, api_key="k", api_key_env="E"
        )


async def test_no_auth_is_allowed_for_local_servers() -> None:
    captured: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _static(
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
            captured,
        )
    )
    async with OpenAICompatClient(
        chat_model="local", endpoint=BASE, transport=transport
    ) as c:
        await c.complete([{"role": "user", "content": "x"}])
    assert "authorization" not in captured[0].headers


async def test_complete_warns_on_length_finish(caplog: pytest.LogCaptureFixture) -> None:
    transport = httpx.MockTransport(
        _static(
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "cut"},
                            "finish_reason": "length",
                        }
                    ],
                },
            )
        )
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        with caplog.at_level("WARNING"):
            reply = await c.complete([{"role": "user", "content": "x"}])
    assert reply.finish_reason == "length"
    assert any("finish_reason=length" in r.message for r in caplog.records)


async def test_complete_defensive_on_missing_usage() -> None:
    # together.ai / some providers may omit usage entirely. Don't crash.
    transport = httpx.MockTransport(
        _static(
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        reply = await c.complete([{"role": "user", "content": "x"}])
    assert reply.usage.input_tokens is None
    assert reply.usage.output_tokens is None
    assert reply.usage.total is None


async def test_complete_raises_on_empty_choices() -> None:
    transport = httpx.MockTransport(
        _static(httpx.Response(200, json={"choices": []}))
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        with pytest.raises(ModelBackendError, match="no choices"):
            await c.complete([{"role": "user", "content": "x"}])


async def test_complete_raises_on_non_200() -> None:
    transport = httpx.MockTransport(
        _static(httpx.Response(429, text="rate limited"))
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        with pytest.raises(ModelBackendError, match="429"):
            await c.complete([{"role": "user", "content": "x"}])


async def test_stream_parses_sse_deltas_and_done_marker() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"hel"}}]}\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n'
        'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2}}\n'
        "data: [DONE]\n"
    )
    transport = httpx.MockTransport(
        _static(httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"}))
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        chunks = [ch async for ch in await c.stream([{"role": "user", "content": "hi"}])]

    deltas = [ch.delta for ch in chunks if ch.delta]
    assert "".join(deltas) == "hello!"
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens == 5
    assert chunks[-1].usage.output_tokens == 2


async def test_stream_handles_missing_done_marker() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n'
    )
    transport = httpx.MockTransport(
        _static(httpx.Response(200, content=sse.encode()))
    )
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        chunks = [ch async for ch in await c.stream([{"role": "user", "content": "hi"}])]
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"


async def test_embed_preserves_server_index_ordering() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        payload = json.loads(request.content)
        # Return out-of-order on purpose to test reordering.
        out = []
        for i, _ in enumerate(payload["input"]):
            out.append({"embedding": [float(i)], "index": i})
        out.reverse()
        return httpx.Response(200, json={"data": out})

    transport = httpx.MockTransport(handler)
    async with OpenAICompatClient(
        chat_model="m",
        embed_model="text-embedding-3-small",
        endpoint=BASE,
        transport=transport,
    ) as c:
        vecs = await c.embed(["a", "b", "c"])

    assert vecs == [[0.0], [1.0], [2.0]]
    body = json.loads(captured[0].content)
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["a", "b", "c"]


async def test_embed_requires_embed_model() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    async with OpenAICompatClient(
        chat_model="m", endpoint=BASE, transport=transport
    ) as c:
        with pytest.raises(ModelBackendError, match="embed_model"):
            await c.embed(["x"])


async def test_embed_empty_input_returns_empty_without_call() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    async with OpenAICompatClient(
        chat_model="m", embed_model="e", endpoint=BASE, transport=transport
    ) as c:
        assert await c.embed([]) == []
    assert calls == []
