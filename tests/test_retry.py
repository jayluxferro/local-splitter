"""Retry contract tests (PIPELINE-RETRY-SPEC.md).

Covers the three cases the spec requires, for both the shared helper and
the sites it is wired into:

(a) a ``ReadError`` before the first byte is retried and the request
    eventually succeeds;
(b) a non-retryable failure is NOT retried and keeps today's 502 body;
(c) a mid-stream failure yields exactly ONE terminal frame and is never
    retried (no second body).

``str(httpx.ReadError(""))`` is EMPTY on purpose in several of these tests:
that is the shape the proxy actually sees (it wraps ``anyio.EndOfStream``),
and the retry decision must still be made correctly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from local_splitter import retry
from local_splitter.models import ModelBackendError, OpenAICompatClient
from local_splitter.retry import (
    backoff_seconds,
    is_retryable,
    iter_upstream,
    retry_budget,
    retry_upstream,
)

# The transparent-proxy tests already own a fake ``httpx.AsyncClient``; reuse
# it (and its fixture) instead of duplicating it.  pytest registers fixtures by
# their declared name, so the alias does not change how tests request it.
from test_transport_http_sse_envelope import (  # noqa: F401
    _FakeAsyncClient,
    _FakeResponse,
)
from test_transport_http_sse_envelope import tool_client_factory as _upstream_tool_client_factory  # noqa: F401


@pytest.fixture
def tool_client_factory(_upstream_tool_client_factory):  # noqa: F811
    """Re-expose the transparent-proxy fixture under its canonical name."""
    return _upstream_tool_client_factory


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Never sleep in tests; record the delays the helper asked for."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(retry, "_sleep", fake_sleep)
    monkeypatch.delenv("LOCAL_SPLITTER_UPSTREAM_RETRIES", raising=False)
    return delays


# ---------------------------------------------------------------------------
# The predicate is type-based, never string-based
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError(""),
        httpx.ConnectTimeout(""),
        httpx.ReadError(""),
        httpx.ReadTimeout(""),
        httpx.WriteError(""),
        httpx.WriteTimeout(""),
        httpx.PoolTimeout(""),
        httpx.RemoteProtocolError(""),
        httpx.CloseError(""),
    ],
)
def test_retryable_transport_errors_match_by_type_even_with_empty_str(exc: Exception) -> None:
    assert str(exc) in ("", "peer closed")  # some of these carry no message at all
    assert is_retryable(exc) is True


def test_non_retryable_errors_are_not_retried_by_type() -> None:
    request = httpx.Request("POST", "http://upstream.invalid/v1/messages")
    response = httpx.Response(429, request=request)
    # A 4xx/5xx *response* is a real answer — pass it through.
    assert (
        is_retryable(httpx.HTTPStatusError("rate limited", request=request, response=response))
        is False
    )
    # A bare TimeoutException (not a per-phase timeout) is not in the set.
    assert is_retryable(httpx.TimeoutException("")) is False
    assert is_retryable(ValueError("nope")) is False


def test_end_of_stream_is_retryable() -> None:
    import anyio

    assert is_retryable(anyio.EndOfStream()) is True


# ---------------------------------------------------------------------------
# Bounded retry budget + backoff
# ---------------------------------------------------------------------------


def test_retry_budget_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_SPLITTER_UPSTREAM_RETRIES", raising=False)
    assert retry_budget() == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("5", 5), (" 3 ", 3), ("nonsense", 2), ("-4", 0)],
)
def test_retry_budget_env_override(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("LOCAL_SPLITTER_UPSTREAM_RETRIES", raw)
    assert retry_budget() == expected


def test_backoff_is_exponential_jittered_and_never_below_200ms() -> None:
    first = [backoff_seconds(1) for _ in range(200)]
    second = [backoff_seconds(2) for _ in range(200)]
    assert all(0.19 <= d <= 0.31 for d in first)
    assert all(0.39 <= d <= 0.61 for d in second)
    assert all(d >= 0.2 for d in first + second)
    assert backoff_seconds(10) <= 2.0


# ---------------------------------------------------------------------------
# (a) retry-then-succeed / bounded / non-retryable passthrough
# ---------------------------------------------------------------------------


async def test_retry_upstream_retries_read_error_then_succeeds(
    _no_real_sleep: list[float],
) -> None:
    calls: list[int] = []

    async def call() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadError("")  # str() is EMPTY
        return "ok"

    assert await retry_upstream(call, operation="test") == "ok"
    assert len(calls) == 3  # 1 attempt + 2 retries (the default budget)
    assert len(_no_real_sleep) == 2


async def test_retry_upstream_is_bounded_and_reraises_the_last_error() -> None:
    calls: list[int] = []

    async def call() -> None:
        calls.append(1)
        raise httpx.ReadError("")

    with pytest.raises(httpx.ReadError):
        await retry_upstream(call, operation="test")
    assert len(calls) == 3


async def test_retry_upstream_passes_non_retryable_errors_straight_through() -> None:
    calls: list[int] = []

    async def call() -> None:
        calls.append(1)
        raise httpx.HTTPStatusError(
            "bad gateway",
            request=httpx.Request("POST", "http://upstream.invalid"),
            response=httpx.Response(502),
        )

    with pytest.raises(httpx.HTTPStatusError):
        await retry_upstream(call, operation="test")
    assert len(calls) == 1  # no retry at all


async def test_retry_upstream_honours_explicit_zero_budget() -> None:
    calls: list[int] = []

    async def call() -> None:
        calls.append(1)
        raise httpx.ReadError("")

    with pytest.raises(httpx.ReadError):
        await retry_upstream(call, operation="test", max_retries=0)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# (c) iter_upstream: retry before the first item, never after
# ---------------------------------------------------------------------------


async def test_iter_upstream_retries_before_first_item(_no_real_sleep: list[float]) -> None:
    attempts: list[int] = []

    async def stream_factory() -> AsyncIterator[int]:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadError("")
        yield 1
        yield 2

    items = [item async for item in iter_upstream(stream_factory, operation="test")]
    assert items == [1, 2]
    assert len(attempts) == 2


async def test_iter_upstream_does_not_retry_after_first_item(
    _no_real_sleep: list[float],
) -> None:
    attempts: list[int] = []

    async def stream_factory() -> AsyncIterator[int]:
        attempts.append(1)
        yield 1
        raise httpx.ReadError("")

    received: list[int] = []
    with pytest.raises(httpx.ReadError):
        async for item in iter_upstream(stream_factory, operation="test"):
            received.append(item)
    assert received == [1]
    assert len(attempts) == 1  # mid-stream: no second attempt
    assert _no_real_sleep == []


# ---------------------------------------------------------------------------
# Transport wiring: catch-all passthrough (non-streaming, 502 shape preserved)
# ---------------------------------------------------------------------------


def _count_tokens_body() -> bytes:
    return json.dumps({"model": "claude-sonnet-4", "messages": []}).encode()


def test_passthrough_retries_read_error_then_succeeds(
    tool_client_factory, _no_real_sleep: list[float]
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("")  # empty str — the production failure shape
        return _FakeResponse(200, {"content-type": "application/json"}, body=b'{"input_tokens": 7}')

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post(
        "/v1/messages/count_tokens", content=_count_tokens_body(), headers={"x-api-key": "sk"}
    )

    assert r.status_code == 200
    assert r.json() == {"input_tokens": 7}
    assert len(seen) == 2  # the retry actually re-sent the request
    assert len(_no_real_sleep) == 1


def test_passthrough_does_not_retry_non_retryable_error(tool_client_factory, caplog) -> None:
    """A non-retryable httpx failure keeps today's exact 502 body/status."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        raise httpx.DecodingError("bad bytes")  # HTTPError, but not retryable

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post(
        "/v1/messages/count_tokens", content=_count_tokens_body(), headers={"x-api-key": "sk"}
    )

    assert len(seen) == 1
    assert r.status_code == 502
    assert r.json() == {"error": {"type": "upstream_error", "message": "DecodingError: bad bytes"}}


def test_passthrough_exhausts_retries_and_keeps_the_502_shape(tool_client_factory) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        raise httpx.ReadError("")

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post(
        "/v1/messages/count_tokens", content=_count_tokens_body(), headers={"x-api-key": "sk"}
    )

    assert len(seen) == 3  # 1 attempt + 2 retries
    assert r.status_code == 502
    assert r.json() == {"error": {"type": "upstream_error", "message": "ReadError"}}


def test_passthrough_retry_count_follows_env_override(
    tool_client_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_SPLITTER_UPSTREAM_RETRIES", "1")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        raise httpx.ReadError("")

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post(
        "/v1/messages/count_tokens", content=_count_tokens_body(), headers={"x-api-key": "sk"}
    )

    assert len(seen) == 2  # 1 attempt + the single configured retry
    assert r.status_code == 502


def test_transparent_proxy_retries_stream_before_commit(tool_client_factory) -> None:
    """A transport failure BEFORE Gate 1 opens the stream is re-sent."""
    seen: list[httpx.Request] = []
    chunk = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("")  # dies before headers arrive
        return _FakeResponse(200, {"content-type": "text/event-stream"}, chunks=[chunk])

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    with client.stream("POST", "/v1/messages", json=_tool_request(stream=True)) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_raw())

    assert len(seen) == 2
    assert body == chunk


# ---------------------------------------------------------------------------
# Transport wiring: mid-stream failure = one terminal frame, no retry
# ---------------------------------------------------------------------------


def _tool_request(stream: bool = True) -> dict[str, Any]:
    return {
        "model": "claude-3-5-sonnet",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "stream": stream,
    }


def test_transparent_proxy_midstream_failure_is_not_retried(tool_client_factory) -> None:
    partial = (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> _FakeResponse:
        seen.append(request)
        return _FakeResponse(
            200,
            {"content-type": "text/event-stream"},
            chunks=[partial],
            mid_exc=httpx.ReadError(""),  # empty str, mid-stream
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/messages", json=_tool_request(stream=True))

    assert r.status_code == 200
    assert len(seen) == 1  # committed stream → never re-sent
    body = r.text
    assert body.count("event: error") == 1  # exactly one terminal frame
    frames = [b for b in body.split("\n\n") if b.strip()]
    assert frames[-1].startswith("event: error")
    payload = json.loads(frames[-1].split("data: ", 1)[1])
    assert payload["type"] == "error"
    assert payload["error"]["message"].startswith("ReadError")


# ---------------------------------------------------------------------------
# Model-layer wiring: pre-first-chunk retry, mid-stream single failure
# ---------------------------------------------------------------------------


def _sse_body() -> bytes:
    events = [
        {"choices": [{"delta": {"content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


async def test_model_stream_retries_read_error_before_first_chunk(
    _no_real_sleep: list[float],
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadError("")
        return httpx.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )

    client = OpenAICompatClient(
        chat_model="m", endpoint="http://upstream.invalid", transport=httpx.MockTransport(handler)
    )
    chunks = [c async for c in await client.stream([{"role": "user", "content": "hi"}])]
    await client.aclose()

    assert len(calls) == 2
    assert "".join(c.delta for c in chunks if not c.done) == "hello"
    assert chunks[-1].done is True


class _BrokenAsyncStream(httpx.AsyncByteStream):
    """Yield one complete SSE frame, then die mid-stream."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        raise httpx.ReadError("")

    async def aclose(self) -> None:
        pass


async def test_model_stream_midstream_failure_is_not_retried(
    _no_real_sleep: list[float],
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BrokenAsyncStream(),
        )

    client = OpenAICompatClient(
        chat_model="m", endpoint="http://upstream.invalid", transport=httpx.MockTransport(handler)
    )
    received: list[str] = []
    with pytest.raises(ModelBackendError) as excinfo:
        async for chunk in await client.stream([{"role": "user", "content": "hi"}]):
            if not chunk.done:
                received.append(chunk.delta)
    await client.aclose()

    assert received == ["ok"]  # the first chunk reached the caller
    assert len(calls) == 1  # mid-stream: no retry
    assert _no_real_sleep == []
    assert "ReadError" in str(excinfo.value)
