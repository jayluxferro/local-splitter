"""Regression tests for the SSE envelope fix in the transparent proxy.

The tool-bearing Anthropic path bypasses the pipeline through
``_transparent_proxy``.  These tests mock the upstream httpx client so no
network is used and verify the two gates from SPEC-sse-envelope.md:

- Gate 1: no ``200 + text/event-stream`` until the upstream proves it.
- Gate 2: a committed stream always emits at least one terminal frame.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from local_splitter.config import Config, ModelConfig, TacticsConfig, TransportConfig
from local_splitter.pipeline import Pipeline
from local_splitter.transport import create_app

from _fakes import FakeChatClient


def _tool_request(stream: bool = True) -> dict[str, Any]:
    """Minimal Anthropic Messages request that triggers the transparent proxy."""
    return {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "stream": stream,
    }


class _FakeResponse:
    """Stand-in for an ``httpx.Response`` returned by our fake AsyncClient."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        body: bytes | None = None,
        chunks: list[bytes] | None = None,
        mid_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers)
        self._chunks = chunks if chunks is not None else ([body] if body is not None else [])
        self._mid_exc = mid_exc

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    @property
    def content(self) -> bytes:
        return b"".join(self._chunks)

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._mid_exc is not None:
            raise self._mid_exc

    async def aclose(self) -> None:
        pass


class _FakeAsyncClient:
    """Injectable replacement for ``httpx.AsyncClient`` in ``_transparent_proxy``."""

    _handler: Callable[[httpx.Request], _FakeResponse] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: bytes | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Request:
        return httpx.Request(method, url, content=content, json=json, headers=headers)

    async def send(self, request: httpx.Request, *, stream: bool = False) -> _FakeResponse:
        if _FakeAsyncClient._handler is None:
            raise RuntimeError("no upstream handler configured")
        return _FakeAsyncClient._handler(request)

    async def post(
        self,
        url: str,
        *,
        content: bytes | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        return await self.send(
            self.build_request("POST", url, content=content, json=json, headers=headers)
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def tool_client_factory(monkeypatch: pytest.MonkeyPatch):
    """Replace ``httpx.AsyncClient`` and yield a factory that builds a TestClient."""
    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient._handler = None

    def factory() -> TestClient:
        cloud = FakeChatClient(chat_model="fake-cloud")
        cfg = Config(
            cloud=ModelConfig(
                backend="openai_compat",
                endpoint="http://fake-cloud",
                chat_model="fake-cloud-model",
            ),
            local=None,
            transport=TransportConfig(),
            tactics=TacticsConfig(),
        )
        pipeline = Pipeline(cloud=cloud, local=None, config=cfg)
        return TestClient(create_app(pipeline, cfg))

    yield factory
    _FakeAsyncClient._handler = None


def _parse_sse_events(text: str) -> list[tuple[str, str]]:
    """Parse ``event: ...\ndata: ...\n\n`` frames into (event, data) pairs."""
    events: list[tuple[str, str]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_payload = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_payload = line[len("data:") :].strip()
        events.append((event_name, data_payload))
    return events


def test_transparent_proxy_upstream_401_returns_faithful_json(tool_client_factory):
    """Gate 1: upstream 401 application/json must stay 401 application/json."""

    def handler(request: httpx.Request) -> _FakeResponse:
        assert request.url.path == "/v1/messages"
        return _FakeResponse(
            401,
            {"content-type": "application/json", "retry-after": "30"},
            body=json.dumps({"error": "invalid api key"}).encode(),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/messages", json=_tool_request(stream=True))

    assert r.status_code == 401
    assert "application/json" in r.headers["content-type"]
    assert not r.headers["content-type"].startswith("text/event-stream")
    assert r.json()["error"] == "invalid api key"
    assert r.headers.get("retry-after") == "30"


def test_transparent_proxy_200_json_not_labeled_sse(tool_client_factory):
    """Gate 1: upstream 200 application/json for stream:true must not become SSE."""

    def handler(request: httpx.Request) -> _FakeResponse:
        return _FakeResponse(
            200,
            {"content-type": "application/json"},
            body=json.dumps({"foo": "bar"}).encode(),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/messages", json=_tool_request(stream=True))

    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    assert not r.headers["content-type"].startswith("text/event-stream")
    assert r.json() == {"foo": "bar"}


def test_transparent_proxy_sse_happy_path(tool_client_factory):
    """Gate 1 + Gate 2 happy path: real SSE is forwarded incrementally."""
    chunk1 = b"event: content_block_start\ndata: {}\n\n"
    chunk2 = b'event: content_block_delta\ndata: {"delta":{}}\n\n'
    chunk3 = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    def handler(request: httpx.Request) -> _FakeResponse:
        return _FakeResponse(
            200,
            {"content-type": "text/event-stream"},
            chunks=[chunk1, chunk2, chunk3],
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    with client.stream("POST", "/v1/messages", json=_tool_request(stream=True)) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        # Stream the response body to confirm chunks make it through.
        raw_chunks = list(r.iter_raw())

    assert len(raw_chunks) >= 1
    assert b"".join(raw_chunks) == chunk1 + chunk2 + chunk3


def test_transparent_proxy_midstream_reset_emits_terminal_error(tool_client_factory):
    """Gate 2: mid-stream reset yields a terminal ``event: error`` frame."""
    partial = b"garbage partial bytes, not a complete SSE frame\n"

    def handler(request: httpx.Request) -> _FakeResponse:
        return _FakeResponse(
            200,
            {"content-type": "text/event-stream"},
            chunks=[partial],
            mid_exc=httpx.RemoteProtocolError("peer closed connection"),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/messages", json=_tool_request(stream=True))

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(r.text)
    assert len(events) >= 1
    assert any(event == "error" for event, _ in events)
    # The terminal error frame should be the last well-formed event.
    last_event, last_data = events[-1]
    assert last_event == "error"
    payload = json.loads(last_data)
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"


# --- OpenAI transparent proxy (`_transparent_openai_proxy`) -----------------


def _openai_tool_request(stream: bool = True) -> dict[str, Any]:
    """Minimal OpenAI chat request that triggers the transparent proxy."""
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "stream": stream,
    }


def test_openai_transparent_proxy_upstream_401_returns_faithful_json(tool_client_factory):
    """Gate 1: upstream 401 application/json must stay 401 application/json."""

    def handler(request: httpx.Request) -> _FakeResponse:
        assert request.url.path == "/v1/chat/completions"
        return _FakeResponse(
            401,
            {"content-type": "application/json", "retry-after": "30"},
            body=json.dumps({"error": {"message": "invalid api key"}}).encode(),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/chat/completions", json=_openai_tool_request(stream=True))

    assert r.status_code == 401
    assert "application/json" in r.headers["content-type"]
    assert not r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("retry-after") == "30"


def test_openai_transparent_proxy_midstream_reset_emits_terminal_error(tool_client_factory):
    """Gate 2: mid-stream reset yields a terminal error frame then [DONE]."""
    partial = b'data: {"choices":[{"delta":{"content":"ok "}}]}\n\n'

    def handler(request: httpx.Request) -> _FakeResponse:
        return _FakeResponse(
            200,
            {"content-type": "text/event-stream"},
            chunks=[partial],
            mid_exc=httpx.RemoteProtocolError("peer closed connection"),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    r = client.post("/v1/chat/completions", json=_openai_tool_request(stream=True))

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    body = r.text
    assert partial.decode() in body
    assert "data: [DONE]" in body
    # Exactly one terminal error frame, and it precedes [DONE].
    error_frames = [d for e, d in _parse_sse_events(body) if d.startswith('{"error"')]
    assert len(error_frames) == 1
    payload = json.loads(error_frames[0])
    assert payload["error"]["type"] == "api_error"
    assert body.rstrip().endswith("data: [DONE]")


def test_transparent_proxy_counts_passthrough_in_stats(tool_client_factory):
    """Regression: tool-bearing requests bypass the pipeline but must still
    appear in /v1/splitter/stats — agentic chains carry tools on every
    request, so without this the stats endpoint reports zero traffic."""

    def handler(request: httpx.Request) -> _FakeResponse:
        return _FakeResponse(
            200,
            {"content-type": "application/json"},
            body=json.dumps({"ok": True}).encode(),
        )

    _FakeAsyncClient._handler = handler
    client = tool_client_factory()

    snap0 = client.get("/v1/splitter/stats").json()
    assert snap0["total_requests"] == 0

    r = client.post("/v1/messages", json=_tool_request(stream=False))
    assert r.status_code == 200

    snap = client.get("/v1/splitter/stats").json()
    assert snap["total_requests"] == 1
    assert snap["by_served"]["passthrough"] == 1
    # Token usage is unknowable on the byte-passthrough path → zero.
    assert snap["tokens_in_cloud"] == 0

    # The OpenAI tool surface records the same way.
    r = client.post("/v1/chat/completions", json=_openai_tool_request(stream=False))
    assert r.status_code == 200
    snap = client.get("/v1/splitter/stats").json()
    assert snap["total_requests"] == 2
    assert snap["by_served"]["passthrough"] == 2


def test_hop_headers_include_accept_encoding():
    """Regression: accept-encoding is capability-bound — forwarding a client's
    `br` when this venv's httpx lacks brotli makes the upstream send bytes we
    cannot decode before re-serving them."""
    from local_splitter.transport.http_proxy import _HOP_HEADERS

    assert "accept-encoding" in _HOP_HEADERS
