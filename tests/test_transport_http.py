"""Unit tests for the FastAPI HTTP proxy.

We use FastAPI's TestClient (sync httpx under the hood) and point the
pipeline at a FakeChatClient so nothing leaves the process.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from local_splitter.config import Config, ModelConfig, TacticsConfig, TransportConfig
from local_splitter.models import Usage
from local_splitter.pipeline import Pipeline
from local_splitter.transport import create_app

from _fakes import FakeChatClient


def _config(local: bool = True) -> Config:
    cloud = ModelConfig(
        backend="openai_compat",
        endpoint="http://cloud",
        chat_model="gpt-4o-mini",
    )
    local_mc = (
        ModelConfig(backend="ollama", endpoint="http://local", chat_model="llama3.2:3b")
        if local
        else None
    )
    return Config(
        cloud=cloud,
        local=local_mc,
        transport=TransportConfig(),
        tactics=TacticsConfig(),
    )


def _client(
    *,
    cloud_reply: str = "hello from cloud",
    local_reply: str = "hello from local",
    usage: Usage = Usage(input_tokens=42, output_tokens=7),
) -> tuple[TestClient, FakeChatClient, FakeChatClient]:
    cloud = FakeChatClient(chat_model="gpt-4o-mini", reply_content=cloud_reply, usage=usage)
    local = FakeChatClient(chat_model="llama3.2:3b", reply_content=local_reply, usage=usage)
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())
    app = create_app(pipeline, _config())
    return TestClient(app), cloud, local


def test_chat_completions_happy_path() -> None:
    client, cloud, _ = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 64,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from cloud"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 42
    assert body["usage"]["completion_tokens"] == 7
    assert body["usage"]["total_tokens"] == 49
    assert body["splitter"]["served_by"] == "cloud"
    trace = body["splitter"]["pipeline_trace"]
    assert len(trace) == 1
    assert trace[0]["stage"] == "cloud_call"
    assert trace[0]["decision"] == "APPLIED"
    # Backend was actually called.
    assert cloud.calls[0]["temperature"] == 0.2
    assert cloud.calls[0]["max_tokens"] == 64


def test_force_local_routes_to_local_backend() -> None:
    client, cloud, local = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"splitter": {"force_local": True}},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["splitter"]["served_by"] == "local"
    assert body["choices"][0]["message"]["content"] == "hello from local"
    assert body["splitter"]["tokens_local"]["input"] == 42
    assert body["usage"]["prompt_tokens"] == 0  # cloud untouched
    assert cloud.calls == []
    assert len(local.calls) == 1


def test_force_cloud_explicitly() -> None:
    client, _, local = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"splitter": {"force_cloud": True}},
        },
    )
    assert r.json()["splitter"]["served_by"] == "cloud"
    assert local.calls == []


def test_stream_true_rejected_501() -> None:
    client, _, _ = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert r.status_code == 501
    assert "stream" in r.json()["detail"].lower()


def test_empty_messages_rejected_400() -> None:
    client, _, _ = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
    )
    assert r.status_code == 400
    assert "messages" in r.json()["detail"]


def test_invalid_body_rejected_400() -> None:
    client, _, _ = _client()
    r = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_list_models_returns_cloud_and_local() -> None:
    client, _, _ = _client()
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = {m["id"] for m in data}
    assert "gpt-4o-mini" in ids
    assert "llama3.2:3b" in ids
    assert any(m["owned_by"].startswith("cloud") for m in data)
    assert any(m["owned_by"].startswith("local") for m in data)


def test_list_models_cloud_only_when_no_local() -> None:
    cloud = FakeChatClient(chat_model="gpt-4o-mini")
    cfg = _config(local=False)
    pipeline = Pipeline(cloud=cloud, local=None, config=cfg)
    client = TestClient(create_app(pipeline, cfg))
    data = client.get("/v1/models").json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "gpt-4o-mini"


def test_stats_endpoint_reflects_activity() -> None:
    client, _, _ = _client()
    # Pre-activity: zeros.
    snap0 = client.get("/v1/splitter/stats").json()
    assert snap0["total_requests"] == 0
    assert snap0["by_served"] == {}

    for _ in range(2):
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}]},
        )

    snap = client.get("/v1/splitter/stats").json()
    assert snap["total_requests"] == 2
    assert snap["by_served"]["cloud"] == 2
    assert snap["tokens_in_cloud"] == 84
    assert snap["tokens_out_cloud"] == 14
    assert snap["p50_latency_ms"] is not None


def test_healthz() -> None:
    client, _, _ = _client()
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_backend_error_translates_to_502() -> None:
    from local_splitter.models import ModelBackendError

    cloud = FakeChatClient(raise_on_complete=ModelBackendError("upstream 500"))
    pipeline = Pipeline(cloud=cloud, local=None, config=_config(local=False))
    client = TestClient(create_app(pipeline, _config(local=False)))

    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 502
    assert "backend" in r.json()["detail"].lower()


def test_force_local_without_local_backend_returns_400() -> None:
    cloud = FakeChatClient()
    cfg = _config(local=False)
    pipeline = Pipeline(cloud=cloud, local=None, config=cfg)
    client = TestClient(create_app(pipeline, cfg))

    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"splitter": {"force_local": True}},
        },
    )
    assert r.status_code == 400
    assert "local" in r.json()["detail"].lower()
