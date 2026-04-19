"""Unit tests for the FastMCP stdio server.

We don't spawn a stdio subprocess — we call the registered tool
handlers directly via `server.call_tool(name, args)`, which is how
FastMCP dispatches internally. That's enough to verify the tool
surface and each handler's behaviour end-to-end.
"""

from __future__ import annotations

import json

import pytest

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline
from local_splitter.transport import create_mcp_server

from _fakes import FakeChatClient


def _config(*, with_local: bool = True) -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat",
            endpoint="http://cloud",
            chat_model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        local=(
            ModelConfig(
                backend="ollama",
                endpoint="http://127.0.0.1:11434",
                chat_model="llama3.2:3b",
            )
            if with_local
            else None
        ),
        tactics=TacticsConfig(),
    )


def _build(
    *,
    cloud_reply: str = "hi from cloud",
    local_reply: str = "hi from local",
    raise_cloud: Exception | None = None,
) -> tuple[object, FakeChatClient, FakeChatClient]:
    cloud = FakeChatClient(
        chat_model="gpt-4o-mini",
        reply_content=cloud_reply,
        usage=Usage(input_tokens=11, output_tokens=2),
        raise_on_complete=raise_cloud,
    )
    local = FakeChatClient(
        chat_model="llama3.2:3b",
        reply_content=local_reply,
        usage=Usage(input_tokens=5, output_tokens=1),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())
    server = create_mcp_server(pipeline, _config())
    return server, cloud, local


async def _call(server, name: str, args: dict) -> dict:
    """Invoke a FastMCP tool and return its structured dict result.

    `FastMCP.call_tool` returns `(content_list, structured_dict)` in
    recent versions of the SDK.
    """
    result = await server.call_tool(name, args)
    if isinstance(result, tuple):
        content, structured = result
    else:  # pragma: no cover — older SDKs
        content, structured = result, None

    if structured is not None:
        return structured

    # Fallback: parse JSON from the text content block.
    assert content, f"{name} returned no content"
    text = getattr(content[0], "text", None) or str(content[0])
    return json.loads(text)


async def test_registered_tool_names_match_contract() -> None:
    server, _, _ = _build()
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "split.complete",
        "split.transform",
        "split.classify",
        "split.cache_lookup",
        "split.stats",
        "split.config",
    }


async def test_split_complete_passthrough_cloud() -> None:
    server, cloud, _ = _build()
    result = await _call(
        server,
        "split.complete",
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert result["response"] == "hi from cloud"
    assert result["served_by"] == "cloud"
    assert result["finish_reason"] == "stop"
    assert result["tokens"]["input_cloud"] == 11
    assert result["tokens"]["output_cloud"] == 2
    assert result["tokens"]["input_local"] == 0
    assert result["pipeline_trace"][0]["stage"] == "cloud_call"
    assert len(cloud.calls) == 1


async def test_split_complete_model_hint_local() -> None:
    server, cloud, local = _build()
    result = await _call(
        server,
        "split.complete",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model_hint": "local",
        },
    )
    assert result["served_by"] == "local"
    assert result["response"] == "hi from local"
    assert result["tokens"]["input_local"] == 5
    assert cloud.calls == []
    assert len(local.calls) == 1


async def test_split_complete_backend_error_returns_structured_error() -> None:
    server, _, _ = _build(raise_cloud=ModelBackendError("upstream 500"))
    result = await _call(
        server,
        "split.complete",
        {"messages": [{"role": "user", "content": "x"}]},
    )
    assert "error" in result
    assert result["error"]["type"] == "backend_error"
    assert "upstream 500" in result["error"]["message"]


async def test_split_classify_stub_when_disabled() -> None:
    server, _, _ = _build()
    result = await _call(
        server,
        "split.classify",
        {"messages": [{"role": "user", "content": "x"}]},
    )
    assert result["decision"] == "NOT_IMPLEMENTED"
    assert result["stage"] == "t1_route"


async def test_split_classify_live_when_t1_enabled() -> None:
    cloud = FakeChatClient(
        chat_model="gpt-4o-mini",
        usage=Usage(input_tokens=11, output_tokens=2),
    )
    local = FakeChatClient(
        chat_model="llama3.2:3b",
        reply_content="TRIVIAL",
        usage=Usage(input_tokens=5, output_tokens=1),
    )
    cfg = Config(
        cloud=ModelConfig(
            backend="openai_compat",
            endpoint="http://cloud",
            chat_model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        local=ModelConfig(
            backend="ollama",
            endpoint="http://127.0.0.1:11434",
            chat_model="llama3.2:3b",
        ),
        tactics=TacticsConfig(t1_route=True),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=cfg)
    server = create_mcp_server(pipeline, cfg)
    result = await _call(
        server,
        "split.classify",
        {"messages": [{"role": "user", "content": "what is 1+1?"}]},
    )
    assert result["decision"] == "TRIVIAL"
    assert result["stage"] == "t1_classify"
    assert len(local.calls) == 1


async def test_split_cache_lookup_stub() -> None:
    server, _, _ = _build()
    result = await _call(
        server,
        "split.cache_lookup",
        {"messages": [{"role": "user", "content": "x"}]},
    )
    assert result["hit"] is False
    assert result["stage"] == "t3_sem_cache"


async def test_split_stats_reflects_activity() -> None:
    server, _, _ = _build()
    await _call(
        server, "split.complete", {"messages": [{"role": "user", "content": "a"}]}
    )
    await _call(
        server, "split.complete", {"messages": [{"role": "user", "content": "b"}]}
    )
    snap = await _call(server, "split.stats", {})
    assert snap["total_requests"] == 2
    assert snap["by_served"]["cloud"] == 2
    assert snap["tokens_in_cloud"] == 22


async def test_split_config_hides_nothing_sensitive() -> None:
    server, _, _ = _build()
    cfg = await _call(server, "split.config", {})
    # API keys are referenced by env var name, never inlined.
    assert cfg["models"]["cloud"]["api_key_env"] == "OPENAI_API_KEY"
    assert "api_key" not in cfg["models"]["cloud"]
    assert cfg["models"]["local"]["backend"] == "ollama"
    assert cfg["tactics"]["t1_route"] is False
    assert cfg["version"] == 1
    assert "adaptive" in cfg
    assert cfg["adaptive"]["enabled"] is False


async def test_split_complete_invalid_model_hint_raises() -> None:
    server, _, _ = _build()
    # FastMCP wraps handler exceptions into tool error responses; the
    # exception propagates out of call_tool in current SDKs.
    with pytest.raises(Exception):  # noqa: B017 — SDK-specific error type
        await _call(
            server,
            "split.complete",
            {
                "messages": [{"role": "user", "content": "x"}],
                "model_hint": "banana",
            },
        )
