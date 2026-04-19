"""Unit tests for T2 compress — local model prompt compression.

Tests cover:
- apply() compresses eligible messages, skips short ones and last user msg
- Fail-open on local model errors
- Only uses compressed version if actually shorter
- Pipeline integration: T2 modifies messages before cloud call
- T2 disabled / no local backend → skipped
- T1 + T2 composition: trivial bypasses T2, complex goes through T2
- T2 + T3 composition: cache hit bypasses T2
"""

from __future__ import annotations

from pathlib import Path

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.compress import apply

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_SYSTEM = "x" * 300  # exceeds default min_length of 200
SHORT_SYSTEM = "be helpful"  # below min_length


def _config(*, t2: bool = True, t1: bool = False, t3: bool = False, **t2_params) -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m",
        ),
        tactics=TacticsConfig(
            t1_route=t1,
            t2_compress=t2,
            t3_sem_cache=t3,
            params={"t2_compress": t2_params} if t2_params else {},
        ),
    )


# ---------------------------------------------------------------------------
# Unit: apply()
# ---------------------------------------------------------------------------

async def test_apply_compresses_long_system_message() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="compressed version",
        usage=Usage(input_tokens=100, output_tokens=20),
    )
    msgs = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "what is a monad?"},
    ]

    result = await apply(msgs, local=local)

    assert result.messages[0]["content"] == "compressed version"
    assert result.messages[1]["content"] == "what is a monad?"  # untouched
    assert len(result.events) == 1
    assert result.events[0].decision == "APPLIED"
    assert result.events[0].detail["messages_compressed"] == 1
    assert len(local.calls) == 1


async def test_apply_skips_short_messages() -> None:
    local = FakeChatClient(chat_model="local-m")
    msgs = [
        {"role": "system", "content": SHORT_SYSTEM},
        {"role": "user", "content": "hi"},
    ]

    result = await apply(msgs, local=local)

    assert result.messages == msgs  # unchanged
    assert result.events[0].decision == "SKIP"
    assert local.calls == []


async def test_apply_never_compresses_last_user_message() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="short")
    msgs = [
        {"role": "user", "content": LONG_SYSTEM},  # long but it's the last user msg
    ]

    result = await apply(msgs, local=local)

    assert result.messages[0]["content"] == LONG_SYSTEM  # untouched
    assert result.events[0].decision == "SKIP"


async def test_apply_compresses_non_last_long_user_message() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="compressed",
    )
    msgs = [
        {"role": "user", "content": LONG_SYSTEM},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "final question"},
    ]

    result = await apply(msgs, local=local)

    # First user message is long and not last → compressed.
    assert result.messages[0]["content"] == "compressed"
    # Last user message untouched.
    assert result.messages[2]["content"] == "final question"
    assert len(local.calls) == 1


async def test_apply_keeps_original_if_compressed_not_shorter() -> None:
    # Local model returns something longer than original.
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="z" * 500,  # longer than LONG_SYSTEM (300)
    )
    msgs = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "q"},
    ]

    result = await apply(msgs, local=local)

    # Original kept because compressed version is longer.
    assert result.messages[0]["content"] == LONG_SYSTEM
    assert result.events[0].decision == "NOOP"


async def test_apply_error_fails_open() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        raise_on_complete=ModelBackendError("down"),
    )
    msgs = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "q"},
    ]

    result = await apply(msgs, local=local)

    # Original kept, no crash.
    assert result.messages[0]["content"] == LONG_SYSTEM
    assert result.events[0].detail["errors"] == 1


async def test_apply_custom_min_length() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="short")
    msgs = [
        {"role": "system", "content": "a" * 50},  # 50 chars
        {"role": "user", "content": "q"},
    ]

    # Default min_length=200 → skip.
    result1 = await apply(msgs, local=local)
    assert result1.events[0].decision == "SKIP"

    # Custom min_length=10 → compress.
    result2 = await apply(msgs, local=local, params={"min_length": 10})
    assert result2.events[0].decision == "APPLIED"


async def test_apply_uses_temperature_zero() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="compressed")
    msgs = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "q"},
    ]

    await apply(msgs, local=local)

    assert local.calls[0]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Integration: Pipeline.complete with T2
# ---------------------------------------------------------------------------

async def test_pipeline_t2_compresses_before_cloud() -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="answer",
        usage=Usage(input_tokens=20, output_tokens=5),
    )
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="compressed sys",
        usage=Usage(input_tokens=50, output_tokens=10),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "question"},
        ])
    )

    assert resp.served_by == "cloud"
    # Cloud received the compressed system message.
    cloud_msgs = cloud.calls[0]["messages"]
    assert cloud_msgs[0]["content"] == "compressed sys"
    assert cloud_msgs[1]["content"] == "question"  # user msg untouched
    # Trace has compress + cloud stages.
    stages = [e.stage for e in resp.trace]
    assert "t2_compress" in stages
    assert "cloud_call" in stages


async def test_pipeline_t2_disabled_sends_original() -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(chat_model="local-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t2=False))

    await pipeline.complete(
        PipelineRequest(messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "q"},
        ])
    )

    # Cloud received original messages.
    assert cloud.calls[0]["messages"][0]["content"] == LONG_SYSTEM


async def test_pipeline_t2_explicit_hint_bypasses() -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(chat_model="local-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())

    await pipeline.complete(
        PipelineRequest(
            messages=[
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": "q"},
            ],
            model_hint="cloud",
        )
    )

    # Cloud received original (T2 bypassed for explicit hint).
    assert cloud.calls[0]["messages"][0]["content"] == LONG_SYSTEM


# ---------------------------------------------------------------------------
# Composition: T1 + T2
# ---------------------------------------------------------------------------

async def test_t1_trivial_bypasses_t2() -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "local answer"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t2=True, t1=True))

    resp = await pipeline.complete(
        PipelineRequest(messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "hi"},
        ])
    )

    assert resp.served_by == "local"
    assert not any(e.stage == "t2_compress" for e in resp.trace)


async def test_t1_complex_then_t2_compresses() -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="answer",
        usage=Usage(input_tokens=20, output_tokens=5),
    )
    # Call 1: T1 classifier → "COMPLEX"
    # Call 2: T2 compress → "compressed"
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["COMPLEX", "compressed"],
        usage=Usage(input_tokens=10, output_tokens=3),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t2=True, t1=True))

    resp = await pipeline.complete(
        PipelineRequest(messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "question"},
        ])
    )

    assert resp.served_by == "cloud"
    stages = [e.stage for e in resp.trace]
    assert "t1_classify" in stages
    assert "t2_compress" in stages
    assert "cloud_call" in stages
    # Cloud received compressed message.
    assert cloud.calls[0]["messages"][0]["content"] == "compressed"


# ---------------------------------------------------------------------------
# Composition: T3 + T2
# ---------------------------------------------------------------------------

async def test_t3_hit_bypasses_t2(tmp_path: Path) -> None:
    from local_splitter.pipeline.sem_cache import CacheStore

    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(chat_model="local-m")
    store = CacheStore(tmp_path / "cache.sqlite", embed_dim=32)

    cfg = Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m",
            embed_model="nomic-embed-text",
        ),
        tactics=TacticsConfig(
            t2_compress=True,
            t3_sem_cache=True,
            params={"t3_sem_cache": {"similarity_threshold": 0.5}},
        ),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=cfg, cache_store=store)

    msgs = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "what is a monad?"},
    ]

    # First call: cache miss → T2 compress → cloud.
    await pipeline.complete(PipelineRequest(messages=msgs))

    # Second call: cache hit → skip T2 entirely.
    resp2 = await pipeline.complete(PipelineRequest(messages=msgs))
    assert resp2.served_by == "cache"
    assert not any(e.stage == "t2_compress" for e in resp2.trace)
    store.close()


async def test_compress_messages_only_runs_t2() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="z" * 50,
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    cfg = _config(t2=True, min_length=10)
    pipeline = Pipeline(cloud=cloud, local=local, config=cfg)
    msgs = [{"role": "system", "content": LONG_SYSTEM}]
    out, trace = await pipeline.compress_messages_only(msgs)
    assert trace and trace[0].stage == "t2_compress"
    assert isinstance(out, list)


async def test_compress_messages_only_respects_tactics_override() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="z" * 50,
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    cfg = _config(t2=True, min_length=10)
    pipeline = Pipeline(cloud=cloud, local=local, config=cfg)
    msgs = [{"role": "system", "content": LONG_SYSTEM}]
    out, trace = await pipeline.compress_messages_only(
        msgs, tactics_override=frozenset({"t2_compress"})
    )
    assert trace == []
    assert out == msgs
