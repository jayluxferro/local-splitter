"""Unit tests for the Stage 3 passthrough pipeline.

Uses a `FakeChatClient` so the tests don't need httpx. We're testing
orchestration, stats, and error handling — not HTTP.
"""

from __future__ import annotations

import pytest

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineError, PipelineRequest

from _fakes import FakeChatClient


def _cloud_only_config() -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat",
            endpoint="http://x",
            chat_model="cloud-model",
        ),
        tactics=TacticsConfig(),
    )


def _dual_config() -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat",
            endpoint="http://cloud",
            chat_model="cloud-model",
        ),
        local=ModelConfig(
            backend="ollama",
            endpoint="http://127.0.0.1:11434",
            chat_model="local-model",
        ),
        tactics=TacticsConfig(),
    )


async def test_passthrough_forwards_to_cloud_and_records_usage() -> None:
    cloud = FakeChatClient(chat_model="cloud-model")
    pipeline = Pipeline(cloud=cloud, local=None, config=_cloud_only_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=[{"role": "user", "content": "hi"}])
    )

    assert resp.content == "hello"
    assert resp.served_by == "cloud"
    assert resp.finish_reason == "stop"
    assert resp.model == "cloud-model"
    assert resp.usage_cloud.input_tokens == 10
    assert resp.usage_cloud.output_tokens == 3
    assert resp.usage_local.input_tokens is None
    assert len(resp.trace) == 1
    assert resp.trace[0].stage == "cloud_call"
    assert resp.trace[0].decision == "APPLIED"
    assert cloud.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_model_hint_local_uses_local_backend() -> None:
    cloud = FakeChatClient(chat_model="cloud-model")
    local = FakeChatClient(chat_model="local-model", reply_content="local answer")
    pipeline = Pipeline(cloud=cloud, local=local, config=_dual_config())

    resp = await pipeline.complete(
        PipelineRequest(
            messages=[{"role": "user", "content": "hi"}], model_hint="local"
        )
    )

    assert resp.served_by == "local"
    assert resp.model == "local-model"
    assert resp.content == "local answer"
    assert resp.usage_local.input_tokens == 10
    assert resp.usage_cloud.input_tokens is None
    assert cloud.calls == []
    assert local.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert resp.trace[0].stage == "local_call"


async def test_model_hint_local_without_local_backend_raises() -> None:
    cloud = FakeChatClient()
    pipeline = Pipeline(cloud=cloud, local=None, config=_cloud_only_config())
    with pytest.raises(PipelineError, match="no local backend"):
        await pipeline.complete(
            PipelineRequest(
                messages=[{"role": "user", "content": "x"}], model_hint="local"
            )
        )


async def test_backend_error_is_recorded_in_trace_and_reraised() -> None:
    cloud = FakeChatClient(raise_on_complete=ModelBackendError("500 oops"))
    pipeline = Pipeline(cloud=cloud, local=None, config=_cloud_only_config())
    with pytest.raises(ModelBackendError, match="500 oops"):
        await pipeline.complete(
            PipelineRequest(messages=[{"role": "user", "content": "x"}])
        )


async def test_stats_accumulate_across_calls() -> None:
    cloud = FakeChatClient(usage=Usage(input_tokens=20, output_tokens=5))
    pipeline = Pipeline(cloud=cloud, local=None, config=_cloud_only_config())

    for _ in range(3):
        await pipeline.complete(
            PipelineRequest(messages=[{"role": "user", "content": "x"}])
        )

    snap = pipeline.stats()
    assert snap.total_requests == 3
    assert snap.by_served["cloud"] == 3
    assert snap.tokens_in_cloud == 60
    assert snap.tokens_out_cloud == 15
    assert snap.tokens_in_local == 0
    assert snap.latency_sample_size == 3
    assert snap.p50_latency_ms is not None
    assert snap.p99_latency_ms is not None
    assert snap.p99_latency_ms >= snap.p50_latency_ms


async def test_stats_split_between_local_and_cloud() -> None:
    cloud = FakeChatClient(usage=Usage(input_tokens=100, output_tokens=20))
    local = FakeChatClient(usage=Usage(input_tokens=8, output_tokens=2))
    pipeline = Pipeline(cloud=cloud, local=local, config=_dual_config())

    await pipeline.complete(PipelineRequest(messages=[{"role": "user", "content": "a"}]))
    await pipeline.complete(
        PipelineRequest(
            messages=[{"role": "user", "content": "b"}], model_hint="local"
        )
    )

    snap = pipeline.stats()
    assert snap.total_requests == 2
    assert snap.by_served == {"cloud": 1, "local": 1}
    assert snap.tokens_in_cloud == 100
    assert snap.tokens_in_local == 8


async def test_t1_enabled_without_local_falls_through_to_cloud() -> None:
    """T1 enabled in config but no local backend → silently fall through."""
    cfg = Config(
        cloud=ModelConfig(backend="ollama", endpoint="http://x", chat_model="m"),
        tactics=TacticsConfig(t1_route=True),
    )
    cloud = FakeChatClient()
    pipeline = Pipeline(cloud=cloud, local=None, config=cfg)
    resp = await pipeline.complete(
        PipelineRequest(messages=[{"role": "user", "content": "x"}])
    )
    assert resp.served_by == "cloud"
    assert len(cloud.calls) == 1


async def test_pipeline_aclose_closes_both_backends() -> None:
    cloud = FakeChatClient()
    local = FakeChatClient()
    pipeline = Pipeline(cloud=cloud, local=local, config=_dual_config())
    await pipeline.aclose()
    assert cloud.closed is True
    assert local.closed is True
