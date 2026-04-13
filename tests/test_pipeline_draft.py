"""Unit tests for T4 draft — local drafter + cloud reviewer.

Tests cover:
- apply(): approved draft, revised draft, local draft error
- Pipeline integration: T4 replaces direct cloud call
- T4 disabled → direct cloud
- Fail-open: local draft error falls back to direct cloud
- T1 + T4 composition: trivial bypasses T4
"""

from __future__ import annotations

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.draft import apply

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(*, t4: bool = True, t1: bool = False) -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m",
        ),
        tactics=TacticsConfig(t1_route=t1, t4_draft=t4),
    )


_MSGS = [{"role": "user", "content": "explain monads"}]


# ---------------------------------------------------------------------------
# Unit: apply()
# ---------------------------------------------------------------------------

async def test_apply_approved_draft() -> None:
    local = FakeChatClient(
        chat_model="local-m", reply_content="A monad is...",
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="APPROVED\nA monad is...",
        usage=Usage(input_tokens=40, output_tokens=5),
    )

    result = await apply(_MSGS, local=local, cloud=cloud)

    assert result is not None
    assert result.approved is True
    assert result.draft == "A monad is..."
    assert result.review.content == "APPROVED\nA monad is..."
    assert len(result.events) == 2
    assert result.events[0].stage == "t4_draft"
    assert result.events[0].decision == "DRAFTED"
    assert result.events[1].stage == "t4_review"
    assert result.events[1].decision == "APPROVED"


async def test_apply_revised_draft() -> None:
    local = FakeChatClient(
        chat_model="local-m", reply_content="wrong answer",
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="Correct answer here.",
        usage=Usage(input_tokens=40, output_tokens=15),
    )

    result = await apply(_MSGS, local=local, cloud=cloud)

    assert result is not None
    assert result.approved is False
    assert result.events[1].decision == "REVISED"
    assert result.review.content == "Correct answer here."


async def test_apply_local_draft_error_returns_none() -> None:
    local = FakeChatClient(raise_on_complete=ModelBackendError("down"))
    cloud = FakeChatClient(chat_model="cloud-m")

    result = await apply(_MSGS, local=local, cloud=cloud)

    assert result is None  # fail-open
    assert cloud.calls == []  # cloud never called


async def test_apply_cloud_review_error_raises() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="draft")
    cloud = FakeChatClient(raise_on_complete=ModelBackendError("cloud down"))

    import pytest
    with pytest.raises(ModelBackendError, match="cloud down"):
        await apply(_MSGS, local=local, cloud=cloud)


# ---------------------------------------------------------------------------
# Integration: Pipeline.complete with T4
# ---------------------------------------------------------------------------

async def test_pipeline_t4_draft_review_flow() -> None:
    local = FakeChatClient(
        chat_model="local-m", reply_content="draft answer",
        usage=Usage(input_tokens=15, output_tokens=8),
    )
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="APPROVED\ndraft answer",
        usage=Usage(input_tokens=30, output_tokens=5),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "draft+cloud"
    assert resp.content == "APPROVED\ndraft answer"
    assert resp.usage_local.input_tokens == 15
    assert resp.usage_cloud.input_tokens == 30
    stages = [e.stage for e in resp.trace]
    assert "t4_draft" in stages
    assert "t4_review" in stages
    # No direct cloud_call — T4 replaced it.
    assert "cloud_call" not in stages


async def test_pipeline_t4_disabled_uses_direct_cloud() -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="direct answer",
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    local = FakeChatClient(chat_model="local-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t4=False))

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "cloud"
    assert "cloud_call" in [e.stage for e in resp.trace]
    assert local.calls == []


async def test_pipeline_t4_local_error_falls_to_direct_cloud() -> None:
    local = FakeChatClient(raise_on_complete=ModelBackendError("down"))
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="cloud answer",
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    # Fell back to direct cloud call.
    assert resp.served_by == "cloud"
    assert "cloud_call" in [e.stage for e in resp.trace]


async def test_pipeline_t4_explicit_hint_bypasses() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="draft")
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=_MSGS, model_hint="cloud")
    )

    assert resp.served_by == "cloud"
    assert local.calls == []


# ---------------------------------------------------------------------------
# Composition: T1 + T4
# ---------------------------------------------------------------------------

async def test_t1_trivial_bypasses_t4() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "local answer"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t4=True, t1=True))

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "local"
    assert not any(e.stage.startswith("t4_") for e in resp.trace)


async def test_t1_complex_then_t4_draft() -> None:
    # Call 1: T1 classifier → "COMPLEX"
    # Call 2: T4 local draft → "draft"
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["COMPLEX", "my draft"],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="APPROVED\nmy draft",
        usage=Usage(input_tokens=30, output_tokens=5),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_config(t4=True, t1=True))

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "draft+cloud"
    stages = [e.stage for e in resp.trace]
    assert "t1_classify" in stages
    assert "t4_draft" in stages
    assert "t4_review" in stages
