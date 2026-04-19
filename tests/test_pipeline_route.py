"""Unit tests for T1 route — the local classifier triage.

Tests cover:
- classify() in isolation
- apply() end-to-end (TRIVIAL → local answer, COMPLEX → pass-through)
- Fail-open on local model errors
- Unparseable classifier output defaults to COMPLEX
- Pipeline.complete integration with T1 enabled
- Explicit model_hint bypasses T1
- Stats accounting for locally-routed requests
"""

from __future__ import annotations

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.route import (
    _extract_user_text,
    _parse_classification,
    apply,
    classify,
)

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t1_config(*, enabled: bool = True, **params) -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m"
        ),
        tactics=TacticsConfig(
            t1_route=enabled,
            params={"t1_route": params} if params else {},
        ),
    )


_MSGS = [{"role": "user", "content": "what is 2+2?"}]


# ---------------------------------------------------------------------------
# Unit: _parse_classification
# ---------------------------------------------------------------------------

class TestParseClassification:
    def test_trivial(self) -> None:
        assert _parse_classification("TRIVIAL") == "TRIVIAL"

    def test_complex(self) -> None:
        assert _parse_classification("COMPLEX") == "COMPLEX"

    def test_case_insensitive(self) -> None:
        assert _parse_classification("trivial") == "TRIVIAL"
        assert _parse_classification("Complex") == "COMPLEX"

    def test_surrounded_by_noise(self) -> None:
        assert _parse_classification("I think TRIVIAL.") == "TRIVIAL"

    def test_garbage_defaults_to_complex(self) -> None:
        assert _parse_classification("banana") == "COMPLEX"

    def test_empty_defaults_to_complex(self) -> None:
        assert _parse_classification("") == "COMPLEX"


# ---------------------------------------------------------------------------
# Unit: _extract_user_text
# ---------------------------------------------------------------------------

class TestExtractUserText:
    def test_single_user_message(self) -> None:
        assert _extract_user_text([{"role": "user", "content": "hi"}]) == "hi"

    def test_last_user_message_wins(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_user_text(msgs) == "second"

    def test_no_user_returns_empty(self) -> None:
        assert _extract_user_text([{"role": "system", "content": "sys"}]) == ""


# ---------------------------------------------------------------------------
# Async: classify()
# ---------------------------------------------------------------------------

async def test_classify_trivial() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="TRIVIAL")
    decision, event = await classify(_MSGS, local=local)
    assert decision == "TRIVIAL"
    assert event.stage == "t1_classify"
    assert event.decision == "TRIVIAL"
    assert event.tokens_in == 10
    # Classifier call was made with temperature=0 and max_tokens=3
    assert local.calls[0]["temperature"] == 0.0
    assert local.calls[0]["max_tokens"] == 10


async def test_classify_complex() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="COMPLEX")
    decision, event = await classify(_MSGS, local=local)
    assert decision == "COMPLEX"
    assert event.decision == "COMPLEX"


async def test_classify_garbage_defaults_complex() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="maybe???")
    decision, event = await classify(_MSGS, local=local)
    assert decision == "COMPLEX"


async def test_classify_model_error_fails_open() -> None:
    local = FakeChatClient(raise_on_complete=ModelBackendError("unreachable"))
    decision, event = await classify(_MSGS, local=local)
    assert decision == "COMPLEX"
    assert event.decision == "ERROR"
    assert "unreachable" in event.detail["error"]


# ---------------------------------------------------------------------------
# Async: apply()
# ---------------------------------------------------------------------------

async def test_apply_trivial_answers_locally() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "the answer is 4"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    result = await apply(_MSGS, local=local)
    assert result.classification == "TRIVIAL"
    assert result.local_reply is not None
    assert result.local_reply.content == "the answer is 4"
    assert len(result.events) == 2
    assert result.events[0].stage == "t1_classify"
    assert result.events[1].stage == "t1_local_answer"
    assert result.events[1].decision == "APPLIED"
    # Two calls: classify + answer
    assert len(local.calls) == 2


async def test_apply_complex_no_local_reply() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="COMPLEX")
    result = await apply(_MSGS, local=local)
    assert result.classification == "COMPLEX"
    assert result.local_reply is None
    assert len(result.events) == 1
    assert result.events[0].stage == "t1_classify"
    # Only the classifier was called, not the answer path.
    assert len(local.calls) == 1


async def test_apply_classify_error_falls_back_to_complex() -> None:
    local = FakeChatClient(raise_on_complete=ModelBackendError("down"))
    result = await apply(_MSGS, local=local)
    assert result.classification == "COMPLEX"
    assert result.local_reply is None
    assert result.events[0].decision == "ERROR"


async def test_apply_answer_error_falls_back_to_complex() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        raise_sequence=[None, ModelBackendError("answer failed")],
        reply_content="TRIVIAL",
    )
    result = await apply(_MSGS, local=local)
    assert result.classification == "COMPLEX"
    assert result.local_reply is None
    assert len(result.events) == 2
    assert result.events[0].decision == "TRIVIAL"
    assert result.events[1].decision == "ERROR"
    assert "answer failed" in result.events[1].detail["error"]


async def test_apply_forwards_generation_params() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "ok"],
    )
    await apply(
        _MSGS,
        local=local,
        temperature=0.7,
        max_tokens=128,
        seed=42,
    )
    # The answer call (second call) should carry the generation params.
    answer_call = local.calls[1]
    assert answer_call["temperature"] == 0.7
    assert answer_call["max_tokens"] == 128
    assert answer_call["seed"] == 42


async def test_apply_force_complex_tool_name() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "should-not-run"],
    )
    result = await apply(
        _MSGS,
        local=local,
        params={"force_complex_tools": ["Edit"]},
        meta={"tool_name": "Edit"},
    )
    assert result.classification == "COMPLEX"
    assert result.local_reply is None
    assert len(local.calls) == 0
    assert result.events[0].decision == "FORCED_COMPLEX"


async def test_apply_verify_trivial_requires_second_trivial() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "TRIVIAL", "final answer"],
    )
    result = await apply(_MSGS, local=local, params={"verify_trivial": True})
    assert result.classification == "TRIVIAL"
    assert result.local_reply is not None
    assert len(local.calls) == 3


async def test_apply_verify_trivial_mismatch_falls_back() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "COMPLEX"],
    )
    result = await apply(_MSGS, local=local, params={"verify_trivial": True})
    assert result.classification == "COMPLEX"
    assert result.local_reply is None
    assert len(local.calls) == 2


async def test_apply_trivial_threshold_triggers_second_vote() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "TRIVIAL", "final"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    result = await apply(
        _MSGS,
        local=local,
        params={"trivial_threshold": 0.8},
    )
    assert result.classification == "TRIVIAL"
    assert result.local_reply is not None
    assert len(local.calls) == 3


# ---------------------------------------------------------------------------
# Integration: Pipeline.complete with T1 enabled
# ---------------------------------------------------------------------------

async def test_pipeline_t1_trivial_routes_locally() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "local answer"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=_MSGS)
    )

    assert resp.served_by == "local"
    assert resp.content == "local answer"
    assert resp.usage_local.input_tokens == 5
    assert resp.usage_cloud.input_tokens is None
    assert cloud.calls == []
    assert len(local.calls) == 2  # classify + answer
    assert len(resp.trace) == 2
    assert resp.trace[0].stage == "t1_classify"
    assert resp.trace[1].stage == "t1_local_answer"


async def test_pipeline_t1_complex_falls_to_cloud() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="COMPLEX")
    cloud = FakeChatClient(
        chat_model="cloud-m",
        reply_content="cloud answer",
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "cloud"
    assert resp.content == "cloud answer"
    assert resp.usage_cloud.input_tokens == 20
    assert len(resp.trace) == 2  # t1_classify + cloud_call
    assert resp.trace[0].stage == "t1_classify"
    assert resp.trace[0].decision == "COMPLEX"
    assert resp.trace[1].stage == "cloud_call"


async def test_pipeline_t1_classifier_error_falls_to_cloud() -> None:
    local = FakeChatClient(raise_on_complete=ModelBackendError("oops"))
    cloud = FakeChatClient(
        chat_model="cloud-m",
        reply_content="cloud answer",
    )
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "cloud"
    assert resp.trace[0].stage == "t1_classify"
    assert resp.trace[0].decision == "ERROR"
    assert resp.trace[1].stage == "cloud_call"


async def test_pipeline_t1_skipped_for_explicit_cloud_hint() -> None:
    local = FakeChatClient(chat_model="local-m", reply_content="TRIVIAL")
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=_MSGS, model_hint="cloud")
    )

    assert resp.served_by == "cloud"
    assert local.calls == []  # T1 never ran


async def test_pipeline_t1_skipped_for_explicit_local_hint() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_content="direct local",
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    resp = await pipeline.complete(
        PipelineRequest(messages=_MSGS, model_hint="local")
    )

    assert resp.served_by == "local"
    assert resp.content == "direct local"
    # Only one call — direct answer, no classifier.
    assert len(local.calls) == 1
    assert len(resp.trace) == 1
    assert resp.trace[0].stage == "local_call"


async def test_pipeline_t1_disabled_skips_routing() -> None:
    local = FakeChatClient(chat_model="local-m")
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config(enabled=False))

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "cloud"
    assert local.calls == []


async def test_pipeline_t1_no_local_backend_skips_routing() -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    cfg = Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        tactics=TacticsConfig(t1_route=True),
    )
    pipeline = Pipeline(cloud=cloud, local=None, config=cfg)

    resp = await pipeline.complete(PipelineRequest(messages=_MSGS))

    assert resp.served_by == "cloud"


async def test_pipeline_t1_stats_count_locally_routed() -> None:
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "ans"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=_t1_config())

    await pipeline.complete(PipelineRequest(messages=_MSGS))

    snap = pipeline.stats()
    assert snap.total_requests == 1
    assert snap.by_served["local"] == 1
    assert snap.tokens_in_local == 5
    assert snap.tokens_out_local == 2
    assert snap.tokens_in_cloud == 0


async def test_pipeline_adaptive_stats_hint() -> None:
    from local_splitter.config import AdaptiveConfig

    base = _t1_config()
    cfg = Config(
        cloud=base.cloud,
        local=base.local,
        tactics=base.tactics,
        adaptive=AdaptiveConfig(
            enabled=True,
            min_requests=1,
            max_local_fraction=0.2,
        ),
    )
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "a"],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    cloud = FakeChatClient(chat_model="cloud-m")
    pipeline = Pipeline(cloud=cloud, local=local, config=cfg)

    await pipeline.complete(PipelineRequest(messages=_MSGS))
    snap = pipeline.stats()
    assert snap.latency_sample_size >= 1
    assert len(snap.adaptive_hints) == 1
