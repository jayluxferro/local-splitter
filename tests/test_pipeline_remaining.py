"""Unit tests for T5 diff, T6 intent, and T7 batch."""

from __future__ import annotations

import json

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.models import ModelBackendError, Usage
from local_splitter.pipeline import Pipeline, PipelineRequest
from local_splitter.pipeline.batch import apply as batch_apply
from local_splitter.pipeline.diff import apply as diff_apply
from local_splitter.pipeline.intent import apply as intent_apply

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_CODE = "```python\n" + ("x = 1\n" * 200) + "```\n"  # > 500 chars
LONG_QUERY = "I need you to " + "explain this function " * 20  # > 100 chars


def _config(**tactics_kw) -> Config:
    return Config(
        cloud=ModelConfig(backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"),
        local=ModelConfig(backend="ollama", endpoint="http://local", chat_model="local-m"),
        tactics=TacticsConfig(**tactics_kw),
    )


# ===========================================================================
# T5 diff
# ===========================================================================

class TestDiffApply:
    async def test_edit_request_gets_minimized(self) -> None:
        local = FakeChatClient(
            chat_model="local-m", reply_content="minimal diff context",
            usage=Usage(input_tokens=50, output_tokens=10),
        )
        msgs = [
            {"role": "user", "content": f"Please edit this file:\n{LONG_CODE}"},
        ]
        result = await diff_apply(msgs, local=local)
        assert result.events[0].decision == "APPLIED"
        assert len(result.messages[0]["content"]) < len(msgs[0]["content"])

    async def test_non_edit_skipped(self) -> None:
        local = FakeChatClient(chat_model="local-m")
        msgs = [{"role": "user", "content": "what is a monad?"}]
        result = await diff_apply(msgs, local=local)
        assert result.events[0].decision == "SKIP"
        assert local.calls == []

    async def test_error_fails_open(self) -> None:
        local = FakeChatClient(raise_on_complete=ModelBackendError("down"))
        msgs = [{"role": "user", "content": f"edit this:\n{LONG_CODE}"}]
        result = await diff_apply(msgs, local=local)
        assert result.events[0].decision == "ERROR"
        assert result.messages == msgs  # original kept


class TestDiffPipeline:
    async def test_t5_wired_in_pipeline(self) -> None:
        local = FakeChatClient(
            chat_model="local-m", reply_content="minimal diff",
            usage=Usage(input_tokens=50, output_tokens=10),
        )
        cloud = FakeChatClient(
            chat_model="cloud-m", reply_content="done",
            usage=Usage(input_tokens=20, output_tokens=5),
        )
        pipeline = Pipeline(cloud=cloud, local=local, config=_config(t5_diff=True))

        resp = await pipeline.complete(
            PipelineRequest(messages=[
                {"role": "user", "content": f"edit:\n{LONG_CODE}"},
            ])
        )
        assert resp.served_by == "cloud"
        assert any(e.stage == "t5_diff" for e in resp.trace)


# ===========================================================================
# T6 intent
# ===========================================================================

class TestIntentApply:
    async def test_extracts_intent_from_long_query(self) -> None:
        intent_json = json.dumps({
            "intent": "explain",
            "target": "monads",
            "constraints": ["concise"],
            "query": "What is a monad?",
        })
        local = FakeChatClient(
            chat_model="local-m", reply_content=intent_json,
            usage=Usage(input_tokens=30, output_tokens=15),
        )
        msgs = [{"role": "user", "content": LONG_QUERY}]
        result = await intent_apply(msgs, local=local)
        assert result.events[0].decision == "APPLIED"
        assert "Intent: explain" in result.messages[0]["content"]

    async def test_short_query_skipped(self) -> None:
        local = FakeChatClient(chat_model="local-m")
        msgs = [{"role": "user", "content": "hi"}]
        result = await intent_apply(msgs, local=local)
        assert result.events[0].decision == "SKIP"
        assert local.calls == []

    async def test_bad_json_fails_open(self) -> None:
        local = FakeChatClient(
            chat_model="local-m", reply_content="not json at all",
        )
        msgs = [{"role": "user", "content": LONG_QUERY}]
        result = await intent_apply(msgs, local=local)
        assert result.events[0].decision == "PARSE_ERROR"
        assert result.messages == msgs

    async def test_error_fails_open(self) -> None:
        local = FakeChatClient(raise_on_complete=ModelBackendError("down"))
        msgs = [{"role": "user", "content": LONG_QUERY}]
        result = await intent_apply(msgs, local=local)
        assert result.events[0].decision == "ERROR"
        assert result.messages == msgs


class TestIntentPipeline:
    async def test_t6_wired_in_pipeline(self) -> None:
        intent_json = json.dumps({
            "intent": "explain", "target": "x", "constraints": [], "query": "q",
        })
        local = FakeChatClient(
            chat_model="local-m", reply_content=intent_json,
            usage=Usage(input_tokens=30, output_tokens=15),
        )
        cloud = FakeChatClient(
            chat_model="cloud-m", reply_content="answer",
            usage=Usage(input_tokens=20, output_tokens=5),
        )
        pipeline = Pipeline(cloud=cloud, local=local, config=_config(t6_intent=True))

        resp = await pipeline.complete(
            PipelineRequest(messages=[{"role": "user", "content": LONG_QUERY}])
        )
        assert any(e.stage == "t6_intent" for e in resp.trace)


# ===========================================================================
# T7 batch (prompt-cache tagging)
# ===========================================================================

LONG_SYSTEM = "s" * 600  # > MIN_PREFIX_LEN (500)


class TestBatchApply:
    def test_tags_long_system_message(self) -> None:
        msgs = [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "hi"},
        ]
        result = batch_apply(msgs)
        assert result.events[0].decision == "APPLIED"
        assert result.messages[0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in result.messages[1]

    def test_skips_short_system(self) -> None:
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        result = batch_apply(msgs)
        assert result.events[0].decision == "SKIP"

    def test_no_system_messages(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = batch_apply(msgs)
        assert result.events[0].decision == "SKIP"


class TestBatchPipeline:
    async def test_t7_wired_in_pipeline(self) -> None:
        cloud = FakeChatClient(
            chat_model="cloud-m", reply_content="answer",
            usage=Usage(input_tokens=20, output_tokens=5),
        )
        local = FakeChatClient(chat_model="local-m")
        pipeline = Pipeline(cloud=cloud, local=local, config=_config(t7_batch=True))

        resp = await pipeline.complete(
            PipelineRequest(messages=[
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": "hi"},
            ])
        )
        assert any(e.stage == "t7_batch" for e in resp.trace)
        # Cloud received the tagged messages.
        assert cloud.calls[0]["messages"][0].get("cache_control") == {"type": "ephemeral"}
