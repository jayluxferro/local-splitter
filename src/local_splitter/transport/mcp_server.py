"""MCP stdio server built on FastMCP.

Exposes the pipeline as a set of MCP tools for agents that speak MCP
natively (Claude Code, Cursor-via-MCP, Codex CLI MCP, etc.). Tool names
follow the `<project>.<verb>` convention shared with the sibling
`resilient-write` / `llm-redactor` projects (`.agent/memory/gotchas.md`).

Supports two modes:

- **Full mode** (cloud + local configured): ``split.complete`` runs
  tactics and returns the cloud response.
- **Local-only mode** (no cloud backend): ``split.complete`` and
  ``split.transform`` run tactics locally and return either a local
  answer (T1 trivial / T3 cache hit) or the transformed messages
  for the calling agent to send to its own model.

Tools
-----
- `split.complete` — full pipeline; in local-only mode returns answer
  or passthrough with transformed messages.
- `split.transform` — run transforms only, never calls a backend.
  Returns answer or transformed messages.
- `split.classify` — T1 classifier only (TRIVIAL / COMPLEX).
- `split.cache_lookup` — T3 cache read-only lookup.
- `split.stats` — aggregate metrics since process start.
- `split.config` — read-only config view.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from local_splitter.config import Config, TACTIC_DISABLE_NAMES
from local_splitter.models import Message, ModelBackendError
from local_splitter.pipeline import Pipeline, PipelineError, PipelineRequest

_log = logging.getLogger(__name__)


def _mcp_tactics_override(raw: list[str] | None) -> frozenset[str] | None:
    if not raw:
        return None
    out = {str(x).strip() for x in raw if str(x).strip() in TACTIC_DISABLE_NAMES}
    return frozenset(out) if out else None


def create_mcp_server(pipeline: Pipeline, config: Config) -> FastMCP:
    """Build an MCP stdio server that wraps `pipeline`."""
    server = FastMCP("local-splitter")

    @server.tool(
        name="split.complete",
        description=(
            "Run a chat completion through the splitter pipeline. "
            "If a cloud backend is configured, returns the full response. "
            "If running in local-only mode (no cloud), returns a local "
            "response for trivial requests or the transformed prompt for "
            "complex ones (use split.transform for transform-only)."
        ),
    )
    async def split_complete(
        messages: list[Message],
        model_hint: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        session_id: str | None = None,
        tool_name: str | None = None,
        tag: str | None = None,
        tactics_disable: list[str] | None = None,
    ) -> dict[str, Any]:
        if model_hint not in ("auto", "local", "cloud"):
            raise ValueError(f"model_hint must be auto|local|cloud, got {model_hint!r}")

        req = PipelineRequest(
            messages=messages,
            model_hint=model_hint,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            meta={
                "session_id": session_id,
                "tool_name": tool_name,
                "tag": tag,
            },
            tactics_override=_mcp_tactics_override(tactics_disable),
        )

        # Local-only mode: no cloud backend configured.
        if pipeline.cloud is None:
            transformed, trace, local_response = await pipeline.transform(req)
            if local_response is not None:
                return {
                    "response": local_response,
                    "served_by": "local",
                    "action": "answer",
                    "pipeline_trace": [e.as_dict() for e in trace],
                }
            return {
                "response": None,
                "served_by": "none",
                "action": "passthrough",
                "transformed_messages": transformed,
                "note": (
                    "No cloud backend configured. Use the transformed "
                    "messages with your own model."
                ),
                "pipeline_trace": [e.as_dict() for e in trace],
            }

        # Full mode: cloud backend available.
        try:
            resp = await pipeline.complete(req)
        except PipelineError as e:
            return {"error": {"type": "pipeline_error", "message": str(e)}}
        except ModelBackendError as e:
            _log.warning("backend error in split.complete: %s", e)
            return {"error": {"type": "backend_error", "message": str(e)}}

        return {
            "response": resp.content,
            "served_by": resp.served_by,
            "action": "answer",
            "finish_reason": resp.finish_reason,
            "model": resp.model,
            "tokens": {
                "input_cloud": resp.usage_cloud.input_tokens or 0,
                "output_cloud": resp.usage_cloud.output_tokens or 0,
                "input_local": resp.usage_local.input_tokens or 0,
                "output_local": resp.usage_local.output_tokens or 0,
            },
            "latency_ms": round(resp.latency_ms, 3),
            "pipeline_trace": [e.as_dict() for e in resp.trace],
        }

    @server.tool(
        name="split.transform",
        description=(
            "Run tactic transforms on messages without calling any backend. "
            "Returns either a local answer (T1 trivial / T3 cache hit) or "
            "the transformed messages for the caller to send to its own "
            "model. This is the primary tool for local-only mode."
        ),
    )
    async def split_transform(
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        tool_name: str | None = None,
        tag: str | None = None,
        tactics_disable: list[str] | None = None,
    ) -> dict[str, Any]:
        req = PipelineRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            meta={
                "session_id": session_id,
                "tool_name": tool_name,
                "tag": tag,
            },
            tactics_override=_mcp_tactics_override(tactics_disable),
        )
        try:
            transformed, trace, local_response = await pipeline.transform(req)
        except Exception as exc:
            _log.warning("split.transform error: %s", exc)
            return {"error": {"type": "transform_error", "message": str(exc)}}

        if local_response is not None:
            return {
                "action": "answer",
                "response": local_response,
                "served_by": "local",
                "pipeline_trace": [e.as_dict() for e in trace],
            }
        return {
            "action": "passthrough",
            "transformed_messages": transformed,
            "served_by": "none",
            "pipeline_trace": [e.as_dict() for e in trace],
        }

    @server.tool(
        name="split.classify",
        description=(
            "Run only T1 (route classifier) without generating a response. "
            "Returns TRIVIAL or COMPLEX.  Returns NOT_IMPLEMENTED when T1 "
            "is disabled or no local backend is configured."
        ),
    )
    async def split_classify(messages: list[Message]) -> dict[str, Any]:
        if not config.tactics.t1_route or pipeline.local is None:
            return {
                "decision": "NOT_IMPLEMENTED",
                "stage": "t1_route",
                "note": "T1 route is disabled or no local backend configured",
            }
        from local_splitter.pipeline.route import classify

        try:
            decision, event = await classify(
                messages,
                local=pipeline.local,
                params=config.tactics.params.get("t1_route"),
            )
        except Exception as exc:
            _log.warning("split.classify error: %s", exc)
            return {
                "decision": "ERROR",
                "stage": "t1_route",
                "error": str(exc),
            }
        return {
            "decision": decision,
            "stage": "t1_route",
            **event.as_dict(),
        }

    @server.tool(
        name="split.cache_lookup",
        description=(
            "Look up a request in the T3 semantic cache without writing. "
            "Optional ``meta`` (e.g. tool_name, tag, session_id) is passed through "
            "to T3 skip/namespace rules. Returns NOT_IMPLEMENTED when T3 is disabled "
            "or no local backend."
        ),
    )
    async def split_cache_lookup(
        messages: list[Message],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            not config.tactics.t3_sem_cache
            or pipeline.local is None
            or pipeline.cache_store is None
        ):
            return {
                "hit": False,
                "stage": "t3_sem_cache",
                "note": "T3 sem_cache is disabled or no local backend / cache store",
            }
        from local_splitter.pipeline.sem_cache import lookup

        try:
            result = await lookup(
                messages,
                local=pipeline.local,
                store=pipeline.cache_store,
                params=config.tactics.params.get("t3_sem_cache"),
                meta=meta,
            )
        except Exception as exc:
            _log.warning("split.cache_lookup error: %s", exc)
            return {"hit": False, "stage": "t3_sem_cache", "error": str(exc)}

        out: dict[str, Any] = {
            "hit": result.hit,
            "stage": "t3_sem_cache",
        }
        if result.entry is not None:
            out["similarity"] = round(result.entry.similarity, 4)
            out["response_preview"] = result.entry.response[:200]
        if result.events:
            out["event"] = result.events[0].as_dict()
        return out

    @server.tool(
        name="split.stats",
        description="Aggregate pipeline metrics since process start.",
    )
    async def split_stats() -> dict[str, Any]:
        return asdict(pipeline.stats())

    @server.tool(
        name="split.config",
        description="Read-only view of the loaded splitter config.",
    )
    async def split_config() -> dict[str, Any]:
        return _safe_config_view(config)

    return server


def _safe_config_view(config: Config) -> dict[str, Any]:
    """Convert `Config` to a plain dict, hiding secrets.

    We never expose an API key value — only the name of the env var that
    holds it. This matches the OpenAI-compat client's design.
    """
    def model_view(mc) -> dict[str, Any]:
        return {
            "backend": mc.backend,
            "endpoint": mc.endpoint,
            "chat_model": mc.chat_model,
            "embed_model": mc.embed_model,
            "api_key_env": mc.api_key_env,
            "num_ctx": mc.num_ctx,
        }

    return {
        "version": config.version,
        "transport": asdict(config.transport),
        "models": {
            "cloud": model_view(config.cloud) if config.cloud is not None else None,
            "local": model_view(config.local) if config.local is not None else None,
        },
        "tactics": {
            "t1_route": config.tactics.t1_route,
            "t2_compress": config.tactics.t2_compress,
            "t3_sem_cache": config.tactics.t3_sem_cache,
            "t4_draft": config.tactics.t4_draft,
            "t5_diff": config.tactics.t5_diff,
            "t6_intent": config.tactics.t6_intent,
            "t7_batch": config.tactics.t7_batch,
            "params": config.tactics.params,
        },
        "adaptive": {
            "enabled": config.adaptive.enabled,
            "min_requests": config.adaptive.min_requests,
            "max_local_fraction": config.adaptive.max_local_fraction,
        },
        "log_file": str(config.log_file) if config.log_file else None,
    }


__all__ = ["create_mcp_server"]
