"""OpenAI-compatible HTTP proxy in front of the pipeline.

An agent that can be pointed at a custom `OPENAI_API_BASE` can use this
transport transparently:

    export OPENAI_API_BASE=http://127.0.0.1:7788/v1
    export OPENAI_API_KEY=unused

Endpoints
---------
- `POST /v1/chat/completions`: streaming (SSE) and non-streaming.
  Tactics run on the request before streaming begins; T4 (draft-review)
  is skipped in streaming mode.
- `GET  /v1/models`: returns the configured local + cloud models.
- `GET  /v1/splitter/stats`: returns aggregate pipeline metrics.

The `splitter` key on responses is the extension that tactics use to
report what happened. Standard OpenAI clients ignore unknown keys.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from local_splitter import __version__
from local_splitter.config import Config
from local_splitter.models import ModelBackendError
from local_splitter.pipeline import Pipeline, PipelineError, PipelineRequest

_log = logging.getLogger(__name__)


def create_app(pipeline: Pipeline, config: Config) -> FastAPI:
    """Build a FastAPI app that serves the OpenAI-compatible surface."""
    app = FastAPI(
        title="local-splitter",
        version=__version__,
        description=(
            "OpenAI-compatible proxy that splits LLM calls between a local "
            "and cloud model. See AGENT.md for the seven tactics."
        ),
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="messages must be a non-empty list"
            )

        extra_body = body.get("extra_body") or {}
        splitter_opts = extra_body.get("splitter") or {}
        model_hint: Any = "auto"
        if splitter_opts.get("force_local"):
            model_hint = "local"
        elif splitter_opts.get("force_cloud"):
            model_hint = "cloud"

        pipeline_req = PipelineRequest(
            messages=messages,
            model_hint=model_hint,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            stop=body.get("stop"),
            seed=body.get("seed"),
            stream=bool(body.get("stream")),
            meta={
                "tool_name": splitter_opts.get("tool_name"),
                "tag": splitter_opts.get("tag"),
                "model_requested": body.get("model"),
            },
        )

        if pipeline_req.stream:
            return StreamingResponse(
                _sse_generator(pipeline, pipeline_req, body.get("model")),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            pr = await pipeline.complete(pipeline_req)
        except PipelineError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ModelBackendError as e:
            _log.warning("backend error serving /v1/chat/completions: %s", e)
            raise HTTPException(status_code=502, detail=f"backend error: {e}") from e

        return JSONResponse(_pipeline_to_openai(pr, request_model=body.get("model")))

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        created = int(time.time())
        data.append(
            {
                "id": config.cloud.chat_model,
                "object": "model",
                "created": created,
                "owned_by": f"cloud:{config.cloud.backend}",
            }
        )
        if config.local is not None:
            data.append(
                {
                    "id": config.local.chat_model,
                    "object": "model",
                    "created": created,
                    "owned_by": f"local:{config.local.backend}",
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/splitter/stats")
    async def splitter_stats() -> dict[str, Any]:
        return asdict(pipeline.stats())

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


async def _sse_generator(
    pipeline: Pipeline, req: PipelineRequest, request_model: str | None
):
    """Yield OpenAI-compatible SSE chunks from the pipeline's stream."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        async for chunk in pipeline.stream(req):
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model or "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.delta} if chunk.delta else {},
                        "finish_reason": chunk.finish_reason if chunk.done else None,
                    }
                ],
            }
            if chunk.done and chunk.usage:
                data["usage"] = {
                    "prompt_tokens": chunk.usage.input_tokens or 0,
                    "completion_tokens": chunk.usage.output_tokens or 0,
                    "total_tokens": (chunk.usage.input_tokens or 0) + (chunk.usage.output_tokens or 0),
                }
            yield f"data: {json.dumps(data)}\n\n"
    except (PipelineError, ModelBackendError) as e:
        _log.warning("streaming error: %s", e)
        error_data = {"error": {"message": str(e), "type": type(e).__name__}}
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"


def _pipeline_to_openai(pr, *, request_model: str | None) -> dict[str, Any]:
    """Translate a `PipelineResponse` into an OpenAI chat completion dict.

    We report the *served-by* model in the `model` field so the caller
    can tell whether the cloud or the local model was used. The raw
    pipeline trace is attached under `splitter` for tactics debugging.
    """
    completion_tokens = pr.usage_cloud.output_tokens or 0
    prompt_tokens = pr.usage_cloud.input_tokens or 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": pr.model or request_model or "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": pr.content},
                "finish_reason": pr.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "splitter": {
            "served_by": pr.served_by,
            "latency_ms": round(pr.latency_ms, 3),
            "pipeline_trace": [e.as_dict() for e in pr.trace],
            "tokens_local": {
                "input": pr.usage_local.input_tokens or 0,
                "output": pr.usage_local.output_tokens or 0,
            },
        },
    }


__all__ = ["create_app"]
