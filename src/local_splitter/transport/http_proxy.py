"""Multi-API HTTP proxy in front of the pipeline.

Supports both API surfaces so any agent can use local-splitter as a
drop-in replacement:

    # OpenAI-compatible agents (Cursor, Codex CLI, etc.)
    export OPENAI_API_BASE=http://127.0.0.1:7788/v1

    # Anthropic-compatible agents (Claude Code, etc.)
    export ANTHROPIC_BASE_URL=http://127.0.0.1:7788

Endpoints
---------
- `POST /v1/chat/completions`: OpenAI format, streaming + non-streaming.
- `POST /v1/messages`: Anthropic format, streaming + non-streaming.
- `GET  /v1/models`: returns the configured local + cloud models.
- `GET  /v1/splitter/stats`: returns aggregate pipeline metrics.

The `splitter` key on responses is the extension that tactics use to
report what happened. Standard clients ignore unknown keys.
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

import httpx

from local_splitter import __version__
from local_splitter.config import Config
from local_splitter.models import ModelBackendError
from local_splitter.pipeline import Pipeline, PipelineError, PipelineRequest

_log = logging.getLogger(__name__)

# Headers that should not be forwarded between pipeline hops.
_HOP_HEADERS = frozenset({
    "host", "transfer-encoding", "connection",
    "content-length", "content-encoding",
})


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

        # Forward all headers from the incoming request (minus hop-by-hop).
        upstream_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_HEADERS
        }

        # Tool-bearing requests bypass the pipeline (which can't represent
        # tool_calls / tool role messages) and go directly to a backend.
        # Try local ollama first, fall back to cloud transparent proxy.
        if "tools" in body or "functions" in body:
            if config.local is not None and config.local.backend == "ollama":
                try:
                    return await _local_openai_tool_proxy(body, config.local)
                except Exception as exc:
                    _log.warning("local openai tool proxy failed, falling back to cloud: %s", exc)
            if config.cloud is not None:
                return await _transparent_openai_proxy(
                    body, config.cloud.endpoint, upstream_headers,
                )

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
            upstream_headers=upstream_headers,
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

    # ------------------------------------------------------------------ #
    #  Anthropic-compatible surface: POST /v1/messages                    #
    # ------------------------------------------------------------------ #

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        try:
            body = await request.json()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        # Forward all headers from the incoming request (minus hop-by-hop).
        upstream_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_HEADERS
        }

        # Tool-bearing requests bypass the pipeline (which can't represent
        # tool_use / tool_result blocks) and go directly to a backend.
        # Try local ollama first (if available), fall back to cloud.
        if "tools" in body:
            if config.local is not None and config.local.backend == "ollama":
                try:
                    return await _local_tool_proxy(
                        body, config.local, upstream_headers,
                    )
                except Exception as exc:
                    _log.warning("local tool proxy failed, falling back to cloud: %s", exc)
            if config.cloud is not None:
                return await _transparent_proxy(
                    body, config.cloud.endpoint, upstream_headers,
                )

        messages = _anthropic_to_pipeline_messages(body)
        if not messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")

        pipeline_req = PipelineRequest(
            messages=messages,
            model_hint="auto",
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            stop=body.get("stop_sequences"),
            stream=bool(body.get("stream")),
            upstream_headers=upstream_headers,
            meta={"model_requested": body.get("model")},
        )

        if pipeline_req.stream:
            return StreamingResponse(
                _anthropic_sse_generator(pipeline, pipeline_req, body.get("model")),
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
            _log.warning("backend error serving /v1/messages: %s", e)
            raise HTTPException(status_code=502, detail=f"backend error: {e}") from e

        return JSONResponse(_pipeline_to_anthropic(pr, request_model=body.get("model")))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


async def _transparent_proxy(
    body: dict[str, Any],
    upstream_endpoint: str,
    headers: dict[str, str],
) -> StreamingResponse | JSONResponse:
    """Bypass the pipeline — forward the raw request/response unchanged.

    Used for tool-bearing requests that the pipeline can't represent.
    """
    url = f"{upstream_endpoint.rstrip('/')}/v1/messages"
    is_stream = body.get("stream", False)

    _log.debug("transparent proxy → %s (stream=%s)", url, is_stream)

    if is_stream:
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

        async def stream_and_close():
            try:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_and_close(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(url, json=body, headers=headers)

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
    }
    from starlette.responses import Response
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI/Ollama format."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return out


def _anthropic_messages_to_openai(
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert an Anthropic request body's messages to OpenAI/Ollama format.

    Handles text, tool_use, and tool_result content blocks.
    """
    msgs: list[dict[str, Any]] = []

    system = body.get("system")
    if system:
        text = system if isinstance(system, str) else " ".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
        msgs.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            msgs.append({"role": role, "content": str(content)})
            continue

        # Content is a list of blocks — separate text, tool_use, and tool_result.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(
                        b.get("text", "") for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(result_content),
                })

        if role == "assistant" and tool_calls:
            msgs.append({
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else "",
                "tool_calls": tool_calls,
            })
        elif tool_results:
            # tool_result blocks come inside a user message in Anthropic format;
            # in OpenAI format they become separate tool-role messages.
            if text_parts:
                msgs.append({"role": role, "content": "\n".join(text_parts)})
            msgs.extend(tool_results)
        else:
            msgs.append({"role": role, "content": "\n".join(text_parts)})

    return msgs


def _openai_response_to_anthropic(
    data: dict[str, Any],
    request_model: str | None,
) -> dict[str, Any]:
    """Convert an Ollama /api/chat response to Anthropic Messages format."""
    message = data.get("message") or {}
    content_blocks: list[dict[str, Any]] = []

    text = message.get("content", "")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": args,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = "end_turn"
    if any(b["type"] == "tool_use" for b in content_blocks):
        stop_reason = "tool_use"

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": request_model or data.get("model", ""),
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        },
    }


async def _local_tool_proxy(
    body: dict[str, Any],
    local_config: "Config.local.__class__",
    headers: dict[str, str],
) -> JSONResponse | StreamingResponse:
    """Route a tool-bearing request to local ollama with format conversion.

    Converts Anthropic Messages format → Ollama /api/chat format,
    calls ollama, and converts the response back.
    """
    ollama_body: dict[str, Any] = {
        "model": local_config.chat_model,
        "messages": _anthropic_messages_to_openai(body),
        "stream": False,
    }

    tools = body.get("tools")
    if tools:
        ollama_body["tools"] = _anthropic_tools_to_openai(tools)

    tool_choice = body.get("tool_choice")
    if tool_choice:
        ollama_body["tool_choice"] = tool_choice

    if body.get("temperature") is not None:
        ollama_body["options"] = {"temperature": body["temperature"]}

    url = f"{local_config.endpoint.rstrip('/')}/api/chat"
    _log.debug("local tool proxy → %s", url)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(url, json=ollama_body)

    if resp.status_code != 200:
        raise RuntimeError(f"ollama returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    anthropic_resp = _openai_response_to_anthropic(data, body.get("model"))
    return JSONResponse(content=anthropic_resp)


async def _local_openai_tool_proxy(
    body: dict[str, Any],
    local_config: "Config.local.__class__",
) -> JSONResponse:
    """Route an OpenAI-format tool request to local ollama.

    OpenAI and Ollama share the same tool format, so no conversion needed —
    just forward to ollama's /api/chat endpoint.
    """
    ollama_body: dict[str, Any] = {
        "model": local_config.chat_model,
        "messages": body.get("messages", []),
        "stream": False,
    }
    if "tools" in body:
        ollama_body["tools"] = body["tools"]
    if "functions" in body:
        ollama_body["functions"] = body["functions"]
    if "tool_choice" in body:
        ollama_body["tool_choice"] = body["tool_choice"]
    if body.get("temperature") is not None:
        ollama_body["options"] = {"temperature": body["temperature"]}

    url = f"{local_config.endpoint.rstrip('/')}/api/chat"
    _log.debug("local openai tool proxy → %s", url)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(url, json=ollama_body)

    if resp.status_code != 200:
        raise RuntimeError(f"ollama returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    message = data.get("message", {})

    # Build OpenAI-compatible response from ollama response.
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", data.get("model", "")),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": (data.get("prompt_eval_count", 0)
                             + data.get("eval_count", 0)),
        },
    })


async def _transparent_openai_proxy(
    body: dict[str, Any],
    upstream_endpoint: str,
    headers: dict[str, str],
) -> StreamingResponse | JSONResponse:
    """Bypass the pipeline — forward the raw OpenAI request/response unchanged."""
    url = f"{upstream_endpoint.rstrip('/')}/v1/chat/completions"
    is_stream = body.get("stream", False)

    _log.debug("transparent openai proxy → %s (stream=%s)", url, is_stream)

    if is_stream:
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

        async def stream_and_close():
            try:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_and_close(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(url, json=body, headers=headers)

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
    }
    from starlette.responses import Response
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


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


# ---------------------------------------------------------------------- #
#  Anthropic format helpers                                               #
# ---------------------------------------------------------------------- #


def _extract_text(content: Any) -> str:
    """Extract plain text from Anthropic content (string or block array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _anthropic_to_pipeline_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    """Convert Anthropic request body to pipeline message list."""
    messages: list[dict[str, str]] = []

    # Anthropic puts system as a top-level param, not in messages.
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _extract_text(system)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = _extract_text(msg.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "error": "end_turn",
    "unknown": "end_turn",
}


def _pipeline_to_anthropic(pr, *, request_model: str | None) -> dict[str, Any]:
    """Translate a PipelineResponse into an Anthropic Messages response."""
    input_tokens = pr.usage_cloud.input_tokens or 0
    output_tokens = pr.usage_cloud.output_tokens or 0

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": pr.content}],
        "model": pr.model or request_model or "",
        "stop_reason": _STOP_REASON_MAP.get(pr.finish_reason, "end_turn"),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
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


async def _anthropic_sse_generator(
    pipeline: Pipeline, req: PipelineRequest, request_model: str | None
):
    """Yield Anthropic-format SSE events from the pipeline's stream."""
    msg_id = f"msg_{uuid.uuid4().hex}"
    model = request_model or ""

    # message_start
    start_msg = {
        "id": msg_id, "type": "message", "role": "assistant",
        "content": [], "model": model, "stop_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': start_msg})}\n\n"

    # content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    total_output = 0
    try:
        async for chunk in pipeline.stream(req):
            if chunk.delta:
                total_output += len(chunk.delta.split())  # rough token estimate
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk.delta},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            if chunk.done:
                stop_reason = _STOP_REASON_MAP.get(
                    chunk.finish_reason or "stop", "end_turn"
                )
                output_tokens = (
                    chunk.usage.output_tokens if chunk.usage else total_output
                )

                # content_block_stop
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                # message_delta
                msg_delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {"output_tokens": output_tokens},
                }
                yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

                # message_stop
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                return

    except (PipelineError, ModelBackendError) as e:
        _log.warning("anthropic streaming error: %s", e)
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    # If no done chunk was received, close the stream gracefully.
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': total_output}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


__all__ = ["create_app"]
