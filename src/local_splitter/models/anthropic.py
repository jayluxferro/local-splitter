"""Client for the Anthropic Messages API (`/v1/messages`).

Same pattern as `OpenAICompatClient`: raw httpx, no vendor SDK.
Converts the pipeline's internal message format (OpenAI-style role/content)
to/from Anthropic's format (system as top-level param, content blocks).

Auth: uses `x-api-key` header (not Bearer). Key comes from an env var
named by `api_key_env` (default `ANTHROPIC_API_KEY`).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from .base import (
    ChatResponse,
    FinishReason,
    Message,
    ModelBackendError,
    StreamChunk,
    Usage,
)

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _map_stop_reason(raw: Any) -> FinishReason:
    if isinstance(raw, str) and raw in _STOP_REASON_MAP:
        return _STOP_REASON_MAP[raw]
    if raw is None:
        return "stop"
    return "unknown"


def _split_system(
    messages: Sequence[Message],
) -> tuple[str | None, list[dict[str, str]]]:
    """Extract system messages and return (system_text, remaining_messages).

    Anthropic wants system as a top-level param, not in the messages array.
    """
    system_parts: list[str] = []
    remaining: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            remaining.append({"role": msg["role"], "content": msg["content"]})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, remaining


class AnthropicClient:
    """Async client for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        chat_model: str,
        embed_model: str | None = None,
        endpoint: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if api_key and api_key_env:
            raise ValueError("pass either api_key or api_key_env, not both")

        resolved_key: str | None = api_key
        if api_key_env:
            resolved_key = os.environ.get(api_key_env)
            if not resolved_key:
                raise ModelBackendError(
                    f"api_key_env={api_key_env!r} is unset in the environment"
                )

        self.chat_model = chat_model
        self.embed_model = embed_model
        self.endpoint = endpoint.rstrip("/")

        headers: dict[str, str] = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if resolved_key:
            headers["x-api-key"] = resolved_key
        if extra_headers:
            headers.update({k.lower(): v for k, v in extra_headers.items()})

        self._http = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AnthropicClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ chat

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        stream: bool,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        system, conv_messages = _split_system(messages)
        body: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": conv_messages,
            "max_tokens": max_tokens or 8192,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if stop is not None:
            body["stop_sequences"] = list(stop)
        if extra:
            for k, v in extra.items():
                body[k] = v
        return body

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
        upstream_headers: Mapping[str, str] | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=False,
            extra=extra,
        )
        req_headers = dict(upstream_headers) if upstream_headers else {}
        try:
            resp = await self._http.post("/v1/messages", json=body, headers=req_headers)
        except httpx.HTTPError as e:
            raise ModelBackendError(f"anthropic chat request failed: {e}") from e

        if resp.status_code != 200:
            raise ModelBackendError(
                f"anthropic /v1/messages returned {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise ModelBackendError(f"anthropic non-JSON body: {e}") from e

        # Extract text from content blocks
        content_blocks = data.get("content") or []
        text_parts = [
            b.get("text", "")
            for b in content_blocks
            if b.get("type") == "text"
        ]
        content = "".join(text_parts)

        usage_raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens"),
            output_tokens=usage_raw.get("output_tokens"),
        )

        finish = _map_stop_reason(data.get("stop_reason"))
        if finish == "length":
            _log.warning(
                "anthropic chat truncated: stop_reason=max_tokens model=%s",
                body["model"],
            )

        return ChatResponse(
            content=content,
            finish_reason=finish,
            usage=usage,
            model=data.get("model", body["model"]),
            raw=data,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
        upstream_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            extra=extra,
        )
        return self._stream_messages(body, upstream_headers=upstream_headers)

    async def _stream_messages(
        self, body: dict[str, Any], *, upstream_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Parse Anthropic SSE into StreamChunks.

        Event types:
            message_start        — carries message metadata + input usage
            content_block_start  — new content block
            content_block_delta  — text delta
            content_block_stop   — block done
            message_delta        — stop_reason + output usage
            message_stop         — stream complete
        """
        headers = {"accept": "text/event-stream"}
        if upstream_headers:
            headers.update(upstream_headers)
        try:
            async with self._http.stream(
                "POST", "/v1/messages", json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors="replace")
                    raise ModelBackendError(
                        f"anthropic stream returned {resp.status_code}: {text[:200]}"
                    )

                usage = Usage()
                finish: FinishReason | None = None
                event_type: str | None = None

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # SSE event type line
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue

                    if not line.startswith("data:"):
                        continue

                    payload = line[5:].strip()
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        _log.warning("anthropic stream: skipping non-JSON payload")
                        continue

                    if event_type == "message_start":
                        msg = event.get("message") or {}
                        u = msg.get("usage") or {}
                        usage = Usage(
                            input_tokens=u.get("input_tokens"),
                            output_tokens=u.get("output_tokens", 0),
                        )

                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        text = delta.get("text", "")
                        if text:
                            yield StreamChunk(delta=text, done=False)

                    elif event_type == "message_delta":
                        delta = event.get("delta") or {}
                        raw_stop = delta.get("stop_reason")
                        if raw_stop:
                            finish = _map_stop_reason(raw_stop)
                        u = event.get("usage") or {}
                        output_tokens = u.get("output_tokens")
                        if output_tokens is not None:
                            usage = Usage(
                                input_tokens=usage.input_tokens,
                                output_tokens=output_tokens,
                            )

                    elif event_type == "message_stop":
                        yield StreamChunk(
                            delta="",
                            done=True,
                            finish_reason=finish or "stop",
                            usage=usage,
                        )
                        return

                # Stream ended without message_stop
                yield StreamChunk(
                    delta="",
                    done=True,
                    finish_reason=finish or "stop",
                    usage=usage,
                )
        except httpx.HTTPError as e:
            raise ModelBackendError(
                f"anthropic stream request failed: {e}"
            ) from e

    # ------------------------------------------------------------------ embed

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        raise ModelBackendError("Anthropic does not support embeddings")


__all__ = ["AnthropicClient"]
