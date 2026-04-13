"""Client for any OpenAI-compatible `/v1/chat/completions` endpoint.

This covers OpenAI itself, Anthropic via LiteLLM, together.ai, vLLM,
llama.cpp server, LM Studio, Ollama's own compatibility layer, and
anyone else who implements the shape.

Defensive notes (from `.agent/memory/gotchas.md`):

- Not every endpoint returns every field. Use `.get()` with defaults.
- `finish_reason == "length"` is silent context truncation — warn loudly.
- `usage` is reported per-response on non-streamed calls; on streamed
  calls it may appear only in the final `[DONE]`-preceding chunk, or
  may be absent entirely. Don't rely on it.
- API keys come from an env var; the caller names the var so we don't
  hardcode `OPENAI_API_KEY` and accidentally leak across providers.
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

_FINISH_REASONS: frozenset[FinishReason] = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "error", "unknown"}
)


def _map_finish_reason(raw: Any) -> FinishReason:
    if raw in _FINISH_REASONS:
        return raw  # type: ignore[return-value]
    if raw is None:
        return "stop"
    return "unknown"


class OpenAICompatClient:
    """Async client for OpenAI-compatible chat + embedding endpoints."""

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
        """Build a client.

        Exactly one of `api_key` or `api_key_env` should be supplied; if
        both are `None` we run unauthenticated (fine for local vLLM /
        llama.cpp servers).
        """
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
            "accept": "application/json",
        }
        if resolved_key:
            headers["authorization"] = f"Bearer {resolved_key}"
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

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ chat

    def _build_chat_body(
        self,
        messages: Sequence[Message],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        seed: int | None,
        extra: Mapping[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": list(messages),
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop is not None:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = seed
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
    ) -> ChatResponse:
        body = self._build_chat_body(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            extra=extra,
            stream=False,
        )
        try:
            resp = await self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise ModelBackendError(f"openai-compat chat request failed: {e}") from e

        if resp.status_code != 200:
            raise ModelBackendError(
                f"openai-compat /chat/completions returned {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise ModelBackendError(f"openai-compat non-JSON body: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise ModelBackendError(f"openai-compat response had no choices: {data}")
        choice0 = choices[0]
        message = choice0.get("message") or {}
        content = message.get("content") or ""
        finish = _map_finish_reason(choice0.get("finish_reason"))

        usage_raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens"),
            output_tokens=usage_raw.get("completion_tokens"),
        )

        if finish == "length":
            _log.warning(
                "openai-compat chat truncated: finish_reason=length model=%s",
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
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_chat_body(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            extra=extra,
            stream=True,
        )
        return self._stream_chat(body)

    async def _stream_chat(self, body: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Parse OpenAI-style SSE into `StreamChunk`s.

        SSE frames look like:

            data: {"choices":[{"delta":{"content":"hel"}, ...}]}
            data: {"choices":[{"delta":{"content":"lo"}, "finish_reason":"stop"}]}
            data: [DONE]
        """
        headers = {"accept": "text/event-stream"}
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors="replace")
                    raise ModelBackendError(
                        f"openai-compat stream returned {resp.status_code}: {text[:200]}"
                    )

                finish: FinishReason | None = None
                usage: Usage | None = None

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        yield StreamChunk(
                            delta="",
                            done=True,
                            finish_reason=finish or "stop",
                            usage=usage,
                        )
                        return
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        _log.warning("openai-compat stream: skipping non-JSON payload")
                        continue

                    # usage sometimes arrives on its own final chunk
                    if "usage" in event and event["usage"]:
                        u = event["usage"]
                        usage = Usage(
                            input_tokens=u.get("prompt_tokens"),
                            output_tokens=u.get("completion_tokens"),
                        )

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice0 = choices[0]
                    delta = (choice0.get("delta") or {}).get("content") or ""
                    fr_raw = choice0.get("finish_reason")
                    if fr_raw is not None:
                        finish = _map_finish_reason(fr_raw)
                        if finish == "length":
                            _log.warning(
                                "openai-compat stream truncated: finish_reason=length model=%s",
                                body["model"],
                            )
                    yield StreamChunk(delta=delta, done=False)

                # Stream ended without an explicit [DONE] marker.
                yield StreamChunk(
                    delta="",
                    done=True,
                    finish_reason=finish or "stop",
                    usage=usage,
                )
        except httpx.HTTPError as e:
            raise ModelBackendError(f"openai-compat stream request failed: {e}") from e

    # ------------------------------------------------------------------ embed

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        model_name = model or self.embed_model
        if not model_name:
            raise ModelBackendError("openai-compat embed requires embed_model to be set")
        if not texts:
            return []

        body = {"model": model_name, "input": list(texts)}
        try:
            resp = await self._http.post("/embeddings", json=body)
        except httpx.HTTPError as e:
            raise ModelBackendError(f"openai-compat embed request failed: {e}") from e

        if resp.status_code != 200:
            raise ModelBackendError(
                f"openai-compat /embeddings returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        entries = data.get("data") or []
        # Preserve server-side ordering via "index" when present.
        ordered: list[tuple[int, list[float]]] = []
        for i, entry in enumerate(entries):
            idx = entry.get("index", i)
            vec = entry.get("embedding")
            if not isinstance(vec, list):
                raise ModelBackendError(f"openai-compat /embeddings bad entry: {entry}")
            ordered.append((idx, [float(x) for x in vec]))
        ordered.sort(key=lambda kv: kv[0])
        return [vec for _, vec in ordered]


__all__ = ["OpenAICompatClient"]
