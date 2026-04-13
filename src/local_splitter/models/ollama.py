"""Ollama REST client.

Talks to the native Ollama API (`/api/chat`, `/api/embed`) rather than
its OpenAI compatibility shim, because the native API exposes options
like `num_ctx` that we rely on (see `.agent/memory/gotchas.md`).

Hard rules enforced here:

- `options.num_ctx` is **always** sent. Ollama's default 2048 silently
  truncates long prompts which would corrupt any compression or intent
  experiment. The default here is 8192; callers can override per-call
  via `extra={"options": {"num_ctx": ...}}`.
- Non-2xx responses raise `ModelBackendError`; tactics above us catch
  it and fail open.
- Timeouts are conservative: local models are fast but cold-start can
  take several seconds (per gotchas.md warmup note).
"""

from __future__ import annotations

import json
import logging
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

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


def _map_done_reason(reason: str | None) -> FinishReason:
    """Map Ollama's `done_reason` onto our normalized `FinishReason`."""
    match reason:
        case "stop" | None:
            return "stop"
        case "length":
            return "length"
        case "load" | "unload":
            return "error"
        case _:
            return "unknown"


class OllamaClient:
    """Async client for the native Ollama REST API."""

    def __init__(
        self,
        *,
        chat_model: str,
        embed_model: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.endpoint = endpoint.rstrip("/")
        self.num_ctx = num_ctx
        self._http = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> OllamaClient:
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
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            # Ollama uses `num_predict` for the max-output-tokens cap.
            options["num_predict"] = max_tokens
        if stop is not None:
            options["stop"] = list(stop)
        if seed is not None:
            options["seed"] = seed

        # Merge caller-supplied options last so they win on conflict.
        extra_opts = (extra or {}).get("options") if extra else None
        if isinstance(extra_opts, Mapping):
            options.update(extra_opts)

        body: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": list(messages),
            "stream": stream,
            "options": options,
        }

        if extra:
            for k, v in extra.items():
                if k == "options":
                    continue
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
            resp = await self._http.post("/api/chat", json=body)
        except httpx.HTTPError as e:
            raise ModelBackendError(f"ollama chat request failed: {e}") from e

        if resp.status_code != 200:
            raise ModelBackendError(
                f"ollama /api/chat returned {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise ModelBackendError(f"ollama returned non-JSON body: {e}") from e

        message = data.get("message") or {}
        content = message.get("content", "")
        finish = _map_done_reason(data.get("done_reason"))

        usage = Usage(
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

        if finish == "length":
            _log.warning("ollama chat truncated: finish_reason=length model=%s", body["model"])

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
        try:
            async with self._http.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors="replace")
                    raise ModelBackendError(
                        f"ollama /api/chat stream returned {resp.status_code}: {text[:200]}"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        _log.warning("ollama stream: skipping non-JSON line")
                        continue

                    delta = (event.get("message") or {}).get("content", "")
                    done = bool(event.get("done"))
                    if not done:
                        yield StreamChunk(delta=delta, done=False)
                        continue

                    finish = _map_done_reason(event.get("done_reason"))
                    usage = Usage(
                        input_tokens=event.get("prompt_eval_count"),
                        output_tokens=event.get("eval_count"),
                    )
                    if finish == "length":
                        _log.warning(
                            "ollama stream truncated: finish_reason=length model=%s",
                            body["model"],
                        )
                    yield StreamChunk(
                        delta=delta,
                        done=True,
                        finish_reason=finish,
                        usage=usage,
                    )
                    return
        except httpx.HTTPError as e:
            raise ModelBackendError(f"ollama stream request failed: {e}") from e

    # ------------------------------------------------------------------ embed

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        model_name = model or self.embed_model
        if not model_name:
            raise ModelBackendError("ollama embed requires embed_model to be set")
        if not texts:
            return []

        body = {"model": model_name, "input": list(texts)}
        try:
            resp = await self._http.post("/api/embed", json=body)
        except httpx.HTTPError as e:
            raise ModelBackendError(f"ollama embed request failed: {e}") from e

        if resp.status_code != 200:
            raise ModelBackendError(
                f"ollama /api/embed returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ModelBackendError(f"ollama /api/embed missing 'embeddings': {data}")
        return [[float(x) for x in vec] for vec in embeddings]


__all__ = ["OllamaClient", "DEFAULT_ENDPOINT", "DEFAULT_NUM_CTX"]
