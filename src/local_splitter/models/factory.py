"""Build a `ChatClient` from a `ModelConfig`.

Kept tiny on purpose — the dispatch is literally two `if`s. Lives here
rather than in `config.py` so `config.py` stays free of httpx/backend
imports (helps tests that only want config parsing).
"""

from __future__ import annotations

from local_splitter.config import ModelConfig

from .anthropic import AnthropicClient
from .base import ChatClient
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient


def build_chat_client(mc: ModelConfig) -> ChatClient:
    """Instantiate the concrete backend described by `mc`."""
    if mc.backend == "ollama":
        return OllamaClient(
            chat_model=mc.chat_model,
            embed_model=mc.embed_model,
            endpoint=mc.endpoint,
            num_ctx=mc.num_ctx,
        )
    if mc.backend == "openai_compat":
        return OpenAICompatClient(
            chat_model=mc.chat_model,
            embed_model=mc.embed_model,
            endpoint=mc.endpoint,
            api_key_env=mc.api_key_env,
        )
    if mc.backend == "anthropic":
        return AnthropicClient(
            chat_model=mc.chat_model,
            embed_model=mc.embed_model,
            endpoint=mc.endpoint,
            api_key_env=mc.api_key_env,
        )
    raise ValueError(f"unknown backend: {mc.backend!r}")


__all__ = ["build_chat_client"]
