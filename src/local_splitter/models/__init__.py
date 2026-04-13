"""Model backends: Ollama + any OpenAI-compatible endpoint.

No vendor SDKs — raw HTTP via httpx only.

The pipeline depends only on `ChatClient` (see `base.py`). `OllamaClient`
and `OpenAICompatClient` are the two concrete implementations; either
may be configured as the local or the cloud model.
"""

from .base import (
    ChatClient,
    ChatResponse,
    FinishReason,
    Message,
    ModelBackendError,
    StreamChunk,
    Usage,
)
from .factory import build_chat_client
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient

__all__ = [
    "ChatClient",
    "ChatResponse",
    "FinishReason",
    "Message",
    "ModelBackendError",
    "OllamaClient",
    "OpenAICompatClient",
    "StreamChunk",
    "Usage",
    "build_chat_client",
]
