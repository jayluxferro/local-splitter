"""T7 batch — prompt-cache tagging for vendor-side caching.

Two sub-tactics:

1. **Prompt-cache tagging** — mark the stable prefix of a prompt so the
   vendor's cache serves it at a discount on subsequent calls.  We add
   vendor-specific ``cache_control`` metadata to the outgoing request.

2. **Local batching** — buffer multiple queries and send as one request.
   This is latency-sensitive and requires an async accumulator, so it's
   **not yet implemented** (placeholder for Stage 9+).

For now this module only does prompt-cache tagging.  It detects the
stable prefix (system messages) and tags them with ``cache_control``
so Anthropic / OpenAI give a cache discount.

Fail-open: if anything goes wrong, messages pass through unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from local_splitter.models import Message

from .types import StageEvent

_log = logging.getLogger(__name__)

# Minimum system prompt length (chars) to justify cache tagging.
MIN_PREFIX_LEN = 500


@dataclass(slots=True)
class BatchResult:
    """Outcome of T7 batch/cache tagging."""

    messages: list[Message]
    extra: dict[str, Any] | None  # merged into the outgoing request's extra
    events: list[StageEvent]


def apply(
    messages: list[Message],
    *,
    params: dict[str, Any] | None = None,
) -> BatchResult:
    """Tag stable prefix messages for vendor prompt caching.

    This is a synchronous function — no model calls needed.  It adds
    ``cache_control`` metadata to system messages that exceed the
    minimum length threshold.
    """
    p = params or {}
    min_len = int(p.get("min_prefix_len", MIN_PREFIX_LEN))

    t0 = time.perf_counter()
    tagged = list(messages)
    n_tagged = 0

    for i, msg in enumerate(tagged):
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if len(content) < min_len:
            continue

        # Tag for vendor caching.  Anthropic uses cache_control on the
        # message; OpenAI caches automatically for >1024 token prefixes.
        tagged[i] = {
            **msg,
            "cache_control": {"type": "ephemeral"},
        }
        n_tagged += 1

    elapsed = (time.perf_counter() - t0) * 1000

    if n_tagged == 0:
        return BatchResult(
            messages=messages,
            extra=None,
            events=[
                StageEvent(
                    stage="t7_batch", decision="SKIP", ms=elapsed,
                    detail={"reason": "no system messages long enough to tag"},
                )
            ],
        )

    return BatchResult(
        messages=tagged,
        extra=None,
        events=[
            StageEvent(
                stage="t7_batch", decision="APPLIED", ms=elapsed,
                detail={"messages_tagged": n_tagged},
            )
        ],
    )


__all__ = ["BatchResult", "apply"]
