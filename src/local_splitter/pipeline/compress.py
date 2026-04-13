"""T2 compress — local model rewrites long context to a shorter form.

Before the request reaches the cloud, eligible messages (system prompts,
long chat history, retrieved docs) are compressed by the local model.
The cloud sees a trimmed prompt and charges fewer input tokens.

Only messages exceeding a minimum length are compressed.  The last user
message is never compressed — it's the actual query.

Fail-open: if the local model errors on any message, the original
content is kept unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from local_splitter.models import ChatClient, Message, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

COMPRESS_SYSTEM = (
    "Compress the following text to the shortest form that preserves all "
    "information a language model needs to answer the user. Remove filler, "
    "repetition, and instructions that do not change the output. Preserve "
    "file paths, variable names, error messages, and numeric values exactly. "
    "Do not paraphrase technical terms.\n\n"
    "Target: about {ratio_pct}% of the original length.\n\n"
    "Respond with ONLY the compressed text, nothing else."
)

DEFAULT_RATIO_TARGET = 0.5
DEFAULT_MIN_LENGTH = 200  # characters


@dataclass(slots=True)
class CompressResult:
    """Outcome of T2 compression."""

    messages: list[Message]
    events: list[StageEvent]


def _is_compressible(msg: Message, index: int, total: int, min_len: int) -> bool:
    """Decide whether a message is eligible for compression."""
    content = msg.get("content", "")
    if len(content) < min_len:
        return False
    # Never compress the last user message — that's the actual query.
    if index == total - 1 and msg.get("role") == "user":
        return False
    return True


async def apply(
    messages: list[Message],
    *,
    local: ChatClient,
    params: dict[str, Any] | None = None,
) -> CompressResult:
    """Compress eligible messages using the local model.

    Returns modified messages and a single summary StageEvent.
    """
    p = params or {}
    ratio = float(p.get("ratio_target", DEFAULT_RATIO_TARGET))
    min_len = int(p.get("min_length", DEFAULT_MIN_LENGTH))
    ratio_pct = int(ratio * 100)
    compress_prompt = COMPRESS_SYSTEM.format(ratio_pct=ratio_pct)

    total = len(messages)
    eligible = [
        i for i, msg in enumerate(messages) if _is_compressible(msg, i, total, min_len)
    ]

    if not eligible:
        return CompressResult(
            messages=messages,
            events=[
                StageEvent(
                    stage="t2_compress",
                    decision="SKIP",
                    ms=0.0,
                    detail={"reason": "no messages eligible for compression"},
                )
            ],
        )

    compressed = list(messages)  # shallow copy
    n_compressed = 0
    n_errors = 0
    total_original_chars = 0
    total_compressed_chars = 0
    total_tokens_in = 0
    total_tokens_out = 0
    t_start = time.perf_counter()

    for i in eligible:
        original_content = messages[i].get("content", "")

        try:
            resp = await local.complete(
                [
                    {"role": "system", "content": compress_prompt},
                    {"role": "user", "content": original_content},
                ],
                temperature=0.0,
            )
        except ModelBackendError as exc:
            _log.warning("T2 compress failed on message %d, keeping original: %s", i, exc)
            n_errors += 1
            continue

        new_content = resp.content.strip()
        total_tokens_in += resp.usage.input_tokens or 0
        total_tokens_out += resp.usage.output_tokens or 0

        # Only use compressed version if it's actually shorter.
        if len(new_content) < len(original_content):
            compressed[i] = {**messages[i], "content": new_content}
            total_original_chars += len(original_content)
            total_compressed_chars += len(new_content)
            n_compressed += 1
        else:
            total_original_chars += len(original_content)
            total_compressed_chars += len(original_content)

    elapsed = (time.perf_counter() - t_start) * 1000
    actual_ratio = (
        total_compressed_chars / total_original_chars
        if total_original_chars > 0
        else 1.0
    )

    decision = "APPLIED" if n_compressed > 0 else "NOOP"
    event = StageEvent(
        stage="t2_compress",
        decision=decision,
        ms=elapsed,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        detail={
            "messages_eligible": len(eligible),
            "messages_compressed": n_compressed,
            "errors": n_errors,
            "original_chars": total_original_chars,
            "compressed_chars": total_compressed_chars,
            "ratio": round(actual_ratio, 3),
        },
    )

    return CompressResult(messages=compressed, events=[event])


__all__ = ["CompressResult", "apply"]
