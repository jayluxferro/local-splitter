"""T5 diff — minimal-diff extraction for edit requests.

For edit requests (detected by the presence of code blocks or edit
instructions), the local model identifies the relevant hunks and
replaces the full file contents with a minimal diff context.  The
cloud then sees a surgical change, not the whole file.

Fail-open: if detection or extraction fails, the original messages
are passed through unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from local_splitter.models import ChatClient, Message, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

DIFF_SYSTEM = (
    "You are a code-edit minimizer. The user has a request that includes "
    "file contents. Extract ONLY the relevant code sections (3 lines of "
    "context around each change). Output the minimal context needed for "
    "another model to apply the edit. Preserve file paths and line numbers.\n\n"
    "If the request is not a code edit, output it unchanged."
)

MIN_EDIT_LEN = 500  # chars — only attempt diff extraction on long messages


@dataclass(slots=True)
class DiffResult:
    """Outcome of T5 diff extraction."""

    messages: list[Message]
    events: list[StageEvent]


def _looks_like_edit(messages: list[Message]) -> bool:
    """Heuristic: does the conversation look like a code-edit request?"""
    text = " ".join(m.get("content", "") for m in messages).lower()
    edit_signals = ("```", "apply_patch", "edit", "change", "replace", "fix", "modify")
    has_signal = any(s in text for s in edit_signals)
    has_length = any(len(m.get("content", "")) > MIN_EDIT_LEN for m in messages)
    return has_signal and has_length


async def apply(
    messages: list[Message],
    *,
    local: ChatClient,
    params: dict[str, Any] | None = None,
) -> DiffResult:
    """Extract minimal diff context from edit requests.

    Non-edit requests pass through unchanged.
    """
    if not _looks_like_edit(messages):
        return DiffResult(
            messages=messages,
            events=[
                StageEvent(
                    stage="t5_diff", decision="SKIP", ms=0.0,
                    detail={"reason": "not an edit request"},
                )
            ],
        )

    # Find the longest message (likely the file contents) and compress it.
    longest_idx = max(range(len(messages)), key=lambda i: len(messages[i].get("content", "")))
    original_content = messages[longest_idx].get("content", "")

    t0 = time.perf_counter()
    try:
        resp = await local.complete(
            [
                {"role": "system", "content": DIFF_SYSTEM},
                {"role": "user", "content": original_content},
            ],
            temperature=0.0,
        )
    except ModelBackendError as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T5 diff extraction failed, keeping original: %s", exc)
        return DiffResult(
            messages=messages,
            events=[
                StageEvent(
                    stage="t5_diff", decision="ERROR", ms=elapsed,
                    detail={"error": str(exc)},
                )
            ],
        )

    elapsed = (time.perf_counter() - t0) * 1000
    new_content = resp.content.strip()

    # Only use minimized version if actually shorter.
    if len(new_content) < len(original_content):
        compressed = list(messages)
        compressed[longest_idx] = {**messages[longest_idx], "content": new_content}
        shrink = len(new_content) / len(original_content)
    else:
        compressed = messages
        shrink = 1.0

    return DiffResult(
        messages=compressed,
        events=[
            StageEvent(
                stage="t5_diff",
                decision="APPLIED" if shrink < 1.0 else "NOOP",
                ms=elapsed,
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
                detail={
                    "original_len": len(original_content),
                    "diff_len": len(new_content),
                    "shrink_factor": round(shrink, 3),
                },
            )
        ],
    )


__all__ = ["DiffResult", "apply"]
