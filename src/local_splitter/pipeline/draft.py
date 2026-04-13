"""T4 draft — local model drafts, cloud model reviews.

Instead of asking the cloud to generate from scratch, the local model
produces a draft and the cloud reviews or patches it.  When the draft
is good the cloud's output token count drops sharply.

If the cloud responds with "APPROVED" followed by the draft unchanged,
we serve the draft directly.  Otherwise we serve the cloud's corrected
version.

Fail-open: if the local model can't produce a draft, we skip the
draft-review flow and let the cloud generate from scratch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from local_splitter.models import ChatClient, ChatResponse, Message, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

DRAFT_SYSTEM = (
    "Draft a response to the user's request. Be concise and accurate. "
    "If you are unsure, give your best attempt — it will be reviewed."
)

REVIEW_SYSTEM = (
    "You are reviewing a draft response to the user's request.\n\n"
    "If the draft is correct and complete, respond with exactly 'APPROVED' "
    "on the first line, followed by the draft unchanged.\n\n"
    "If the draft is incorrect or incomplete, respond with the corrected "
    "version only. Do not explain your changes."
)


@dataclass(slots=True)
class DraftResult:
    """Outcome of the T4 draft-review stage."""

    draft: str
    review: ChatResponse
    approved: bool
    events: list[StageEvent]


def _build_review_messages(
    original_messages: list[Message], draft: str
) -> list[Message]:
    """Build the messages for the cloud review call.

    The cloud sees: the review system prompt, then the original
    conversation, then the draft as an assistant turn to review.
    """
    msgs: list[Message] = [{"role": "system", "content": REVIEW_SYSTEM}]
    msgs.extend(original_messages)
    msgs.append(
        {"role": "assistant", "content": f"[DRAFT]\n{draft}"}
    )
    msgs.append(
        {
            "role": "user",
            "content": "Review the draft above. APPROVED + draft if correct, or corrected version.",
        }
    )
    return msgs


def _is_approved(content: str, draft: str) -> bool:
    """Check whether the cloud approved the draft."""
    stripped = content.strip()
    return stripped.upper().startswith("APPROVED")


async def apply(
    messages: list[Message],
    *,
    local: ChatClient,
    cloud: ChatClient,
    params: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    extra: Any = None,
) -> DraftResult | None:
    """Run T4 draft-review.

    Returns a ``DraftResult`` on success, or ``None`` if the local
    draft failed (fail-open — caller should fall back to direct cloud).
    """
    # --- Step 1: local draft ---
    draft_messages: list[Message] = [
        {"role": "system", "content": DRAFT_SYSTEM},
        *messages,
    ]

    t0 = time.perf_counter()
    try:
        draft_resp = await local.complete(
            draft_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ModelBackendError as exc:
        _log.warning("T4 local draft failed, skipping draft-review: %s", exc)
        return None

    draft_ms = (time.perf_counter() - t0) * 1000
    draft_text = draft_resp.content.strip()

    draft_event = StageEvent(
        stage="t4_draft",
        decision="DRAFTED",
        ms=draft_ms,
        tokens_in=draft_resp.usage.input_tokens,
        tokens_out=draft_resp.usage.output_tokens,
    )

    # --- Step 2: cloud review ---
    review_messages = _build_review_messages(messages, draft_text)

    t1 = time.perf_counter()
    try:
        review_resp = await cloud.complete(
            review_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra=extra,
        )
    except ModelBackendError:
        # Cloud review failed — still raise so the pipeline's error
        # handling catches it (we don't fail-open on cloud errors).
        raise

    review_ms = (time.perf_counter() - t1) * 1000
    approved = _is_approved(review_resp.content, draft_text)

    review_event = StageEvent(
        stage="t4_review",
        decision="APPROVED" if approved else "REVISED",
        ms=review_ms,
        tokens_in=review_resp.usage.input_tokens,
        tokens_out=review_resp.usage.output_tokens,
        detail={
            "draft_len": len(draft_text),
            "review_len": len(review_resp.content),
        },
    )

    return DraftResult(
        draft=draft_text,
        review=review_resp,
        approved=approved,
        events=[draft_event, review_event],
    )


__all__ = ["DraftResult", "apply"]
