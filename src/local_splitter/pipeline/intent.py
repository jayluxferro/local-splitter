"""T6 intent — structured intent extraction.

The local model parses free-text prompts into structured intent fields
(intent, target, constraints).  The cloud receives a tight template
rather than chatty prose, reducing input tokens.

Fail-open: if extraction fails or the output can't be parsed, the
original messages are passed through unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from local_splitter.models import ChatClient, Message, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

INTENT_SYSTEM = (
    "Extract the user's intent from their message. Output a JSON object "
    "with these fields:\n"
    '  "intent": one of "explain", "refactor", "debug", "generate", '
    '"rename", "search", "edit", "other"\n'
    '  "target": the file, symbol, or region referenced\n'
    '  "constraints": list of specific requirements\n'
    '  "query": the essential question in one sentence\n\n'
    "Output valid JSON only, no explanation."
)

TEMPLATE = (
    "Intent: {intent}\n"
    "Target: {target}\n"
    "Constraints: {constraints}\n"
    "Query: {query}"
)

MIN_INTENT_LEN = 100  # chars — don't extract intent from short messages


@dataclass(slots=True)
class IntentResult:
    """Outcome of T6 intent extraction."""

    messages: list[Message]
    events: list[StageEvent]


def _render_template(parsed: dict[str, Any]) -> str:
    """Render extracted intent as a compact template."""
    constraints = parsed.get("constraints", [])
    if isinstance(constraints, list):
        constraints_str = "; ".join(str(c) for c in constraints) if constraints else "none"
    else:
        constraints_str = str(constraints)
    return TEMPLATE.format(
        intent=parsed.get("intent", "other"),
        target=parsed.get("target", "unspecified"),
        constraints=constraints_str,
        query=parsed.get("query", ""),
    )


async def apply(
    messages: list[Message],
    *,
    local: ChatClient,
    params: dict[str, Any] | None = None,
) -> IntentResult:
    """Extract structured intent from the last user message.

    Only processes the last user message if it exceeds the minimum
    length.  Other messages are left unchanged.
    """
    # Find the last user message.
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return IntentResult(
            messages=messages,
            events=[StageEvent(stage="t6_intent", decision="SKIP", ms=0.0,
                               detail={"reason": "no user message"})],
        )

    p = params or {}
    min_len = int(p.get("min_length", MIN_INTENT_LEN))
    original = messages[last_user_idx].get("content", "")

    if len(original) < min_len:
        return IntentResult(
            messages=messages,
            events=[StageEvent(stage="t6_intent", decision="SKIP", ms=0.0,
                               detail={"reason": "user message too short"})],
        )

    t0 = time.perf_counter()
    try:
        resp = await local.complete(
            [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": original},
            ],
            temperature=0.0,
        )
    except ModelBackendError as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T6 intent extraction failed, keeping original: %s", exc)
        return IntentResult(
            messages=messages,
            events=[StageEvent(stage="t6_intent", decision="ERROR", ms=elapsed,
                               detail={"error": str(exc)})],
        )

    elapsed = (time.perf_counter() - t0) * 1000

    # Try to parse JSON from the response.
    try:
        parsed = json.loads(resp.content.strip())
    except json.JSONDecodeError:
        _log.warning("T6 intent returned non-JSON, keeping original: %r", resp.content[:100])
        return IntentResult(
            messages=messages,
            events=[StageEvent(stage="t6_intent", decision="PARSE_ERROR", ms=elapsed,
                               detail={"raw": resp.content[:200]})],
        )

    template_text = _render_template(parsed)

    # Only use template if shorter.
    if len(template_text) < len(original):
        modified = list(messages)
        modified[last_user_idx] = {**messages[last_user_idx], "content": template_text}
    else:
        modified = messages

    return IntentResult(
        messages=modified,
        events=[
            StageEvent(
                stage="t6_intent",
                decision="APPLIED" if len(template_text) < len(original) else "NOOP",
                ms=elapsed,
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
                detail={
                    "intent": parsed.get("intent", "other"),
                    "original_len": len(original),
                    "template_len": len(template_text),
                },
            )
        ],
    )


__all__ = ["IntentResult", "apply"]
