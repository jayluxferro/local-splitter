"""T1 route — local classifier that triages TRIVIAL vs COMPLEX.

The local model reads the incoming request and emits TRIVIAL or COMPLEX.
Trivial requests are answered by the local model directly and never reach
the cloud.  Complex requests flow on to the next pipeline stage.

Fail-open: if the local model is unreachable or returns garbage, the
request is classified as COMPLEX and proceeds to the cloud unchanged
(ARCHITECTURE.md principle 2).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from local_splitter.models import ChatClient, ChatResponse, Message, ModelBackendError

from .types import StageEvent

_log = logging.getLogger(__name__)

Classification = Literal["TRIVIAL", "COMPLEX"]

# Few-shot examples that reliably steer small models (llama3.2:3b etc.)
# into single-token classification without preamble.
CLASSIFIER_FEWSHOT: list[Message] = [
    {"role": "user", "content": "Classify: What is 2+2? Answer TRIVIAL or COMPLEX only."},
    {"role": "assistant", "content": "TRIVIAL"},
    {"role": "user", "content": "Classify: Design a distributed consensus algorithm for a multi-region database. Answer TRIVIAL or COMPLEX only."},
    {"role": "assistant", "content": "COMPLEX"},
    {"role": "user", "content": "Classify: Rename variable x to count. Answer TRIVIAL or COMPLEX only."},
    {"role": "assistant", "content": "TRIVIAL"},
    {"role": "user", "content": "Classify: Refactor the auth module to support OAuth2 with PKCE flow across three services. Answer TRIVIAL or COMPLEX only."},
    {"role": "assistant", "content": "COMPLEX"},
]

_DECISION_RE = re.compile(r"\b(TRIVIAL|COMPLEX)\b", re.IGNORECASE)


@dataclass(slots=True)
class RouteResult:
    """Outcome of the T1 routing stage."""

    classification: Classification
    local_reply: ChatResponse | None
    events: list[StageEvent]


def _extract_user_text(messages: list[Message]) -> str:
    """Return the last user message's content for the classifier."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _parse_classification(raw: str) -> Classification:
    """Parse classifier output.  Defaults to COMPLEX on any ambiguity."""
    match = _DECISION_RE.search(raw)
    if match:
        return match.group(1).upper()  # type: ignore[return-value]
    _log.warning(
        "T1 classifier returned unparseable output: %r; defaulting to COMPLEX", raw
    )
    return "COMPLEX"


async def classify(
    messages: list[Message],
    *,
    local: ChatClient,
    params: dict[str, Any] | None = None,
) -> tuple[Classification, StageEvent]:
    """Classify a request as TRIVIAL or COMPLEX.

    On local-model failure returns COMPLEX (fail-open).
    """
    user_text = _extract_user_text(messages)
    classifier_messages: list[Message] = [
        *CLASSIFIER_FEWSHOT,
        {"role": "user", "content": f"Classify: {user_text} Answer TRIVIAL or COMPLEX only."},
    ]

    t0 = time.perf_counter()
    try:
        resp = await local.complete(
            classifier_messages,
            temperature=0.0,
            max_tokens=10,
        )
    except ModelBackendError as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T1 classifier failed, defaulting to COMPLEX: %s", exc)
        return "COMPLEX", StageEvent(
            stage="t1_classify",
            decision="ERROR",
            ms=elapsed,
            detail={"error": str(exc)},
        )

    elapsed = (time.perf_counter() - t0) * 1000
    classification = _parse_classification(resp.content)
    return classification, StageEvent(
        stage="t1_classify",
        decision=classification,
        ms=elapsed,
        tokens_in=resp.usage.input_tokens,
        tokens_out=resp.usage.output_tokens,
    )


async def apply(
    messages: list[Message],
    *,
    local: ChatClient,
    params: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RouteResult:
    """Run T1 routing end-to-end.

    Classifies the request, and if TRIVIAL answers it locally.
    On any local-model error the request falls back to COMPLEX.
    """
    classification, classify_event = await classify(
        messages, local=local, params=params
    )
    events: list[StageEvent] = [classify_event]

    if classification != "TRIVIAL":
        return RouteResult(classification="COMPLEX", local_reply=None, events=events)

    # --- answer locally ---
    t0 = time.perf_counter()
    try:
        reply = await local.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            extra=extra,
        )
    except ModelBackendError as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.warning("T1 local answer failed, falling back to COMPLEX: %s", exc)
        events.append(
            StageEvent(
                stage="t1_local_answer",
                decision="ERROR",
                ms=elapsed,
                detail={"error": str(exc)},
            )
        )
        return RouteResult(classification="COMPLEX", local_reply=None, events=events)

    elapsed = (time.perf_counter() - t0) * 1000
    events.append(
        StageEvent(
            stage="t1_local_answer",
            decision="APPLIED",
            ms=elapsed,
            tokens_in=reply.usage.input_tokens,
            tokens_out=reply.usage.output_tokens,
        )
    )
    return RouteResult(classification="TRIVIAL", local_reply=reply, events=events)


__all__ = ["Classification", "RouteResult", "apply", "classify"]
