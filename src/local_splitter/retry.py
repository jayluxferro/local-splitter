"""Bounded, pre-commit-only retry for upstream transport failures.

Every stage in the manifold chain used to turn a single transient
``httpx.ReadError`` into a terminal ``502``: DeepSeek dropping a socket or
a worker recycling killed the user's whole request.  This module retries
such failures — but only when it is provably safe to do so.

The three rules (see PIPELINE-RETRY-SPEC.md):

1. **Type, never string.**  ``str(httpx.ReadError)`` is EMPTY (it wraps
   ``anyio.EndOfStream``), so any ``"ReadError" in str(exc)`` test silently
   never fires — that was the original bug.  The decision here is made
   purely by ``isinstance``.
2. **Pre-commit only.**  :func:`retry_upstream` retries a whole call;
   :func:`iter_upstream` retries a stream ONLY while it has not yielded a
   single item yet.  Once a byte reaches the client, a failure must stay a
   single terminal frame — never a second body.
3. **Bounded + spaced.**  ``max_retries`` defaults to 2 (3 attempts) and is
   overridable with ``<SERVICE>_UPSTREAM_RETRIES``.  Backoff is exponential
   with jitter (~0.25s, ~0.5s, cap 2s) and never faster than 200ms, so a
   flapping peer is not hammered.

Note on POST: the usual "never retry a POST" rule assumes a non-idempotent
side effect.  Every upstream call in this proxy is a *model request* whose
effect is "generate a completion"; re-sending it before any response bytes
have been consumed is safe (at worst the peer generates twice and we keep
one answer).  That is the only reason POST retries are allowed here, and it
is why the retry is never allowed after commit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import anyio
import httpx

_log = logging.getLogger(__name__)

#: Service name used for the env override: ``LOCAL_SPLITTER_UPSTREAM_RETRIES``.
SERVICE = "LOCAL_SPLITTER"

#: Retries after the initial attempt (so 2 retries == 3 attempts).
DEFAULT_MAX_RETRIES = 2

_BASE_BACKOFF_S = 0.25
_MIN_BACKOFF_S = 0.2
_MAX_BACKOFF_S = 2.0

#: Retryable failures — matched by TYPE.  ``httpx.HTTPStatusError`` is
#: deliberately absent: a 4xx/5xx *response* is a real answer, pass it
#: through.  ``TimeoutException`` itself is absent too — only the per-phase
#: timeout types below are retryable, so a blanket timeout on an already
#: committed request is not re-sent.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.CloseError,
    anyio.EndOfStream,
)

T = TypeVar("T")

#: Indirection so tests can neutralise sleeping without touching real time.
_sleep = asyncio.sleep


def retry_budget(service: str = SERVICE) -> int:
    """Max retries for *service*, from ``<SERVICE>_UPSTREAM_RETRIES``.

    Unset / unparsable / negative values fall back to the default of 2.
    """
    raw = os.environ.get(f"{service}_UPSTREAM_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_RETRIES
    return max(0, value)


def is_retryable(exc: BaseException) -> bool:
    """True when *exc* is a transient transport failure we may safely re-send."""
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


def backoff_seconds(attempt: int) -> float:
    """Delay before retry number *attempt* (1-based): exponential + jitter."""
    base = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** (attempt - 1)))
    jittered = base * (0.8 + random.random() * 0.4)  # +/-20%
    return max(_MIN_BACKOFF_S, min(_MAX_BACKOFF_S, jittered))


async def retry_upstream(
    call: Callable[[], Awaitable[T]],
    *,
    operation: str,
    service: str = SERVICE,
    logger: logging.Logger | None = None,
    max_retries: int | None = None,
) -> T:
    """Await ``call()``, retrying retryable transport errors a bounded number of times.

    Only safe for calls whose result is not yet visible downstream (a request
    that has not been sent, or a response body that has not been returned).
    Non-retryable exceptions — and the final retryable one — propagate
    unchanged, so callers keep their existing failure body/status.
    """
    budget = retry_budget(service) if max_retries is None else max(0, max_retries)
    log = logger or _log
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — re-raised unless retryable
            if not is_retryable(exc) or attempt >= budget:
                raise
            attempt += 1
            delay = backoff_seconds(attempt)
            # Log the TYPE only: str() is empty for several of these, and no
            # request/response content belongs in a retry log line.
            log.warning(
                "%s: retrying upstream after %s (attempt %d/%d, backoff %.2fs)",
                operation,
                type(exc).__name__,
                attempt,
                budget,
                delay,
            )
            await _sleep(delay)


async def iter_upstream(
    stream: Callable[[], AsyncIterator[T]],
    *,
    operation: str,
    service: str = SERVICE,
    logger: logging.Logger | None = None,
    max_retries: int | None = None,
) -> AsyncIterator[T]:
    """Yield from ``stream()``, retrying ONLY before the first item is yielded.

    ``stream`` is a zero-argument factory so every attempt gets a fresh
    async iterator (and a fresh upstream request).  Once one item has been
    yielded the stream is committed: a later failure propagates untouched so
    the caller can emit exactly one terminal error frame.
    """
    budget = retry_budget(service) if max_retries is None else max(0, max_retries)
    log = logger or _log
    attempt = 0
    while True:
        committed = False
        inner = stream()
        try:
            async for item in inner:
                committed = True
                yield item
            return
        except Exception as exc:  # noqa: BLE001 — re-raised unless retryable
            if committed or not is_retryable(exc) or attempt >= budget:
                raise
            attempt += 1
            delay = backoff_seconds(attempt)
            log.warning(
                "%s: retrying upstream after %s (attempt %d/%d, backoff %.2fs)",
                operation,
                type(exc).__name__,
                attempt,
                budget,
                delay,
            )
            await _sleep(delay)
        finally:
            # Close the attempt deterministically (releases the upstream
            # response/connection) instead of waiting for GC — the caller may
            # have stopped consuming because its client went away.
            aclose = getattr(inner, "aclose", None)
            if aclose is not None:
                await aclose()


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "RETRYABLE_EXCEPTIONS",
    "SERVICE",
    "backoff_seconds",
    "is_retryable",
    "iter_upstream",
    "retry_budget",
    "retry_upstream",
]
