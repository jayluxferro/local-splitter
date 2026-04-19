"""Judge-model quality evaluation — pairwise preference scoring.

Sends the original prompt plus two candidate responses (baseline and
splitter) to a judge model in randomised order.  The judge picks A, B,
or TIE.  We aggregate win rates across all samples.

The judge prompt follows Zheng et al. (2023) "Judging LLM-as-a-Judge"
with position-debiased scoring: each pair is judged twice with swapped
order, and only consistent verdicts count.

Usage::

    verdicts = await judge_quality(
        samples, baseline_results, treatment_results, judge=cloud_client
    )
    print(quality_summary(verdicts))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from local_splitter.models import ChatClient, ModelBackendError

from .types import SampleResult, WorkloadSample

_log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are an impartial judge evaluating two AI assistant responses to a "
    "user prompt. Compare Response A and Response B on correctness, "
    "completeness, and helpfulness. Output ONLY one of: A, B, or TIE. "
    "Do not explain your reasoning."
)

JUDGE_TEMPLATE = (
    "### User Prompt\n{prompt}\n\n"
    "### Response A\n{response_a}\n\n"
    "### Response B\n{response_b}\n\n"
    "Which response is better? Answer with exactly one word: A, B, or TIE."
)


@dataclass(slots=True)
class JudgeVerdict:
    """Result of judging one sample pair."""

    sample_id: str
    winner: str  # "baseline", "treatment", "tie", "inconsistent", "error"
    raw_ab: str | None = None  # verdict when baseline=A
    raw_ba: str | None = None  # verdict when baseline=B


def _parse_verdict(text: str) -> str | None:
    """Extract A, B, or TIE from judge output."""
    text = text.strip().upper()
    if text in ("A", "B", "TIE"):
        return text
    for token in text.split():
        token = token.strip(".,;:!?")
        if token in ("A", "B", "TIE"):
            return token
    return None


async def _judge_once(
    prompt: str,
    response_a: str,
    response_b: str,
    judge: ChatClient,
) -> str | None:
    """Run one judge call. Returns A/B/TIE or None on failure."""
    user_content = JUDGE_TEMPLATE.format(
        prompt=prompt, response_a=response_a, response_b=response_b
    )
    try:
        resp = await judge.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return _parse_verdict(resp.content)
    except ModelBackendError as e:
        _log.warning("judge call failed: %s", e)
        return None


async def judge_pair(
    sample: WorkloadSample,
    baseline: SampleResult,
    treatment: SampleResult,
    judge: ChatClient,
) -> JudgeVerdict:
    """Judge one sample with position-debiased scoring.

    Calls the judge twice: once with baseline as A, once with baseline as B.
    Only consistent verdicts count.
    """
    prompt = "\n".join(
        m.get("content", "") for m in sample.messages if m.get("role") == "user"
    )

    if not baseline.content or not treatment.content:
        return JudgeVerdict(sample_id=sample.id, winner="error")
    if baseline.error or treatment.error:
        return JudgeVerdict(sample_id=sample.id, winner="error")

    # Pass 1: baseline=A, treatment=B
    v1 = await _judge_once(prompt, baseline.content, treatment.content, judge)
    # Pass 2: baseline=B, treatment=A
    v2 = await _judge_once(prompt, treatment.content, baseline.content, judge)

    if v1 is None or v2 is None:
        return JudgeVerdict(sample_id=sample.id, winner="error", raw_ab=v1, raw_ba=v2)

    # Reconcile: v1 is from baseline=A, v2 is from baseline=B (swapped)
    # Map both to "baseline wins" / "treatment wins" / "tie"
    def normalize(v: str, baseline_is_a: bool) -> str:
        if v == "TIE":
            return "tie"
        if baseline_is_a:
            return "baseline" if v == "A" else "treatment"
        else:
            return "baseline" if v == "B" else "treatment"

    n1 = normalize(v1, baseline_is_a=True)
    n2 = normalize(v2, baseline_is_a=False)

    if n1 == n2:
        winner = n1
    else:
        winner = "inconsistent"

    return JudgeVerdict(sample_id=sample.id, winner=winner, raw_ab=v1, raw_ba=v2)


async def judge_quality(
    samples: list[WorkloadSample],
    baseline_results: list[SampleResult],
    treatment_results: list[SampleResult],
    judge: ChatClient,
) -> list[JudgeVerdict]:
    """Judge all samples, matching by sample_id."""
    baseline_map = {r.sample_id: r for r in baseline_results}
    treatment_map = {r.sample_id: r for r in treatment_results}

    verdicts: list[JudgeVerdict] = []
    for sample in samples:
        bl = baseline_map.get(sample.id)
        tr = treatment_map.get(sample.id)
        if bl is None or tr is None:
            verdicts.append(JudgeVerdict(sample_id=sample.id, winner="error"))
            continue
        v = await judge_pair(sample, bl, tr, judge)
        verdicts.append(v)

    return verdicts


def quality_summary(verdicts: list[JudgeVerdict]) -> dict[str, Any]:
    """Aggregate verdicts into a summary."""
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.winner] = counts.get(v.winner, 0) + 1

    total = len(verdicts)
    valid = total - counts.get("error", 0) - counts.get("inconsistent", 0)

    return {
        "total": total,
        "valid": valid,
        "baseline_wins": counts.get("baseline", 0),
        "treatment_wins": counts.get("treatment", 0),
        "ties": counts.get("tie", 0),
        "inconsistent": counts.get("inconsistent", 0),
        "errors": counts.get("error", 0),
        "treatment_win_rate": (
            counts.get("treatment", 0) / valid if valid > 0 else 0.0
        ),
        "baseline_win_rate": (
            counts.get("baseline", 0) / valid if valid > 0 else 0.0
        ),
        "tie_rate": counts.get("tie", 0) / valid if valid > 0 else 0.0,
    }


__all__ = [
    "JudgeVerdict",
    "judge_pair",
    "judge_quality",
    "quality_summary",
]
