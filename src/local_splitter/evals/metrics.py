"""Metric computation for eval runs.

Takes raw per-sample results and produces aggregate summaries and
cross-run comparisons.
"""

from __future__ import annotations

from collections import Counter

from .types import RunSummary, SampleResult


def _percentile(sorted_values: list[float], q: float) -> float:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    idx = min(n - 1, max(0, round(q * (n - 1))))
    return sorted_values[idx]


def compute_summary(results: list[SampleResult]) -> RunSummary:
    """Compute aggregate metrics from per-sample results."""
    n = len(results)
    n_errors = sum(1 for r in results if r.error is not None)
    tokens_in_cloud = sum(r.tokens_in_cloud for r in results)
    tokens_out_cloud = sum(r.tokens_out_cloud for r in results)
    tokens_in_local = sum(r.tokens_in_local for r in results)
    tokens_out_local = sum(r.tokens_out_local for r in results)

    served: Counter[str] = Counter()
    latencies: list[float] = []
    for r in results:
        served[r.served_by] += 1
        latencies.append(r.latency_ms)

    avg = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p50 = _percentile(sorted_lat, 0.50) if sorted_lat else None
    p99 = _percentile(sorted_lat, 0.99) if sorted_lat else None

    return RunSummary(
        n_samples=n,
        n_errors=n_errors,
        tokens_in_cloud=tokens_in_cloud,
        tokens_out_cloud=tokens_out_cloud,
        tokens_in_local=tokens_in_local,
        tokens_out_local=tokens_out_local,
        served_by=dict(served),
        latency_avg_ms=avg,
        latency_p50_ms=p50,
        latency_p99_ms=p99,
    )


def token_savings_pct(baseline: RunSummary, treatment: RunSummary) -> float:
    """Cloud token savings as a percentage.

    ``(baseline_cloud_total - treatment_cloud_total) / baseline_cloud_total``

    Returns 0.0 if the baseline has no cloud tokens (avoids division by zero).
    """
    base_total = baseline.tokens_in_cloud + baseline.tokens_out_cloud
    treat_total = treatment.tokens_in_cloud + treatment.tokens_out_cloud
    if base_total == 0:
        return 0.0
    return (base_total - treat_total) / base_total * 100.0


def cost_estimate(
    summary: RunSummary,
    *,
    cloud_input_per_mtok: float = 0.15,
    cloud_output_per_mtok: float = 0.60,
    local_input_per_mtok: float = 0.0,
    local_output_per_mtok: float = 0.0,
) -> float:
    """Estimated dollar cost using a rate card (per million tokens).

    Local tokens default to $0 (self-hosted). Cloud rates default to
    gpt-4o-mini pricing as of 2025-Q1.
    """
    cost = (
        summary.tokens_in_cloud * cloud_input_per_mtok / 1_000_000
        + summary.tokens_out_cloud * cloud_output_per_mtok / 1_000_000
        + summary.tokens_in_local * local_input_per_mtok / 1_000_000
        + summary.tokens_out_local * local_output_per_mtok / 1_000_000
    )
    return cost


def routing_accuracy(
    results: list[SampleResult],
    labels: dict[str, bool],
) -> dict[str, float]:
    """Compute T1 routing accuracy against ground-truth labels.

    ``labels`` maps sample_id → True if the sample *should* have been
    routed locally (i.e. it's trivial).  Returns precision, recall, and
    F1 for the "local" class.

    Samples missing from ``labels`` are excluded.
    """
    tp = fp = fn = tn = 0
    for r in results:
        if r.sample_id not in labels:
            continue
        should_be_local = labels[r.sample_id]
        was_local = r.served_by == "local"
        if should_be_local and was_local:
            tp += 1
        elif should_be_local and not was_local:
            fn += 1
        elif not should_be_local and was_local:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


__all__ = [
    "compute_summary",
    "cost_estimate",
    "routing_accuracy",
    "token_savings_pct",
]
