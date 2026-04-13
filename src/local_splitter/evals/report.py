"""Report generation — CSV export and markdown comparison tables.

Chart generation (matplotlib/seaborn) will be added when the paper
figures are needed. For now we produce CSV + markdown, which is enough
for the runner to produce a human-readable summary.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .metrics import cost_estimate, token_savings_pct
from .types import RunResult


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "run_id",
    "subset_name",
    "workload",
    "local_model",
    "cloud_model",
    "n_samples",
    "n_errors",
    "tokens_in_cloud",
    "tokens_out_cloud",
    "tokens_in_local",
    "tokens_out_local",
    "served_local",
    "served_cloud",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p99_ms",
    "cost_usd",
]


def to_csv(runs: list[RunResult], path: Path) -> None:
    """Write one-row-per-run CSV to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in runs:
            writer.writerow(_run_to_row(r))


def to_csv_string(runs: list[RunResult]) -> str:
    """Return the CSV as a string (useful for tests / stdout)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for r in runs:
        writer.writerow(_run_to_row(r))
    return buf.getvalue()


def _run_to_row(r: RunResult) -> dict[str, object]:
    s = r.summary
    return {
        "run_id": r.run_id,
        "subset_name": r.subset_name,
        "workload": r.workload,
        "local_model": r.local_model,
        "cloud_model": r.cloud_model,
        "n_samples": s.n_samples,
        "n_errors": s.n_errors,
        "tokens_in_cloud": s.tokens_in_cloud,
        "tokens_out_cloud": s.tokens_out_cloud,
        "tokens_in_local": s.tokens_in_local,
        "tokens_out_local": s.tokens_out_local,
        "served_local": s.served_by.get("local", 0),
        "served_cloud": s.served_by.get("cloud", 0),
        "latency_avg_ms": round(s.latency_avg_ms, 2),
        "latency_p50_ms": round(s.latency_p50_ms, 2) if s.latency_p50_ms else "",
        "latency_p99_ms": round(s.latency_p99_ms, 2) if s.latency_p99_ms else "",
        "cost_usd": round(cost_estimate(s), 6),
    }


# ---------------------------------------------------------------------------
# Markdown comparison table
# ---------------------------------------------------------------------------

def comparison_table(
    baseline: RunResult,
    treatments: list[RunResult],
) -> str:
    """Markdown table comparing treatments against a baseline run.

    Columns: subset name, cloud tokens saved (%), local tokens used,
    requests served locally (%), estimated cost, cost savings (%).
    """
    b = baseline.summary
    base_cloud = b.tokens_in_cloud + b.tokens_out_cloud
    base_cost = cost_estimate(b)

    lines: list[str] = []
    lines.append(
        "| Subset | Cloud tokens | Saved (%) | Local tokens | Local % | Cost ($) | Cost saved (%) |"
    )
    lines.append(
        "|--------|-------------|-----------|-------------|---------|----------|----------------|"
    )

    # Baseline row.
    lines.append(
        f"| baseline | {base_cloud:,} | — | 0 | 0% | {base_cost:.6f} | — |"
    )

    for t in treatments:
        s = t.summary
        cloud_total = s.tokens_in_cloud + s.tokens_out_cloud
        local_total = s.tokens_in_local + s.tokens_out_local
        savings = token_savings_pct(b, s)
        tcost = cost_estimate(s)
        cost_saved = ((base_cost - tcost) / base_cost * 100) if base_cost > 0 else 0.0
        local_pct = (
            s.served_by.get("local", 0) / s.n_samples * 100
            if s.n_samples > 0
            else 0.0
        )
        lines.append(
            f"| {t.subset_name} | {cloud_total:,} | {savings:.1f}% "
            f"| {local_total:,} | {local_pct:.0f}% "
            f"| {tcost:.6f} | {cost_saved:.1f}% |"
        )

    return "\n".join(lines)


__all__ = ["comparison_table", "to_csv", "to_csv_string"]
