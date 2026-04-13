"""Evaluation harness: runner, metrics, quality judge, report generation.

See docs/EVALUATION.md for the 25-subset matrix design.

Quick start::

    from local_splitter.evals import run_matrix, comparison_table

    results = await run_matrix(samples, cloud=cloud, local=local, base_config=cfg)
    baseline, *treatments = results
    print(comparison_table(baseline, treatments))
"""

from .metrics import compute_summary, cost_estimate, routing_accuracy, token_savings_pct
from .report import comparison_table, to_csv, to_csv_string
from .runner import TACTIC_SUBSETS, run_matrix, run_single
from .types import (
    RunResult,
    RunSummary,
    SampleResult,
    WorkloadSample,
    append_results_jsonl,
    load_workload,
    save_workload,
)

__all__ = [
    "TACTIC_SUBSETS",
    "RunResult",
    "RunSummary",
    "SampleResult",
    "WorkloadSample",
    "append_results_jsonl",
    "comparison_table",
    "compute_summary",
    "cost_estimate",
    "load_workload",
    "routing_accuracy",
    "run_matrix",
    "run_single",
    "save_workload",
    "to_csv",
    "to_csv_string",
    "token_savings_pct",
]
