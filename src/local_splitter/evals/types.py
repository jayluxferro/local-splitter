"""Data types for the evaluation harness.

These are the internal currency of the eval system — workload samples go
in, run results come out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_splitter.models import Message


@dataclass(slots=True)
class WorkloadSample:
    """One request in a workload dataset."""

    id: str
    workload: str
    messages: list[Message]
    reference_response: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "workload": self.workload,
            "messages": self.messages,
        }
        if self.reference_response is not None:
            d["reference_response"] = self.reference_response
        if self.labels:
            d["labels"] = self.labels
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadSample:
        return cls(
            id=data["id"],
            workload=data["workload"],
            messages=data["messages"],
            reference_response=data.get("reference_response"),
            labels=data.get("labels", {}),
        )


@dataclass(slots=True)
class SampleResult:
    """Result of running one sample through the pipeline."""

    sample_id: str
    content: str
    served_by: str
    tokens_in_cloud: int
    tokens_out_cloud: int
    tokens_in_local: int
    tokens_out_local: int
    latency_ms: float
    trace: list[dict[str, Any]]
    error: str | None = None

    @property
    def tokens_cloud_total(self) -> int:
        return self.tokens_in_cloud + self.tokens_out_cloud

    @property
    def tokens_local_total(self) -> int:
        return self.tokens_in_local + self.tokens_out_local


@dataclass(slots=True)
class RunSummary:
    """Aggregate metrics for one eval run (one tactic subset × one workload)."""

    n_samples: int
    n_errors: int
    tokens_in_cloud: int
    tokens_out_cloud: int
    tokens_in_local: int
    tokens_out_local: int
    served_by: dict[str, int]
    latency_avg_ms: float
    latency_p50_ms: float | None
    latency_p99_ms: float | None


@dataclass(slots=True)
class RunResult:
    """Full result of one eval run: config + per-sample results + summary."""

    run_id: str
    subset_name: str
    workload: str
    local_model: str
    cloud_model: str
    samples: list[SampleResult]
    summary: RunSummary


# ---------------------------------------------------------------------------
# JSONL I/O for workloads
# ---------------------------------------------------------------------------

def load_workload(path: Path) -> list[WorkloadSample]:
    """Load samples from a JSONL file (one JSON object per line)."""
    samples: list[WorkloadSample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(WorkloadSample.from_dict(json.loads(line)))
    return samples


def save_workload(samples: list[WorkloadSample], path: Path) -> None:
    """Save samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# JSONL I/O for results
# ---------------------------------------------------------------------------

def append_results_jsonl(
    results: list[SampleResult],
    path: Path,
    *,
    run_id: str,
    subset_name: str,
) -> None:
    """Append per-sample results to a JSONL log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in results:
            row = asdict(r)
            row["run_id"] = run_id
            row["subset_name"] = subset_name
            f.write(json.dumps(row) + "\n")


__all__ = [
    "RunResult",
    "RunSummary",
    "SampleResult",
    "WorkloadSample",
    "append_results_jsonl",
    "load_workload",
    "save_workload",
]
