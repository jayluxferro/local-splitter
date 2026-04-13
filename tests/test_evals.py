"""Tests for the evaluation harness.

Exercises the full runner → metrics → report pipeline using
FakeChatClient so nothing leaves the process.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from local_splitter.config import Config, ModelConfig, TacticsConfig
from local_splitter.evals import (
    RunResult,
    SampleResult,
    WorkloadSample,
    comparison_table,
    compute_summary,
    cost_estimate,
    load_workload,
    routing_accuracy,
    run_matrix,
    run_single,
    save_workload,
    to_csv,
    to_csv_string,
    token_savings_pct,
)
from local_splitter.models import Usage
from local_splitter.pipeline import Pipeline

from _fakes import FakeChatClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _samples(n: int = 5, workload: str = "wl_test") -> list[WorkloadSample]:
    return [
        WorkloadSample(
            id=f"{workload}_{i:03d}",
            workload=workload,
            messages=[{"role": "user", "content": f"question {i}"}],
            labels={"trivial": i % 2 == 0},
        )
        for i in range(n)
    ]


def _base_config(*, t1: bool = False) -> Config:
    return Config(
        cloud=ModelConfig(
            backend="openai_compat", endpoint="http://cloud", chat_model="cloud-m"
        ),
        local=ModelConfig(
            backend="ollama", endpoint="http://local", chat_model="local-m"
        ),
        tactics=TacticsConfig(t1_route=t1),
    )


# ---------------------------------------------------------------------------
# types: WorkloadSample serialization
# ---------------------------------------------------------------------------

class TestWorkloadIO:
    def test_round_trip_jsonl(self, tmp_path: Path) -> None:
        samples = _samples(3)
        p = tmp_path / "test.jsonl"
        save_workload(samples, p)
        loaded = load_workload(p)
        assert len(loaded) == 3
        assert loaded[0].id == "wl_test_000"
        assert loaded[2].messages[0]["content"] == "question 2"

    def test_labels_preserved(self, tmp_path: Path) -> None:
        samples = _samples(2)
        p = tmp_path / "test.jsonl"
        save_workload(samples, p)
        loaded = load_workload(p)
        assert loaded[0].labels["trivial"] is True
        assert loaded[1].labels["trivial"] is False

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert load_workload(p) == []


# ---------------------------------------------------------------------------
# metrics: compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def test_basic_aggregation(self) -> None:
        results = [
            SampleResult(
                sample_id="s1", content="a", served_by="cloud",
                tokens_in_cloud=100, tokens_out_cloud=20,
                tokens_in_local=0, tokens_out_local=0,
                latency_ms=10.0, trace=[],
            ),
            SampleResult(
                sample_id="s2", content="b", served_by="local",
                tokens_in_cloud=0, tokens_out_cloud=0,
                tokens_in_local=15, tokens_out_local=5,
                latency_ms=5.0, trace=[],
            ),
        ]
        s = compute_summary(results)
        assert s.n_samples == 2
        assert s.n_errors == 0
        assert s.tokens_in_cloud == 100
        assert s.tokens_out_cloud == 20
        assert s.tokens_in_local == 15
        assert s.tokens_out_local == 5
        assert s.served_by == {"cloud": 1, "local": 1}
        assert s.latency_avg_ms == 7.5

    def test_errors_counted(self) -> None:
        results = [
            SampleResult(
                sample_id="s1", content="", served_by="error",
                tokens_in_cloud=0, tokens_out_cloud=0,
                tokens_in_local=0, tokens_out_local=0,
                latency_ms=0.0, trace=[], error="boom",
            ),
        ]
        s = compute_summary(results)
        assert s.n_errors == 1


# ---------------------------------------------------------------------------
# metrics: token_savings_pct
# ---------------------------------------------------------------------------

class TestTokenSavings:
    def test_fifty_percent_savings(self) -> None:
        baseline = compute_summary([
            SampleResult("s1", "a", "cloud", 100, 100, 0, 0, 10.0, []),
        ])
        treatment = compute_summary([
            SampleResult("s1", "a", "cloud", 50, 50, 10, 5, 10.0, []),
        ])
        assert token_savings_pct(baseline, treatment) == 50.0

    def test_hundred_percent_savings(self) -> None:
        baseline = compute_summary([
            SampleResult("s1", "a", "cloud", 100, 50, 0, 0, 10.0, []),
        ])
        treatment = compute_summary([
            SampleResult("s1", "a", "local", 0, 0, 20, 10, 5.0, []),
        ])
        assert token_savings_pct(baseline, treatment) == 100.0

    def test_zero_baseline_returns_zero(self) -> None:
        baseline = compute_summary([
            SampleResult("s1", "a", "local", 0, 0, 10, 5, 10.0, []),
        ])
        treatment = compute_summary([
            SampleResult("s1", "a", "local", 0, 0, 10, 5, 10.0, []),
        ])
        assert token_savings_pct(baseline, treatment) == 0.0


# ---------------------------------------------------------------------------
# metrics: cost_estimate
# ---------------------------------------------------------------------------

class TestCostEstimate:
    def test_cloud_only_cost(self) -> None:
        s = compute_summary([
            SampleResult("s1", "a", "cloud", 1_000_000, 1_000_000, 0, 0, 10.0, []),
        ])
        # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
        assert abs(cost_estimate(s) - 0.75) < 1e-9

    def test_local_is_free_by_default(self) -> None:
        s = compute_summary([
            SampleResult("s1", "a", "local", 0, 0, 1_000_000, 1_000_000, 10.0, []),
        ])
        assert cost_estimate(s) == 0.0


# ---------------------------------------------------------------------------
# metrics: routing_accuracy
# ---------------------------------------------------------------------------

class TestRoutingAccuracy:
    def test_perfect_accuracy(self) -> None:
        results = [
            SampleResult("s1", "a", "local", 0, 0, 10, 5, 5.0, []),
            SampleResult("s2", "b", "cloud", 20, 10, 0, 0, 10.0, []),
        ]
        labels = {"s1": True, "s2": False}
        acc = routing_accuracy(results, labels)
        assert acc["accuracy"] == 1.0
        assert acc["precision"] == 1.0
        assert acc["recall"] == 1.0
        assert acc["f1"] == 1.0

    def test_all_wrong(self) -> None:
        results = [
            SampleResult("s1", "a", "cloud", 20, 10, 0, 0, 10.0, []),
            SampleResult("s2", "b", "local", 0, 0, 10, 5, 5.0, []),
        ]
        labels = {"s1": True, "s2": False}
        acc = routing_accuracy(results, labels)
        assert acc["accuracy"] == 0.0
        assert acc["tp"] == 0
        assert acc["fp"] == 1
        assert acc["fn"] == 1

    def test_missing_labels_excluded(self) -> None:
        results = [
            SampleResult("s1", "a", "local", 0, 0, 10, 5, 5.0, []),
            SampleResult("s2", "b", "cloud", 20, 10, 0, 0, 10.0, []),
        ]
        labels = {"s1": True}  # s2 not labelled
        acc = routing_accuracy(results, labels)
        assert acc["tp"] == 1
        assert acc["tn"] == 0  # s2 excluded, not counted


# ---------------------------------------------------------------------------
# runner: run_single
# ---------------------------------------------------------------------------

async def test_run_single_collects_all_samples() -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="cloud answer",
        usage=Usage(input_tokens=20, output_tokens=5),
    )
    pipeline = Pipeline(cloud=cloud, local=None, config=_base_config())
    samples = _samples(4)

    results = await run_single(samples, pipeline=pipeline, subset_name="baseline")

    assert len(results) == 4
    assert all(r.served_by == "cloud" for r in results)
    assert all(r.tokens_in_cloud == 20 for r in results)
    assert results[0].sample_id == "wl_test_000"
    assert results[3].sample_id == "wl_test_003"


async def test_run_single_logs_to_jsonl(tmp_path: Path) -> None:
    cloud = FakeChatClient(chat_model="cloud-m", usage=Usage(input_tokens=10, output_tokens=3))
    pipeline = Pipeline(cloud=cloud, local=None, config=_base_config())
    log = tmp_path / "runs.jsonl"

    await run_single(
        _samples(2), pipeline=pipeline, run_id="test_run",
        subset_name="baseline", log_path=log,
    )

    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["run_id"] == "test_run"
    assert row["subset_name"] == "baseline"
    assert row["tokens_in_cloud"] == 10


async def test_run_single_handles_pipeline_error() -> None:
    from local_splitter.models import ModelBackendError

    cloud = FakeChatClient(raise_on_complete=ModelBackendError("503"))
    pipeline = Pipeline(cloud=cloud, local=None, config=_base_config())

    results = await run_single(_samples(2), pipeline=pipeline)

    assert len(results) == 2
    assert all(r.error is not None for r in results)
    assert all(r.served_by == "error" for r in results)


# ---------------------------------------------------------------------------
# runner: run_matrix
# ---------------------------------------------------------------------------

async def test_run_matrix_baseline_and_t1() -> None:
    cloud = FakeChatClient(
        chat_model="cloud-m", reply_content="cloud answer",
        usage=Usage(input_tokens=50, output_tokens=10),
    )
    # Local client: classifier always returns "TRIVIAL", then answers.
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["TRIVIAL", "local ans"] * 5,  # 5 samples × 2 calls
        usage=Usage(input_tokens=8, output_tokens=2),
    )
    samples = _samples(5)
    subsets = {
        "baseline": TacticsConfig(),
        "T1_only": TacticsConfig(t1_route=True),
    }
    runs = await run_matrix(
        samples, cloud=cloud, local=local, base_config=_base_config(),
        subsets=subsets,
    )

    assert len(runs) == 2
    assert runs[0].subset_name == "baseline"
    assert runs[1].subset_name == "T1_only"

    # Baseline: everything goes to cloud.
    baseline = runs[0]
    assert baseline.summary.served_by.get("cloud", 0) == 5
    assert baseline.summary.tokens_in_cloud == 250  # 5 × 50
    assert baseline.summary.tokens_out_cloud == 50   # 5 × 10

    # T1: everything classified TRIVIAL, served locally.
    t1 = runs[1]
    assert t1.summary.served_by.get("local", 0) == 5
    assert t1.summary.tokens_in_cloud == 0
    assert t1.summary.tokens_in_local == 40  # 5 × 8

    # Token savings should be 100%.
    savings = token_savings_pct(baseline.summary, t1.summary)
    assert savings == 100.0


async def test_run_matrix_baseline_always_first() -> None:
    cloud = FakeChatClient(chat_model="cloud-m")
    local = FakeChatClient(
        chat_model="local-m",
        reply_sequence=["COMPLEX"] * 5,
    )
    samples = _samples(5)

    # Pass subsets in non-baseline-first order.
    subsets = {
        "T1_only": TacticsConfig(t1_route=True),
        "baseline": TacticsConfig(),
    }
    runs = await run_matrix(
        samples, cloud=cloud, local=local,
        base_config=_base_config(), subsets=subsets,
    )

    assert runs[0].subset_name == "baseline"


async def test_run_matrix_logs_to_file(tmp_path: Path) -> None:
    cloud = FakeChatClient(chat_model="cloud-m", usage=Usage(input_tokens=10, output_tokens=3))
    log = tmp_path / "runs.jsonl"

    await run_matrix(
        _samples(2), cloud=cloud, local=None,
        base_config=_base_config(),
        subsets={"baseline": TacticsConfig()},
        log_path=log,
    )

    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2  # 2 samples × 1 subset


# ---------------------------------------------------------------------------
# report: CSV
# ---------------------------------------------------------------------------

class TestCSVReport:
    def _make_run(self, subset: str, cloud_in: int, cloud_out: int) -> RunResult:
        results = [
            SampleResult(
                "s1", "a", "cloud" if cloud_in > 0 else "local",
                cloud_in, cloud_out,
                0 if cloud_in > 0 else 15, 0 if cloud_in > 0 else 5,
                10.0, [],
            )
        ]
        return RunResult(
            run_id=f"run_{subset}",
            subset_name=subset,
            workload="wl_test",
            local_model="local-m",
            cloud_model="cloud-m",
            samples=results,
            summary=compute_summary(results),
        )

    def test_csv_string_has_header_and_rows(self) -> None:
        runs = [self._make_run("baseline", 100, 20), self._make_run("T1", 0, 0)]
        text = to_csv_string(runs)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["subset_name"] == "baseline"
        assert rows[1]["subset_name"] == "T1"
        assert int(rows[0]["tokens_in_cloud"]) == 100
        assert int(rows[1]["tokens_in_cloud"]) == 0

    def test_csv_file(self, tmp_path: Path) -> None:
        runs = [self._make_run("baseline", 50, 10)]
        p = tmp_path / "results.csv"
        to_csv(runs, p)
        assert p.exists()
        text = p.read_text()
        assert "baseline" in text


# ---------------------------------------------------------------------------
# report: comparison_table
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def test_markdown_table_structure(self) -> None:
        results_base = [
            SampleResult("s1", "a", "cloud", 100, 50, 0, 0, 10.0, []),
        ]
        results_t1 = [
            SampleResult("s1", "a", "local", 0, 0, 20, 10, 5.0, []),
        ]
        baseline = RunResult(
            "r1", "baseline", "wl_test", "local-m", "cloud-m",
            results_base, compute_summary(results_base),
        )
        t1 = RunResult(
            "r2", "T1_only", "wl_test", "local-m", "cloud-m",
            results_t1, compute_summary(results_t1),
        )
        table = comparison_table(baseline, [t1])

        assert "| baseline |" in table
        assert "| T1_only |" in table
        assert "100.0%" in table  # 100% cloud tokens saved
