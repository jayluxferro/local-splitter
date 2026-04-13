"""Evaluation runner — orchestrates runs across tactic subsets.

The runner takes a list of workload samples and a set of tactic-subset
configurations, creates a Pipeline for each subset, runs every sample
through it, and returns structured results.

Usage from code::

    results = await run_matrix(
        samples,
        cloud=cloud_client,
        local=local_client,
        base_config=config,
    )

Usage from CLI (planned)::

    uv run local-splitter eval --workload path/to/samples.jsonl
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from pathlib import Path

from local_splitter.config import Config, TacticsConfig
from local_splitter.models import ChatClient, ModelBackendError
from local_splitter.pipeline import Pipeline, PipelineError, PipelineRequest

from .metrics import compute_summary
from .types import (
    RunResult,
    SampleResult,
    WorkloadSample,
    append_results_jsonl,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tactic subsets — extended as more tactics are implemented.
# ---------------------------------------------------------------------------

TACTIC_SUBSETS: dict[str, TacticsConfig] = {
    "baseline": TacticsConfig(),
    "T1_only": TacticsConfig(t1_route=True),
    "T2_only": TacticsConfig(t2_compress=True),
    "T3_only": TacticsConfig(t3_sem_cache=True),
    "T1_T3": TacticsConfig(t1_route=True, t3_sem_cache=True),
    "T1_T2": TacticsConfig(t1_route=True, t2_compress=True),
    "T4_only": TacticsConfig(t4_draft=True),
    "T5_only": TacticsConfig(t5_diff=True),
    "T6_only": TacticsConfig(t6_intent=True),
    "T7_only": TacticsConfig(t7_batch=True),
    "T1_T2_T3": TacticsConfig(t1_route=True, t2_compress=True, t3_sem_cache=True),
    "T1_T3_T4": TacticsConfig(t1_route=True, t3_sem_cache=True, t4_draft=True),
    "T1_T2_T3_T6": TacticsConfig(
        t1_route=True, t2_compress=True, t3_sem_cache=True, t6_intent=True,
    ),
    "all": TacticsConfig(
        t1_route=True, t2_compress=True, t3_sem_cache=True,
        t4_draft=True, t5_diff=True, t6_intent=True, t7_batch=True,
    ),
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

async def run_single(
    samples: list[WorkloadSample],
    *,
    pipeline: Pipeline,
    run_id: str | None = None,
    subset_name: str = "unknown",
    log_path: Path | None = None,
) -> list[SampleResult]:
    """Run every sample through ``pipeline`` and return per-sample results."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    results: list[SampleResult] = []
    for sample in samples:
        req = PipelineRequest(messages=sample.messages)
        try:
            resp = await pipeline.complete(req)
            sr = SampleResult(
                sample_id=sample.id,
                content=resp.content,
                served_by=resp.served_by,
                tokens_in_cloud=resp.usage_cloud.input_tokens or 0,
                tokens_out_cloud=resp.usage_cloud.output_tokens or 0,
                tokens_in_local=resp.usage_local.input_tokens or 0,
                tokens_out_local=resp.usage_local.output_tokens or 0,
                latency_ms=resp.latency_ms,
                trace=[e.as_dict() for e in resp.trace],
            )
        except (PipelineError, ModelBackendError) as exc:
            _log.warning("sample %s error: %s", sample.id, exc)
            sr = SampleResult(
                sample_id=sample.id,
                content="",
                served_by="error",
                tokens_in_cloud=0,
                tokens_out_cloud=0,
                tokens_in_local=0,
                tokens_out_local=0,
                latency_ms=0.0,
                trace=[],
                error=str(exc),
            )
        results.append(sr)

    if log_path is not None:
        append_results_jsonl(results, log_path, run_id=run_id, subset_name=subset_name)

    return results


# ---------------------------------------------------------------------------
# Matrix run
# ---------------------------------------------------------------------------

async def run_matrix(
    samples: list[WorkloadSample],
    *,
    cloud: ChatClient,
    local: ChatClient | None,
    base_config: Config,
    subsets: dict[str, TacticsConfig] | None = None,
    log_path: Path | None = None,
    cache_store: object | None = None,
) -> list[RunResult]:
    """Run every tactic subset against the workload samples.

    Returns one ``RunResult`` per subset, each containing per-sample
    results and an aggregate summary.  The ``"baseline"`` subset (all
    tactics disabled) is always included as the first result.

    If ``cache_store`` is provided, subsets with T3 enabled get a fresh
    cache per run (evict all before starting).
    """
    if subsets is None:
        subsets = TACTIC_SUBSETS

    # Ensure baseline is always first.
    ordered: list[tuple[str, TacticsConfig]] = []
    if "baseline" in subsets:
        ordered.append(("baseline", subsets["baseline"]))
    for name, tc in subsets.items():
        if name != "baseline":
            ordered.append((name, tc))

    run_batch_id = uuid.uuid4().hex[:8]
    results: list[RunResult] = []

    for subset_name, tactics in ordered:
        run_id = f"{run_batch_id}_{subset_name}"
        cfg = replace(base_config, tactics=tactics)

        # Give T3-enabled subsets the cache store; others get None.
        cs = cache_store if tactics.t3_sem_cache else None
        # Evict old entries so each subset starts fresh.
        if cs is not None and hasattr(cs, "evict_expired"):
            cs.evict_expired(ttl=0)

        pipeline = Pipeline(cloud=cloud, local=local, config=cfg, cache_store=cs)

        try:
            sample_results = await run_single(
                samples,
                pipeline=pipeline,
                run_id=run_id,
                subset_name=subset_name,
                log_path=log_path,
            )
        finally:
            # Don't close shared clients — pipeline doesn't own them in
            # eval context.  Just let it be GC'd.
            pass

        summary = compute_summary(sample_results)
        workload = samples[0].workload if samples else "unknown"

        results.append(
            RunResult(
                run_id=run_id,
                subset_name=subset_name,
                workload=workload,
                local_model=local.chat_model if local else "",
                cloud_model=cloud.chat_model,
                samples=sample_results,
                summary=summary,
            )
        )
        _log.info(
            "run %s/%s: %d samples, cloud_in=%d cloud_out=%d local_in=%d local_out=%d",
            subset_name,
            workload,
            summary.n_samples,
            summary.tokens_in_cloud,
            summary.tokens_out_cloud,
            summary.tokens_in_local,
            summary.tokens_out_local,
        )

    return results


__all__ = ["TACTIC_SUBSETS", "run_matrix", "run_single"]
