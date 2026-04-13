#!/usr/bin/env python3
"""Run the full evaluation suite against live backends.

Requires:
- Ollama running with llama3.2:3b + nomic-embed-text
- OPENAI_API_KEY set (for gpt-4o-mini cloud backend)

Usage:
    uv run python evals/run_eval.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from local_splitter.config import TacticsConfig, load_config
from local_splitter.evals import (
    TACTIC_SUBSETS,
    comparison_table,
    cost_estimate,
    load_workload,
    routing_accuracy,
    run_matrix,
    to_csv,
    token_savings_pct,
)
from local_splitter.models import build_chat_client
from local_splitter.pipeline.sem_cache import CacheStore

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

OUTPUT = Path(".local_splitter/eval")
WORKLOADS = Path("evals/workloads")

# Focus on the most informative subsets.
EVAL_SUBSETS = {
    "baseline": TacticsConfig(),
    "T1_only": TacticsConfig(t1_route=True),
    "T2_only": TacticsConfig(t2_compress=True),
    "T3_only": TacticsConfig(t3_sem_cache=True),
    "T4_only": TacticsConfig(t4_draft=True),
    "T1_T3": TacticsConfig(t1_route=True, t3_sem_cache=True),
    "T1_T2": TacticsConfig(t1_route=True, t2_compress=True),
    "T1_T2_T3": TacticsConfig(t1_route=True, t2_compress=True, t3_sem_cache=True),
    "all": TacticsConfig(
        t1_route=True, t2_compress=True, t3_sem_cache=True,
        t4_draft=True, t5_diff=True, t6_intent=True, t7_batch=True,
    ),
}


async def main() -> None:
    # Use eval-specific config (Ollama for both backends to avoid SSL issues).
    config_path = Path("evals/config_eval.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")
    config = load_config(config_path)
    cloud = build_chat_client(config.cloud)
    local = build_chat_client(config.local) if config.local else None

    if local is None:
        print("ERROR: local model not configured")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "runs.jsonl"

    wl_files = sorted(WORKLOADS.glob("wl*.jsonl"))
    if not wl_files:
        print(f"ERROR: no workload files in {WORKLOADS}")
        return

    all_runs = []
    summaries: dict[str, dict] = {}

    for wl_path in wl_files:
        samples = load_workload(wl_path)
        wl_name = wl_path.stem
        print(f"\n{'='*60}")
        print(f"  {wl_name}: {len(samples)} samples")
        print(f"{'='*60}")

        # Fresh cache store per workload.
        cache_store = CacheStore(OUTPUT / f"cache_{wl_name}.sqlite", embed_dim=768)

        try:
            runs = await run_matrix(
                samples,
                cloud=cloud,
                local=local,
                base_config=config,
                subsets=EVAL_SUBSETS,
                log_path=log_path,
                cache_store=cache_store,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            cache_store.close()
            continue

        all_runs.extend(runs)

        # Print comparison table.
        if len(runs) >= 2:
            print()
            print(comparison_table(runs[0], runs[1:]))
            print()

            baseline = runs[0]
            for run in runs[1:]:
                savings = token_savings_pct(baseline.summary, run.summary)
                local_pct = (
                    run.summary.served_by.get("local", 0)
                    + run.summary.served_by.get("cache", 0)
                ) / max(run.summary.n_samples, 1) * 100
                print(
                    f"  {run.subset_name:15s}: "
                    f"{savings:6.1f}% cloud savings, "
                    f"{local_pct:5.1f}% served locally/cached, "
                    f"avg {run.summary.latency_avg_ms:.0f}ms"
                )

            # Routing accuracy for T1 subsets.
            labels = {
                s.id: s.labels.get("trivial", False)
                for s in samples if "trivial" in s.labels
            }
            if labels:
                for run in runs:
                    if "T1" in run.subset_name:
                        acc = routing_accuracy(run.samples, labels)
                        print(
                            f"  {run.subset_name} routing: "
                            f"acc={acc['accuracy']:.2f} "
                            f"prec={acc['precision']:.2f} "
                            f"recall={acc['recall']:.2f} "
                            f"F1={acc['f1']:.2f}"
                        )

        # Store summary for paper.
        summaries[wl_name] = {
            "n_samples": len(samples),
            "subsets": {},
        }
        for run in runs:
            s = run.summary
            base_cloud = runs[0].summary.tokens_in_cloud + runs[0].summary.tokens_out_cloud
            run_cloud = s.tokens_in_cloud + s.tokens_out_cloud
            summaries[wl_name]["subsets"][run.subset_name] = {
                "tokens_in_cloud": s.tokens_in_cloud,
                "tokens_out_cloud": s.tokens_out_cloud,
                "tokens_in_local": s.tokens_in_local,
                "tokens_out_local": s.tokens_out_local,
                "served_by": s.served_by,
                "latency_avg_ms": round(s.latency_avg_ms, 1),
                "cost_usd": round(cost_estimate(s), 6),
                "cloud_savings_pct": round(
                    (base_cloud - run_cloud) / base_cloud * 100 if base_cloud > 0 else 0, 1
                ),
            }

        cache_store.close()

    # Write CSV.
    if all_runs:
        csv_path = OUTPUT / "results.csv"
        to_csv(all_runs, csv_path)
        print(f"\nCSV: {csv_path}")

    # Write summary JSON for paper.
    summary_path = OUTPUT / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"JSONL log: {log_path}")

    # Cleanup.
    await cloud.aclose()
    await local.aclose()


if __name__ == "__main__":
    asyncio.run(main())
