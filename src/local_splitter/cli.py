"""Typer CLI entrypoint.

Subcommands:

    local-splitter serve-http  --config PATH     # FastAPI OpenAI-compat proxy
    local-splitter serve-mcp   --config PATH     # MCP stdio server
    local-splitter eval        --workload PATH   # run evaluation harness
    local-splitter --version                     # print version
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from local_splitter import __version__
from local_splitter.config import Config, ConfigError, load_config
from local_splitter.models import build_chat_client
from local_splitter.pipeline import Pipeline

app = typer.Typer(
    name="local-splitter",
    help="MCP shim that splits LLM requests between a local and cloud model.",
    no_args_is_help=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: FBT001 — typer flag
        False,
        "--version",
        "-V",
        help="Print version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """local-splitter — see AGENT.md for the research design."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _load(config_path: Path | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


def _build_pipeline(config: Config) -> Pipeline:
    cloud = build_chat_client(config.cloud)
    local = build_chat_client(config.local) if config.local is not None else None
    return Pipeline(cloud=cloud, local=local, config=config)


@app.command("serve-http")
def serve_http(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml (defaults: $LOCAL_SPLITTER_CONFIG, .local_splitter/config.yaml, ./config.yaml).",
        exists=False,
        dir_okay=False,
    ),
    host: str | None = typer.Option(None, "--host", help="Override transport.http_host."),
    port: int | None = typer.Option(None, "--port", help="Override transport.http_port."),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Run the FastAPI OpenAI-compatible proxy."""
    import uvicorn

    logging.basicConfig(level=log_level.upper())
    config = _load(config_path)
    pipeline = _build_pipeline(config)

    # Late import to keep CLI import cheap.
    from local_splitter.transport import create_app

    app_fastapi = create_app(pipeline, config)
    bind_host = host or config.transport.http_host
    bind_port = port or config.transport.http_port

    typer.echo(
        f"local-splitter http proxy on http://{bind_host}:{bind_port}/v1  "
        f"(cloud={config.cloud.chat_model}, "
        f"local={config.local.chat_model if config.local else 'none'})"
    )
    uvicorn.run(app_fastapi, host=bind_host, port=bind_port, log_level=log_level.lower())


@app.command("serve-mcp")
def serve_mcp(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml.",
        exists=False,
        dir_okay=False,
    ),
    log_level: str = typer.Option("warning", "--log-level"),
) -> None:
    """Run the MCP stdio server."""
    logging.basicConfig(level=log_level.upper())
    config = _load(config_path)
    pipeline = _build_pipeline(config)

    from local_splitter.transport import create_mcp_server

    server = create_mcp_server(pipeline, config)
    asyncio.run(server.run_stdio_async())


@app.command("eval")
def eval_cmd(
    workload: list[Path] = typer.Option(
        ...,
        "--workload",
        "-w",
        help="Path to a workload JSONL file. Can be specified multiple times.",
        exists=True,
        dir_okay=False,
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml.",
        exists=False,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path(".local_splitter/eval"),
        "--output",
        "-o",
        help="Output directory for results CSV and JSONL log.",
    ),
    subsets: str | None = typer.Option(
        None,
        "--subsets",
        "-s",
        help="Comma-separated subset names to run (default: all defined subsets).",
    ),
    log_level: str = typer.Option("warning", "--log-level"),
) -> None:
    """Run the evaluation harness on one or more workloads."""
    logging.basicConfig(level=log_level.upper())
    config = _load(config_path)
    pipeline_config = config

    cloud = build_chat_client(config.cloud)
    local_client = build_chat_client(config.local) if config.local is not None else None

    from local_splitter.evals import (
        TACTIC_SUBSETS,
        comparison_table,
        load_workload,
        run_matrix,
        to_csv,
        token_savings_pct,
    )

    # Resolve subsets.
    if subsets:
        names = [s.strip() for s in subsets.split(",")]
        chosen = {}
        for name in names:
            if name not in TACTIC_SUBSETS:
                typer.secho(f"unknown subset: {name}", fg=typer.colors.RED, err=True)
                typer.secho(f"available: {', '.join(TACTIC_SUBSETS)}", err=True)
                raise typer.Exit(code=2)
            chosen[name] = TACTIC_SUBSETS[name]
        # Always include baseline.
        if "baseline" not in chosen:
            chosen = {"baseline": TACTIC_SUBSETS["baseline"], **chosen}
    else:
        chosen = None  # use all

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "runs.jsonl"

    all_runs = []
    for wl_path in workload:
        samples = load_workload(wl_path)
        if not samples:
            typer.secho(f"empty workload: {wl_path}", fg=typer.colors.YELLOW, err=True)
            continue

        typer.echo(f"\n--- {wl_path.stem}: {len(samples)} samples ---")

        # Build a cache store for T3 if needed.
        cache_store = None
        if config.tactics.t3_sem_cache or (chosen and any(
            tc.t3_sem_cache for tc in (chosen or TACTIC_SUBSETS).values()
        )):
            from local_splitter.pipeline.sem_cache import CacheStore
            cache_db = output / f"cache_{wl_path.stem}.sqlite"
            embed_dim = 768  # nomic-embed-text default
            cache_store = CacheStore(cache_db, embed_dim=embed_dim)

        async def _run():
            return await run_matrix(
                samples,
                cloud=cloud,
                local=local_client,
                base_config=pipeline_config,
                subsets=chosen,
                log_path=log_path,
                cache_store=cache_store,
            )

        runs = asyncio.run(_run())
        all_runs.extend(runs)

        # Print comparison table.
        if len(runs) >= 2:
            baseline_run = runs[0]
            treatment_runs = runs[1:]
            typer.echo(comparison_table(baseline_run, treatment_runs))
            typer.echo("")
            for t in treatment_runs:
                savings = token_savings_pct(baseline_run.summary, t.summary)
                typer.echo(f"  {t.subset_name}: {savings:.1f}% cloud token savings")

        if cache_store is not None:
            cache_store.close()

    # Write CSV.
    if all_runs:
        csv_path = output / "results.csv"
        to_csv(all_runs, csv_path)
        typer.echo(f"\nResults written to {csv_path}")
        typer.echo(f"JSONL log: {log_path}")


if __name__ == "__main__":
    app()
