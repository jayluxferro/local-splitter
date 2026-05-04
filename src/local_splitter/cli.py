"""Typer CLI entrypoint.

Subcommands:

    local-splitter serve-http  --config PATH     # FastAPI OpenAI-compat proxy
    local-splitter serve-mcp   --config PATH     # MCP stdio server
    local-splitter transform                     # one-shot transform (for hooks)
    local-splitter eval        --workload PATH   # run evaluation harness
    local-splitter --version                     # print version
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import replace
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
    """local-splitter — see README.md and docs/ARCHITECTURE.md."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _load(config_path: Path | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


def _build_pipeline(config: Config) -> Pipeline:
    cloud = build_chat_client(config.cloud) if config.cloud is not None else None
    local = build_chat_client(config.local) if config.local is not None else None
    cache_store = None
    if (
        config.tactics.t3_sem_cache
        and local is not None
        and config.local is not None
        and config.local.embed_model
    ):
        from local_splitter.pipeline.sem_cache import CacheStore

        state_dir = Path.cwd() / ".local_splitter"
        state_dir.mkdir(parents=True, exist_ok=True)
        cache_store = CacheStore(state_dir / "cache.sqlite", embed_dim=768)
    return Pipeline(cloud=cloud, local=local, config=config, cache_store=cache_store)


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
    upstream: str | None = typer.Option(
        None, "--upstream", help="Override the cloud upstream URL"
    ),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Run the FastAPI OpenAI-compatible proxy."""
    import uvicorn

    logging.basicConfig(level=log_level.upper())
    config = _load(config_path)
    if upstream and config.cloud:
        config = replace(config, cloud=replace(config.cloud, endpoint=upstream))
    pipeline = _build_pipeline(config)

    # Late import to keep CLI import cheap.
    from local_splitter.transport import create_app

    app_fastapi = create_app(pipeline, config)
    bind_host = host or config.transport.http_host
    bind_port = port or config.transport.http_port

    cloud_name = config.cloud.chat_model if config.cloud else "none"
    local_name = config.local.chat_model if config.local else "none"
    typer.echo(
        f"local-splitter http proxy on http://{bind_host}:{bind_port}/v1  "
        f"(cloud={cloud_name}, local={local_name})"
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


@app.command("transform")
def transform_cmd(
    config_path: Path | None = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml.",
        exists=False, dir_okay=False,
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p",
        help="Prompt text (alternative to stdin).",
    ),
    log_level: str = typer.Option("warning", "--log-level"),
) -> None:
    """One-shot transform for hook integration.

    Reads a prompt from --prompt or stdin, runs tactic transforms
    (T1 route, T2 compress, T3 cache, T5 diff), and prints JSON
    to stdout. Designed for use in Claude Code hooks.

    \b
    Input: plain text prompt, or JSON {"messages": [...]}
    Output JSON:
      {"action": "answer", "response": "..."}        — answered locally
      {"action": "passthrough", "messages": [...]}    — send these to your model

    \b
    Example hook usage:
      local-splitter transform -p "what is 2+2" --config config.yaml
    """
    logging.basicConfig(level=log_level.upper())
    config = _load(config_path)
    pipeline = _build_pipeline(config)

    # Read input.
    if prompt is not None:
        raw_input = prompt
    elif not sys.stdin.isatty():
        raw_input = sys.stdin.read()
    else:
        typer.secho(
            "error: provide --prompt or pipe input via stdin",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    raw_input = raw_input.strip()
    if not raw_input:
        typer.secho("error: empty input", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Parse input: JSON messages array or plain text.
    messages: list[dict[str, str]]
    try:
        parsed = json.loads(raw_input)
        if isinstance(parsed, dict) and "messages" in parsed:
            messages = parsed["messages"]
        elif isinstance(parsed, list):
            messages = parsed
        else:
            messages = [{"role": "user", "content": raw_input}]
    except json.JSONDecodeError:
        messages = [{"role": "user", "content": raw_input}]

    from local_splitter.pipeline import PipelineRequest

    req = PipelineRequest(messages=messages)

    async def _run():
        return await pipeline.transform(req)

    try:
        transformed, trace, local_response = asyncio.run(_run())
    except Exception as exc:
        result = {"action": "error", "error": str(exc)}
        typer.echo(json.dumps(result))
        raise typer.Exit(code=1) from exc

    if local_response is not None:
        result = {
            "action": "answer",
            "response": local_response,
            "served_by": "local",
            "trace": [e.as_dict() for e in trace],
        }
    else:
        result = {
            "action": "passthrough",
            "messages": transformed,
            "trace": [e.as_dict() for e in trace],
        }

    typer.echo(json.dumps(result))


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


@app.command("demo")
def demo_command() -> None:
    """Print a concise first-run checklist (install, models, config, tests)."""
    typer.echo("local-splitter — quick checklist\n")
    typer.echo("  1. uv sync")
    typer.echo("  2. ollama pull llama3.2:3b && ollama pull nomic-embed-text")
    typer.echo("  3. cp config.example.yaml config.yaml  # set cloud endpoint + api_key_env")
    typer.echo("  4. uv run pytest -q")
    typer.echo("  5. uv run local-splitter serve-http --config config.yaml")
    typer.echo("\nOptional:")
    typer.echo("  • uv run python scripts/trace_report.py .local_splitter/eval/runs.jsonl -o /tmp/trace.html")
    typer.echo("  • JSON Schemas for MCP tools: schemas/mcp-tools.json")
    typer.echo("  • examples/openai_chat.sh — minimal curl to the proxy")


if __name__ == "__main__":
    app()
