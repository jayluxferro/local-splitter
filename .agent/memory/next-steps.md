---
name: next steps
description: Staged task list for building out local-splitter
type: project
---

# Next steps

## Status as of scaffold-time

- Research brief: complete (`docs/`)
- Agent handoff docs: complete (`AGENT.md`, `.agent/memory/`)
- Paper skeleton: `paper/paper.tex` stubbed
- Code: not started
- Evaluation harness: not started
- Workload captures: not started

## Stage 1 — Scaffold the Python project

1. `pyproject.toml` using `hatch` or `uv`. Only runtime deps:
   - `httpx` (HTTP client)
   - `mcp` (MCP SDK)
   - `pyyaml` (config)
   - `typer` + `rich` (CLI)
   - `fastapi` + `uvicorn` (HTTP proxy)
   - `sqlite-vec` (semantic cache)
2. `src/local_splitter/` skeleton (`__init__.py`, `config.py`,
   `transport/`, `models/`, `pipeline/`, `evals/`).
3. `tests/` with pytest + pytest-asyncio.
4. `README.md` update once installable.

Ask Jay which of `hatch` / `uv` he prefers before committing to a
build backend.

## Stage 2 — Model backends

1. `src/local_splitter/models/base.py` — `ChatClient` protocol with
   `complete`, `embed`, and `stream` methods.
2. `src/local_splitter/models/ollama.py` — Ollama REST client.
3. `src/local_splitter/models/openai_compat.py` — OpenAI-compatible
   client (`/v1/chat/completions`, `/v1/embeddings`).
4. Integration test: stand up Ollama with `llama3.2:3b` and
   `nomic-embed-text`, round-trip a chat and an embedding.
5. Optional smoke test against a real cloud endpoint (gpt-4o-mini)
   gated on an environment variable so CI doesn't burn tokens.

## Stage 3 — Transport layer

1. `src/local_splitter/transport/mcp_server.py` — MCP stdio server
   exposing `split.complete`, `split.classify`, `split.cache_lookup`,
   `split.stats`, `split.config`.
2. `src/local_splitter/transport/http_proxy.py` — FastAPI server
   exposing `/v1/chat/completions`, `/v1/models`,
   `/v1/splitter/stats`.
3. Both transports call the same pipeline orchestrator.

## Stage 4 — Pipeline: implement T1 first

Start with **T1 (route)** because it's the simplest and most
impactful. Build it end-to-end including the local model call,
classifier prompt, decision logic, and metric logging. Measure it on
a tiny workload to verify the pipeline works before adding more
tactics.

1. `src/local_splitter/pipeline/route.py` — classifier + local answer.
2. `src/local_splitter/pipeline/__init__.py` — orchestrator that
   calls stages in order, respecting config.
3. First real measurement: run T1 alone on WL3 (general chat, 50
   samples) and report tokens-saved / quality-delta. This is the
   first paper-worthy data point.

## Stage 5 — Pipeline: T3, T2, T5

In this order because the cache must run before compression (see
`decisions.md`), and diff is the highest-impact tactic for the
edit-heavy workload.

1. `src/local_splitter/pipeline/sem_cache.py` — sqlite-vec backed,
   embedding-keyed.
2. `src/local_splitter/pipeline/compress.py`.
3. `src/local_splitter/pipeline/diff.py`.
4. Run T1+T3, T1+T2, T1+T5 pairwise and report.

## Stage 6 — Pipeline: T4, T6, T7

The remaining three tactics, in any order.

1. `src/local_splitter/pipeline/draft.py`.
2. `src/local_splitter/pipeline/intent.py`.
3. `src/local_splitter/pipeline/batch.py`.

## Stage 7 — Workload capture

Each of WL1..WL4 needs ~200 samples. Options:

1. Capture real sessions from a proxy (Jay already has proxy-atlas
   for this). Scrub via `llm-redactor` or presidio.
2. Generate synthetic workloads from public datasets:
   - **HumanEval** or **MBPP** for code-completion (WL1 surrogate)
   - **CodeXGLUE** or **CoNaLa** for explanation (WL2 surrogate)
   - **OpenAssistant** or **ShareGPT** for general chat (WL3
     surrogate)
   - **MS MARCO** for RAG-heavy (WL4 surrogate)

Jay will prefer real captures for authenticity, but synthetic is
fine for the MVP. Confirm before committing.

## Stage 8 — Evaluation harness

1. `evals/runner.py` — takes a config, a workload, a tactic subset,
   and outputs a CSV row per sample.
2. `evals/metrics.py` — token counting (via `tiktoken` or local
   tokeniser), latency, cost.
3. `evals/quality.py` — judge-model A/B comparison.
4. `evals/report.py` — aggregates and produces figures.

## Stage 9 — First full-matrix evaluation

Run the 25-subset matrix on WL3 (general chat, cheapest to iterate)
with one local model (llama3.2:3b) and one cloud model (gpt-4o-mini).
Sanity check results, fix bugs, re-run.

## Stage 10 — Paper first draft

Once the first results are in, draft the paper. Fill out the stubs
in `paper/paper.tex`. Add figures. Send to Jay for review.

## Stage 11 — Full evaluation matrix

Run the full WL × local × cloud × subset matrix. ~300 runs total.
Budget: ~1 day of wall clock with a cheap cloud model.

## Stage 12 — Paper polish + arXiv submission

Related work section, limitations section, abstract tuning, figure
polish. arXiv submission.

## Hard rules for the entire build

- Every PR that changes a tactic must include a measurement showing
  before/after.
- Every tactic must be independently togglable.
- Every failure mode gets recorded in `docs/FAILURES.md` as it's
  discovered, so the paper's discussion section has material.
- Never add a tactic T8 without user agreement.
- Never train or fine-tune anything.
- Every claim in the paper must be reproducible from the committed
  workloads and runner.

## When to push back on scope

If Jay asks you to:
- **"Support more backends"** → push back. Ollama + OpenAI-compatible
  is the contract.
- **"Add an eighth tactic"** → push back gently. Ask what problem it
  solves that the existing seven don't.
- **"Skip the evaluation and ship the tool"** → push back firmly.
  The evaluation *is* the paper, and the paper *is* the deliverable.
- **"Use a larger local model by default"** → push back. We measure
  size-dependence; default should be the smallest that works on
  laptop hardware.
