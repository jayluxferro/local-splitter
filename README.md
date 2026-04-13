# local-splitter

An MCP-compatible **outbound LLM request shim** that cuts cloud token usage
by running a local small model (via Ollama or any OpenAI-compatible endpoint)
as a *triage layer* in front of a frontier cloud model.

Individual tactics (routing, compression, caching, local drafting) are known
in isolation. What is *not* well-documented is how they combine on a realistic
coding-agent workload, and which combinations give the best marginal savings
vs. quality loss. This project answers that question empirically and writes
the results up as an arXiv paper.

## Quick start

Requires **Python 3.12+**, [`uv`](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com/) running locally.

```sh
# Install
uv sync

# Pull the required local models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml: set your cloud endpoint + API key env var

# Run tests
uv run pytest -q

# Start the proxy
uv run local-splitter serve-http --config config.yaml
```

## How it works

`local-splitter` sits between your coding agent and the cloud LLM. It
exposes two interfaces:

1. **OpenAI-compatible HTTP proxy** (`/v1/chat/completions`) — point any
   agent at `OPENAI_API_BASE=http://127.0.0.1:7788/v1`
2. **MCP stdio server** — register with any MCP-aware agent (Claude Code,
   Cursor, etc.)

Both interfaces feed the same internal pipeline:

```
Request → T1 route → T3 cache → T2 compress → T6 intent
        → T5 diff  → T7 batch → T4 draft    → Cloud
```

Each tactic is independently togglable via config. Disabled tactics are
zero-cost pass-throughs.

## The seven tactics

| # | Name | Type | What it does |
|---|------|------|-------------|
| **T1** | `route` | short-circuit | Local model classifies requests as TRIVIAL/COMPLEX. Trivials answered locally — never hit the cloud. |
| **T2** | `compress` | transform | Local model shortens long prompts (system prompts, history, RAG chunks) before they reach the cloud. |
| **T3** | `sem-cache` | short-circuit | Semantic similarity cache (SQLite + sqlite-vec). Near-duplicate queries return cached responses. |
| **T4** | `draft` | replace | Local model drafts the answer; cloud model reviews/patches it instead of generating from scratch. |
| **T5** | `diff` | transform | For code-edit requests, extracts minimal diff context so the cloud only sees the surgical change. |
| **T6** | `intent` | transform | Parses verbose free-text prompts into structured intent fields — cloud gets a tight template. |
| **T7** | `batch` | tag | Tags stable prompt prefixes with `cache_control` for vendor-side caching discounts. |

**Fail-open everywhere**: if the local model is unreachable or returns
garbage, every tactic defaults to passing the request through to the cloud
unchanged.

## Configuration

Two sets of presets — pick based on how you're using the splitter.

### Proxy mode (transparent interceptor)

For agents pointed at the splitter's HTTP endpoint. Requires a cloud
backend — the splitter calls it on your behalf.

| Preset | Tactics | Savings | Use case |
|--------|---------|---------|----------|
| [`proxy/conservative`](configs/proxy/conservative.yaml) | T1 | 29-69% | Safest — only routes trivials locally |
| [`proxy/recommended`](configs/proxy/recommended.yaml) | T1+T2 | 45-79% | **Best default** — route + compress |
| [`proxy/max-savings`](configs/proxy/max-savings.yaml) | T1+T2+T3 | 43-80% | Adds caching — best for repetitive workloads |
| [`proxy/rag-heavy`](configs/proxy/rag-heavy.yaml) | T1+T2+T3+T4+T5 | 51% on RAG | Long-context workloads with retrieved chunks |

```sh
cp configs/proxy/recommended.yaml config.yaml
# Edit: set your cloud endpoint + API key env var
```

### MCP mode (agent is the cloud model)

For Claude Code, Cursor, and other MCP-aware agents. **No cloud backend
needed** — the splitter answers trivials locally and returns compressed
prompts for the agent's own model.

| Preset | Tactics | Savings | Use case |
|--------|---------|---------|----------|
| [`mcp/conservative`](configs/mcp/conservative.yaml) | T1 | 29-69% | Safest — complex requests pass through untouched |
| [`mcp/recommended`](configs/mcp/recommended.yaml) | T1+T2 | 45-79% | **Best default** — route + compress |
| [`mcp/max-savings`](configs/mcp/max-savings.yaml) | T1+T2+T3 | 43-80% | Adds caching — compounds with query repetition |
| [`mcp/rag-heavy`](configs/mcp/rag-heavy.yaml) | T1+T2+T3+T5 | 44-51% | Long-context RAG workloads |

```sh
cp configs/mcp/recommended.yaml config.yaml
# No cloud config needed — just Ollama
```

### Eval results per workload

Evaluated with llama3.2:3b (local) and gemma3:4b (cloud), 10 samples
per workload, mean of 2 runs:

| Config | WL1 (edit) | WL2 (explain) | WL3 (chat) | WL4 (RAG) | Avg |
|--------|-----------|---------------|------------|-----------|-----|
| `conservative` (T1) | 29% | 69% | 59% | 38% | 49% |
| **`recommended`** (T1+T2) | **45%** | **79%** | 57% | 44% | **56%** |
| `max-savings` (T1+T2+T3) | 43% | **80%** | **60%** | 44% | **56%** |
| `rag-heavy` (proxy, +T4+T5) | 29% | 72% | 59% | **51%** | 53% |
| `rag-heavy` (mcp, +T5) | — | — | — | 44-51% | — |

Key observations:
- **Start with `recommended`** (T1+T2). 56% average savings, works
  across all workload types.
- `max-savings` adds T3 caching — same average but compounds over
  repeated queries (support bots, multi-user teams).
- `rag-heavy` proxy wins on WL4 because T4 (draft-review) helps with
  long outputs. MCP mode skips T4 (the agent is the reviewer).
- `conservative` still saves 49% — use if quality is the top priority.
- Quality cost: baseline wins ~3x more judge verdicts on explanation-heavy
  workloads. Acceptable on edit/RAG workloads. See the paper for details.

Config resolution order: explicit `--config` flag > `$LOCAL_SPLITTER_CONFIG`
env var > `.local_splitter/config.yaml` > `./config.yaml`.

## Usage

Start the proxy, then point your agent at it. The proxy speaks both the
**OpenAI** and **Anthropic** API formats, so any agent works transparently.

```sh
uv run local-splitter serve-http --config config.yaml
```

### Claude Code

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:7788 claude
```

### Cursor / Continue / any OpenAI-compatible agent

Set in the agent's settings:

```
API Base: http://127.0.0.1:7788/v1
API Key:  <your real cloud key>
Model:    gpt-4o-mini
```

Or via environment:

```sh
export OPENAI_API_BASE=http://127.0.0.1:7788/v1
export OPENAI_API_KEY=your-key
```

### Codex CLI

```sh
OPENAI_API_BASE=http://127.0.0.1:7788/v1 codex
```

### As an MCP server (local-only mode)

When the agent (Claude Code, Cursor) **is** the cloud model, the splitter
only needs a local model. No cloud backend required — the splitter routes
trivials locally and returns transformed prompts for complex requests.

```sh
cp configs/mcp/recommended.yaml config.yaml
uv run local-splitter serve-mcp --config config.yaml
```

Register with Claude Code by adding to `~/.claude.json` or your project's
MCP config:

```json
{
  "mcpServers": {
    "local-splitter": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/local-splitter",
               "local-splitter", "serve-mcp", "--config", "config.yaml"]
    }
  }
}
```

**How it works in local-only mode:**

1. Agent calls `split.transform` with the prompt
2. If T1 classifies as TRIVIAL → returns `{"action": "answer", "response": "..."}` (answered locally, no cloud tokens spent)
3. If COMPLEX → returns `{"action": "passthrough", "transformed_messages": [...]}` (compressed/optimized prompt for the agent's own model)

Exposed MCP tools:
- `split.complete` — full pipeline; returns local answer or transformed messages
- `split.transform` — transforms only, never calls a backend
- `split.classify` — run T1 classifier only (TRIVIAL / COMPLEX)
- `split.cache_lookup` — check T3 cache without writing
- `split.stats` — aggregate metrics since startup
- `split.config` — read-only config view

### CLI transform (for hooks)

One-shot transform for integration with agent hooks. Reads a prompt,
runs tactics, prints JSON to stdout.

```sh
# Plain text prompt
echo "what is 2+2" | local-splitter transform --config config.yaml
# → {"action": "answer", "response": "2 + 2 = 4", "served_by": "local", ...}

# Complex prompt passes through
echo "refactor the auth middleware..." | local-splitter transform -c config.yaml
# → {"action": "passthrough", "messages": [...], ...}

# Or use --prompt flag
local-splitter transform -p "explain merge sort" -c config.yaml
```

Output is always one JSON object:
- `{"action": "answer", "response": "..."}` — answered locally, use directly
- `{"action": "passthrough", "messages": [...]}` — send to your model

#### Claude Code hook setup

Add to your Claude Code `settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task|Bash|Edit|Write",
        "hook": "echo \"$PROMPT\" | local-splitter transform -c /path/to/config.yaml"
      }
    ]
  }
}
```

### API surfaces

| Endpoint | Format | Streaming |
|----------|--------|-----------|
| `POST /v1/chat/completions` | OpenAI | SSE (`stream: true`) |
| `POST /v1/messages` | Anthropic | SSE (`stream: true`) |
| `GET /v1/models` | OpenAI | — |
| `GET /v1/splitter/stats` | JSON | — |
| `GET /healthz` | JSON | — |

Both surfaces add a `splitter` key to responses with observability data:

```json
{
  "splitter": {
    "served_by": "local",
    "latency_ms": 42.3,
    "pipeline_trace": [
      { "stage": "t1_classify", "decision": "TRIVIAL", "ms": 12.1 },
      { "stage": "t1_local_answer", "decision": "APPLIED", "ms": 30.2 }
    ],
    "tokens_local": { "input": 15, "output": 8 }
  }
}
```

### Force routing

Override the pipeline per-request:

```python
# Via HTTP extra_body (OpenAI surface)
{"extra_body": {"splitter": {"force_local": True}}}   # bypass pipeline, use local
{"extra_body": {"splitter": {"force_cloud": True}}}   # bypass pipeline, use cloud

# Via MCP model_hint
{"model_hint": "local"}
{"model_hint": "cloud"}
```

## Evaluation

The evaluation harness measures per-tactic and per-combination savings across
four workload classes:

| Workload | Description | Trivial% |
|----------|------------|----------|
| WL1 edit-heavy | Refactoring sessions, many file edits | ~25% |
| WL2 explain | "What does X do" onboarding questions | ~45% |
| WL3 chat | General-purpose mixed chat | ~50% |
| WL4 RAG | Long system prompts with retrieved chunks | ~20% |

### Running evals

```sh
# Specific subsets on specific workloads
uv run local-splitter eval \
  -w evals/workloads/wl3_chat.jsonl \
  --config config.yaml \
  --subsets baseline,T1_only,T1_T2_T3

# All subsets on all workloads (full matrix)
uv run local-splitter eval \
  -w evals/workloads/wl1_edit.jsonl \
  -w evals/workloads/wl2_explain.jsonl \
  -w evals/workloads/wl3_chat.jsonl \
  -w evals/workloads/wl4_rag.jsonl \
  --config config.yaml

# Full eval script (produces paper-ready summary)
uv run python evals/run_eval.py

# Run specific subsets only
uv run python evals/run_eval.py T5_only T6_only T7_only

# Include judge-model quality evaluation (pairwise A/B comparison)
uv run python evals/run_eval.py --quality
```

Results land in `.local_splitter/eval/`:
- `results.csv` — one row per (workload × tactic subset)
- `runs.jsonl` — per-sample detail log
- `summary.json` — aggregates for the paper (includes quality verdicts with `--quality`)

### Available tactic subsets

`baseline`, `T1_only`, `T2_only`, `T3_only`, `T4_only`, `T5_only`,
`T6_only`, `T7_only`, `T1_T2`, `T1_T3`, `T1_T2_T3`, `T1_T3_T4`,
`T1_T2_T3_T6`, `all`

## Project structure

```
src/local_splitter/
├── cli.py                  # Typer CLI (serve-http, serve-mcp, eval)
├── config.py               # YAML config loader
├── models/                 # Backend implementations
│   ├── base.py             #   ChatClient protocol + data types
│   ├── ollama.py           #   Ollama native API client
│   ├── openai_compat.py    #   OpenAI-compatible client
│   └── factory.py          #   Build client from config
├── pipeline/               # The seven tactics + orchestrator
│   ├── __init__.py         #   Pipeline orchestrator
│   ├── types.py            #   PipelineRequest/Response, StageEvent
│   ├── route.py            #   T1 — local classifier
│   ├── compress.py         #   T2 — prompt compression
│   ├── sem_cache.py        #   T3 — semantic cache (sqlite-vec)
│   ├── draft.py            #   T4 — draft + review
│   ├── diff.py             #   T5 — minimal diff extraction
│   ├── intent.py           #   T6 — intent extraction
│   └── batch.py            #   T7 — prompt-cache tagging
├── transport/              # External interfaces
│   ├── http_proxy.py       #   FastAPI OpenAI-compat proxy
│   └── mcp_server.py       #   FastMCP stdio server
└── evals/                  # Evaluation harness
    ├── types.py            #   WorkloadSample, SampleResult, RunResult
    ├── runner.py           #   Matrix runner + tactic subsets
    ├── metrics.py          #   Token savings, cost, routing accuracy
    ├── quality.py          #   Judge-model pairwise quality evaluation
    └── report.py           #   CSV + markdown export

evals/workloads/            # Evaluation datasets (JSONL)
paper/                      # LaTeX paper
tests/                      # 171 tests
```

## Development

```sh
uv sync                  # install deps
uv run pytest -q         # run tests (171 tests, <1s)
uv run ruff check src/   # lint
```

### Adding a new tactic

1. Create `src/local_splitter/pipeline/<name>.py` with an `apply()` function
2. Wire it into `Pipeline.complete()` in `pipeline/__init__.py`
3. Add the config flag to `TacticsConfig` in `config.py`
4. Add tests in `tests/test_pipeline_<name>.py`
5. Add eval subsets in `evals/runner.py`

Every tactic must:
- Emit a `StageEvent` for observability
- Fail open on errors (pass request through unchanged)
- Use `temperature=0` for deterministic classifier/extractor calls

## Paper

The paper skeleton is in `paper/paper.tex`. It measures all seven tactics
individually, in pairs, and in greedy-additive subsets across the four
workload classes above. See `docs/EVALUATION.md` for the research questions
and success criteria.

## Results

Evaluated with **llama3.2:3b** (local) and **gemma3** (cloud via Ollama),
10 samples per workload:

| Subset | WL1 edit | WL2 explain | WL3 chat | WL4 RAG |
|--------|---------|-------------|----------|---------|
| T1 route | 28.2% | 61.9% | 55.4% | 33.7% |
| T2 compress | 20.6% | 20.3% | — | 15.5% |
| T1+T2 | **46.9%** | **79.4%** | 59.5% | 45.3% |
| T1+T2+T3 | 42.4% | 79.1% | **60.6%** | 45.6% |
| all | 31.0% | 72.7% | 60.3% | **51.8%** |

**Key findings:**
- **T1+T2 is the best 2-tactic combination** (47–79% cloud token savings)
- T1 alone routes 50–70% of requests locally (28–62% savings)
- T1+T2+T3 adds semantic caching on top — marginal on single-pass, stronger with repetition
- On WL4 (RAG-heavy), **"all" beats T1+T2+T3** (51.8% vs 45.6%) because
  T4 draft-review helps when outputs are long
- **Optimal subset is workload-dependent** — the paper's key finding

## License

MIT

## Status

- [x] Python scaffold (uv, package skeleton, 175 tests)
- [x] Model backends (Ollama + OpenAI-compatible)
- [x] Transport layer (MCP stdio + HTTP proxy: OpenAI + Anthropic)
- [x] Streaming support (SSE, both API surfaces)
- [x] All seven tactics implemented and tested
- [x] Evaluation harness (runner, metrics, quality judge, CSV/markdown)
- [x] CLI eval command + seed workloads (4 classes, 40 samples)
- [x] Evaluation with real numbers (Ollama llama3.2:3b + gemma3)
- [x] Paper with results (tables, figures, quality evaluation)
- [ ] arXiv submission
