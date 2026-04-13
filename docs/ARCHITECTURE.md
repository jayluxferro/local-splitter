# Architecture

`local-splitter` is a single process that exposes two interfaces to the
outside world:

1. **An MCP server over stdio** — for agents that speak MCP natively
   (Claude Code, Cursor-via-MCP, Codex CLI MCP, etc.).
2. **An HTTP proxy listening on `localhost:<port>`** — speaks the
   OpenAI-compatible `/v1/chat/completions` shape, so any agent that can
   be pointed at a custom API base (`OPENAI_API_BASE=http://localhost:<port>`)
   can use it transparently.

Both interfaces feed the same internal pipeline.

```
┌─────────────────────────┐
│  agent outbound request │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  stage 1: classify      │  T1 route  (local model 2-token classifier)
└────────────┬────────────┘
             │
   ┌─────────┴──────────┐
   │  TRIVIAL           │  COMPLEX
   ▼                    ▼
┌─────────┐   ┌──────────────────────────┐
│ local   │   │  stage 2: cache lookup   │  T3 sem-cache  (local embed)
│ respond │   └────────────┬─────────────┘
└─────────┘                │
   ▲               ┌───────┴────────┐
   │               │ HIT            │  MISS
   │               ▼                ▼
   │          ┌─────────┐   ┌──────────────────────────┐
   │          │ serve   │   │  stage 3: compress       │  T2 compress
   │          │ cached  │   └────────────┬─────────────┘
   │          └─────────┘                │
   │                                     ▼
   │                        ┌──────────────────────────┐
   │                        │  stage 4: intent extract │  T6 intent
   │                        └────────────┬─────────────┘
   │                                     │
   │                                     ▼
   │                        ┌──────────────────────────┐
   │                        │  stage 5: local draft    │  T4 draft
   │                        └────────────┬─────────────┘
   │                                     │
   │                                     ▼
   │                        ┌──────────────────────────┐
   │                        │  stage 6: diff rewrite   │  T5 diff
   │                        └────────────┬─────────────┘
   │                                     │
   │                                     ▼
   │                        ┌──────────────────────────┐
   │                        │  stage 7: batch queue    │  T7 batch
   │                        └────────────┬─────────────┘
   │                                     │
   │                                     ▼
   │                        ┌──────────────────────────┐
   │                        │  upstream cloud model    │
   │                        └────────────┬─────────────┘
   │                                     │
   │          ┌──────────────────────────┘
   │          │
   │          ▼
   │     ┌─────────┐
   └─────┤ cache   │  (write on MISS)
         │ store   │
         └─────────┘
```

Each stage is **independently togglable** via config. The evaluation
harness (`evals/`) runs fixed workloads against every subset of
`{T1,...,T7}` to measure marginal contribution.

## Core components

### 1. The transport layer (`src/local_splitter/transport/`)

- `mcp_server.py` — MCP stdio server. Exposes `split.complete`,
  `split.cache_lookup`, `split.classify`, `split.stats`.
- `http_proxy.py` — FastAPI server. Speaks `/v1/chat/completions` and
  `/v1/models`. Forwards to the same pipeline as the MCP surface.

### 2. The model registry (`src/local_splitter/models/`)

Two backends, both implementing a common `ChatClient` interface:

- `ollama.py` — talks to `http://localhost:11434/api/*`.
- `openai_compat.py` — talks to any `/v1/chat/completions` endpoint with
  a bearer token or no auth.

Config picks which backend provides the **local** model and which
provides the **cloud** model. The cloud target can itself be
OpenAI-compatible (so the user can put Anthropic behind LiteLLM if they
want).

### 3. The pipeline (`src/local_splitter/pipeline/`)

One file per tactic:

- `route.py` — T1 classify TRIVIAL / COMPLEX
- `sem_cache.py` — T3 local embedding + sqlite-vec store
- `compress.py` — T2 summarise long context
- `intent.py` — T6 structured intent extraction
- `draft.py` — T4 draft-then-review
- `diff.py` — T5 minimal diff for edits
- `batch.py` — T7 batching + prompt-cache helper

Each file exports a single `apply(request, config) -> request | response`
function. The orchestrator in `pipeline/__init__.py` calls them in
order, skipping any tactic disabled in config.

### 4. The evaluation harness (`evals/`)

- `workloads/` — fixed input datasets (see `docs/EVALUATION.md`).
- `metrics.py` — token counter, latency, quality scorer.
- `runner.py` — runs a config matrix across tactic subsets.
- `report.py` — outputs CSV + charts for the paper.

## Design principles

1. **Every stage is observable.** Each tactic emits a `stage_result`
   event with `{tokens_in, tokens_out, latency_ms, decision}`. The
   evaluation harness and the paper both rely on these events.
2. **Failure is local.** If the local model is unreachable, every tactic
   must fail open — pass the request through to the cloud unchanged —
   and log the degradation. Users should never be stuck.
3. **The cloud call is the last resort.** Every stage either answers
   the request, transforms the request, or passes it through. No stage
   makes a *parallel* cloud call.
4. **No global state beyond the cache.** All configuration is per-call
   or per-config-file. Makes testing trivial.
5. **Deterministic where possible.** Temperature-0 for the local
   classifier and intent extractor. Cache keys are content hashes, not
   timestamps. Makes the evaluation reproducible.

## Config model

```yaml
version: 1
transport:
  mcp: true
  http: true
  http_port: 7788

models:
  local:
    backend: ollama
    endpoint: http://127.0.0.1:11434
    chat_model: llama3.2:3b
    embed_model: nomic-embed-text
  cloud:
    backend: openai_compat
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    chat_model: gpt-4o-mini

pipeline:
  t1_route:      { enabled: true,  trivial_threshold: 0.8 }
  t2_compress:   { enabled: true,  max_tokens: 4000, ratio_target: 0.5 }
  t3_sem_cache:  { enabled: true,  similarity_threshold: 0.92, ttl: 86400 }
  t4_draft:      { enabled: false }
  t5_diff:       { enabled: true }
  t6_intent:     { enabled: false }
  t7_batch:      { enabled: false, window_ms: 250, max_size: 8 }

evaluation:
  log_file: .local_splitter/runs.jsonl
```

## State directory

`.local_splitter/` in the working directory:

```
.local_splitter/
├── runs.jsonl              # every pipeline event for eval replay
├── cache.sqlite            # T3 semantic cache (sqlite + sqlite-vec)
├── metrics.jsonl           # per-call tokens/latency/cost
└── config.yaml             # optional per-workspace override
```
