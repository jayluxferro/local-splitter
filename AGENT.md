# AGENT.md — briefing for the next agent

You are picking up work on **local-splitter**, an MCP shim that reduces
cloud LLM token usage by routing requests through a local small model
first. This document is the handoff.

Read this file, then the five files under `docs/`, then the five files
under `.agent/memory/`. Then propose your first action and wait for the
user to confirm.

## What this project is

`local-splitter` sits between a coding agent (Claude Code, Cursor, Codex,
Copilot CLI) and its cloud model endpoint. It intercepts outbound LLM
requests and decides — on a per-request basis — how to split the work
between a cheap local model and the expensive cloud model.

There are **seven tactics** (T1–T7), each of which attacks a different
slice of token waste:

| Tactic | Purpose |
|---|---|
| T1 `route` | Send trivial requests to a local model entirely |
| T2 `compress` | Shrink long prompts before sending |
| T3 `sem-cache` | Serve near-duplicate queries from cache |
| T4 `draft` | Local drafts, cloud reviews |
| T5 `diff` | Minimal-diff edits instead of full-file rewrites |
| T6 `intent` | Structured intent extraction instead of free-form prose |
| T7 `batch` | Accumulate queries and exploit vendor prompt caching |

The research question is: **which tactic wins on which workload, and
does combining them give super-linear returns or diminishing returns?**

This is a **research-first** project. The code exists to support an
evaluation, and the evaluation exists to support a paper. The paper is
the deliverable.

## Current status

- **Design**: frozen for the seven tactics. See `docs/TACTICS.md`.
- **Code**: not started.
- **Evaluation harness**: not started. See `docs/EVALUATION.md` for the
  design.
- **Paper skeleton**: `paper/paper.tex` with section stubs.

The MVP you should build first is `T1 route + T3 sem-cache + T2 compress`
— the three tactics with the highest measured savings ceiling. Add
`T4 draft` and `T5 diff` once the measurement harness works. `T6 intent`
and `T7 batch` are last because they interact with other tactics in
subtle ways and are harder to evaluate in isolation.

## Compatibility contract

The shim must work with:

1. **Ollama** via its native `/api/chat` and `/api/generate` REST endpoints.
2. **Any OpenAI-compatible API** — the `/v1/chat/completions` shape. This
   includes OpenAI itself, Anthropic via `LiteLLM`, together.ai, vLLM,
   llama.cpp server, LM Studio, and most other open-weight hosts.

The project must **not** require a specific vendor SDK. Build against the
raw HTTP contract with `httpx` and let users plug in whatever endpoint.

## Your first 30 minutes

1. Read `README.md`, `AGENT.md` (this file), `docs/ARCHITECTURE.md`.
2. Read `docs/TACTICS.md` — the seven tactics in detail with pseudocode.
3. Read `docs/API.md` — MCP tool surface + HTTP proxy surface.
4. Read `docs/EVALUATION.md` — how we'll measure which tactic wins.
5. Read `docs/PAPER.md` — the arXiv paper outline.
6. Read `.agent/memory/origin.md` — why this project exists (spun out of
   the conversation that followed the LLM CLI telemetry report).
7. Read `.agent/memory/decisions.md` — what was considered and rejected.
8. Read `.agent/memory/next-steps.md` — the concrete task list.
9. Read `.agent/memory/user-profile.md` — how to communicate with Jay.
10. Read `.agent/memory/gotchas.md` — non-obvious lessons from adjacent
    work.

## Hard rules

- **The seven tactics are the contract.** Do not add an eighth without
  explicit user agreement. Do not silently merge two of them.
- **Every tactic must be independently togglable.** The point of the
  research is to measure them individually and in combination.
- **All claims of token savings must be measured, not estimated.** Every
  PR that changes a tactic must include a benchmark result from
  `evals/` showing the before/after token count on a fixed workload.
- **Must work with local + cloud endpoints.** The test suite runs against
  Ollama locally AND against at least one OpenAI-compatible cloud model.
  If you break one, you break the project.
- **No vendor lock-in.** Don't import `anthropic` or `openai` SDKs. Use
  `httpx` against the raw HTTP shape. Users pick their models via env
  vars or config.
- **The paper is a deliverable.** Treat `paper/paper.tex` as a
  first-class artefact, not documentation. Every design decision lands in
  both code and paper.

## What this project is NOT

- **Not a proxy for privacy.** See the sibling project `llm-redactor` for
  that. `local-splitter` can reduce exfiltration *as a side effect* of
  keeping trivials local, but its goal is tokens, not privacy.
- **Not a load balancer.** We don't do cross-provider failover or multi-
  model ensembling. One local model + one cloud model is the model.
- **Not a training project.** We never train or fine-tune anything. We
  only route, compress, cache, draft, diff, extract, and batch.
- **Not a chat UI.** Users interact through their existing agents; we are
  invisible infrastructure.

## The paper

We're aiming for a cs.CL or cs.DC arXiv submission with the form:

> "Seven ways to cut cloud LLM token usage with a local small model:
> a measurement study on coding-agent workloads."

See `docs/PAPER.md` for the outline and `paper/paper.tex` for the
skeleton. The novelty is **empirical**, not theoretical — every
individual tactic exists in the literature or in blog posts; the
contribution is a rigorous measurement of which wins on which workload
and how they compose.

## How to talk to the user

Direct, terse, proof > promises, no preamble. See
`.agent/memory/user-profile.md` for the full voice guide.
