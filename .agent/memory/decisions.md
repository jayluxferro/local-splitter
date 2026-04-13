---
name: design decisions
description: What was considered, what was chosen, and what was rejected
type: project
---

# Decisions

## Overall shape

### Decision: seven independent tactics, not one monolithic strategy

**Why**: the research question is *which tactic wins on which
workload*. If we collapse them into a single strategy, we lose the
ability to measure them individually. Each tactic must be
independently togglable and independently measurable.

**Rejected alternatives**:
- *One "smart router" that does everything*. Rejected because we
  can't attribute wins or losses to specific mechanisms.
- *Fewer tactics (e.g. five)*. Rejected because each of the seven
  targets a distinct source of waste and the interaction matrix
  shows they don't strictly dominate each other.

### Decision: dual transport (MCP + OpenAI-compatible HTTP)

**Why**: some agents speak MCP natively (Claude Code, Cursor). Others
can be pointed at a custom `OPENAI_API_BASE` but don't know what MCP
is (Continue, Aider, Cline via OpenRouter). Supporting both means the
splitter is drop-in for every common coding-agent setup.

**Rejected alternatives**:
- *MCP only*. Rejected because OpenAI-compatibility gives us a much
  wider agent surface.
- *HTTP only*. Rejected because MCP-native agents wouldn't benefit
  from the agent-specific metadata that MCP carries (session id,
  tool name).
- *Anthropic-native API shape*. Rejected because the industry has
  converged on OpenAI-compatible as the lingua franca.

## Model backends

### Decision: Ollama and OpenAI-compatible, nothing else

**Why**: Ollama is the de-facto standard for local models on laptops.
OpenAI-compatible is the de-facto standard for cloud LLM APIs.
Together they cover ~95% of real deployments. Adding more backends
is scope creep that doesn't serve the research question.

**Rejected alternatives**:
- *Direct llama.cpp integration*. Rejected — Ollama wraps llama.cpp
  and exposes a nicer API. Users who want llama.cpp can run it as an
  OpenAI-compatible server (it has one built in).
- *Anthropic SDK directly*. Rejected — users can point at Anthropic
  via LiteLLM if they want, via the OpenAI-compatible shape.
- *Hugging Face Inference API*. Rejected for the same reason as
  llama.cpp; it's an OpenAI-compatible endpoint already.

### Decision: no model training or fine-tuning

**Why**: this is a research project about *deployment strategies*,
not model improvement. Training adds scope, latency, and a GPU
requirement the project deliberately avoids.

## Research / paper shape

### Decision: arXiv first, venue later

**Why**: arXiv lets us publish on our timeline and accumulate
citations while we refine. A conference submission can happen
afterwards without conflict (arXiv doesn't preclude most venues).

**Rejected alternatives**:
- *Workshop submission first*. Rejected because workshops are
  slow and a paper on arXiv already lives as a preprint.
- *Direct conference submission (NeurIPS, ICLR)*. Rejected because
  the review cycles are too long and the paper might be scooped.

### Decision: primary metric is tokens, not dollars

**Why**: dollar rates change; token counts are durable measurements.
We report both, but the headline results are in tokens.

### Decision: quality measured by judge-model proxy + spot-checks

**Why**: full human evaluation at scale is too expensive for a
solo research project. A judge-model proxy (e.g. Claude 3.5 Sonnet
as judge) is the standard in the field, and spot-checks catch
obvious regressions.

**Rejected alternatives**:
- *Full human evaluation*. Rejected due to cost. Noted as future
  work.
- *No quality evaluation*. Rejected as unpublishable — saying "we
  saved X% tokens" without a quality delta is meaningless.

### Decision: four workload classes (edit-heavy, explanation-heavy,
general chat, RAG-heavy)

**Why**: these four classes span the coding-agent space while
remaining tractable. Each is captured from real sessions (scrubbed)
and each has distinct statistical properties.

**Rejected alternatives**:
- *Single synthetic workload*. Rejected because a single workload
  is easy to overfit to.
- *All-real capture corpus*. Rejected because curation and scrubbing
  are expensive. Four focused classes is the right budget.

## Implementation

### Decision: Python with `httpx`, `mcp`, `ollama`, `sqlite-vec`

**Why**: Python has the best ML ecosystem, `httpx` is the right
async HTTP client, `mcp` is the official MCP SDK, `sqlite-vec` is a
single-file vector store that doesn't require a separate database
server.

**Rejected alternatives**:
- *TypeScript / Node.js*. Rejected because `sqlite-vec` and model
  inference tooling are weaker there.
- *Rust*. Rejected because it would slow prototyping and the
  performance-critical path is already inside the model servers, not
  our glue.

### Decision: SQLite + `sqlite-vec` for the T3 cache, not ChromaDB / Qdrant / FAISS

**Why**: SQLite is a single file with no daemon. Cache lives in
`.local_splitter/cache.sqlite` next to the journal. No setup, no
migration path, no network dependency.

### Decision: one config file (`config.yaml`), no CLI flags for pipeline config

**Why**: the pipeline is a fixed 7-stage graph with per-stage
enables. A YAML file is the natural shape. CLI flags proliferate
fast and become untestable.

## Evaluation

### Decision: 25 runs per (workload × model combo), not 128

**Why**: 128 = 2^7 is exhaustive but unnecessary. Singletons tell us
the individual strength, pairs tell us interactions, greedy-additive
tells us the good combination, and full-set tells us the ceiling.
That's ~25 runs and captures the interesting points.

### Decision: workloads live in the repo, not externally

**Why**: reviewers need to reproduce. External datasets rot.
Workloads are small (~200 samples each) and redacted.

### Decision: report every failed run, not just successful ones

**Why**: "T4 hurt quality on WL3 by 8%" is a research finding.
Hiding failures produces a worse paper.
