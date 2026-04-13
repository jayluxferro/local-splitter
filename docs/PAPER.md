# Paper outline

Working title:

> **Local-Splitter: A Measurement Study of Seven Tactics for Reducing
> Cloud LLM Token Usage on Coding-Agent Workloads**

Target venue: **arXiv, cs.CL** (primary) and **cs.DC** (secondary).
Long-term: a systems / ML-infrastructure workshop (MLSys, NeurIPS
Efficient ML workshop, etc.) once the paper has been on arXiv for a
few months and absorbed feedback.

## Abstract (target: 200 words)

We present a systematic measurement of seven tactics for reducing
cloud LLM token usage when a small local model can act as a triage
layer in front of a frontier cloud model. The tactics are:

1. local routing (trivial queries answered entirely locally),
2. local prompt compression,
3. semantic caching on local embeddings,
4. local drafting with cloud review,
5. minimal-diff edits,
6. structured intent extraction,
7. batching and vendor prompt caching.

We evaluate all seven individually, in pairs, and in a greedy-additive
subset across four coding-agent workload classes (edit-heavy,
explanation-heavy, general chat, RAG-heavy) using Ollama for the local
model and OpenAI-compatible endpoints for the cloud model. We report
tokens saved, dollar cost, latency, and blind quality deltas.

Our headline result: **(to be filled in post-experiment)**. Individual
tactics reach up to X% token reduction with Y% quality loss. The best
subset for edit-heavy workloads is {T1, T3, T5}, reaching Z% savings
at W% quality loss. No single combination dominates across all
workloads, supporting the conclusion that **tactic selection should be
workload-aware**.

Code, data, and trained evaluation harness are released under the
MIT license at github.com/jayluxferro/local-splitter.

## Outline

### 1. Introduction

- Cloud LLM costs have become a meaningful line item in dev-tool
  operational budgets.
- Local small models (< 3B) are now capable enough to act as a triage
  layer for a large fraction of requests.
- This paper asks: **given a local model and a cloud model, what is
  the best way to split work between them?**
- Contributions: (a) seven concrete tactics, (b) a rigorous per-tactic
  and per-combination evaluation on four workloads, (c) the
  observation that optimal subset is workload-dependent, (d) an
  open-source reference implementation.

### 2. Related Work

- **Speculative decoding** (Leviathan et al., 2023 — fast inference
  via a draft model) — we adopt the concept at the application layer
  instead of the token layer.
- **Prompt compression** (LLMLingua, Jiang et al., 2023) — we
  replicate and compare.
- **Semantic caching** (GPTCache) — we extend with quality
  measurement.
- **Routing and cascaded models** (FrugalGPT, RouteLLM) — we compare.
- **Prompt caching** (Anthropic cache_control, OpenAI automatic
  caching) — we integrate.
- What we add: rigorous *combined* evaluation, not individual claims;
  workload-differentiated results; open harness.

### 3. Tactics

Short subsection per tactic with formalisation and pseudocode.
Content matches `docs/TACTICS.md`.

### 4. System design

- Architecture diagram (reuse `docs/ARCHITECTURE.md`).
- Pipeline ordering and interaction constraints.
- Dual transport (MCP + HTTP proxy).
- Ollama + OpenAI-compatible backend abstraction.

### 5. Evaluation

- Four workload classes (WL1..WL4 from `EVALUATION.md`).
- Model matrix (local × cloud).
- Primary metrics: tokens, cost, latency, quality.
- Secondary metrics per tactic.
- 25 tactic-subset runs per (workload, model) combination.

### 6. Results

- **6.1 Singleton results**: bar chart per workload of each tactic
  alone, tokens-saved vs quality-delta.
- **6.2 Pair interactions**: cases of super-/sub-linearity.
- **6.3 Greedy-additive subset**: best subset per workload.
- **6.4 Pareto frontier** across all runs.
- **6.5 Minimum-viable local model**: what happens at 1B vs 3B vs 7B.
- **6.6 Vendor prompt-cache integration**: how much does T7 add on
  top of everything else?

### 7. Discussion

- **Why optimal subset varies by workload**: because each tactic
  targets a different slice of waste (input tokens, output tokens,
  context overhead, per-call fixed cost).
- **What this means for coding-agent vendors**: specific recommendations
  by agent type.
- **Failure modes**: workloads where the splitter *loses* — add too
  much latency, degrade quality too much, etc.

### 8. Limitations

- Workloads are captured from a narrow set of real sessions; broader
  captures would strengthen external validity.
- Quality evaluation uses judge-model (and human spot checks) but a
  full human evaluation at scale is future work.
- We only evaluate text-only workloads. Vision / audio agents are out
  of scope.

### 9. Conclusion

Restate contributions. The splitter saves ~X–Y% tokens at ≤ Z%
quality loss on realistic workloads, and optimal tactic selection is
workload-dependent. Open-source release enables future reproduction.

### Appendices

- **A**. Full metric tables per workload.
- **B**. Example prompts and responses per tactic.
- **C**. Reproducibility checklist.
- **D**. Hardware details.

## Schedule

Rough milestones (days from project start):

| Day | Milestone |
|---:|---|
| 0–3 | Scaffold code, implement T1, run single workload end-to-end |
| 3–7 | Implement T2, T3, T5 |
| 7–10 | Implement T4, T6, T7 |
| 10–14 | Evaluation harness + first full run on WL1 |
| 14–18 | Full eval matrix |
| 18–22 | Paper first draft |
| 22–25 | Internal review, figures, polish |
| 25–28 | arXiv submission |

This is optimistic; real schedule will likely be 1.5–2× longer
because of infra hiccups and workload curation. The paper submission
is the gate.
