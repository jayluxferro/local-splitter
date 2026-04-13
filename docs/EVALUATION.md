# Evaluation

This is the part that makes it a research project rather than a
utility. The paper lives or dies on the strength of this evaluation.

## Research questions

**RQ1**: How much does each tactic save in isolation, on a realistic
coding-agent workload, measured in input tokens, output tokens, and
dollar cost?

**RQ2**: How much quality is lost per tactic, measured by a blind
pairwise preference test against the baseline?

**RQ3**: Do tactics compose super-linearly, linearly, or
sub-linearly? Specifically, does `{T1, T3}` save more than
`T1` and `T3` independently?

**RQ4**: What is the optimal tactic subset for a given workload class?
Is there a universal best combination, or is it workload-dependent?

**RQ5**: What is the minimum-viable local model size? Does a 1B model
work, or do you need a 3B / 7B / 14B?

## Workload classes

The evaluation runs on four distinct workload classes. Each class is
captured from a real coding-agent session (or a synthetic surrogate
that matches real session statistics).

### WL1 — coding agent: edit-heavy

Captured from `claude-code` / `cursor-cli` during real refactoring
sessions. Characterised by many file edits, moderate context window,
heavy tool use.

- Typical prompt: 8–20K tokens
- Typical completion: 200–1500 tokens
- Edit-request fraction: ~60%
- Trivial fraction: ~25%

### WL2 — coding agent: explanation-heavy

"Explain this file", "what does X do", "how does Y work". Captured
from an onboarding scenario where a new engineer asks an agent to
walk them through a codebase.

- Typical prompt: 4–12K tokens
- Typical completion: 500–3000 tokens
- Edit-request fraction: ~5%
- Trivial fraction: ~45%

### WL3 — mixed chat

General-purpose chat, not coding-specific. Includes both short and
long turns.

- Typical prompt: 500–4000 tokens
- Typical completion: 100–1500 tokens
- Edit-request fraction: 0%
- Trivial fraction: ~50%

### WL4 — RAG-heavy

Retrieval-augmented. Long system prompt with multiple retrieved
chunks, user asks a focused question.

- Typical prompt: 10–40K tokens
- Typical completion: 100–800 tokens
- Edit-request fraction: 0%
- Trivial fraction: ~20%

**Each workload has at least 200 samples**. They live as JSON files
under `evals/workloads/<wl_name>/` and are regenerated from captures
using a scrubbing script that removes any real identifiers.

## Metrics

### Primary

- **Tokens saved** — `(baseline_total - splitter_total) / baseline_total`
- **Dollar cost saved** — apply the cloud vendor's published rate card
  to the token deltas
- **Latency delta** — median, p95, p99 over the workload (splitter may
  be slower per-call due to local classification, but cheaper per-call
  overall)
- **Quality delta** — blind pairwise preference from a human rater (or
  a large judge model as a proxy) comparing splitter output to
  baseline on 100 held-out samples per workload

### Secondary

- **Routing accuracy** (T1) — fraction of TRIVIAL/COMPLEX calls that
  match ground truth
- **Compression ratio** (T2) — `compressed_tokens / original_tokens`
- **Cache hit rate** (T3)
- **Draft acceptance rate** (T4) — fraction of drafts the cloud
  approves unchanged
- **Diff shrink factor** (T5) — `diff_prompt_tokens / full_file_tokens`
- **Intent extraction F1** (T6) — agreement with hand-labelled intents
- **Batch fill rate** (T7)

## Models

The evaluation runs with a matrix of local × cloud models:

**Local (via Ollama)**
- `llama3.2:1b`
- `llama3.2:3b`
- `qwen2.5:3b`
- `phi3.5:3.8b`
- `gemma2:2b`

**Cloud (via OpenAI-compatible endpoint)**
- `gpt-4o-mini`
- `gpt-4o`
- `claude-3-5-haiku` (via LiteLLM if needed)
- `claude-3-5-sonnet` (via LiteLLM if needed)

Every workload × local × cloud × tactic-subset combination is a single
evaluation run. We report the full matrix and the ANOVA breakdown.

## The tactic subset matrix

There are 7 tactics, so 2^7 = 128 possible subsets. We don't run all
of them; instead:

1. **Singletons** (7 runs): each tactic on, others off. Measures T_i
   in isolation.
2. **Pairs** of tactics that are known to interact (see interaction
   matrix in `TACTICS.md`): ~10 combinations.
3. **Greedy-additive** (up to 7 runs): start with the best singleton,
   add the tactic that most improves the primary metric, repeat.
4. **Full set** (1 run): all tactics on, to measure the ceiling.

Total: ~25 runs per workload class × 4 classes × (say) 3 model
combinations = ~300 runs. Each run is a few hundred samples.
Budget-friendly if we keep the workloads small.

## The report

`evals/runner.py` emits a CSV with one row per
`(workload, local_model, cloud_model, tactic_subset)` combination,
columns for all primary and secondary metrics.

`evals/report.py` generates:

- One bar chart per workload: tactic subset on x-axis, tokens saved
  on y-axis.
- One scatter per workload: quality delta (x) vs tokens saved (y).
- One table per workload: full matrix of metrics.
- A Pareto frontier across all runs.

All figures land in `paper/figures/` and are referenced from
`paper/paper.tex`.

## Reproducibility

- Every run is seeded (local model temperature=0 where possible).
- Every run records the local and cloud model versions + timestamps.
- Workload captures include a hash so we can verify we're running the
  same inputs on re-runs.
- Results JSON + CSV + figures are committed so reviewers can diff.

## Ethics & data handling

Workload captures must be scrubbed before commit. Use a PII classifier
(the sibling `llm-redactor` project's pipeline once it exists, or
`presidio` for now) and a hand review.

No real user prompts. No real source code unless it's already public.

## What success looks like

The paper is publishable if we can show:

1. **At least one tactic saves ≥ 30% tokens on ≥ one workload with
   ≤ 5% quality loss**. (This is easy to hit — T1 alone probably
   clears it.)
2. **At least one tactic combination is Pareto-better than any single
   tactic** on at least one workload.
3. **The optimal subset differs between at least two workloads**.
   (This is the "it depends" result that's publishable as a
   contribution.)
4. **The minimum-viable local model size is reportable** — we can say
   "T1 works with 1B but not smaller" or similar.
