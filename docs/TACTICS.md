# Tactics — deep dive

Each tactic is independently togglable. The research question is
which wins alone and which combinations compose.

## T1 — `route` (local classifier)

### What it does

A small local model reads the incoming request and emits a single
token: `TRIVIAL` or `COMPLEX`. Trivial requests are answered by the
local model directly and never reach the cloud. Complex requests flow
on to the next stage.

### Why it works

A lot of coding-agent traffic is structurally trivial: short completions,
single-word renames, "what does this file do", boilerplate generation.
Frontier cloud models are massively overpowered for these — a 3B local
model gives acceptable answers for about a third to half of them.

### How to implement

```python
# src/local_splitter/pipeline/route.py

TRIVIAL_PROMPT = """You are a request classifier. Read the user request
and output exactly one token: TRIVIAL or COMPLEX.

TRIVIAL means: short completion, rename, typo fix, boilerplate, simple
lookup, restatement, or a question a junior engineer could answer in
under 10 seconds.

COMPLEX means: multi-step reasoning, ambiguous requirements, refactoring
that touches multiple files, novel algorithm, anything that needs
frontier-level capability.

Request:
{request}

Output (TRIVIAL or COMPLEX only):"""

def apply(request, config):
    if not config.t1_route.enabled:
        return request

    classification = local_chat(
        model=config.models.local.chat_model,
        prompt=TRIVIAL_PROMPT.format(request=request.user_text),
        max_tokens=3,
        temperature=0.0,
    ).strip().upper()

    if classification == "TRIVIAL":
        answer = local_chat(
            model=config.models.local.chat_model,
            prompt=request.user_text,
            max_tokens=request.max_tokens,
        )
        return Response(answer=answer, served_by="local")

    return request
```

### Expected savings

On a mixed coding-agent workload: **30–50% of requests never hit the
cloud**. Token savings ≈ the fraction routed locally (minus the tiny
cost of the classifier call, which is amortised across everything).

### Risks

- **False positives**: a "TRIVIAL" call that actually needed the cloud
  gives a degraded answer. Mitigate with a confidence margin: if the
  classifier says TRIVIAL with logprob < threshold, escalate anyway.
- **Classifier drift**: cheap local models occasionally refuse to follow
  the format. Sanitise with a strict regex on the output, default to
  COMPLEX on any parse failure.

### Evaluation

- **Routing accuracy**: compare against human-labelled ground truth on
  a fixed benchmark set.
- **Quality loss**: blind A/B where human raters compare local-answered
  TRIVIALs to the cloud's answer.
- **Token savings**: input + output tokens that never left the local
  machine, as a percentage of the baseline.

---

## T2 — `compress` (prompt compression)

### What it does

Before the request goes out, a local model rewrites the context section
(system prompt, chat history, retrieved docs, file contents) to a
shorter form preserving semantic meaning. The cloud model then sees a
trimmed prompt and charges less input tokens.

### Why it works

Context windows in agent prompts are huge and often very repetitive.
A typical coding-agent system prompt is 3–8K tokens of boilerplate that
could be summarised to 400 tokens without losing the load-bearing
instructions.

### How to implement

Two modes:

1. **Static compression** — do it once at session start on the system
   prompt. Cache the compressed form.
2. **Dynamic compression** — run on every call for the chat history
   and retrieved docs, because they change each turn.

```python
COMPRESS_PROMPT = """Compress the following text to the shortest form
that preserves all information a language model needs to answer the
user. Remove filler, repetition, and instructions that do not change
the output. Preserve file paths, variable names, error messages, and
numeric values exactly. Do not paraphrase technical terms.

Target length: about {target_tokens} tokens.

Text:
{text}

Compressed:"""
```

### Expected savings

30–70% input-token reduction on typical coding-agent prompts. Output
tokens unaffected. Net savings ≈ input_saved / total_tokens.

### Risks

- **Information loss**: a compressed prompt may drop a critical detail.
- **Quality floor**: if the local model is weaker than the cloud, it
  may drop exactly the details the cloud would have needed.

Mitigate by benchmarking against a held-out quality set and rolling
back compression if quality drops by more than X%.

---

## T3 — `sem-cache` (semantic cache)

### What it does

Every outbound request gets embedded by a local embedding model, and
responses are stored in a vector index keyed by the embedding. On
subsequent similar queries, if the cosine similarity is above a
threshold, serve the cached response directly.

### Why it works

Even inside a single session, agents frequently re-ask variants of the
same question ("explain this file", "what does X do", "how does Y
work"). Across sessions, users often return to the same questions.
Semantic caching catches near-duplicates that exact-string caching
misses.

### How to implement

Stack: `sqlite` + `sqlite-vec` extension for the vector store,
`nomic-embed-text` via Ollama for embeddings. About 300 lines of Python.

```python
def apply(request, config):
    if not config.t3_sem_cache.enabled:
        return request

    embedding = embed(request.user_text)
    hit = cache.nearest(embedding, threshold=config.t3_sem_cache.similarity_threshold)
    if hit:
        return Response(answer=hit.response, served_by="cache", cache_hit=True)

    return request   # miss — proceed and cache on return
```

### Expected savings

5–40% hit rate depending on workload. Support-style workloads hit much
harder than code-generation workloads. We'll measure both.

### Risks

- **Stale answers**: cached responses for code questions can become
  stale as the codebase evolves. Mitigate with per-workspace cache
  namespaces and a short TTL.
- **Privacy leak**: cache entries persist across sessions. Add an
  explicit "don't cache" flag for sensitive prompts.

---

## T4 — `draft` (local drafter + cloud reviewer)

### What it does

Instead of asking the cloud to generate an answer from scratch, ask
the local model to draft one, then ask the cloud to *review or patch*
the draft. The cloud's output tokens drop sharply because it's editing
rather than authoring.

### Why it works

Output tokens are typically more expensive than input tokens, and most
of the answer is usually correct on first pass. If the local model
produces a 90%-correct draft, the cloud's job is a 10%-correction,
which is much cheaper.

### How to implement

```python
LOCAL_DRAFT_PROMPT = "Draft a response to the following request. Be concise.\n\n{request}"
CLOUD_REVIEW_PROMPT = """The user asked:
{request}

A draft response was produced:
{draft}

Review the draft. If it is correct and complete, respond with
'APPROVED' followed by the draft unchanged. If it is incorrect or
incomplete, respond with the corrected version. Do not explain your
changes."""
```

### Expected savings

On output-token-heavy workloads (code generation, explanation),
**40–70% output token reduction** when the local draft is good.

### Risks

- **Garbage in, garbage out**: if the local draft is hallucinatory,
  the cloud spends more tokens correcting it than it would have spent
  writing from scratch. Cap attempts; fall back to direct cloud call.
- **Review-prompt overhead**: the reviewer prompt adds a few hundred
  input tokens. On short requests this is a net loss.

---

## T5 — `diff` (minimal-diff edits)

### What it does

For edit requests ("change X to Y in this file"), the local model
computes the minimal diff needed. The cloud then receives only the
diff context + instruction, not the full file contents.

### Why it works

Typical coding-agent file edits send the entire file (thousands of
tokens) even when the edit is a 3-line change. The minimal-diff
approach shrinks that to the ~50 tokens around the change.

### How to implement

Detect edit requests by the presence of `<file>` blocks or
`apply_patch`-style tool calls. The local model uses `difflib` or a
lightweight parser to produce a unified diff context, and the cloud
model's prompt becomes "apply this change to this diff context".

```python
def apply(request, config):
    if not config.t5_diff.enabled: return request
    if not is_edit_request(request): return request

    hunks = local_identify_edit_hunks(request)
    minimal_context = extract_hunk_context(hunks, window=3)
    request = request.replace_context(minimal_context)
    return request
```

### Expected savings

**5–10× token reduction** on edit requests specifically. If ~20% of
requests are edits, net savings are ~60–80% on that slice × 20% = 12–16%
overall.

### Risks

- **Context underflow**: sometimes the cloud needs broader context to
  reason about whether the edit is correct. Make the window size
  configurable and measure the quality delta.
- **Parser brittleness**: JSON/XML files have different edit semantics
  than Python. Start with plain-text diffs; add language awareness
  later.

---

## T6 — `intent` (structured intent extraction)

### What it does

Before sending a free-text prompt to the cloud, a local model parses
it into a structured `{intent, target, constraints}` dict. The cloud
prompt becomes a filled-in template rather than chatty prose.

### Why it works

User prompts are verbose. Most of the verbosity is framing ("Could
you help me with...", "I'd like to understand why...", "So I'm
working on..."). The actual information content is usually 20% of the
prompt. Structured extraction strips the framing.

### How to implement

```python
INTENT_SCHEMA = {
    "intent": "explain | refactor | debug | generate | rename | search",
    "target": "file or symbol or region",
    "constraints": "list of strings",
}

def apply(request, config):
    if not config.t6_intent.enabled: return request
    extracted = local_chat_json(
        model=config.models.local.chat_model,
        prompt=f"Extract intent from:\n{request.user_text}",
        schema=INTENT_SCHEMA,
    )
    request.user_text = render_template(extracted)
    return request
```

### Expected savings

20–30% input token reduction, with a corresponding reduction in
output tokens because the cloud responds more directly.

### Risks

- **Intent misclassification**: the biggest risk. A missed intent
  produces an off-topic answer.
- **Template rigidity**: some user requests don't fit any of the
  predefined intents. Fall back to passing through raw.

---

## T7 — `batch` (batching + prompt cache)

### What it does

Two related sub-tactics:

1. **Local batching** — if the user fires multiple short queries in
   quick succession, buffer them briefly and send as one request with
   "answer all of these" framing.
2. **Prompt caching** — tag the stable prefix of a prompt (system
   prompt, codebase context) so the vendor's cache serves it at a
   discount on subsequent calls. Anthropic has
   `cache_control: ephemeral` and `OpenAI` has automatic prompt
   caching for >1024-token prefixes.

### Why it works

Batch submission amortises per-call overhead. Prompt caching shifts
cost from per-token to per-cache-miss. Both are vendor-native features
that agents rarely use explicitly.

### How to implement

Batching is tricky — needs a latency budget (don't wait > 300ms) and
only works for independent queries. Use a window of 250 ms, max 8
queries.

Prompt caching is easy — add vendor-specific `cache_control` headers
to the outgoing request when the request's prefix matches a previous
call's prefix above a threshold.

### Expected savings

- Prompt caching: up to **90%** savings on the stable prefix (the
  vendor discount).
- Batching: 10–20% savings from amortising fixed overhead.

### Risks

- **Latency increase** for batching. Users notice.
- **Cache mismatches** — different models have different cache
  formats. Keep the logic behind an abstraction and only enable for
  vendors we've tested.

---

## Interaction matrix

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| **T1** | — | serial: T1 then T2 | serial: T1 then T3 | conflicts: T4 duplicates local work | orthogonal | serial | orthogonal |
| **T2** | | — | T3 before T2 (cache raw) | serial | orthogonal | T6 before T2 | orthogonal |
| **T3** | | | — | serial | serial | serial | orthogonal |
| **T4** | | | | — | T5 before T4 | orthogonal | orthogonal |
| **T5** | | | | | — | orthogonal | orthogonal |
| **T6** | | | | | | — | orthogonal |
| **T7** | | | | | | | — |

Key interaction concerns:

- **T1 + T4 are partial substitutes**. Both use the local model to
  short-circuit the cloud. If T1 handles the easy cases, T4 has fewer
  remaining cases to help with.
- **T3 must run before T2**. Cache keys must be computed on the
  uncompressed request, else compression noise misses valid cache hits.
- **T6 must run before T2**. Compressing a structured template doesn't
  help; extract intent first, then compress any remaining free text.
- **T5 must run before T4**. If it's an edit request, diff-reduce it
  first, then draft-review the minimized form.

These ordering constraints define the pipeline order in
`ARCHITECTURE.md`.
