---
name: gotchas and lessons learned
description: Non-obvious things that would otherwise burn time
type: feedback
---

# Gotchas

## Ollama

- `/api/chat` and `/api/generate` are **not** OpenAI-compatible out of
  the box. The response shape is different. Ollama has an
  OpenAI-compatibility layer at `/v1/chat/completions` — use that when
  you want uniform handling, but be aware that it doesn't expose
  every Ollama feature (e.g. `options.num_ctx`).
- Ollama defaults to `num_ctx=2048`, which silently truncates long
  prompts. Always set `options.num_ctx` explicitly for the local
  model calls we make in T1/T2/T6.
- Ollama models preload on first request and take several seconds
  (the "cold start"). For latency benchmarking, do a warmup call
  first and exclude it from metrics.
- `ollama pull` a model once before the evaluation starts so the
  benchmark doesn't include the download.

## OpenAI-compatible APIs

- Not all "OpenAI-compatible" endpoints are equal. Anthropic-via-LiteLLM
  doesn't support `logprobs`. together.ai supports it but with
  different field names. Be defensive.
- `finish_reason: length` is silent context truncation. Always check
  it and log a warning.
- The `usage.prompt_tokens` count from the cloud is authoritative for
  billing. Our local tiktoken count may differ by 1--5 tokens. Report
  cloud-side numbers in the paper.

## Semantic caching

- Exact-string hashing catches nothing in real agent workloads because
  session IDs and timestamps vary. You must embed to get meaningful
  hits.
- Cosine similarity thresholds are touchy. 0.92 is the initial
  threshold but sweep it in the eval (0.85, 0.90, 0.92, 0.95) and
  report the quality / hit-rate trade-off.
- `sqlite-vec` requires loading the extension at connection time with
  `conn.load_extension`. Python's default SQLite has extension loading
  disabled on many platforms; use `pysqlite3-binary` to get a build
  that supports it, or fall back to `faiss` if that's a hassle.

## Quality evaluation

- Judge models have biases. A judge that's the same family as the
  candidate will prefer it. Use a different family (e.g. use
  Claude-3.5 to judge GPT outputs, or vice versa).
- Judge models hallucinate "scores" — always force them into a
  structured A/B/tie/confidence schema, not a free-form rating.
- Tie rates tell you something. If the judge ties > 40% of the time,
  the quality delta is genuinely small and you should report that.

## Token counting

- `tiktoken` gives OpenAI tokeniser counts. For Llama/Qwen/Phi tokens,
  use the local model's own tokeniser via `transformers` or Ollama's
  `/api/tokenize` (if available).
- Reporting "tokens saved" is meaningless unless you specify whose
  tokeniser. Always say "cloud model's tokeniser" and stick to it.

## Latency measurement

- Ollama first-token latency on a warm model is 50-200ms on an M-series
  Mac. Include it in measurements but annotate the warmup.
- Network latency to the cloud dominates for small requests. Don't
  report latency improvements that are actually just "we avoided the
  round trip" — that's valid but call it out.
- p99 is the right tail to report. Mean is misleading.

## MCP SDK

- The Python MCP SDK has changed shape more than once in 2024-2026.
  Pin the version and note it in `pyproject.toml`.
- Stdio transport doesn't support large responses well — chunk
  responses if they exceed 1 MB or they get truncated.
- MCP tool names are case-sensitive and should be `dot.separated`.
  The convention across resilient-write / local-splitter / llm-redactor
  is `<project>.<verb>`. Follow it.

## Config management

- Don't load `config.yaml` lazily per request. Parse once at startup
  and keep it in memory. Lazy parsing triples per-request latency.
- Environment variables override config file. Config file overrides
  defaults. Document the precedence clearly.

## LaTeX paper

- arXiv wants a single .tex file plus `.bbl` (not `.bib`) for the
  bibliography unless you configure it otherwise. Run `latex` + `bibtex`
  locally and commit the `.bbl` so arXiv doesn't have to resolve refs
  itself.
- arXiv rejects `\usepackage{minted}` unless you pre-render code as
  `\verbatim` or images. Use `listings` instead for code samples.
- Figures under 5 MB ideally, < 10 MB hard limit per-figure on arXiv.
  Use vector PDFs, not PNG screenshots.

## POSIX shell subshells

- `while read f; do total=$((total+x)); done < <(find ...)` only works
  in bash. In POSIX sh, write stats to a tempfile.

## macOS `find`

- Use `-size +100k` not `-size +102400c`. Both work but `+100k` is
  portable.
- `find -n` is not a flag; don't let your CLI pass unparsed args to
  find.

## Reproducibility

- Every eval run records local model version, cloud model version,
  git SHA, workload hash, and wall-clock timestamp. Without these,
  you can't rerun.
- Seed the local model's temperature to 0 for classifiers. Generation
  tactics (T2, T4) can use non-zero temperature but record the seed.
