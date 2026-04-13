---
name: origin story
description: Where local-splitter came from and what the motivating question is
type: project
---

# Origin

## The conversation that produced this project

In April 2026 I was helping the user (Jay Lux Ferro, `@sperixlabs`)
finish a technical report on LLM CLI telemetry. Near the end of that
session, Jay asked two big follow-up research questions:

1. *"Using local models to reduce token usage — what's possible and
   what can be done?"*
2. *"Hiding LLM data reading format or tokenization so it doesn't
   exist in plain text when sending data to LLMs — what's possible?"*

I sketched seven tactics for question~(1) and eight options for
question~(2). The user said: *"let's work on all of them"* and
immediately asked me to spin the two research areas out into their
own projects.

This repo is the project for question (1). The sibling
`/Users/jay/dev/ml/mcp/llm-redactor` is the project for question (2).

## What this project is for

A coding agent (Claude Code, Cursor, Codex CLI, Copilot CLI) makes
many LLM requests per task. Frontier cloud models are great but
expensive. A 1--7~B local model on a modern MacBook is fast and free
but can't match a frontier model on hard problems. The natural
question is: \emph{can we use both, and if so, how should we split
the work between them?}

This repo answers that question by:

1. Implementing seven distinct tactics for splitting work between a
   local and cloud model.
2. Running each tactic (and combinations) against four realistic
   coding-agent workloads.
3. Measuring tokens saved, latency, and quality.
4. Publishing the results as an arXiv paper.

## Why it's a research project and not just a utility

Each of the seven tactics has shown up in the literature or in blog
posts, but:

- Nobody has measured all seven on the same benchmark.
- Nobody has measured their pairwise interactions.
- Nobody has published a workload-differentiated ranking that says
  "for edit-heavy workloads, use {T1, T3, T5}; for explanation-heavy
  workloads, use {T1, T2, T3}."

That third result is the novel contribution. The expected finding is
that the optimal subset depends on the workload, and the paper's
practical value comes from giving practitioners a decision rule based
on workload characteristics.

## Why it lives in `~/dev/ml/mcp/`

The user already has a set of related MCP-adjacent research projects
in this directory:

- `resilient-write` — durable write surface (filter-block recovery).
- `local-splitter` — this project (token reduction).
- `llm-redactor` — sibling (privacy-preserving LLM requests).

All three are outbound-LLM-request-pipeline projects. They may end up
sharing a common library later, but for now each is independent.

## Constraints from the user

- Must work with **Ollama** and **any OpenAI-compatible API**. No
  vendor SDK dependency.
- The tool eventually gets published; the paper is the deliverable.
- Test which tactics actually work before claiming anything.
- "It will be a novel idea" — Jay wants this to be original research,
  not a rehash.

## Stakes

Jay is a security / privacy researcher with a personal blog
(`sperixlabs.org`). This project's findings will inform how he and
other practitioners choose model mixes in their daily work. If the
paper holds up, it becomes a citable reference for anyone deploying
coding agents in production.
