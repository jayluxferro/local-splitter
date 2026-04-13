---
name: user profile
description: How to work with Jay (shared voice guide across sibling projects)
type: user
---

# Working with Jay

## Who they are

- **Jay Lux Ferro** (`jay@sperixlabs.org`, `@sperixlabs`, github
  `jayluxferro`).
- Personal cybersecurity research blog at `sperixlabs.org`.
- Focus areas: mobile reverse engineering, telemetry analysis,
  LLM tooling, privacy-preserving systems, local-first AI.
- Active adjacent projects: `proxy-atlas` (unreleased MITM capture
  indexer), `ollama-forge` (PyPI, local model pipelines),
  `resilient-write` (sibling MCP project for durable writes),
  `llm-redactor` (sibling MCP project for privacy).
- Runs on macOS / Apple Silicon. Uses bun, Hugo, Python 3.12+.

## How they work

- **Direct and terse.** Expects the same in return. No preamble,
  no filler, no "I'm happy to help" energy.
- **Wants proof, not promises.** "I built X, here's the output"
  beats "I will build X". Screenshots, greps, hashes, file listings
  are appreciated.
- **Names trade-offs explicitly.** If you're choosing X over Y, say
  why in one sentence and move on.
- **Tracks tasks.** Uses the agent's task list heavily. Expects you
  to update it as you go.
- **Asks before risky actions.** Especially destructive git operations,
  deletes, force-pushes. Default to dry-run or preview.
- **Confirms understanding by doing the next thing.** Rarely says
  "ok" — instead asks the next question. Silence = acceptance.

## Established preferences

- **Code style**: no emojis unless asked, short functions, no
  speculative abstractions. Comments only where non-obvious.
- **Documentation style**: structured markdown with tables,
  code fences for examples, explicit "what this is NOT" sections.
- **Scripts**: POSIX `sh`, `set -eu`, colour-coded status markers.
- **Python**: 3.12+, type hints, `pyproject.toml`, stdlib when
  reasonable.
- **Deploy / publish**: one-command flows that do everything with
  `--dry-run` / `--no-push` flags for safety.

## Things to avoid

- **Repetition**. If you already said "X is Y", don't say it again.
- **Summarising at the end of every response**.
- **Over-explaining technical basics**. Meet them at the level.
- **Decorative language** ("beautifully", "elegant", "seamlessly"
  get ignored).
- **Unsolicited features**. Do exactly what was asked. Offer
  follow-ups in one short list at the end if relevant.

## What to do on a new session

1. Read `AGENT.md`.
2. Read the five files under `.agent/memory/`.
3. Check git status. If the working tree is dirty, figure out why
   before touching anything.
4. Open by saying what you read and what you plan to do, in under
   10 lines. Wait for confirmation.

## One-liner mental model

Build the smallest useful thing first, ship it, verify it on the
wire, then ask what's next. Never build past the current stage
without confirmation.
