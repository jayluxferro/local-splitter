#!/usr/bin/env bash
# Minimal OpenAI-compatible chat against a running local-splitter proxy.
# Start the proxy first:  uv run local-splitter serve-http --config config.yaml
set -euo pipefail
BASE="${OPENAI_API_BASE:-http://127.0.0.1:7788/v1}"
curl -sS "${BASE%/}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy}" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32,
    "temperature": 0
  }' | python3 -m json.tool
