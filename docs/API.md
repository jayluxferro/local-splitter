# API

`local-splitter` exposes two parallel interfaces that feed the same
pipeline.

## 1. MCP interface (stdio)

For agents that speak MCP natively.

### `split.complete`
Main entry. Runs the full pipeline and returns the final response.

**Input**
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "model_hint": "auto | local | cloud",
  "max_tokens": 2048,
  "temperature": 0.7,
  "stream": false,
  "meta": {
    "tool_name": "coding_agent",
    "session_id": "..."
  }
}
```

**Output**
```json
{
  "response": "...",
  "served_by": "local | cache | cloud | draft+cloud",
  "tokens": {
    "input_cloud": 0,
    "output_cloud": 0,
    "input_local": 432,
    "output_local": 80
  },
  "latency_ms": 184,
  "pipeline_trace": [
    {"stage": "t1_route",     "decision": "COMPLEX", "ms": 42},
    {"stage": "t3_sem_cache", "decision": "MISS",    "ms": 6},
    {"stage": "t2_compress",  "decision": "APPLIED", "ms": 120,
     "tokens_in": 4200, "tokens_out": 1600},
    {"stage": "t5_diff",      "decision": "SKIPPED (not edit)"},
    {"stage": "cloud_call",   "decision": "APPLIED", "ms": 800}
  ]
}
```

### `split.classify`
Runs only T1 without generating a response. For debugging and for
agents that want to short-circuit without delegating generation.

### `split.cache_lookup`
Runs only T3. Returns the cached response (if any) without writing to
the cache.

### `split.stats`
Returns aggregate metrics since process start:
- total requests
- requests by `served_by` class
- total input/output tokens saved vs baseline
- average latency per pipeline stage

### `split.config`
Returns the current config (read-only).

---

## 2. HTTP proxy interface (`POST /v1/chat/completions`)

OpenAI-compatible. An agent points at `http://localhost:7788/v1` instead
of the real upstream, and everything works transparently.

### Request shape

Standard OpenAI chat completion. Any field not listed below is forwarded
unchanged to the upstream cloud call.

### Extra fields recognised by the proxy

| Field | Purpose |
|---|---|
| `extra_body.splitter.force_local` | `true` forces T1 to answer locally |
| `extra_body.splitter.force_cloud` | `true` bypasses the whole pipeline |
| `extra_body.splitter.tactics` | array of enabled tactic names for this call only (overrides config) |
| `extra_body.splitter.tag` | opaque string used in metrics for workload segmentation |

### Response shape

Standard OpenAI chat completion, plus:

```json
{
  "choices": [...],
  "usage": {
    "prompt_tokens": 200,
    "completion_tokens": 80,
    "total_tokens": 280
  },
  "splitter": {
    "served_by": "cloud",
    "pipeline_trace": [...],
    "tokens_saved": {"input": 1800, "output": 120}
  }
}
```

### `GET /v1/models`

Returns both the local and the cloud model so agents see them.

### `GET /v1/splitter/stats`

Same payload as `split.stats` from the MCP interface.

---

## 3. Ollama-native passthrough (optional)

For agents that speak Ollama's native API (`/api/chat`,
`/api/generate`), the proxy also accepts those shapes and translates
them to the internal request format.

Enable via `transport.ollama_compat: true` in config.
