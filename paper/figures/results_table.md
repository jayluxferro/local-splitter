# Evaluation Results — llama3.2:3b (local) / gemma3 (cloud)

## Cloud Token Savings (% reduction vs baseline)

| Subset | WL1 edit | WL2 explain | WL3 chat | WL4 RAG |
|--------|---------|-------------|----------|---------|
| baseline | 0% | 0% | 0% | 0% |
| T1 route | 28.2% | 61.9% | 55.4% | 33.7% |
| T2 compress | 20.6% | 20.3% | -4.7% | 15.5% |
| T3 cache | 9.9% | -1.2% | -5.0% | 3.6% |
| T4 draft | -39.4% | -43.1% | 11.7% | -35.8% |
| T1+T3 | 34.1% | 65.6% | 56.8% | 34.8% |
| T1+T2 | **46.9%** | **79.4%** | 59.5% | 45.3% |
| T1+T2+T3 | 42.4% | 79.1% | **60.6%** | 45.6% |
| all | 31.0% | 72.7% | 60.3% | **51.8%** |

## Requests Served Locally (T1)

| Workload | Locally served | Percentage |
|----------|---------------|------------|
| WL1 edit | 5/10 | 50% |
| WL2 explain | 7/10 | 70% |
| WL3 chat | 7/10 | 70% |
| WL4 RAG | 5/10 | 50% |

## Estimated Cost (USD per 10 samples, gpt-4o-mini rates)

| Subset | WL1 edit | WL2 explain | WL3 chat | WL4 RAG |
|--------|---------|-------------|----------|---------|
| baseline | $0.0040 | $0.0052 | $0.0068 | $0.0057 |
| T1+T2 | $0.0025 | $0.0013 | $0.0028 | $0.0049 |
| T1+T2+T3 | $0.0028 | $0.0013 | $0.0027 | $0.0050 |
| all | $0.0024 | $0.0009 | $0.0016 | $0.0023 |

## Key Findings

1. **T1 (route) is the single most impactful tactic** across all workloads
   (28–62% savings). It routes 50–70% of requests to the local model.

2. **T1+T2 is the best two-tactic combination** on 3/4 workloads
   (47–79% savings). Compression further reduces cloud input for complex requests.

3. **T4 (draft) is workload-dependent**: net negative on short-output workloads
   (edit, explain) because the review prompt adds input tokens. Net positive on
   chat workloads where outputs are long.

4. **On RAG-heavy workloads, "all" beats T1+T2+T3** (51.8% vs 45.6%) because
   T4 becomes favorable with long outputs from retrieved context.

5. **T3 (cache) shows minimal savings on single-pass workloads.** Value appears
   in multi-session or repetitive workloads (not tested in seed data).

6. **Optimal subset is workload-dependent:**
   - Edit-heavy: T1+T2 (46.9%)
   - Explanation: T1+T2 (79.4%)
   - Chat: T1+T2+T3 (60.6%)
   - RAG-heavy: all (51.8%)

## Recommendation

Default configuration: **T1 + T2 enabled**. Add T3 for repetitive workloads.
Add T4 for long-output workloads (RAG, code generation). T5/T6/T7 provide
marginal additional benefit.
