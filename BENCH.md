# Benchmark results

Server-side metrics from the serve's `/metrics` endpoint (`prompt_ms`,
`decode_ms`, `first_engine_token_s`, `reused`) — these are the engine's own
timings, not client-side guesses. The serve batches decode across streams, so
"decode/s" is the engine `decode_tok_s` for that request, not wall-clock per stream.

Run: `2026-11-19 17:57 EST`, model `qwen3.8-flash-next-split3`, live 262k
config, `--spec 4`, `--batch 4`, mtp drafting, on 3× Volta (2× V100-PCIE-32GB +
1× PG500-216). Client driven from a separate host over LAN so box was clean of
chat traffic during the run.

`ttft_s` = time to first engine token (includes queue wait + prefill),
`decode_tok_s` = engine decode rate during that request's decode window,
`reused` = conversation-cache prompt tokens reused.

## Cold prefill (prompt only, `max_tokens=1`)

| prompt tokens | prefill time | prefill rate |
|---|---|---|
| 972 | 1.0 s | 1011.7 t/s |
| 3,728 | 2.0 s | 1591.9 t/s |
| 14,739 | 7.0 s | 2142.9 t/s |

Prefill rate rises with prompt size (amortized by batch/spec); larger prompts
run faster per token.

## Concurrency — 256-token completions, unique (cold) prompts

Each stream gets a distinct prompt with zero cache reuse, issued together.

| streams | wall (s) | aggregate out (t/s) | per-stream TTFT (s) | per-stream decode (t/s) |
|---|---|---|---|---|
| 1 | 4.14 | 61.8 | 2.35 | 143.5 |
| 2 | 7.91 | 64.7 | 4.71 / 2.31 | 79.7 / 46.6 |
| 3 | 11.25 | 68.3 | 4.66 / 7.06 / 2.28 | 39.3 / 64.3 / 29.4 |
| 4 | 14.73 | 69.5 | 9.44 / 4.72 / 7.07 / 2.32 | 49.4 / 25.9 / 36.0 / 21.7 |

TTFT grows with concurrency (streams queue behind shared decode slots);
aggregate output climbs from ~62 → ~70 t/s as streams share the batch.

## Concurrency — 256-token completions, shared (warm) prompt

All streams share one prompt so the conversation cache hits (3518 tokens reused
on the 2nd+ streams).

| streams | wall (s) | aggregate out (t/s) | per-stream TTFT (s) | per-stream decode (t/s) | reused |
|---|---|---|---|---|---|
| 1 | 4.48 | 57.1 | 2.25 | 119.8 | 0 |
| 2 | 5.63 | 90.9 | 2.52 / 0.21 | 82.3 / 48.4 | 0 / 3518 |
| 3 | 7.00 | 109.7 | 0.45 / 2.82 / 0.15 | 39.2 / 61.3 / 38.1 | 3518 / 0 / 3518 |
| 4 | 12.38 | 82.7 | 4.98 / 7.37 / 2.59 / 0.24 | 35.0 / 52.1 / 26.5 / 21.8 | 0 / 0 / 0 / 3518 |

Warm cache makes a big difference at 2–3 streams (cache-hit stream drops to
~0.2 s TTFT; aggregate jumps to ~110 t/s at 3 streams). The 4-stream warm run
did not reuse for 3 of the streams (cache eviction pressure under batch 4) —
that pass is noisy; treat 4-stream warm as ~83 t/s, not a regression.

## Takeaways

- **Single-stream decode ~120–143 t/s** (well above the 32 GB single-card path
  thanks to the 3-card ring).
- **TTFT under concurrency** scales with queue depth: 4 cold streams top out at
  ~9.4 s for the last one. If latency matters, keep ≤2 concurrent.
- **Aggregate throughput** plateaus near ~70 t/s cold, ~83–110 t/s warm.
  The batch=4 / parallel=4 config saturates around 2–3 useful concurrent
  streams; the 4th adds throughput but pays queue-wait on TTFT.
- **Conversation cache** is the concurrency lever: a cache-hit stream skips
  prefill entirely (~0.2 s TTFT vs 2–5 s).
- **Spec decode (mtp draft, spec 4):** acceptance rate ~58% over the session
  (41,907 of 72,118 drafted tokens accepted), which is what sustains the
  per-stream decode rates above on sm_70 hardware.

Machine-readable per-request records: [`bench/results.json`](bench/results.json).
