### Table A. Memory and compute across 20,000 simulated sessions (824,728 turns)

| Mode | Mean KV per request | KV vs append-only | KV vs compaction | Computed tokens, warm cache, vs append-only | Computed tokens, no cache, vs append-only | Prefix-cache hit rate |
|---|---|---|---|---|---|---|
| Append-only | 30.96 GB | — | +174.5% | — | — | 95.9% |
| Append-only + compaction | 11.28 GB | -63.6% | — | +144.8% | -63.6% | 72.6% |
| Mode A: Authoritative Tail | 3.50 GB | -88.7% | -69.0% | +3.6% | -88.7% | 62.6% |
| Mode B: Lazy Tail, rebase after 10 diffs | 4.71 GB | -84.8% | -58.2% | -88.1% | -84.8% | 96.8% |
| Mode B without rebasing | 3.98 GB | -87.1% | -64.7% | -89.4% | -87.1% | 96.6% |

Mode B diff chains (most diffs applied to one file in a session): with rebasing p50/p90/p99/max = 4/10/10/10; without rebasing = 4/99/99/99. Rebases: 36,771. Diffs verified byte-exact: 547,785 of 547,785. Prompt-only reconstruction: 5,516 of 5,516.

### Table B. By workload profile (change vs append-only)

| Profile | Mode A KV | Mode B KV | Mode A compute, warm cache | Mode B compute, warm cache | Mode A compute, no cache | Mode B compute, no cache |
|---|---|---|---|---|---|---|
| Paper | -88.3% | -88.1% | +2.4% | -91.5% | -88.3% | -88.1% |
| Realistic | -87.2% | -80.3% | +5.1% | -84.0% | -87.2% | -80.3% |
| Runaway | -89.2% | -84.7% | +4.6% | -85.1% | -89.2% | -84.7% |
| All | -88.7% | -84.8% | +3.6% | -88.1% | -88.7% | -84.8% |

### Table C. Simulated 16-GPU H100 cluster, 512 agents, prefix caching on, 1 hour

| Metric | Append-only + compaction | Mode A | Mode B |
|---|---|---|---|
| Agent turns per hour | 3,883 | 13,854 | 12,456 |
| p95 turn latency (s) | 689 | 190 | 210 |
| p95 time to first token (s) | 655 | 107 | 128 |
| Requests running at once | 3.54 | 14.06 | 12.84 |
| KV per running request (GB) | 14.10 | 3.55 | 3.88 |
| KV pool utilisation | 99.7% | 99.7% | 99.7% |
| Prefix-cache hit rate | 0.3% | 0.6% | 0.6% |
| Prompt tokens computed per turn | 43,069 | 17,462 | 19,414 |
| Generated tokens/s (cluster) | 335 | 824 | 739 |
| Compaction calls | 204 | 0 | 0 |

Relative to append-only + compaction: Mode A 3.57× turns/h, p95 -72%; Mode B 3.21× turns/h, p95 -70%.

### Table D. Agent turns per GPU-hour by serving scenario (single simulated H100)

| Prefix cache | Agents / GPU | Append-only + compaction | Mode A | Mode B | B vs A |
|---|---|---|---|---|---|
| on | 1 | 297 | 434 | 587 | 1.35× |
| on | 2 | 514 | 829 | 1,118 | 1.35× |
| on | 4 | 295 | 1,168 | 2,054 | 1.76× |
| on | 8 | 182 | 1,303 | 1,398 | 1.07× |
| on | 16 | 197 | 949 | 756 | 0.80× |
| on | 32 | 266 | 943 | 829 | 0.88× |
| on | 64 | 331 | 1,023 | 930 | 0.91× |
| off | 1 | 86 | 304 | 202 | 0.66× |
| off | 2 | 135 | 457 | 324 | 0.71× |
| off | 4 | 180 | 666 | 440 | 0.66× |
| off | 8 | 178 | 773 | 590 | 0.76× |
| off | 16 | 195 | 836 | 675 | 0.81× |
| off | 32 | 267 | 942 | 827 | 0.88× |
| off | 64 | 334 | 1,022 | 926 | 0.91× |

### Table E. Mechanical needle test (300 cases, one external one-line change)

| Prompt | Mean prompt tokens | Correct file | Correct line | Correct text | Diffs to apply (mean / max) |
|---|---|---|---|---|---|
| Append-only | 217,963 | 100% | 100% | 100% | — |
| Mode A (stubs + live tail) | 24,723 | 100% | 0% | 0% | — |
| Mode A + R(h) | 28,443 | 100% | 100% | 100% | — |
| Mode B (rebase after 10) | 25,158 | 100% | 100% | 100% | 1.6 / 5 |
| Mode B (no rebase) | 25,158 | 100% | 100% | 100% | 1.6 / 5 |

### Table F. Recommended mode by scenario

| Scenario | Recommended | Evidence |
|---|---|---|
| Prefix cache on, light load (1–8 agents per GPU, cache stays warm) | Mode B | B completes 1.07–1.76× the turns of A (Table D) |
| Prefix cache on, heavy load (16+ agents per GPU, cache evicted) | Mode A | B completes 0.80–0.91× the turns of A; cluster: 0.90× (Table C) |
| No prefix cache, or requests not routed to the same cache | Mode A | B completes 0.66–0.91× the turns of A (Table D) |
| Billed API with discounted cached input tokens | Mode B | B computes 88.1% fewer prompt tokens than append-only with a warm cache; A computes +3.6% (Table A) |
| Lowest KV memory per request | Mode A | A 3.50 GB vs B 4.71 GB per request (Table A) |
| Long edit loops, accuracy-critical work | Mode A, or Mode B with N=5 | Live test, line after anchor (Table G): A 93.3%, B 90.0% with 10 diffs, 93.3% with N=5; differences of 1–2 of 30 cases, within noise |
| Model must see exactly what changed, without retrieval | Mode B | B shows the changed lines directly; A needs R(h) for that (Table E) |

### Table G. Live-LLM diff-chain test (deepseek-v4.1-flash, 30 cases, file has 10 edits)

| Question | Prompt | Diffs to apply | Correct | 95% CI |
|---|---|---|---|---|
| Line after anchor | Append-only | 0 | 29/30 (96.7%) | 83%–99% |
| Line after anchor | Mode A | 0 | 28/30 (93.3%) | 79%–98% |
| Line after anchor | Mode B, N=10 | 10 | 27/30 (90.0%) | 74%–97% |
| Line after anchor | Mode B, N=5 | 4 | 28/30 (93.3%) | 79%–98% |
| Line number | Append-only | 0 | 9/30 (30.0%) | 17%–48% |
| Line number | Mode A | 0 | 4/30 (13.3%) | 5%–30% |
| Line number | Mode B, N=10 | 10 | 17/30 (56.7%) | 39%–73% |
