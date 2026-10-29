# 05 — Results explained

The key numbers of the published run, what they mean in plain words, and where each one comes from. All tables are generated from `published-results/` by `python -m experiments.tradeoff_tables`; the full set is in [`published-results/tradeoff_tables.md`](../published-results/tradeoff_tables.md).

## 1. Nothing is lost

| Check | Result | Source |
|---|---|---|
| Mode A prompts expanded back through the vault, compared byte for byte with the original | 11,032 / 11,032 identical | `fleet_meta.json` → `expand_ok` |
| Mode A live copies matching the agent's real file and its fingerprint | 2,061,819 / 2,061,819 | `live_ok` |
| Mode B diffs that reproduce the exact new file when applied | 547,785 / 547,785 | `lazy_diffs_verified` |
| Mode B prompts from which every file can be rebuilt from the prompt text alone | 5,516 / 5,516 | `lazy_ok` |
| Gateway requests verified lossless before forwarding | 1,280 / 1,280 | `gateway_loadtest.json` |

**In plain words:** in every case checked, the exact original could be recovered. HashedContext removes repetition, not information.

## 2. Memory: how much GPU memory each request needs

Mean KV-cache per request across 824,728 turns (mean prompt length × 163,840 bytes per token):

| Layout | Memory per request | vs. append-only | vs. append-only + compaction |
|---|---:|---:|---:|
| Append-only | 30.96 GB | — | — |
| Append-only + compaction | 11.28 GB | −63.6% | — |
| **Mode A** | **3.50 GB** | **−88.7%** | **−69.0%** |
| **Mode B** (rebase after 10) | 4.71 GB | −84.8% | −58.2% |
| Mode B (no rebase) | 3.98 GB | −87.1% | −64.7% |

**In plain words:** a request that would need about 31 GB of GPU memory needs about 3.5 GB with Mode A. Compaction alone gets to about 11 GB, by throwing information away.

**Why is Mode B a bit bigger?** It keeps the first full copy of each file and every diff; rebasing every 10 diffs adds extra full copies.

**Source:** `fleet_turns.csv.gz` (columns `conv`, `conv_with_compaction`, `hc_paper_stub`, `lazy`, `lazy_inf`).

## 3. Compute: how much text the server must actually process

A server with a **prefix cache** does not recompute the beginning of a prompt it has just seen. So the real work per request is "prompt length minus what the cache already has":

| Layout | Warm cache (ideal, never evicted) | No cache |
|---|---:|---:|
| Append-only | baseline | baseline |
| **Mode A** | **+3.6%** | **−88.7%** |
| **Mode B** | **−88.1%** | −84.8% |

**In plain words:** append-only prompts only ever grow at the end, so a warm cache reuses almost all of them. Mode A rewrites the end of the prompt every turn, so with a warm cache it actually costs slightly *more* work than append-only. Mode B never rewrites anything, so it gets both short prompts *and* cache reuse. Without a cache, everything has to be computed, and the shortest prompt wins (Mode A).

**Source:** `fleet_turns.csv.gz` (`*_cache_hit` columns: an ideal per-session cache in 16-token blocks).

## 4. Throughput: what this means for a real server

From the simulated 16-GPU cluster serving 512 agents, prefix cache on, one hour (paper Table 4):

| | Append-only + compaction | Mode A | Mode B |
|---|---:|---:|---:|
| Agent turns completed per hour | 3,883 | **13,854** (3.57×) | 12,456 (3.21×) |
| 95th-percentile turn latency | 689 s | **190 s** | 210 s |
| Requests running at once | 3.5 | **14.1** | 12.8 |
| Memory per running request | 14.1 GB | **3.55 GB** | 3.88 GB |
| Prefix-cache hit rate | 0.3% | 0.6% | 0.6% |

**In plain words:** with less memory per request, about four times as many requests fit on each GPU at once, so roughly 3–3.6× more work gets done per hour. On this busy cluster the cache keeps getting evicted (hit rates below 1%), so Mode B's cache advantage disappears and Mode A's smaller memory wins.

**When does Mode B win?** On a single GPU with fewer agents, the cache stays warm (paper Table 5):

| Prefix cache | Agents per GPU | Mode B's turns, relative to Mode A |
|---|---|---:|
| on | 1–8 | **1.07–1.76×** (Mode B wins) |
| on | 16–64 | 0.80–0.91× (Mode A wins) |
| off | 1–64 | 0.66–0.91× (Mode A wins) |

**Source:** `serving_all.json`.

## 5. Change detection: does the model see what changed?

Mechanical needle test, 300 cases (paper Table 6):

| Layout | Right file | Right line | Right new text |
|---|---:|---:|---:|
| Append-only | 100% | 100% | 100% |
| Mode A | 100% | **0%** | **0%** |
| Mode A + fetch old version from the vault | 100% | 100% | 100% |
| Mode B | 100% | 100% | 100% |

**In plain words:** in Mode A, a changed fingerprint tells the model *which* file changed, but the old text is in the vault, not the prompt, so it cannot see *what* changed unless the old version is fetched. Mode B sends the changed lines, so it can.

**Source:** `needle_mechanical.json`.

## 6. Live model: can a real LLM follow Mode B's diffs?

`deepseek-v4.1-flash`, 30 cases, a file edited 10 times; question: "what is the line right after line X?" (paper Table 8):

| Layout | Diffs to apply | Correct | 95% range |
|---|---:|---:|---|
| Append-only | 0 | 29 / 30 (96.7%) | 83–99% |
| Mode A | 0 | 28 / 30 (93.3%) | 79–98% |
| Mode B, rebase after 10 | 10 | 27 / 30 (90.0%) | 74–97% |
| Mode B, rebase after 5 | 4 | 28 / 30 (93.3%) | 79–98% |

**In plain words:** the model coped with 10 diffs about as well as with a full copy. The differences are one or two questions out of 30, too small to tell apart with this many cases. One question failed under every layout, so it is just hard for this model. For accuracy-critical work, rebasing every 5 diffs matched Mode A and cost about 16% more tokens than rebasing every 10.

**A test that did not work, reported for transparency:** an earlier version asked "what is the text of line 56?". Even with the whole file in view, the model got only 4 of 30 right in Mode A: it was testing line *counting*, not the layouts. It is kept in `chain_llm.json` but not used to compare modes.

**Source:** `chain_llm_anchor.json`, `chain_llm_anchor_rebase5.json`, and the per-question answers in `chain_llm_anchor*_cases.jsonl`.

## 7. Other measured facts

| Fact | Value | Source |
|---|---|---|
| Stub size (the paper's example) | 67 tokens | `micro.json` |
| Stub size across real file paths | mean 67.8, range 53–88 tokens | `fleet_meta.json` |
| Mode B "still current" stub | mean 72.8 tokens | `fleet_meta.json` → `sigma_lazy_mean` |
| Mode B diff block | median 104 tokens | `diff_block_tokens` |
| SHA-256 speed (one core, no SHA extensions) | ~1.8 ms per MB | `micro.json` |
| Vault size per session | 0.36 MB of unique content on average | `fleet_meta.json` |
| Cost growth in stuck loops | still quadratic in the number of turns, but with a coefficient ~25–30× smaller | `fleet_turns.csv.gz`, `paper_math.json` |

## What these results do not show

See [09 — Limitations](09-limitations.md). In short: the agents are scripted, the GPUs are simulated, and the live-model test is small.
