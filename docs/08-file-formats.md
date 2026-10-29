# 08 — File formats

Every file in `published-results/` (and in `results/` after your own run). All token counts use the `cl100k_base` tokenizer.

## `fleet_turns.csv.gz` — one row per agent turn (824,728 rows)

Gzip-compressed CSV. Open with `pandas.read_csv("fleet_turns.csv.gz")` or `zcat fleet_turns.csv.gz | head`.

| Column | Meaning |
|---|---|
| `session` | Session id, 0–19,999. Session *i* is identical in every run. |
| `profile` | Agent behaviour: `paper`, `realistic` or `runaway` (see [04](04-experiments.md)). |
| `repo` | Repository: 0 = flask, 1 = express, 2 = zod, 3 = axios. |
| `M` | Number of files in the agent's working set. |
| `P` | Number of turns in the session. |
| `k` | Turn number, 1 … P. |
| `conv` | Prompt tokens, **append-only**. |
| `hc_paper_stub` | Prompt tokens, **Mode A** with the paper's stub format. |
| `hc_compact_stub` | Prompt tokens, Mode A with the compact ~20-token stub. |
| `conv_with_compaction` | Prompt tokens, append-only with compaction (summarise to 30% when the prompt would exceed 128,000 tokens). |
| `conv_file` | Tokens of file content in the append-only prompt. |
| `hc_file` | Tokens of file content (stubs + live tail) in the Mode A prompt. |
| `dialogue` | Tokens of this turn's narrative (reasoning, tool calls, edit diffs, test output). |
| `generated` | Tokens the agent "wrote" this turn (used for output cost). |
| `conv_cache_hit` | Tokens of the append-only prompt an ideal per-session prefix cache could reuse from the previous turn (whole 16-token blocks). |
| `hc_cache_hit` | Same for Mode A. |
| `eff_cache_hit` | Same for append-only + compaction (0 on a turn where compaction happened). |
| `aux_in`, `aux_out` | Input and output tokens of the summarisation call on a compaction turn, otherwise 0. |
| `lazy` | Prompt tokens, **Mode B** (rebase after 10 diffs). |
| `lazy_cache_hit` | Reusable prefix for Mode B. |
| `lazy_file` | Tokens of file content (full blocks, stubs, diffs) in the Mode B prompt. |
| `lazy_inf` | Prompt tokens, Mode B without rebasing. |
| `lazy_inf_cache_hit` | Reusable prefix for Mode B without rebasing. |

Useful derived quantities:
- **KV memory per request** = prompt tokens × 163,840 bytes.
- **Tokens the server must compute** (warm cache) = prompt tokens − cache hit (+ `aux_in` for compaction).

## `fleet_meta.json` — fleet totals and correctness counters

| Field | Meaning |
|---|---|
| `sessions`, `turns`, `workers`, `wall_seconds` | Size and run time of the fleet run. |
| `file_observations`, `observed_bytes` | How many file reads, and their total size. |
| `fleet_vault_unique_objects`, `fleet_vault_unique_bytes` | Distinct file versions across all sessions. |
| `per_session_vault_unique_bytes` | Sum over sessions of each session's vault size. |
| `edits`, `check_failures` | Edits applied by agents, and how many failed the syntax check (expected: runaway agents break code on purpose). |
| `live_ok` / `live_total` | Mode A live copies matching the real file. |
| `expand_ok` / `expand_total` | Mode A prompts that expand back byte-exact (sampled sessions). |
| `lazy_ok` / `lazy_total` | Mode B prompts from which every file rebuilds exactly (sampled). |
| `lazy_full_blocks`, `lazy_stubs`, `lazy_diffs`, `lazy_rebases` | How Mode B rendered file reads. |
| `lazy_diffs_verified` | Mode B diffs verified by applying them back and checking SHA-256. |
| `lazy_max_diff_chain`, `lazy_max_diff_chain_no_rebase` | Longest diff chain per session: 50th/90th/99th/100th percentile, with and without rebasing. |
| `sigma_paper`, `sigma_paper_mean`, `sigma_compact_mean` | Stub size in tokens (percentiles 0/5/50/95/100, means). |
| `sigma_lazy`, `sigma_lazy_mean`, `diff_block_tokens` | Mode B stub and diff block sizes. |
| `live_header_mean` | Tokens of the `[LIVE_FILE]` header. |
| `compactions`, `dropped_tokens` | Compaction events in the compaction baseline, and tokens summarised away. |
| `N_mean_by_profile` | Mean file size in tokens per profile. |
| `tok_err` | Pairs (segment-sum count, full-prompt count) used to check token counting. |

## `serving_all.json` — GPU simulator, one record per configuration

| Field | Meaning |
|---|---|
| `kind` | `sweep` (one GPU) or `cluster` (one of 16 GPUs). |
| `mode` | `conv` (append-only + compaction), `hc` (Mode A), `hc_compact`, `lazy` (Mode B). |
| `apc`, `agents`, `seed` | Prefix cache on/off, agents on this GPU, random seed. |
| `horizon`, `sim_time` | Simulated seconds (3,600). |
| `turns`, `turns_per_hour` | Agent turns completed. |
| `gen_tok_per_s` | Output tokens generated per second. |
| `prompt_tokens`, `prefill_tokens` | Prompt tokens sent, and tokens actually computed after cache reuse. |
| `apc_hit_rate` | Share of prompt tokens served from the prefix cache. |
| `mean_batch`, `mean_queue` | Average requests running and waiting. |
| `mean_kv_util`, `cap_blocks` | Average share of the KV pool in use, and pool size in 16-token blocks. |
| `evictions`, `rejected` | Cache blocks evicted; requests too large to ever fit. |
| `lat_p50`, `lat_p95`, `ttft_p50`, `ttft_p95` | Turn latency and time to first token, median and 95th percentile (seconds). |
| `prefill_busy_frac` | Share of GPU time spent on prefill. |
| `aux_calls` | Compaction summarisation calls. |
| `wall` | Real seconds the simulation took (machine-dependent). |

## `gateway_loadtest.json` — HTTP load test

Top level: `baseline`, `gateway`, `edge`, each with `clients`, `turns`, `requests`, `wall_s`, `rps` and `rows` (one per request): `client`, `k` (turn), `rtt_ms`, `in_bytes` / `out_bytes` (body received by / forwarded from the gateway), `gw_us` / `parse_us` / `xform_us` / `verify_us` / `ser_us` (gateway time split), `verified` (running count of lossless checks passed), `vault_objects`, `vault_bytes`, `prompt_tokens` (counted by the backend).

## Needle and diff-chain results

- **`needle_mechanical.json`:** per layout (`conv`, `hc`, `hc_resolve`, `lazy`, `lazy_no_rebase`): `n`, `path_acc`, `line_acc`, `text_acc`, `mean_prompt_tokens`, and for Mode B `diffs_to_apply_mean` / `max`.
- **`chain_llm_anchor.json`, `chain_llm_anchor_rebase5.json`, `chain_llm.json`:** live diff-chain results. Per layout: `n`, `correct`, `api_errors`, `accuracy`, `mean_prompt_tokens`, `diffs_to_apply_*`. Plus `grader` (model name) and `settings` (cases, diffs, rebase, question type).
- **`chain_llm*_cases.jsonl`:** one JSON object per question and layout: `case`, `cond`, `ok`, `truth` (path, line, expected text, anchor line), `answer` (parsed), `raw` (the model's reply, first 2,000 characters).

## Other files

| File | Content |
|---|---|
| `micro.json` | `sha256` (seconds and µs/MB per block size), `sigma` (stub sizes per tokenizer), `payload` (JSON request size and parse time per turn). |
| `paper_math.json` | Each worked example of the paper: the printed value and the recomputed one. |
| `summary.json` | Aggregates used for the figures: totals per profile, reduction by turn depth, runaway-loop costs, serving and gateway summaries. |
| `tradeoff_tables.md` / `.tex` | The paper's dual-mode tables, for reading and for LaTeX. |
| `figures/*.png` | Prompt length per turn, runaway-loop cost, serving throughput, gateway payload. |
| `logs/` | Console output of the published runs; see [`logs/README.md`](../published-results/logs/README.md). |
