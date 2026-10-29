# Published results

The raw output of the run behind the HashedContext paper (v0.3), exactly as produced. Your own runs go to `../results/`; this folder is never overwritten, so you can always compare against it (`make compare`).

**Machine:** Intel Core i7-10700 (8 cores, 16 threads, no SHA extensions), 31 GB RAM, Linux, Python 3.12.3, the package versions in `../requirements.txt`. **Date:** 2026-10-29.

## What is where

| File | Produced by | Paper | Content |
|---|---|---|---|
| `fleet_turns.csv.gz` | E1 agent fleet | Tables 1–3 | One row per turn: prompt tokens of every layout, cache reuse, compaction |
| `fleet_meta.json` | E1 | Tables 1–3 | Totals and correctness counters (losslessness, diff verification, stub sizes) |
| `serving_all.json` | E2 serving simulator | Tables 4–5 | One record per simulated GPU configuration |
| `gateway_loadtest.json` | E3 HTTP load test | §9 | Per-request bytes, timings, token counts, lossless checks |
| `micro.json` | E4 | §2–3, §9 | Stub sizes, SHA-256 speed, JSON payload sizes |
| `paper_math.json` | E5 | throughout | Each worked example of the paper, printed vs recomputed |
| `needle_mechanical.json` | needle test | Table 6 | What each layout reveals about an external one-line change |
| `chain_llm_anchor.json` | live diff-chain test | Table 8 | Live model, 30 cases, all three layouts, rebase after 10 |
| `chain_llm_anchor_rebase5.json` | live diff-chain test | Table 8 | Live model, Mode B with rebase after 5 |
| `chain_llm.json` | live diff-chain test | §11 | The "line N" variant (measures line counting; not used to compare modes) |
| `chain_llm*_cases.jsonl` | live tests | — | Every question, expected answer and raw model reply |
| `summary.json` | analysis | figures | Aggregates used for the figures |
| `tradeoff_tables.md`, `.tex` | table generator | Tables 2–8 | The dual-mode tables |
| `figures/` | analysis | — | Charts (below) |
| `logs/` | console | — | Console output of each run, see [`logs/README.md`](logs/README.md) |
| `VERIFICATION_REPORT.md` | — | — | The claim-by-claim audit of the earlier paper versions that led to v0.3 |

Field-by-field descriptions: [`../docs/08-file-formats.md`](../docs/08-file-formats.md). What the numbers mean: [`../docs/05-results-explained.md`](../docs/05-results-explained.md).

**Live-model details.** `deepseek-v4.1-flash` through the OpenAI-compatible endpoint `https://api.aisa.one/v1`, temperature 0, 0 API errors. No API key appears in any file.

## Figures

**Prompt length per turn** (median over sessions; append-only grows without limit, HashedContext stays small):

![Prompt length per turn](figures/fig1_prompt_per_turn.png)

**Cost of agents stuck in a loop** (still quadratic, but far flatter):

![Runaway loop cost](figures/fig2_runaway_cost.png)

**Simulated GPU throughput and latency** (prefix cache on and off):

![Serving](figures/fig3_serving.png)

**Request size through the HTTP gateway:**

![Gateway payload](figures/fig4_gateway_payload.png)
