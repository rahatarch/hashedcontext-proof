# Logs

Console output of the published runs, unedited.

| Log | Run |
|---|---|
| `e1_lazy_stdout.log`, `e1_lazy_time.log` | **Final** agent fleet run (all layouts including Mode B), 20,000 sessions on 14 workers: 13 min 40 s. Produced `fleet_turns.csv.gz` and `fleet_meta.json`. |
| `e1_stdout.log` | The **first** fleet run (before Mode B existed), 2 workers, ~28 min. Its append-only and Mode A token totals are identical to the final run. |
| `e2_stdout.log`, `e2_time.log` | GPU serving simulator, all 104 configurations, 12 workers: 1 min 6 s. |
| `loadtest_stdout.log` | HTTP gateway load test, 64 clients × 20 turns, three setups. |
| `e4_stdout.log` | Micro-benchmarks. |
| `e5_stdout.log` | Paper arithmetic. |
| `needle_stdout.log` | Mechanical needle test, 300 cases, five layouts. |
| `chain_llm_anchor_run.log` | Live diff-chain test, "line after anchor", three layouts, 30 cases. One line per answer: `OK` or `BAD`. |
| `chain_llm_anchor_rebase5_run.log` | Same, Mode B with rebase after 5. |
| `chain_llm_run.log` | Live diff-chain test, "line N" variant. |

`*_time.log` files are GNU `time -v` output (wall time, CPU use, peak memory).
