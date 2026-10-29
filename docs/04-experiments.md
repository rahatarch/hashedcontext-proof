# 04 — The experiments

Each experiment answers one question. This page explains what it does, how, and which files it writes. Paper section and table numbers refer to v0.3.

## E1 — Agent fleet: how many tokens and how much memory?

**Command:** `python -m experiments.e1_fleet 20000 [workers]` · **Code:** `experiments/e1_fleet.py`, `sim/fleet.py`, `sim/agent.py`

**Question.** Across many realistic coding sessions, how long is each prompt under each layout, how much of it is file content, and how much could a prefix cache reuse?

**How.** 20,000 simulated agents each work on one of four real repositories for 5 to 100 turns. Every turn an agent reads files, writes some reasoning, applies a real edit to a real file (with a real diff) and runs a real check (Python's `compile()`, or bracket matching for JavaScript/TypeScript). Agents follow one of three behaviour profiles:

| Profile | Share | Behaviour |
|---|---|---|
| `paper` | ~40% | 5 files, re-reads all of them every turn, 5–30 turns |
| `realistic` | ~40% | 3–8 files, discovers them gradually, re-reads ~35% per turn, 10–60 turns |
| `runaway` | ~20% | 2 files, stuck in an edit → test fails → re-read loop for 100 turns |

For **every turn**, the same conversation is assembled in six ways and counted with a real tokenizer: append-only, append-only + compaction, Mode A (paper stub), Mode A (compact stub), Mode B (rebase after 10 diffs), Mode B (no rebase).

**Correctness checks (built in).** For every file in every Mode A prompt, the live copy is checked against the agent's actual file. On a sample of sessions, the Mode A prompt is expanded back through the vault and compared byte for byte with the append-only prompt. The Mode B prompt is rebuilt from its own text and compared with the actual files. Every Mode B diff is applied back and its SHA-256 checked. Token counts are cross-checked against tokenizing the full prompt in one piece.

**Writes:** `results/fleet_turns.csv.gz` (one row per turn, see [08](08-file-formats.md)), `results/fleet_meta.json` (totals and check counters).

**Feeds:** paper Tables 1–3 and the memory numbers in the abstract.

## E2 — GPU serving simulator: what does it mean for a real server?

**Command:** `python -m experiments.e2_serving all` · **Code:** `experiments/e2_serving.py`, `sim/serving.py`

**Question.** If these agents talk to an 80 GB H100 GPU serving a 70-billion-parameter model, how many turns per hour get done, and how fast?

**How.** A discrete-event simulator of one GPU, modelled on vLLM: a fixed pool of KV-cache memory in 16-token blocks; a prefix cache that matches real token IDs block by block and evicts least-recently-used blocks; continuous batching; and timing from the GPU's published compute and memory bandwidth (a "roofline" model). The agents are the same as in E1, now generating real token streams. Each configuration simulates one hour.

- **Sweep:** 1 to 64 agents per GPU, prefix cache on and off, four layouts (append-only + compaction, Mode A, Mode A compact, Mode B).
- **Cluster:** 16 GPUs × 32 agents = 512 agents, for append-only + compaction, Mode A and Mode B.

**Writes:** `results/serving_all.json` (one record per configuration).

**Feeds:** paper Tables 4 and 5, the 3.57×/3.21× throughput numbers, and the mode recommendations.

## E3 — HTTP gateway load test: does it work in a real deployment?

**Command:** `python -m gateway.loadtest 64 20` · **Code:** `gateway/`

**Question.** Can HashedContext sit in front of a model server as a gateway, for unmodified clients, and what does that cost?

**How.** Starts a real HTTP gateway and a stand-in model server (which counts tokens like an OpenAI-compatible API) on your machine. 64 concurrent clients run 20-turn sessions in three setups: plain pass-through, HashedContext in the gateway, and HashedContext in the client. Every gateway request is transformed, expanded back and verified to be lossless before it is forwarded.

**Writes:** `results/gateway_loadtest.json`.

## E4 — Micro-benchmarks: how big is a stub, how fast is hashing?

**Command:** `python -m experiments.e4_micro` · **Writes:** `results/micro.json`

Stub size in tokens (for the paper's own example and for every real file path), SHA-256 speed, and JSON payload size and parse time per turn.

## E5 — Paper arithmetic: are the printed numbers right?

**Command:** `python -m experiments.e5_paper_math 67` · **Writes:** `results/paper_math.json`

Recomputes every worked example printed in the paper (memory per token, decode time, batch sizes, cost examples, payload sizes, …) from its stated inputs, then again with the measured stub size of 67 tokens.

## Needle test: is the information still there?

**Command:** `python -m llm_eval.needle_eval --mechanical --cases 300` · **Writes:** `results/needle_mechanical.json`

One line of one file is changed by "someone else" just before the last turn. Question: which file, which line, what is the new text? A *mechanical grader* answers using only the prompt text. That shows what information each layout contains:

- Append-only: everything is there.
- Mode A: shows **which** file changed (its fingerprint differs) but not **what** changed, because the old version is in the vault, not the prompt.
- Mode A + R(h): the old version is fetched from the vault, so everything is there.
- Mode B: shows the changed line directly.

## Diff-chain test: can the reader follow 10 diffs?

**Command:** `python -m llm_eval.chain_eval --mechanical --cases 100 --question anchor` · **Writes:** `results/chain_mechanical_anchor.json`

A stuck agent edits the same file 10 times. The question names a unique line of the file and asks for the line right after it. Under Mode B, answering requires applying all 10 diffs. The mechanical grader always succeeds, which confirms the test is built correctly; the interesting result comes from running it against a **live LLM** ([07](07-live-llm-tests.md)).

## Analysis and tables

```bash
python -m experiments.analyze          # results/summary.json and figures/*.png
python -m experiments.tradeoff_tables  # results/tradeoff_tables.md and .tex (paper Tables 2–8)
```
