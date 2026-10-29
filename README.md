# HashedContext — Reproducibility Package

[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23668957-blue)](https://doi.org/10.5281/zenodo.23668957)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-100%25%20Pass-brightgreen)](published-results/VERIFICATION_REPORT.md)

**Everything needed to check the results of the HashedContext paper independently: the engine, the simulators, the experiments, the raw logs of the published run, and a comparison script that reports, check by check, whether your run matches ours.**

📄 Paper (latest version): **[doi.org/10.5281/zenodo.23668957](https://doi.org/10.5281/zenodo.23668957)**
🗂️ Paper repository: [github.com/rahatarch/hashedcontext](https://github.com/rahatarch/hashedcontext)

---

## The One-Command Proof

```bash
git clone https://github.com/rahatarch/hashedcontext-proof && cd hashedcontext-proof
make setup && make full
```

That is the whole verification. `make setup` creates the virtual environment, installs the pinned dependencies of the published run, and fetches the four pinned upstream repositories at their recorded commits. `make full` then:

- runs the **20,000-agent simulation** — 824,728 turns across four real open-source codebases, in both HashedContext modes and the conventional append-only baseline;
- replays the GPU serving simulator, the HTTP gateway load test, the micro-benchmarks, and the paper's worked arithmetic;
- finishes with `All 19 exact checks match the published results.`

All 19 checks are **exact**: every deterministic output is compared against [`published-results/`](published-results/) row by row and column by column. Timings and live-LLM results are reported side by side and never counted as passes or failures. Expect 20 minutes to 2 hours depending on your machine. If any check fails, [docs/03-reproduce.md](docs/03-reproduce.md) explains how to diagnose it.

---

## What is this about, in one minute?

AI coding assistants (Claude Code, Cursor, Cline, …) work in turns. On every turn they send the model the whole conversation so far, **including every file they have read, again and again**, even when the file has not changed.

```
Turn 1:  [instructions] [file A]
Turn 2:  [instructions] [file A] [file A]
Turn 3:  [instructions] [file A] [file A] [file A]          ← same file, three copies
...
Turn 30: [instructions] [file A] × 30                        ← mostly repeated text
```

The model's server has to keep all of that text in fast GPU memory (the "KV-cache"). Repeated copies waste that memory, so each GPU can serve fewer users.

**HashedContext** replaces the repeated copies with a short fingerprint (a SHA-256 hash, ~67 tokens instead of thousands) and keeps every real version safely in a local store, so nothing is ever lost. It comes in two flavours:

- **Mode A — Authoritative Tail:** old copies become fingerprints; one complete, current copy of each file sits at the end.
- **Mode B — Lazy Tail:** the conversation is never rewritten; each file is sent once, and afterwards only the *changed lines* are sent.

New to all this? Start with **[docs/01-the-idea.md](docs/01-the-idea.md)** and keep the **[glossary](docs/02-glossary.md)** open.

## What the published run found

Measured on 20,000 simulated coding agents working on four real open-source projects (824,728 turns), plus a simulated 16-GPU server. Full tables: [published-results/tradeoff_tables.md](published-results/tradeoff_tables.md).

| | Mode A | Mode B |
|---|---:|---:|
| GPU memory (KV-cache) needed per request, vs. sending everything | **−88.7%** | −84.8% |
| Text the server must process, when its cache is warm | +3.6% | **−88.1%** |
| Agent turns completed per hour on a busy 16-GPU cluster | **3.57×** | 3.21× |
| Every past file version recoverable exactly | ✅ | ✅ |

Neither mode wins everywhere: **Mode B is better when the server's cache stays warm, Mode A under heavy load or without a cache.** What these numbers mean, and what they do *not* mean: [docs/05-results-explained.md](docs/05-results-explained.md).

## How to verify it — pick your level

| Level | What you do | Time | You need |
|---|---|---|---|
| **1. Read** | Open the published logs and tables in [`published-results/`](published-results/) | minutes | a browser |
| **2. Quick check** | `make setup && make quick` — runs every deterministic part at small scale and compares it, row by row, with the published run | ~5 min | Python 3.10+, git, internet |
| **3. Full reproduction** | `make full` — regenerates every published number except the live-LLM test | 20 min – 2 h | a multi-core machine |
| **4. Live model** | `make live` — asks a real LLM about files rebuilt from diffs | ~15 min | an API key for any OpenAI-compatible endpoint |

```bash
git clone <this repository> && cd hashedcontext-reproducibility
make setup      # creates .venv, installs pinned packages, downloads the 4 test repositories
make quick      # ~5 minutes; ends with a PASS/FAIL report
```

The comparison script checks **exact** equality for everything that is deterministic (the simulated agents use fixed random seeds, so your numbers should match ours bit for bit), and only shows timing-dependent numbers side by side. Step-by-step instructions and troubleshooting: **[docs/03-reproduce.md](docs/03-reproduce.md)**.

## What's in this folder

```
hashedcontext-reproducibility/
├── README.md                  ← you are here
├── docs/                      ← plain-language documentation (start with 01)
├── hashedcontext/             ← the engine: hashing, vault, stubs, Mode A and Mode B
├── sim/                       ← simulated coding agents, the agent fleet, the GPU serving simulator
├── gateway/                   ← a real HTTP gateway that applies HashedContext to unmodified clients
├── llm_eval/                  ← needle test and diff-chain test (mechanical grader or a live LLM)
├── experiments/               ← one script per experiment, plus analysis and table generation
├── tests/                     ← unit and property tests (losslessness, diffs, vault, gateway)
├── scripts/                   ← quick check, full reproduction, live test, comparison
├── tokenizer_cache/           ← tokenizer files, bundled so token counts work offline
├── published-results/         ← raw results, logs and figures of the published run
├── Makefile                   ← `make help` lists every command
└── requirements.txt           ← exact package versions used for the published run
```

Your own runs go to `results/` and `figures/`; `published-results/` is never overwritten.

## Documentation

| | |
|---|---|
| [01 — The idea](docs/01-the-idea.md) | The problem and both modes, explained without jargon |
| [02 — Glossary](docs/02-glossary.md) | Every technical term used here, in plain words |
| [03 — Reproduce the results](docs/03-reproduce.md) | Install, quick check, full run, expected output, troubleshooting |
| [04 — The experiments](docs/04-experiments.md) | What each experiment measures, how, and which file it writes |
| [05 — Results explained](docs/05-results-explained.md) | The key numbers, how to read them, and where each comes from |
| [06 — Code tour](docs/06-code-tour.md) | How the engine and simulators are put together |
| [07 — Live-LLM tests](docs/07-live-llm-tests.md) | Running the tests against a real model, safely |
| [08 — File formats](docs/08-file-formats.md) | Every column and field in the published results |
| [09 — Limitations](docs/09-limitations.md) | What this evidence does and does not show |
| [Verification report](published-results/VERIFICATION_REPORT.md) | The claim-by-claim audit of the earlier paper versions that led to v0.3 |

## Honest limits, up front

- The coding agents are **scripted programs, not real LLMs**. Token, memory and cost accounting is exact; reasoning quality is not tested by the simulation.
- GPU results come from a **simulator** using NVIDIA H100 constants, not from a physical GPU.
- The live-LLM test covers **one model and 30 cases per condition**, so it can only detect large differences.

More in [docs/09-limitations.md](docs/09-limitations.md).

## License

- **Code** (everything except `published-results/`, `docs/` and `tokenizer_cache/`): [MIT](LICENSE)
- **Results and documentation** (`published-results/`, `docs/`): [CC BY 4.0](LICENSE-DATA.md)
- **Tokenizer files** (`tokenizer_cache/`): the `cl100k_base` and `o200k_base` encodings distributed with OpenAI's [tiktoken](https://github.com/openai/tiktoken) (MIT); bundled only so the package works offline.

## Citation

If you use this package, please cite the paper ([`CITATION.cff`](CITATION.cff)):

> Hasan, R. (2026). *Scaling GPU KV-Cache Efficiency and Inference Concurrency for Autonomous Agents with HashedContext: A Dual-Mode Architecture.* Zenodo. https://doi.org/10.5281/zenodo.23668957

## Contact

Rahat Hasan · Akkhar-Labs · [rahathasan5773@gmail.com](mailto:rahathasan5773@gmail.com) · [github.com/rahatarch](https://github.com/rahatarch)
