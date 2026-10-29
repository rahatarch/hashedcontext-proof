# 03 — Reproduce the results

This page takes you from a fresh machine to a PASS/FAIL report comparing your run with the published one.

## What you need

- **Linux, macOS, or Windows with WSL2.** (Native Windows is untested.)
- **Python 3.10 or newer**, with the `venv` module. On Debian/Ubuntu: `sudo apt install python3-venv`.
- **git** and **make**.
- **Internet access** during setup only: to install Python packages and download the four open-source repositories the agents work on (~40 MB).
- **Disk:** about 1 GB including the virtual environment and results.
- **CPU:** any. More cores make the full run faster; results do not depend on the number of cores.

No GPU is needed: the GPU experiments are simulated.

## 1. Set up (once)

```bash
cd hashedcontext-reproducibility
make setup
```

This creates a private Python environment in `.venv/`, installs the exact package versions listed in `requirements.txt`, and downloads four public repositories at pinned commits into `corpus/`:

| Repository | Commit |
|---|---|
| [pallets/flask](https://github.com/pallets/flask) | `d086db856be187255b8ec61ef409357393020f32` |
| [expressjs/express](https://github.com/expressjs/express) | `1c6f1b0fe4000977b9a7b4f34097e4d527dd9824` |
| [colinhacks/zod](https://github.com/colinhacks/zod) | `b1f6ba06c5de0e600640917615e5df70fa73f060` |
| [axios/axios](https://github.com/axios/axios) | `9d11464a75ee5c644c0308d742f58755c0482594` |

Pinned commits matter: the simulated agents read and edit these files, so a different version would give different token counts.

## 2. Quick check (~1–5 minutes)

```bash
make quick
```

It runs the tests, the first 200 sessions of the agent fleet, the paper-arithmetic check, the micro-benchmarks, and small versions of the needle and diff-chain tests, then compares everything with `published-results/`. The end of the output looks like this:

```
Agent fleet (E1)
  PASS  200 sessions, 7,754 turns, every column  identical to published rows
  PASS  Mode B: every sampled prompt rebuilds every file exactly  51/51
  PASS  Mode A: every sampled prompt expands back byte-exact  102/102
  ...
Micro-benchmarks (E4)
  PASS  stub sizes in tokens (the paper's example stub and real paths)  paper's example stub = 67 tokens
  INFO  SHA-256 speed (depends on your CPU)  yours 1.76 ms/MB, published 1.82 ms/MB

All 13 exact checks match the published results.
```

On a 16-thread desktop, `make setup && make quick` took 54 seconds from scratch.

**Why can 200 sessions be compared with a 20,000-session run?** Every session gets its own fixed random seed, so session 17 is identical whether you run 200 sessions or 20,000. The comparison checks every column of every turn you ran against the same rows of the published run.

## 3. Full reproduction (~20 minutes to 2 hours)

```bash
make full
```

| Step | What runs | Time on 16 threads |
|---|---|---|
| Tests | unit and property tests | seconds |
| E1 agent fleet | 20,000 sessions, 824,728 turns, all layouts | ~14 min |
| E2 serving simulator | 56-point single-GPU sweep + 16-GPU cluster | ~1–2 min |
| E3 gateway load test | 3 × 1,280 real HTTP requests | ~1 min |
| E4 / E5 | micro-benchmarks, paper arithmetic | seconds |
| Needle and diff-chain tests | mechanical graders | ~2 min |
| Analysis | summary, figures, trade-off tables | ~1 min |

The fleet uses all but two of your CPU threads by default. To choose: `PY=.venv/bin/python .venv/bin/python -m experiments.e1_fleet 20000 4` (4 workers). On 2 cores expect roughly 1–1.5 hours for the fleet alone.

After a full run you get, in `results/` and `figures/`, every file in `published-results/` except the live-LLM results, and the comparison report at the end.

On a 16-thread desktop, `make full` took 17 minutes from a fresh setup, and ended with:

```
Agent fleet (E1)
  PASS  20,000 sessions, 824,728 turns, every column  identical to published rows
  PASS  Mode B: every sampled prompt rebuilds every file exactly  5,516/5,516
  PASS  Mode A: every sampled prompt expands back byte-exact  11,032/11,032
GPU serving simulator (E2)
  PASS  104 simulator configurations (turns, tokens, cache hits, batch, latency)  identical
HTTP gateway load test (E3)
  PASS  every gateway request verified lossless  1280/1280
  ...
All 19 exact checks match the published results.
```

## 4. Live-LLM test (optional)

See [07 — Live-LLM tests](07-live-llm-tests.md). It needs an API key and costs a small amount of money.

## Running single experiments

```bash
.venv/bin/python -m pytest -q tests                        # tests only
.venv/bin/python -m experiments.e1_fleet 2000              # fleet with 2,000 sessions
HC_HORIZON=600 .venv/bin/python -m experiments.e2_serving cluster   # shorter serving run (won't match published)
.venv/bin/python -m llm_eval.needle_eval --mechanical --cases 300
.venv/bin/python -m llm_eval.chain_eval --mechanical --cases 100 --question anchor
make compare                                                # compare whatever is in results/
```

Each experiment is described in [04 — The experiments](04-experiments.md).

## Reading the report

- **PASS:** your output is identical to the published output (within 10⁻⁹ relative error for decimals).
- **FAIL:** something differs. See below.
- **INFO:** machine-dependent (timings, requests per second) or from a live model. Shown side by side and never counted as a failure.

## If a check fails

Almost always it is one of these:

1. **Different package versions.** Token counts depend on `tiktoken`. Make sure you used `make setup` (which installs the pinned versions) and are running `.venv/bin/python`.
2. **Different repository versions.** Delete `corpus/` and run `.venv/bin/python -m sim.fetch_corpus` again. It checks out the pinned commits.
3. **Line endings.** On Windows, git may convert line endings in `corpus/`, which changes token counts. Use WSL2, or set `git config --global core.autocrlf false` before setup.
4. **Changed code.** If you modified anything under `hashedcontext/`, `sim/` or `llm_eval/`, differences are expected.

The report prints the first differing session, turn and column, which usually points straight at the cause. If none of the above explains it, please open an issue with the report output, your OS, and `.venv/bin/pip freeze`.

## Cleaning up

```bash
make clean-results   # deletes results/ and figures/ (published-results/ is never touched)
rm -rf .venv corpus  # removes the environment and the downloaded repositories
```
