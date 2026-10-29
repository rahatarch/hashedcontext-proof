# 06 — Code tour

How the pieces fit together, for anyone who wants to read or reuse the code. Everything is plain Python 3; the only heavy dependency is `tiktoken` for token counting.

## The big picture

```
                    ┌───────────────────────── hashedcontext/ (the engine) ─────────────────────────┐
 sim/agent.py       │  core.py   hash gate · HashVault · stubs · Mode A assembly · expand · gateway  │
 (scripted agents)  │  lazy.py   Mode B: line diffs · LazyContext (rebase) · replay                   │
       │            │  tokens.py exact token counting (bundled tokenizer files)                       │
       ▼            └─────────────────────────────────────────────────────────────────────────────────┘
 sim/fleet.py   ── builds every layout for every turn, counts tokens, runs correctness checks ──► E1
 sim/serving.py ── feeds the token streams into a simulated GPU with a prefix cache ──────────► E2
 gateway/       ── real HTTP gateway applying the engine to unmodified OpenAI-style clients ────► E3
 llm_eval/      ── needle and diff-chain tests, mechanical grader or live LLM
 experiments/   ── one entry point per experiment, analysis, table generation
```

## `hashedcontext/` — the engine

### `core.py` (Mode A and shared parts)

| Piece | What it does |
|---|---|
| `sha256_hex(data)` | The hash gate: the fingerprint of a file's bytes. |
| `HashVault` | Content-addressed store `fingerprint → exact bytes`, in memory and optionally on disk (git-style `ab/cdef…` folders). Every `resolve()` re-checks the fingerprint, so corrupted data is never returned. |
| `render_stub(path, h, compact=False)` | The `[HASH_STUB]` block (paper format, ~67 tokens) or the compact `[#<16 hex> path]` form (~20 tokens). |
| `render_live(file)` | The `[LIVE_FILE]` block at the tail: path, fingerprint, full content ("self-attesting": anyone can re-hash it). |
| `assemble_conventional(system, turns)` | The append-only baseline. |
| `assemble_hashed(system, turns, vault)` | **Mode A** (Algorithm 1 in the paper): history with stubs, the current turn, then the live tail. |
| `expand(prompt, vault)` | The inverse of Mode A: turns every stub back into the file. Used to prove losslessness: `expand(assemble_hashed(...)) == assemble_conventional(...)`. |
| `oracle_read(prompt)` | What can be learned from a Mode A prompt alone (live files and stub history), without the vault. |
| `gateway_transform` / `gateway_expand` | Mode A applied to an OpenAI-style `messages` list, and its exact inverse. |

### `lazy.py` (Mode B)

| Piece | What it does |
|---|---|
| `line_diff(old, new)` | Changed lines only: hunks `@@ i,n @@` ("replace n lines from line i") followed by `-` and `+` lines. |
| `apply_line_diff(old, hunks)` | Applies a diff; raises an error if the old lines do not match. |
| `LazyContext(max_diffs_before_rebase=10)` | Remembers what the model has already seen and renders each new file observation as a full block, a stub ("Turn X content is still current"), a diff, or a full block again after N diffs (rebase). **Every diff is applied back and its SHA-256 checked before it is used.** |
| `replay(prompt)` | Rebuilds every file from the Mode B prompt text alone, checking every fingerprint along the way. |
| `assemble_lazy(system, turns, max_diffs_before_rebase=10)` | The full Mode B prompt (Algorithm 2 in the paper). |

### `tokens.py`

`Tokenizer` wraps `tiktoken` with a cache. The tokenizer files are bundled in `tokenizer_cache/` so counts work offline and do not depend on a download.

## `sim/` — agents and simulators

- **`agent.py` — `CodingAgent`.** Holds a copy-on-write workspace over a real repository. `step(k)` produces turn *k*: it applies the edit decided last turn (a real change to the file, recorded as a real unified diff), runs a real check, chooses which files to read according to its profile, and writes some reasoning text. It returns a `Turn(text, files)`: the narrative plus the file contents read this turn.
- **`corpus.py` / `fetch_corpus.py`.** The four pinned repositories and the code that downloads them.
- **`fleet.py` — `run_session(spec)`.** Runs one agent session and, for each turn, computes the token count of every layout, the file tokens, the cache-reusable prefix, the compaction baseline, and the correctness checks. `run_fleet(n)` runs `n` sessions in parallel (`ProcessPoolExecutor`). Sessions are generated from a fixed seed, so session *i* is always the same.
- **`serving.py` — `simulate(mode, n_agents, ...)`.** A discrete-event GPU simulator: `KVPool` (16-token blocks, reference counting, prefix matching on chained block hashes, LRU eviction), continuous batching, and roofline timing from the `HW` constants (80 GB, 3.35 TB/s, 30 GB weights, 163,840 bytes of KV per token). `AgentCtx` turns an agent into the exact token stream a client would send under each layout.

## `gateway/`

- **`servers.py`.** An aiohttp gateway (pass-through or HashedContext, lossless check on every request) and a stand-in backend that counts prompt tokens like an OpenAI-compatible API.
- **`loadtest.py`.** Starts both, runs concurrent unmodified clients, and records bytes, timings and token counts.

## `llm_eval/`

- **`needle_eval.py`.** The needle test for all layouts, and `ask_llm()`, the small OpenAI-compatible client used for live runs (with retries; the API key is never printed).
- **`chain_eval.py`.** The diff-chain test. `--question anchor` or `line`, `--rebase N`, `--conds conv,hc,lazy`, mechanical or live grader. Each layout's format is described accurately in its system prompt, and every answer is logged.

## `experiments/`

`e1_fleet.py`, `e2_serving.py`, `e4_micro.py`, `e5_paper_math.py` (entry points); `analyze.py` (summary and figures); `tradeoff_tables.py` (the paper's dual-mode tables, Markdown and LaTeX). E3 lives in `gateway/loadtest.py`.

## `tests/test_core.py`

Property tests on randomly generated sessions: Mode A round-trips byte-exact (300 sessions), Mode B prompts are strict prefixes of each other and rebuild exactly (300 sessions), line diffs round-trip (500 random pairs), the vault deduplicates and verifies, change detection works from the prompt alone, and the gateway transform is lossless.

## Using the engine in your own code

```python
from hashedcontext.core import FileObs, HashVault, Turn, assemble_hashed, expand, assemble_conventional
from hashedcontext.lazy import assemble_lazy, replay

turns = [Turn("read config\n", [FileObs("app/config.py", "DEBUG = False\n")]),
         Turn("set debug\n",   [FileObs("app/config.py", "DEBUG = True\n")])]

vault = HashVault()
mode_a = "".join(assemble_hashed("SYSTEM\n", turns, vault))       # stubs + live tail
assert expand(mode_a, vault) == "".join(assemble_conventional("SYSTEM\n", turns))

mode_b = assemble_lazy("SYSTEM\n", turns)                          # full once, then a diff
assert replay(mode_b)["app/config.py"] == "DEBUG = True\n"
```
