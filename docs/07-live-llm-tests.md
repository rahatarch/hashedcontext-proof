# 07 — Live-LLM tests

The simulations measure tokens, memory and throughput exactly, but they cannot tell whether a **real model** reads HashedContext prompts correctly. The live tests send the prompts to a real model through any OpenAI-compatible API.

## What the diff-chain test checks

Mode B's main risk is that the model must rebuild a file from its last full copy plus a chain of diffs. The test creates sessions where a file is edited 10 times, then asks:

> In the current version of `<file>`, find the line that is exactly `<line X>`. What is the exact text of the line immediately after it?

- **Append-only and Mode A** contain the whole current file, so the model only has to find line X.
- **Mode B** contains the first version plus 10 diffs, so the model has to work out where every change landed.

Each layout's format is explained to the model in its system prompt, so a low score reflects the layout, not confusing instructions.

## Run it

**1. Store your API key in a private file** (never put it on the command line, where it ends up in shell history):

```bash
mkdir -p ~/.config/hashedcontext && chmod 700 ~/.config/hashedcontext
nano ~/.config/hashedcontext/api_key          # paste the key on one line, save
chmod 600 ~/.config/hashedcontext/api_key
```

**2. Run** with your provider's base URL and model name:

```bash
HC_API_BASE="https://api.deepseek.com"  HC_MODEL="deepseek-chat"  make live
```

`HC_API_BASE` may include or omit a trailing `/v1`. Use `HC_API_KEY_FILE=/path/to/file` to read the key from somewhere else, and `CASES=100` for a larger sample.

**What it sends.** About 120 requests and roughly 2.3 million input tokens for 30 cases: 90 requests for the three layouts (append-only prompts are ~43K tokens, Mode A and B ~9K), plus 30 for Mode B with a rebase every 5 diffs. Check your provider's pricing first. The prompts contain code from the four public repositories.

**What it writes.** `results/chain_llm_anchor.json` and `results/chain_llm_anchor_rebase5.json` (scores), plus `*_cases.jsonl` files with every question, the expected answer and the model's raw reply, so every score can be audited. The key is not written anywhere.

## Reading the result

`make live` ends with your scores next to the published ones (`deepseek-v4.1-flash`: Mode A 28/30, Mode B 27/30 at 10 diffs and 28/30 at 5). With 30 cases, a difference of one or two answers is noise: the 95% range of 27/30 is roughly 74–97%. To detect a real gap of a few percentage points you need hundreds of cases.

Useful comparisons:
- **Mode B vs. Mode A** at the same model: does applying diffs cost accuracy?
- **Mode B at N=10 vs. N=5:** does rebasing more often help?
- **Append-only:** the upper reference (full information, but the longest prompt).

## Other live runs

```bash
# needle test with a live model (not run for the published results)
HC_API_BASE=... HC_MODEL=... HC_API_KEY="$(cat ~/.config/hashedcontext/api_key)" \
  .venv/bin/python -m llm_eval.needle_eval --cases 30

# the "line N" variant of the diff-chain test (measures line counting; see 05, section 6)
... .venv/bin/python -m llm_eval.chain_eval --cases 30 --question line
```
