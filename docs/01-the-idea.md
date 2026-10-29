# 01 — The idea

*No background needed. Terms in **bold** are explained in the [glossary](02-glossary.md).*

## How an AI coding assistant talks to a model

A coding assistant does not "remember" anything between steps. Every time it asks the model something, it sends the **whole conversation so far**: the task, every message, every tool result, and every file it has read. The model reads all of it, then writes the next step.

A typical session goes like this:

1. read `auth.py` and `session.py`, think, propose an edit
2. apply the edit, run the tests, read both files again to check
3. tests fail; read the files again, edit again
4. … and so on, for 10, 30, sometimes 100 turns

Each "read the file again" adds a **full copy** of the file to the conversation. After 30 turns, a 4,000-word file may appear 30 times, although only a few lines ever changed.

## Why that is expensive

Text is measured in **tokens** (roughly ¾ of a word). For every token in the conversation, the model's server keeps a block of numbers in fast GPU memory, the **KV-cache**: about 160 KB per token for a large model. A 200,000-token conversation therefore occupies about 32 GB of a GPU's 80 GB, for a single user.

So repeated copies are not just wasted text. They use up the scarcest resource on the server, and the server can run far fewer users at the same time.

The common fix today is **compaction**: when the conversation gets too long, ask the model to summarise it and throw the details away. That saves memory, but summaries lose exact details (variable names, line numbers), and producing them costs extra model calls.

## The HashedContext idea

Keep exactly the information the model needs, and nothing twice.

1. **Fingerprint every file.** Each time a file is read, compute its **SHA-256 hash**: a 64-character fingerprint that changes completely if even one character of the file changes.
2. **Replace repeated copies with the fingerprint.** If the model has already seen this exact version, send a short **stub** (~67 tokens) instead of the whole file:
   ```
   [HASH_STUB]
   path: src/auth/session.ts
   sha256: e4b2c91f7a62d88d3e91b40c21a7f05b8823190dfca91283c4b58e99128741aa
   state: CACHED
   [/HASH_STUB]
   ```
3. **Never lose anything.** Every version of every file is kept in a local **vault**, indexed by its fingerprint. Any old version can be put back exactly, at any time. This is why the method is called **lossless**: unlike a summary, nothing is thrown away.

The conversation itself (the task, the reasoning, the edits, the test output) is always kept word for word. Only repeated file contents are deduplicated.

## Two ways to lay out the prompt: Mode A and Mode B

There is a design choice in *where* the current version of each file appears, and it matters for the server.

### Mode A — Authoritative Tail

Every old copy becomes a stub. The **current** version of each file the assistant just read is placed, in full, at the very end of the prompt.

```
[task] [step 1] [stub A] [step 2] [stub A] [step 3] … [current turn] [FULL CURRENT A]
```

- ✅ The model always sees one complete, up-to-date copy of every file. Nothing to reconstruct.
- ✅ Uses the least memory.
- ❌ Each turn, the previous "full copy at the end" turns into a stub, so the prompt's ending changes every turn. Servers with a **prefix cache** (which reuse the work done on the unchanged beginning of a prompt) have to redo that part.

### Mode B — Lazy Tail

The conversation is **never rewritten**, only extended. A file is sent in full the first time. After that:

- if it has not changed, a stub says *"Turn X content is still current"*;
- if it has changed, only the **changed lines** are sent (a **line diff**).

```
[task] [step 1] [FULL A] [step 2] [stub: A unchanged] [step 3] [diff: 2 lines of A changed] …
```

- ✅ Each prompt starts exactly like the previous one, so a prefix cache can reuse almost all of it: the server processes ~88% fewer tokens.
- ✅ The model sees exactly which lines changed.
- ❌ To know a file's current state, the model has to mentally apply every diff since the last full copy. Long chains of diffs might confuse it ("lost in the middle"). To limit this, after `max_diffs_before_rebase` diffs (default 10) the full file is sent again (a **rebase**).

### Which one is better?

It depends on the server, which is why the paper proposes both:

| Situation | Better mode |
|---|---|
| The server's prefix cache stays warm (few users per GPU, requests from one user go to the same machine, API pricing that discounts cached tokens) | **Mode B** |
| The server is busy and its cache keeps getting evicted, or there is no prefix cache | **Mode A** |
| Lowest possible memory | **Mode A** |
| Accuracy-critical work with long edit loops | **Mode A**, or Mode B with a rebase every 5 diffs |

The measurements behind this table are explained in [05 — Results explained](05-results-explained.md).

## Where it runs

Either inside the assistant itself (an editor extension or command-line agent: "edge-native") or as a **gateway** in front of the model server that rewrites incoming requests ("gateway-native"). Neither the model nor the server software needs to change.

## What this package lets you check

That the method is lossless; how much memory and computation it saves, for both modes, on realistic workloads; how that translates into server throughput; and how a real model copes with Mode B's diffs. [03 — Reproduce](03-reproduce.md) shows how.
