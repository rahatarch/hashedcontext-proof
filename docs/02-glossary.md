# 02 — Glossary

Plain-language definitions of every term used in this package, in alphabetical order.

**Agent / coding agent.** A program that uses an LLM to work on code in turns: read files, think, edit, run tests, repeat. In this package the agents are *scripted* (see *Simulated agent*).

**Append-only (context).** The usual way assistants build prompts: every turn adds new text at the end and nothing is ever removed, so every file read stays in the conversation as a full copy. This is the main baseline the paper compares against.

**Append-only + compaction.** Append-only, but when the conversation reaches the model's limit (here 128,000 tokens) it is summarised down to 30% of the limit. A more realistic baseline, because plain append-only eventually stops fitting.

**Authoritative tail.** In Mode A, the full current copy of each file placed at the very end ("tail") of the prompt. "Authoritative" because it is the one copy the model should trust.

**Baseline.** What a result is compared against. Here: append-only, or append-only + compaction.

**Batch / batch size.** How many users' requests a GPU processes together at the same moment. Bigger batches mean more throughput, and the batch size is limited by GPU memory.

**BPE tokenizer.** The algorithm that splits text into tokens. This package uses `cl100k_base` (GPT-4-era) for exact counts; `o200k_base` gives nearly identical stub sizes.

**Cache hit / hit rate.** The share of a prompt the server did not have to recompute because it was found in the *prefix cache*.

**Compute / prefill tokens.** The tokens the server actually has to process for a request, i.e. the prompt minus whatever the prefix cache already holds. "Compute" in this package means these tokens.

**Concept DOI.** A permanent identifier that always points to the newest version of a publication. The paper's is 10.5281/zenodo.23668957.

**Content-addressed storage.** A store where each item is found by a fingerprint of its content (here SHA-256), not by a name. Identical content is stored once; any change gives a new address. Git works this way.

**Decode.** The phase where the model writes its answer, one token at a time. Each step reads the model weights and the whole KV-cache from GPU memory.

**Deterministic.** Produces exactly the same output every time it is run with the same inputs. All simulated experiments here use fixed random seeds and are deterministic, which is why your run can be compared to ours bit for bit.

**Diff / line diff.** A description of how one version of a file becomes another: "remove these lines here, add these lines there". Mode B sends diffs instead of whole files. Here diffs are line-based, not syntax-aware.

**Diff chain.** In Mode B, the sequence of diffs sent for a file since its last full copy. To know the current file, the model must apply all of them in order.

**Edge-native / gateway-native.** Where HashedContext runs: inside the client (the assistant, "edge") or in a proxy in front of the model server ("gateway").

**Fleet.** The 20,000 simulated agent sessions used for the main measurements.

**GPU / H100 / HBM.** The graphics processor that runs the model; the NVIDIA H100 is a common data-centre GPU with 80 GB of fast memory (HBM). The simulator uses its published numbers.

**Hash / SHA-256.** A function that turns any data into a fixed 64-character fingerprint. Any change to the data changes the fingerprint completely, and two different files practically never share one.

**Hash stub / `[HASH_STUB]`.** The short block (~67 tokens) that replaces a repeated file copy. It contains the file's path and its SHA-256 fingerprint.

**KV-cache.** The working memory a model keeps on the GPU for every token of the prompt (keys and values of the attention mechanism). About 160 KB per token for the 70-billion-parameter model assumed here. It is the main thing HashedContext saves.

**Live LLM / live test.** A test that sends prompts to a real model through an API, as opposed to the *mechanical grader*.

**LLM.** Large language model, e.g. GPT, Claude, DeepSeek.

**Lossless.** Nothing is lost: the original can be rebuilt exactly, byte for byte. Checked here on hundreds of thousands of prompts.

**"Lost in the middle".** The known tendency of LLMs to overlook information buried in the middle of long prompts. The concern for Mode B is that a model might misapply a long diff chain.

**`max_diffs_before_rebase` (N).** Mode B setting: after this many diffs to a file, its next change is sent as a full copy again. Default 10; 5 is recommended for accuracy-critical work.

**Mechanical grader.** A small program that answers the test questions using only the prompt text. It shows what information a prompt contains. It is not a model, so it never "gets confused".

**Mode A / Mode B.** The two layouts of HashedContext. See [01 — The idea](01-the-idea.md).

**Needle test.** One line in one file is changed behind the agent's back; the question is "what changed?". Shows whether each layout still contains that information.

**Prefill.** The phase where the model reads the prompt before answering. Its cost grows with the number of tokens that have to be processed.

**Prefix cache.** A server feature (e.g. in vLLM or SGLang) that keeps the KV-cache of recent prompts, so a new prompt that starts with the same text does not need that part recomputed. Under memory pressure, old entries are evicted.

**Profile.** The behaviour of a simulated agent: *paper* (re-reads all 5 files every turn), *realistic* (reads lazily, re-reads some files), *runaway* (stuck in an edit–test-fail loop for 100 turns).

**Rebase.** In Mode B, sending a file in full again after `max_diffs_before_rebase` diffs, so the diff chain restarts from zero.

**Simulated agent.** A scripted program that behaves like a coding agent (reads real files, makes real edits, runs real syntax checks) but does not use an LLM. Lets 20,000 sessions run cheaply and reproducibly.

**Token.** The unit LLMs read and write; roughly ¾ of an English word, or a few characters of code.

**Turn.** One request–response cycle between the agent and the model.

**Vault (HashVault).** The local content-addressed store holding every file version ever seen, so any stub can be turned back into the exact file.
