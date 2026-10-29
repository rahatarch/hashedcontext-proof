# 09 — Limitations

What this evidence shows, what it does not, and why. Read this before quoting any number.

## The agents are scripted, not LLMs

The 20,000 agents are programs that behave like coding agents: they read real files, make real edits and run real checks. They do not think.

- **What this is good for:** everything that depends only on *what is sent*: token counts, memory, cache reuse, payload sizes, costs, losslessness. These are exact.
- **What it cannot show:** whether a real model *reasons* as well from a HashedContext prompt as from a full one. Agent behaviour (how often files are re-read, how big edits are) is modelled, and real agents may differ.

## The GPUs are simulated

The serving results come from a simulator using an NVIDIA H100's published memory size, bandwidth and compute, and vLLM-style memory management. It captures the first-order effects (memory limits batch size; the prefix cache saves recomputation; eviction under pressure), but not everything a production server does (kernel details, scheduling policies, network, multi-GPU communication). Treat throughput numbers as well-founded estimates, not hardware measurements.

## The live-LLM test is small

One model (`deepseek-v4.1-flash`), 30 questions per layout. That can detect large problems but not small ones: Mode A's 28/30 and Mode B's 27/30 have overlapping 95% ranges (about 79–98% and 74–97%). Other models, longer chains, other question types and other tasks are untested. An earlier version of the question ("what is line N?") turned out to measure line counting, which shows how easily such tests can measure the wrong thing.

## Mode B's diffs are line-based

Diffs list changed lines by position, not syntax-aware changes. Lines are located only by line number, which is precisely the kind of reasoning models can find hard over long chains; hence the rebase setting.

## The workload is coding on four repositories

Python (flask), JavaScript (express, axios) and TypeScript (zod). Other languages, much larger files, or non-coding agents may behave differently, though the mechanism does not depend on the language.

## The "ideal cache" accounting is an idealisation

The warm-cache compute numbers in the fleet results assume each session has its own prefix cache that never evicts. This is the *best case* for cache reuse. The serving simulator shows what happens under real memory pressure, where reuse largely disappears for every layout.

## The pilot session in the paper is a single run

The 57.7% reduction measured on DeepSeek V4 comes from one session per layout. It illustrates the mechanism, not an effect size. Its raw API logs are not part of this package.

## What would make the evidence stronger

- Live-LLM runs with hundreds of cases, several models, and real agent tasks (not only targeted questions).
- Measurements on physical GPUs with a production server (vLLM or SGLang).
- Real agent traces instead of scripted agents.

Contributions in any of these directions are very welcome.
