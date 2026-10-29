"""
Diff-chain fidelity test for Mode B (Lazy Tail + line diff).

Each case is a stuck agent that edits the same file every turn, so by the last
turn the file has a chain of `--diffs` line diffs since its last full body
(default 10, the rebase bound). The question asks for the exact CURRENT text of
a line that was changed by an earlier edit, so under Mode B the reader must
apply the whole chain (line numbers shift as lines are inserted and removed).

Conditions:
  conv  append-only (every read is a full copy)
  hc    Mode A (stubs + one complete current copy at the tail)
  lazy  Mode B (first full body + diffs; rebase after 10)

Graders:
  --mechanical  deterministic reader using only the prompt text (checks the harness;
                a program applies any chain perfectly, so this is NOT the fidelity result)
  (default)     a live LLM via any OpenAI-compatible endpoint:
                HC_API_BASE, HC_API_KEY, HC_MODEL  (same as needle_eval)

usage:
  python -m llm_eval.chain_eval --mechanical --cases 100
  HC_API_BASE=https://api.deepseek.com HC_API_KEY=sk-... HC_MODEL=deepseek-chat \\
    python -m llm_eval.chain_eval --cases 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from hashedcontext.core import HashVault, assemble_conventional, assemble_hashed, oracle_read
from hashedcontext.lazy import BLOCK_RE, assemble_lazy, replay
from hashedcontext.tokens import Tokenizer
from llm_eval.needle_eval import ask_llm
from sim.agent import SYSTEM_PROMPT, CodingAgent

# Each prompt layout is described accurately to the model, so a low score reflects the layout,
# not misleading instructions.
_AGENT = ("You are an autonomous software engineering agent. You can call tools: "
          "read_file(path), edit_file(path, diff), run_tests(target). Work step by step, "
          "keep edits minimal, verify each change with tests, and explain your reasoning. ")
SYSTEMS = {
    "conv": _AGENT + ("File contents are provided inside <file> blocks. If a file appears several "
                      "times, the last copy is its current content.\n"),
    "hc": SYSTEM_PROMPT,  # Mode A: stubs + [LIVE_FILE] blocks at the end
    "lazy": _AGENT + (
        "Files are provided in three forms. A [FILE] block contains a file's full content. "
        "A [HASH_STUB] block saying 'Turn X content is still current' means the file has not changed "
        "since turn X. A [FILE_DIFF] block lists changes relative to the previous version of that file: "
        "each hunk header '@@ i,n @@' means replace the n lines starting at line i (1-based line numbers "
        "of the previous version) with the lines marked '+'; lines marked '-' are the lines removed; "
        "n=0 means insert before line i. To get a file's current content, start from its last [FILE] "
        "block and apply every later [FILE_DIFF] block for that path, in order.\n"),
}
from sim.corpus import load_repos

QUESTION = ("\nUSER: Report the exact current text of line {line} of {path}, as the file is now "
            "(after every edit so far). Reply with JSON only: {{\"text\": \"<the line, exact>\"}}\n")
# --question anchor: no line counting needed when the full current file is visible
QUESTION_ANCHOR = ("\nUSER: In the current version of {path} (after every edit so far), find the line that is "
                   "exactly `{anchor}`. What is the exact text of the line immediately after it? "
                   "Reply with JSON only: {{\"text\": \"<the line, exact>\"}}\n")
FILE_RE = re.compile(r'<file path="([^"]*)">\n(.*?)\n</file>\n', re.S)


def build_case(seed, repos, tok, n_diffs=10, question="line"):
    r = random.Random(seed)
    for attempt in range(50):
        repo = repos[r.randrange(len(repos))]
        # runaway agents edit ws[0] at the start of every turn from turn 2 on and re-read it each turn,
        # so after P turns the file carries P-1 diffs since its first full body
        a = CodingAgent(seed * 100 + attempt, repo, "runaway", r, M=2, P=n_diffs + 1, p_edit=1.0, tok=tok)
        turns = [a.step(k) for k in range(1, n_diffs + 2)]
        p = a.working_set[0]
        orig, cur = repo.files[p].split("\n"), a.file(p).split("\n")
        changed = [i for i, l in enumerate(cur) if l not in set(orig) and l.strip()]
        if question == "anchor":  # the line before the target must be non-blank and unique in the file
            changed = [i for i in changed if i > 0 and cur[i - 1].strip() and cur.count(cur[i - 1]) == 1
                       and "`" not in cur[i - 1]]
        if changed:
            i = r.choice(changed)
            q = (QUESTION_ANCHOR.format(anchor=cur[i - 1], path=p) if question == "anchor"
                 else QUESTION.format(line=i + 1, path=p))
            turns[-1].text += q
            return dict(turns=turns, truth=dict(path=p, line=i + 1, text=cur[i],
                                                anchor=cur[i - 1] if question == "anchor" else None))
    raise RuntimeError("no changed line found")


def prompts(case, rebase=10):
    turns = case["turns"]
    return dict(conv="".join(assemble_conventional(SYSTEMS["conv"], turns)),
                hc="".join(assemble_hashed(SYSTEMS["hc"], turns, HashVault())),
                lazy=assemble_lazy(SYSTEMS["lazy"], turns, max_diffs_before_rebase=rebase))


def chain_depth(prompt, path):
    n = 0
    for m in BLOCK_RE.finditer(prompt):
        if m.group("fp") == path:
            n = 0
        elif m.group("dp") == path:
            n += 1
    return n


def mechanical(cond, prompt, truth):
    p, line = truth["path"], truth["line"]
    if cond == "lazy":
        body = replay(prompt).get(p)
    elif cond == "hc":
        body = oracle_read(prompt)[0].get(p)
    else:
        copies = [b for q, b in FILE_RE.findall(prompt) if q == p]
        body = copies[-1] if copies else None
    if body is None:
        return None
    lines = body.split("\n")
    if truth.get("anchor") is not None:  # find the anchor by text, return the next line
        idx = [i for i, l in enumerate(lines) if l == truth["anchor"]]
        return {"text": lines[idx[0] + 1]} if len(idx) == 1 and idx[0] + 1 < len(lines) else None
    return {"text": lines[line - 1]} if 0 < line <= len(lines) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--diffs", type=int, default=10)
    ap.add_argument("--mechanical", action="store_true")
    ap.add_argument("--rebase", type=int, default=10, help="Mode B max_diffs_before_rebase")
    ap.add_argument("--question", choices=("line", "anchor"), default="line",
                    help="line: text of line N (needs line counting); anchor: line after a unique line")
    ap.add_argument("--conds", default="conv,hc,lazy", help="comma-separated subset of conv,hc,lazy")
    args = ap.parse_args()
    tok = Tokenizer()
    repos = load_repos(tok)
    agg = {c: dict(ok=0, n=0, tokens=0, chain=[], errors=0) for c in ("conv", "hc", "lazy")}
    tag = ("mechanical" if args.mechanical else "llm") + ("_anchor" if args.question == "anchor" else "")
    suffix = "" if args.rebase == 10 else f"_rebase{args.rebase}"
    cases_path = os.path.join(os.path.dirname(__file__), "..", "results", f"chain_{tag}{suffix}_cases.jsonl")
    conds = set(args.conds.split(","))
    log = open(cases_path, "w")
    for s in range(args.cases):
        case = build_case(s, repos, tok, args.diffs, args.question)
        for cond, pr in prompts(case, args.rebase).items():
            if cond not in conds:
                continue
            raw = None
            a = agg[cond]
            if args.mechanical:
                ans, usage = mechanical(cond, pr, case["truth"]), {}
            else:
                try:
                    ans, usage, raw = ask_llm(pr, system=SYSTEMS[cond], return_text=True)
                except RuntimeError as e:  # API failure: excluded from accuracy, reported separately
                    a["errors"] += 1
                    log.write(json.dumps(dict(case=s, cond=cond, error=str(e)[:300])) + "\n")
                    print(f"case {s} {cond}: API error", flush=True)
                    continue
            a["n"] += 1
            a["tokens"] += usage.get("prompt_tokens") or tok.count(pr)
            ok = bool(ans) and isinstance(ans, dict) and str(ans.get("text") or "").strip() == case["truth"]["text"].strip()
            a["ok"] += int(ok)
            log.write(json.dumps(dict(case=s, cond=cond, ok=ok, truth=case["truth"], answer=ans,
                                      raw=(raw or "")[:2000])) + "\n")
            log.flush()
            if not args.mechanical:
                print(f"case {s:2d} {cond:5s} {'OK ' if ok else 'BAD'}", flush=True)
            if cond == "lazy":
                a["chain"].append(chain_depth(pr, case["truth"]["path"]))
    log.close()
    res = {c: dict(n=a["n"], correct=a["ok"], api_errors=a["errors"],
                   accuracy=a["ok"] / a["n"] if a["n"] else None,
                   mean_prompt_tokens=a["tokens"] / a["n"] if a["n"] else None,
                   **({"diffs_to_apply_mean": sum(a["chain"]) / len(a["chain"]),
                       "diffs_to_apply_min": min(a["chain"]), "diffs_to_apply_max": max(a["chain"])}
                      if a["chain"] else {})) for c, a in agg.items() if c in conds}
    res["grader"] = "mechanical" if args.mechanical else os.environ.get("HC_MODEL", "llm")
    res["settings"] = dict(cases=args.cases, diffs=args.diffs, rebase=args.rebase, question=args.question)
    print(json.dumps(res, indent=1))
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"chain_{tag}{suffix}.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
