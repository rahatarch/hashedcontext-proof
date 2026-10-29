"""
Needle-in-a-haystack fidelity test (paper Sec. 11.3), three prompt conditions:

  conv        full append-only history
  hc          HashedContext prompt only (stubs + live tail)
  hc_resolve  HashedContext + on-demand R(h): the previous version of every file
              whose live digest differs from its latest stub is resolved from the
              vault and injected for this single turn (paper Sec. 3.5)

A single line of one file is mutated externally (no diff in the narrative)
before the final turn. Task: name the file, the line number, the new text.

Two graders:
  --mechanical  deterministic reader that may use ONLY the prompt text
                (proves what information is present in each prompt)
  (default)     a real LLM via any OpenAI-compatible endpoint:
                HC_API_BASE (e.g. https://api.deepseek.com), HC_API_KEY, HC_MODEL

usage:
  python -m llm_eval.needle_eval --mechanical --cases 500
  HC_API_BASE=... HC_API_KEY=... HC_MODEL=deepseek-chat python -m llm_eval.needle_eval --cases 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request

from hashedcontext.core import (FILE_CLOSE, FILE_OPEN, HashVault, assemble_conventional, assemble_hashed,
                                oracle_read, sha256_hex)
from hashedcontext.lazy import BLOCK_RE, assemble_lazy, replay
from hashedcontext.tokens import Tokenizer
from sim.agent import SYSTEM_PROMPT, CodingAgent
from sim.corpus import load_repos

QUESTION = ("\nUSER: Before this turn, exactly one line in exactly one of the files above was modified by "
            "someone else (not by you). Identify it. Reply with JSON only: "
            '{"path": "<file path>", "line": <1-based line number>, "new_text": "<the new line, exact>"}\n')


def build_case(seed, repos, tok):
    r = random.Random(seed)
    P = r.randint(4, 20)
    a = CodingAgent(seed, repos[r.randrange(len(repos))], "paper", r, M=5, P=P, p_edit=0.3, min_file_tok=1500, tok=tok)
    turns = [a.step(k) for k in range(1, P)]
    a.pending_edit = None
    p = r.choice(a.working_set)
    lines = a.file(p).split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    i = r.choice(idx)
    tag = f"needle{r.randint(1000, 9999)}"
    lines[i] = lines[i] + (f"  # {tag}" if p.endswith(".py") else f" // {tag}")
    a.ws[p] = "\n".join(lines)
    last = a.step(P)
    last.text += QUESTION
    turns.append(last)
    return dict(turns=turns, truth=dict(path=p, line=i + 1, new_text=lines[i]))


def prompts(case):
    turns = case["turns"]
    v = HashVault()
    conv = "".join(assemble_conventional(SYSTEM_PROMPT, turns))
    hc = "".join(assemble_hashed(SYSTEM_PROMPT, turns, v))
    live, hist = oracle_read(hc)
    extra = []
    for p, body in live.items():
        if hist.get(p) and hist[p][-1] != sha256_hex(body):
            old = v.resolve(hist[p][-1])  # R(h), injected for this turn only
            extra.append(f"[RESOLVED {hist[p][-1][:16]}]\n" + FILE_OPEN.format(path=p) + old + FILE_CLOSE)
    i = hc.index("[LIVE_FILE]")
    hc_res = hc[:i] + "".join(extra) + hc[i:]
    # Mode B (Lazy Tail + line diff): default rebase after 10 diffs, and no rebasing
    lazy = assemble_lazy(SYSTEM_PROMPT, turns)
    lazy_inf = assemble_lazy(SYSTEM_PROMPT, turns, max_diffs_before_rebase=None)
    return dict(conv=conv, hc=hc, hc_resolve=hc_res, lazy=lazy, lazy_no_rebase=lazy_inf)


def chain_depth(prompt, path):
    """Diff blocks the reader must apply to the last full body of `path` (prompt text only)."""
    n = 0
    for m in BLOCK_RE.finditer(prompt):
        if m.group("fp") == path:
            n = 0
        elif m.group("dp") == path:
            n += 1
    return n


# ------------------------------------------------------------- mechanical grader
FILE_RE = re.compile(r'<file path="([^"]*)">\n(.*?)\n</file>\n', re.S)


def mechanical(cond, prompt):
    if cond.startswith("lazy"):
        # rebuild every file from the prompt text before and after the final turn
        cut = prompt.index(QUESTION)
        before, after = replay(prompt[:cut]), replay(prompt)
        for p, body in after.items():
            if p in before and before[p] != body:
                return _diffline(p, before[p], body)
        return None
    if cond == "conv":
        copies = {}
        for m in FILE_RE.finditer(prompt):
            copies.setdefault(m.group(1), []).append(m.group(2))
        for p, cs in copies.items():
            if len(cs) >= 2 and cs[-1] != cs[-2]:
                return _diffline(p, cs[-2], cs[-1])
        return None
    live, hist = oracle_read(prompt)
    changed = [p for p, b in live.items() if hist.get(p) and hist[p][-1] != sha256_hex(b)]
    if not changed:
        return None
    p = changed[0]
    if cond == "hc":
        return dict(path=p, line=None, new_text=None)  # knows WHICH file, cannot see old bytes
    old = {m.group(1): m.group(2) for m in FILE_RE.finditer(prompt)}.get(p)
    return _diffline(p, old, live[p]) if old is not None else dict(path=p, line=None, new_text=None)


def _diffline(p, a, b):
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return dict(path=p, line=i + 1, new_text=y)
    return dict(path=p, line=min(len(la), len(lb)) + 1, new_text=lb[-1])


# ------------------------------------------------------------- LLM grader
def chat_url():
    """Accept a base with or without a trailing /v1 (e.g. https://api.deepseek.com or https://host/v1)."""
    base = os.environ["HC_API_BASE"].rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def ask_llm(prompt, system=SYSTEM_PROMPT, retries=4, return_text=False):
    import time
    key, model = os.environ["HC_API_KEY"], os.environ.get("HC_MODEL", "deepseek-chat")
    body = json.dumps({"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(chat_url(), body,
                                     {"content-type": "application/json", "authorization": f"Bearer {key}"})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=600))
            break
        except Exception as e:  # transient API/network errors: retry with backoff; never print the key
            if attempt == retries:
                raise RuntimeError(f"API request failed after {retries + 1} attempts: {type(e).__name__}: {e}") from None
            time.sleep(2 ** attempt * 3)
    txt = data["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        ans = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        ans = None
    if return_text:
        return ans, data.get("usage", {}), txt
    return ans, data.get("usage", {})


def score(ans, truth):
    if not ans:
        return dict(path=False, line=False, text=False)
    return dict(path=ans.get("path") == truth["path"],
                line=ans.get("line") == truth["line"],
                text=(ans.get("new_text") or "").strip() == truth["new_text"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--mechanical", action="store_true")
    args = ap.parse_args()
    tok = Tokenizer()
    repos = load_repos(tok)
    conds = ("conv", "hc", "hc_resolve", "lazy", "lazy_no_rebase")
    agg = {c: dict(path=0, line=0, text=0, tokens=0, n=0, chain=[]) for c in conds}
    for s in range(args.cases):
        case = build_case(s, repos, tok)
        for cond, pr in prompts(case).items():
            if args.mechanical:
                ans, usage = mechanical(cond, pr), {}
            else:
                ans, usage = ask_llm(pr)
            sc = score(ans, case["truth"])
            a = agg[cond]
            a["n"] += 1
            a["tokens"] += usage.get("prompt_tokens") or tok.count(pr)
            for k in ("path", "line", "text"):
                a[k] += int(sc[k])
            if cond.startswith("lazy"):
                a["chain"].append(chain_depth(pr, case["truth"]["path"]))
    res = {c: dict(n=a["n"], path_acc=a["path"] / a["n"], line_acc=a["line"] / a["n"], text_acc=a["text"] / a["n"],
                   mean_prompt_tokens=a["tokens"] / a["n"],
                   **({"diffs_to_apply_mean": sum(a["chain"]) / len(a["chain"]), "diffs_to_apply_max": max(a["chain"])}
                      if a["chain"] else {})) for c, a in agg.items()}
    print(json.dumps(res, indent=1))
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "needle_mechanical.json" if args.mechanical else "needle_llm.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
