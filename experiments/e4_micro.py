"""E4 - micro-measurements: SHA-256 cost, stub size sigma, JSON payload bytes/parse time.
usage: python -m experiments.e4_micro"""
import hashlib
import json
import os
import random
import time

import numpy as np

from hashedcontext.core import HashVault, gateway_transform, render_stub, sha256_hex
from hashedcontext.tokens import Tokenizer
from sim.agent import SYSTEM_PROMPT, CodingAgent
from sim.corpus import load_repos

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def bench(fn, reps):
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def main():
    res = {}
    # 1) SHA-256 throughput
    sha = {}
    for size in (4 << 10, 64 << 10, 1 << 20, 32 << 20):
        buf = os.urandom(size)
        reps = max(3, (64 << 20) // size)
        dt = bench(lambda: hashlib.sha256(buf).digest(), reps)
        sha[str(size)] = dict(seconds=dt, us_per_MB=dt / (size / 2**20) * 1e6, GB_per_s=size / dt / 1e9)
    res["sha256"] = sha

    # 2) stub size sigma over every corpus path, two tokenizers
    sig = {}
    for name in ("cl100k_base", "o200k_base"):
        tok = Tokenizer(name)
        repos = load_repos(tok)
        paths = [p for r in repos for p in r.candidates]
        hs = [sha256_hex(p + str(i)) for i, p in enumerate(paths)]
        paper = [tok.count(render_stub(p, h)) for p, h in zip(paths, hs)]
        comp = [tok.count(render_stub(p, h, True)) for p, h in zip(paths, hs)]
        hexonly = [tok.count(h) for h in hs]
        sig[name] = dict(paper_mean=float(np.mean(paper)), paper_p5_p95=np.percentile(paper, [5, 95]).tolist(),
                         compact_mean=float(np.mean(comp)), hex64_mean=float(np.mean(hexonly)), n_paths=len(paths),
                         paper_listing1=tok.count("[HASH_STUB]\npath: src/auth/session.ts\nsha256: e4b2c91f7a62d88d3e91b40c21a7f05b"
                                                  "8823190dfca91283c4b58e99128741aa\nstate: CACHED\n[/HASH_STUB]"))
    res["sigma"] = sig

    # 3) JSON payload size / parse time per turn (paper profile, M=5)
    tok = Tokenizer()
    repos = load_repos(tok)
    try:
        import orjson
    except ImportError:
        orjson = None
    agent = CodingAgent(0, repos[2], "paper", random.Random(3), M=5, P=30, p_edit=0.5, min_file_tok=2000, tok=tok)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows = []
    for k in range(1, 31):
        t = agent.step(k)
        calls = [{"id": f"c{k}_{j}", "type": "function",
                  "function": {"name": "read_file", "arguments": json.dumps({"path": f.path})}}
                 for j, f in enumerate(t.files)]
        msgs.append({"role": "assistant", "content": t.text, "tool_calls": calls})
        for c, f in zip(calls, t.files):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f.content})
        conv = json.dumps({"model": "m", "messages": msgs}).encode()
        hcm = gateway_transform(msgs, HashVault())
        hc = json.dumps({"model": "m", "messages": hcm}).encode()
        ntok_conv = sum(tok.count(m["content"]) for m in msgs)
        ntok_hc = sum(tok.count(m["content"]) for m in hcm)
        reps = 5 if len(conv) > 1e6 else 20
        row = dict(k=k, conv_bytes=len(conv), hc_bytes=len(hc), conv_tokens=ntok_conv, hc_tokens=ntok_hc,
                   json_conv_ms=bench(lambda: json.loads(conv), reps) * 1e3,
                   json_hc_ms=bench(lambda: json.loads(hc), reps) * 1e3)
        if orjson:
            row.update(orjson_conv_ms=bench(lambda: orjson.loads(conv), reps) * 1e3,
                       orjson_hc_ms=bench(lambda: orjson.loads(hc), reps) * 1e3)
        hb = b"".join(m["content"].encode() for m in msgs if m["role"] == "tool")
        row["sha_all_file_bytes_ms"] = bench(lambda: hashlib.sha256(hb).digest(), 3) * 1e3
        rows.append(row)
    res["payload"] = rows
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "micro.json"), "w"), indent=1)
    print(json.dumps({"sha256": sha, "sigma": sig}, indent=1))
    for r in rows[::5] + [rows[-1]]:
        print(r)


if __name__ == "__main__":
    main()
