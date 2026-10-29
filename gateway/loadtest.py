"""
E3 - gateway load test over real HTTP.

Spawns backend + gateway processes, then runs N concurrent unmodified agent
clients (OpenAI chat format, full append-only history every turn).
Runs twice: gateway in passthrough mode (baseline) and hashed mode.

usage: python -m gateway.loadtest [clients] [turns]
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import time

from aiohttp import ClientSession, ClientTimeout

from hashedcontext.tokens import Tokenizer
from sim.agent import SYSTEM_PROMPT, CodingAgent
from sim.corpus import load_repos

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
PY = sys.executable


def wait_port(port):
    import socket
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(port)


async def client(cid, agent: CodingAgent, url, sess, rows, edge=False):
    from hashedcontext.core import HashVault, gateway_transform
    vault = HashVault()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for k in range(1, agent.P + 1):
        t = agent.step(k)
        calls = [{"id": f"c{cid}_{k}_{j}", "type": "function",
                  "function": {"name": "read_file", "arguments": json.dumps({"path": f.path})}}
                 for j, f in enumerate(t.files)]
        msgs.append({"role": "assistant", "content": t.text, **({"tool_calls": calls} if calls else {})})
        for c, f in zip(calls, t.files):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f.content})
        send = gateway_transform(msgs, vault) if edge else msgs  # edge-native mode: HC in the client
        body = json.dumps({"model": "agent-70b", "messages": send}).encode()
        t0 = time.perf_counter()
        async with sess.post(url, data=body, headers={"content-type": "application/json"}) as r:
            data = await r.json()
            h = r.headers
        rows.append(dict(client=cid, k=k, rtt_ms=(time.perf_counter() - t0) * 1e3,
                         in_bytes=int(h["x-hc-in-bytes"]), out_bytes=int(h["x-hc-out-bytes"]),
                         gw_us=float(h["x-hc-gw-us"]), parse_us=float(h["x-hc-parse-us"]),
                         xform_us=float(h["x-hc-xform-us"]), verify_us=float(h["x-hc-verify-us"]),
                         ser_us=float(h["x-hc-ser-us"]), verified=int(h["x-hc-verified"]),
                         vault_objects=int(h["x-hc-vault-objects"]), vault_bytes=int(h["x-hc-vault-bytes"]),
                         prompt_tokens=data["usage"]["prompt_tokens"]))


async def run_mode(name, n_clients, turns, repos, tok):
    mode = {"baseline": "passthrough", "gateway": "hashed", "edge": "passthrough"}[name]
    edge = name == "edge"
    bport, gport = 18001, 18002
    env = dict(os.environ, PYTHONPATH=os.path.dirname(OUT))
    b = subprocess.Popen([PY, "-m", "gateway.servers", "backend", str(bport)], env=env)
    g = subprocess.Popen([PY, "-m", "gateway.servers", "gateway", str(gport),
                          f"http://127.0.0.1:{bport}/v1/chat/completions", mode], env=env)
    try:
        wait_port(bport)
        wait_port(gport)
        rows = []
        agents = [CodingAgent(i, repos[i % len(repos)], "paper", random.Random(500 + i), M=5, P=turns,
                              p_edit=0.5, min_file_tok=2000, tok=tok) for i in range(n_clients)]
        t0 = time.perf_counter()
        async with ClientSession(timeout=ClientTimeout(total=3600)) as sess:
            await asyncio.gather(*[client(i, a, f"http://127.0.0.1:{gport}/v1/chat/completions", sess, rows, edge)
                                   for i, a in enumerate(agents)])
        wall = time.perf_counter() - t0
        return dict(mode=name, clients=n_clients, turns=turns, wall_s=wall, requests=len(rows),
                    rps=len(rows) / wall, rows=rows)
    finally:
        for p in (g, b):
            p.terminate()
            p.wait()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tok = Tokenizer()
    repos = load_repos(tok)
    res = {}
    for mode in ("baseline", "gateway", "edge"):
        r = asyncio.run(run_mode(mode, n, turns, repos, tok))
        rows = r["rows"]
        print(f"{mode:12s} requests={r['requests']} wall={r['wall_s']:.1f}s rps={r['rps']:.1f} "
              f"in={sum(x['in_bytes'] for x in rows)/1e9:.2f}GB out={sum(x['out_bytes'] for x in rows)/1e9:.3f}GB "
              f"backend_tokens={sum(x['prompt_tokens'] for x in rows)/1e6:.1f}M "
              f"verified={max(x['verified'] for x in rows)}", flush=True)
        res[mode] = r
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gateway_loadtest.json"), "w"))


if __name__ == "__main__":
    main()
