"""
Real HTTP deployment of HashedContext in Gateway-Native mode (paper Sec. 10.2).

    unmodified agent client --(full OpenAI JSON)--> GATEWAY --(deduplicated)--> BACKEND

backend : OpenAI-compatible /v1/chat/completions; counts prompt tokens with a
          real BPE tokenizer and returns `usage` (stands in for vLLM/SGLang).
gateway : parses the JSON body, applies gateway_transform (SHA-256 hash-gate,
          vault, stubs, live tail), verifies losslessness of EVERY request with
          gateway_expand, forwards, and reports timings in response headers.
          mode=passthrough forwards the body untouched (baseline).
"""
from __future__ import annotations

import json
import sys
import time

from aiohttp import ClientSession, ClientTimeout, web

from hashedcontext.core import HashVault, gateway_expand, gateway_transform
from hashedcontext.tokens import Tokenizer


def render_messages(msgs):
    parts = []
    for m in msgs:
        parts.append(f"<|{m['role']}|>\n")
        if m.get("content"):
            parts.append(m["content"])
        for tc in m.get("tool_calls") or []:
            parts.append(f"\n<tool_call>{tc['function']['name']}({tc['function']['arguments']})</tool_call>")
        parts.append("\n")
    return parts


def make_backend():
    tok = Tokenizer()

    async def chat(req: web.Request):
        body = await req.read()
        data = json.loads(body)
        n = sum(tok.count(p) for p in render_messages(data["messages"]))
        return web.json_response({
            "id": "cmpl", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1},
        })

    app = web.Application(client_max_size=1 << 30)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def make_gateway(backend_url: str, mode: str):
    vault = HashVault()
    stats = dict(requests=0, verified=0)

    async def on_start(app):
        app["s"] = ClientSession(timeout=ClientTimeout(total=600))

    async def on_stop(app):
        await app["s"].close()

    async def chat(req: web.Request):
        body = await req.read()
        t0 = time.perf_counter()
        if mode == "passthrough":
            # a production gateway still deserialises + validates the body (paper Sec. 9)
            data = json.loads(body)
            assert isinstance(data.get("messages"), list)
            out = body
            t_parse, t_xform, t_verify, t_ser = time.perf_counter() - t0, 0.0, 0.0, 0.0
        else:
            data = json.loads(body)
            t1 = time.perf_counter()
            msgs = gateway_transform(data["messages"], vault)
            t2 = time.perf_counter()
            ok = gateway_expand(msgs, vault) == data["messages"]  # lossless check, every request
            t3 = time.perf_counter()
            stats["requests"] += 1
            stats["verified"] += int(ok)
            if not ok:
                return web.Response(status=500, text="lossless check failed")
            out = json.dumps({**data, "messages": msgs}).encode()
            t4 = time.perf_counter()
            t_parse, t_xform, t_verify, t_ser = t1 - t0, t2 - t1, t3 - t2, t4 - t3
        t_gw = time.perf_counter() - t0
        async with app["s"].post(backend_url, data=out, headers={"content-type": "application/json"}) as r:
            resp = await r.read()
        return web.Response(body=resp, content_type="application/json", headers={
            "x-hc-in-bytes": str(len(body)), "x-hc-out-bytes": str(len(out)),
            "x-hc-gw-us": f"{t_gw*1e6:.1f}", "x-hc-parse-us": f"{t_parse*1e6:.1f}",
            "x-hc-xform-us": f"{t_xform*1e6:.1f}", "x-hc-verify-us": f"{t_verify*1e6:.1f}",
            "x-hc-ser-us": f"{t_ser*1e6:.1f}", "x-hc-verified": str(stats["verified"]),
            "x-hc-vault-objects": str(len(vault)), "x-hc-vault-bytes": str(vault.unique_bytes)})

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_post("/v1/chat/completions", chat)
    return app


if __name__ == "__main__":
    role, port = sys.argv[1], int(sys.argv[2])
    if role == "backend":
        web.run_app(make_backend(), port=port, print=None)
    else:
        web.run_app(make_gateway(sys.argv[3], sys.argv[4]), port=port, print=None)
