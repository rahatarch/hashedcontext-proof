"""
GPU serving simulator (one inference replica), vLLM-style:

  * Paged KV-cache: 16-token blocks, fixed pool = (HBM - weights) / block bytes
  * Automatic Prefix Caching: chained block hashes over REAL token IDs,
    longest-prefix match, ref-counted sharing, LRU eviction of unpinned blocks
  * Continuous batching, FIFO admission gated by free KV blocks
  * Roofline timing: prefill = FLOPs / (peak * MFU) incl. attention term;
    decode step = (weights + every running sequence's KV bytes) / HBM BW

Hardware/model constants are the paper's own (Sec. 2, 5):
  H100 80 GB, 3.35 TB/s, 70B-class GQA model in FP8, 30 GB weights,
  L=80, H_kv=8, d_h=128, b=1  ->  163,840 B of KV per token.

The same agent fleet (identical seeds -> identical behaviour) is replayed under
each context architecture; only the prompt sent to the engine differs.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import List, Optional

from hashedcontext.core import (FILE_CLOSE, FILE_OPEN, LIVE_CLOSE, LIVE_OPEN, render_stub,
                                sha256_hex)
from hashedcontext.lazy import DEFAULT_MAX_DIFFS, LazyContext
from sim.agent import SYSTEM_PROMPT, CodingAgent

BLOCK = 16


@dataclass
class HW:
    hbm: float = 80e9
    weights: float = 30e9
    kv_per_token: int = 2 * 80 * 8 * 128 * 1  # 163,840 B
    bw: float = 3.35e12
    peak_flops: float = 1979e12  # H100 FP8 dense
    mfu: float = 0.40
    params: float = 70e9
    L: int = 80
    d_model: int = 8192
    max_seqs: int = 256
    ctx_limit: int = 128_000

    @property
    def cap_blocks(self) -> int:
        return int((self.hbm - self.weights) // (self.kv_per_token * BLOCK))

    def prefill_time(self, new: int, ctx: int) -> float:
        flops = 2 * self.params * new + 4 * self.L * self.d_model * new * (ctx - new / 2)
        return flops / (self.peak_flops * self.mfu)

    def step_time(self, kv_tokens: int) -> float:
        return (self.weights + kv_tokens * self.kv_per_token) / self.bw


class Chain:
    """Incremental vLLM-style chained block hashing."""
    __slots__ = ("hashes", "carry", "last", "ntok")

    def __init__(self):
        self.hashes: List[int] = []
        self.carry: List[int] = []
        self.last = 0
        self.ntok = 0

    def copy(self) -> "Chain":
        c = Chain()
        c.hashes, c.carry, c.last, c.ntok = list(self.hashes), list(self.carry), self.last, self.ntok
        return c

    def extend(self, ids):
        self.ntok += len(ids)
        buf = self.carry + list(ids) if self.carry else list(ids)
        n = len(buf) // BLOCK * BLOCK
        h = self.last
        hs = self.hashes
        for i in range(0, n, BLOCK):
            h = hash((h, tuple(buf[i:i + BLOCK])))
            hs.append(h)
        self.last = h
        self.carry = buf[n:]


class KVPool:
    def __init__(self, cap: int, apc: bool):
        self.cap, self.apc = cap, apc
        self.ref = {}
        self.lru: "OrderedDict[int,None]" = OrderedDict()
        self.used = 0  # all allocated blocks (pinned + cached-unpinned + anonymous)
        self.hit_tokens = 0
        self.evictions = 0

    def match(self, hashes) -> int:
        if not self.apc:
            return 0
        n = 0
        for h in hashes:
            if h in self.ref:
                n += 1
            else:
                break
        return n

    def try_admit(self, req) -> bool:
        m = self.match(req.hashes)
        anon = (1 if req.carry else 0) + math.ceil(req.gen / BLOCK)
        need = len(req.hashes) - m + anon
        if need > self.cap:
            req.too_big = True
            return False
        free = self.cap - self.used
        if need > free:
            pinned_m = set(req.hashes[:m])
            evictable = sum(1 for h in self.lru if h not in pinned_m) if need - free > 0 else 0
            if need > free + evictable:
                return False
            while self.cap - self.used < need:
                for h in self.lru:
                    if h not in pinned_m:
                        break
                del self.lru[h]
                del self.ref[h]
                self.used -= 1
                self.evictions += 1
        for h in req.hashes:
            r = self.ref.get(h)
            if r is None:
                self.ref[h] = 1
                self.used += 1
            else:
                if r == 0:
                    self.lru.pop(h, None)
                self.ref[h] = r + 1
        self.used += anon
        req.anon = anon
        req.matched = m
        self.hit_tokens += m * BLOCK
        return True

    def release(self, req):
        for h in req.hashes:
            r = self.ref[h] - 1
            if r == 0:
                if self.apc:
                    self.ref[h] = 0
                    self.lru[h] = None
                else:
                    del self.ref[h]
                    self.used -= 1
            else:
                self.ref[h] = r
        self.used -= req.anon


@dataclass
class Req:
    agent: int
    hashes: list
    carry: list
    ntok: int
    gen: int
    t_arr: float
    aux: bool = False
    anon: int = 0
    matched: int = 0
    rem: int = 0
    too_big: bool = False
    t_start: float = 0.0


class AgentCtx:
    """Builds the token stream an agent sends, under a given context mode."""

    def __init__(self, agent: CodingAgent, mode: str, tok, hw: HW, max_diffs=DEFAULT_MAX_DIFFS):
        self.a, self.mode, self.tok, self.hw = agent, mode, tok, hw
        self.max_diffs = max_diffs
        self.k = 0
        self.reset()

    def reset(self):
        self.hist = Chain()
        self.hist.extend(self.tok.ids(SYSTEM_PROMPT))
        if self.mode == "lazy":  # Mode B: history is rebuilt from scratch, files shown in full again
            self.lz = LazyContext(self.max_diffs)

    def next_request(self, now: float) -> List[Req]:
        a, tok = self.a, self.tok
        self.k += 1
        t = a.step(self.k)
        gen = max(32, tok.count(a.generated_text))
        u_ids = tok.ids(t.text)
        reqs = []
        if self.mode == "conv":
            add = list(u_ids)
            for f in t.files:
                add += tok.ids(FILE_OPEN.format(path=f.path)) + tok.ids(f.content) + tok.ids(FILE_CLOSE)
            if self.hist.ntok + len(add) > self.hw.ctx_limit:  # post-hoc compaction
                reqs.append(Req(a.id, list(self.hist.hashes), list(self.hist.carry), self.hist.ntok,
                                2000, now, aux=True))
                summary = tok.enc.encode_ordinary(("SUMMARY of earlier work: " + t.text) * 400)[: int(0.3 * self.hw.ctx_limit)]
                self.reset()
                self.hist.extend(summary)
            self.hist.extend(add)
            reqs.append(Req(a.id, list(self.hist.hashes), list(self.hist.carry), self.hist.ntok, gen, now))
        elif self.mode == "lazy":
            # Mode B: append-only stream; unchanged re-reads -> stub, changes -> line diff
            add = list(u_ids)
            for f in t.files:
                add += tok.ids(self.lz.render(self.k, f.path, f.content)[1])
            if self.hist.ntok + len(add) > self.hw.ctx_limit * 0.9:
                self.reset()
                add = list(u_ids)
                for f in t.files:
                    add += tok.ids(self.lz.render(self.k, f.path, f.content)[1])
            self.hist.extend(add)
            reqs.append(Req(a.id, list(self.hist.hashes), list(self.hist.carry), self.hist.ntok, gen, now))
        else:
            compact = self.mode == "hc_compact"
            cur = self.hist.copy()
            cur.extend(u_ids)
            stubs = []
            for f in t.files:
                h = sha256_hex(f.content)
                cur.extend(tok.ids(LIVE_OPEN.format(path=f.path, h=h)) + tok.ids(f.content) + tok.ids(LIVE_CLOSE))
                stubs += tok.ids(render_stub(f.path, h, compact))
            reqs.append(Req(a.id, cur.hashes, cur.carry, cur.ntok, gen, now))
            self.hist.extend(list(u_ids) + stubs)
            if self.hist.ntok > self.hw.ctx_limit * 0.9:
                self.reset()
        return reqs


def simulate(mode: str, n_agents: int, horizon: float, tok, repos, hw: HW = HW(), apc: bool = True,
             seed: int = 11, session_P=(10, 60)):
    rng = random.Random(seed)
    pool = KVPool(hw.cap_blocks, apc)

    def new_agent(i, s):
        r = random.Random(s)
        u = r.random()
        prof = "realistic" if u < 0.4 else ("paper" if u < 0.8 else "runaway")
        M = 5 if prof == "paper" else (2 if prof == "runaway" else r.randint(3, 8))
        P = 100 if prof == "runaway" else r.randint(*session_P)
        ag = CodingAgent(i, repos[r.randrange(len(repos))], prof, r, M=M, P=P,
                         p_edit=0.5 if prof != "runaway" else 1.0,
                         min_file_tok=2000 if prof == "paper" else 0, tok=tok)
        return AgentCtx(ag, mode, tok, hw)

    seeds = [rng.randrange(1 << 30) for _ in range(n_agents)]
    gen_count = [0] * n_agents
    ctxs = [new_agent(i, seeds[i]) for i in range(n_agents)]
    arrivals = [(rng.uniform(0, 5), i) for i in range(n_agents)]
    heapq.heapify(arrivals)
    pending = {}  # agent -> remaining reqs in a multi-request turn
    queue: deque = deque()
    running: List[Req] = []
    t = 0.0
    st = dict(turns=0, aux=0, gen_tokens=0, prompt_tokens=0, prefill_tokens=0, lat=[], busy_prefill=0.0,
              batch_area=0.0, kv_area=0.0, rejected=0, queue_area=0.0, ttft=[])

    def arrive(i, now):
        c = ctxs[i]
        if c.k >= c.a.P:  # session finished -> start a new one with the next seed
            gen_count[i] += 1
            ctxs[i] = c = new_agent(i, seeds[i] * 1000003 + gen_count[i])
        rq = c.next_request(now)
        queue.append(rq[0])
        if len(rq) > 1:
            pending[i] = rq[1:]

    while t < horizon:
        while arrivals and arrivals[0][0] <= t:
            ta, i = heapq.heappop(arrivals)
            arrive(i, ta)
        admitted = []
        while queue and len(running) + len(admitted) < hw.max_seqs:
            r = queue[0]
            if pool.try_admit(r):
                queue.popleft()
                admitted.append(r)
            else:
                if r.too_big:
                    queue.popleft()
                    st["rejected"] += 1
                    continue
                break
        if admitted:
            dt = 0.0
            for r in admitted:
                new = r.ntok - r.matched * BLOCK
                st["prefill_tokens"] += new
                st["prompt_tokens"] += r.ntok
                dt += hw.prefill_time(new, r.ntok)
                r.rem = r.gen
            st["batch_area"] += len(running) * dt
            st["queue_area"] += len(queue) * dt
            st["kv_area"] += pool.used * dt
            st["busy_prefill"] += dt
            t += dt
            for r in admitted:
                st["ttft"].append(t - r.t_arr)
            running += admitted
            continue
        if not running:
            if not arrivals:
                break
            t = max(t, arrivals[0][0])
            continue
        kv_tok = sum(r.ntok + (r.gen - r.rem) for r in running)
        step = hw.step_time(kv_tok)
        steps = min(r.rem for r in running)
        if arrivals and arrivals[0][0] < t + steps * step:
            steps = max(1, math.ceil((arrivals[0][0] - t) / step))
        dt = steps * step
        st["batch_area"] += len(running) * dt
        st["queue_area"] += len(queue) * dt
        st["kv_area"] += pool.used * dt
        t += dt
        st["gen_tokens"] += steps * len(running)
        still = []
        for r in running:
            r.rem -= steps
            if r.rem <= 0:
                pool.release(r)
                if r.aux:
                    st["aux"] += 1
                    nxt = pending.pop(r.agent)
                    for q in nxt:
                        q.t_arr = t
                    queue.extend(nxt)  # follow-up turn request
                else:
                    st["turns"] += 1
                    st["lat"].append(t - r.t_arr)
                    heapq.heappush(arrivals, (t + rng.uniform(1.0, 6.0), r.agent))
            else:
                still.append(r)
        running = still
    lat = sorted(st["lat"]) or [0]
    ttft = sorted(st["ttft"]) or [0]
    T = max(t, 1e-9)
    return dict(mode=mode, apc=apc, agents=n_agents, horizon=horizon, sim_time=T,
                turns=st["turns"], turns_per_hour=st["turns"] / T * 3600,
                gen_tok_per_s=st["gen_tokens"] / T, aux_calls=st["aux"],
                mean_batch=st["batch_area"] / T, mean_queue=st["queue_area"] / T,
                mean_kv_util=st["kv_area"] / T / pool.cap,
                prefill_tokens=st["prefill_tokens"], prompt_tokens=st["prompt_tokens"],
                apc_hit_rate=pool.hit_tokens / max(1, st["prompt_tokens"]),
                prefill_busy_frac=st["busy_prefill"] / T, evictions=pool.evictions,
                lat_p50=lat[len(lat) // 2], lat_p95=lat[int(len(lat) * 0.95)],
                ttft_p50=ttft[len(ttft) // 2], ttft_p95=ttft[int(len(ttft) * 0.95)],
                rejected=st["rejected"], cap_blocks=pool.cap)
