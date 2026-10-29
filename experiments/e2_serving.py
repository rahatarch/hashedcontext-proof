"""E2 - GPU serving capacity: same agent fleet, three context architectures.
usage: python -m experiments.e2_serving"""
import json
import multiprocessing as mp
import os
import sys
import time

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
HORIZON = float(os.environ.get("HC_HORIZON", 3600))
_G = {}


def _init():
    from hashedcontext.tokens import Tokenizer
    from sim.corpus import load_repos
    _G["tok"] = Tokenizer()
    _G["repos"] = load_repos(_G["tok"])


def _run(cfg):
    from sim.serving import simulate
    t = time.time()
    r = simulate(cfg["mode"], cfg["agents"], HORIZON, _G["tok"], _G["repos"], apc=cfg["apc"], seed=cfg["seed"])
    r.update(cfg, wall=time.time() - t)
    return r


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = []
    if which in ("all", "sweep"):
        for apc in (True, False):
            for n in (1, 2, 4, 8, 16, 32, 64):
                for mode in ("conv", "hc", "hc_compact", "lazy"):
                    cfgs.append(dict(kind="sweep", mode=mode, apc=apc, agents=n, seed=11))
    if which in ("all", "cluster"):
        for rep in range(16):  # 16 GPU replicas x 32 agents = 512 concurrent agents
            for mode in ("conv", "hc", "lazy"):
                cfgs.append(dict(kind="cluster", mode=mode, apc=True, agents=32, seed=1000 + rep))
    res = []
    with mp.Pool(int(os.environ.get("HC_WORKERS", max(1, (os.cpu_count() or 2) - 2))), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, cfgs)):
            res.append(r)
            print(f"[{i+1}/{len(cfgs)}] {r['kind']} {r['mode']:10s} apc={r['apc']!s:5s} agents={r['agents']:3d} "
                  f"turns/h={r['turns_per_hour']:8.1f} p95={r['lat_p95']:7.1f}s hit={r['apc_hit_rate']:.2f} "
                  f"wall={r['wall']:.0f}s", flush=True)
    json.dump(res, open(os.path.join(OUT, f"serving_{which}.json"), "w"), indent=1)
