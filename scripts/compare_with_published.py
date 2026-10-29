"""
Compare your own run (results/) with the published run (published-results/).

    python scripts/compare_with_published.py

What is checked:
  EXACT  deterministic outputs: every fleet session you ran (row by row, every column),
         the GPU serving simulator, the mechanical needle and diff-chain tests, the
         re-derived paper arithmetic, token totals of the gateway load test.
         These must match bit for bit. A mismatch means something differs (code,
         library versions, corpus) and is worth investigating.
  INFO   machine-dependent outputs (timings, throughput) and live-LLM results.
         These are shown side by side and are never counted as failures.

Only files present in results/ are compared, so a partial run is fine.
Exit code: 0 if no EXACT check failed, 1 otherwise.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MINE = os.path.join(ROOT, "results")
PUB = os.path.join(ROOT, "published-results")

GREEN, RED, DIM, BOLD, END = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = BOLD = END = ""

failures = 0
checks = 0


def report(status, name, detail=""):
    global failures, checks
    if status in ("PASS", "FAIL"):
        checks += 1
    if status == "FAIL":
        failures += 1
    color = {"PASS": GREEN, "FAIL": RED}.get(status, DIM)
    print(f"  {color}{status:5s}{END} {name}" + (f"  {DIM}{detail}{END}" if detail else ""))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def close(a, b, rel=1e-9):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


# ---------------------------------------------------------------- fleet
def read_fleet(path, sessions=None):
    rows = {}
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            parts = line.rstrip("\n").split(",")
            sid = int(parts[0])
            if sessions is not None and sid not in sessions:
                continue
            rows[(sid, int(parts[5]))] = parts
    return header, rows


def check_fleet():
    mine = os.path.join(MINE, "fleet_turns.csv.gz")
    if not os.path.exists(mine):
        return
    print(f"\n{BOLD}Agent fleet (E1){END}")
    h_m, rows_m = read_fleet(mine)
    sessions = {k[0] for k in rows_m}
    h_p, rows_p = read_fleet(os.path.join(PUB, "fleet_turns.csv.gz"), sessions)
    if h_m != h_p:
        report("FAIL", "CSV columns", f"yours: {len(h_m)} columns, published: {len(h_p)}")
        return
    missing = [k for k in rows_m if k not in rows_p]
    diff = [k for k in rows_m if k in rows_p and rows_m[k] != rows_p[k]]
    report("FAIL" if missing or diff else "PASS",
           f"{len(sessions):,} sessions, {len(rows_m):,} turns, every column",
           f"{len(diff)} rows differ, {len(missing)} rows not in published" if missing or diff else "identical to published rows")
    if diff:
        k = diff[0]
        cols = [h_m[i] for i, (a, b) in enumerate(zip(rows_m[k], rows_p[k])) if a != b]
        print(f"        first difference: session {k[0]} turn {k[1]}, columns {cols[:6]}")
    meta_m, meta_p = load(os.path.join(MINE, "fleet_meta.json")), load(os.path.join(PUB, "fleet_meta.json"))
    if meta_m and meta_p:
        if meta_m.get("sessions") == meta_p.get("sessions"):
            keys = ["turns", "file_observations", "observed_bytes", "edits", "live_ok", "live_total", "expand_ok",
                    "expand_total", "lazy_ok", "lazy_total", "lazy_diffs", "lazy_diffs_verified", "lazy_rebases",
                    "sigma_paper_mean", "sigma_lazy_mean"]
            bad = [k for k in keys if k in meta_p and not close(meta_m.get(k), meta_p[k])]
            report("FAIL" if bad else "PASS", "fleet summary (losslessness counters, diff counts, stub sizes)",
                   f"differs: {bad}" if bad else "")
        lo, lt = meta_m.get("lazy_ok"), meta_m.get("lazy_total")
        eo, et = meta_m.get("expand_ok"), meta_m.get("expand_total")
        if lt:
            report("PASS" if lo == lt else "FAIL", "Mode B: every sampled prompt rebuilds every file exactly", f"{lo:,}/{lt:,}")
        if et:
            report("PASS" if eo == et else "FAIL", "Mode A: every sampled prompt expands back byte-exact", f"{eo:,}/{et:,}")
        if meta_m.get("wall_seconds"):
            report("INFO", "fleet wall time", f"yours {meta_m['wall_seconds'] / 60:.1f} min on {meta_m.get('workers', '?')} workers"
                   f" ({meta_m.get('sessions'):,} sessions); published {meta_p['wall_seconds'] / 60:.1f} min on "
                   f"{meta_p.get('workers', '?')} workers ({meta_p['sessions']:,} sessions)")


# ---------------------------------------------------------------- serving
def check_serving():
    mine, pub = load(os.path.join(MINE, "serving_all.json")), load(os.path.join(PUB, "serving_all.json"))
    if not mine:
        return
    print(f"\n{BOLD}GPU serving simulator (E2){END}")
    key = lambda r: (r["kind"], r["mode"], r["apc"], r["agents"], r["seed"])
    P = {key(r): r for r in pub}
    fields = ["turns", "prompt_tokens", "prefill_tokens", "apc_hit_rate", "mean_batch", "lat_p95", "turns_per_hour"]
    bad, n = [], 0
    for r in mine:
        p = P.get(key(r))
        if p is None:
            bad.append((key(r), "not in published"))
            continue
        n += 1
        diffs = [f for f in fields if not close(r[f], p[f])]
        if diffs:
            bad.append((key(r), diffs))
    report("FAIL" if bad else "PASS", f"{n} simulator configurations (turns, tokens, cache hits, batch, latency)",
           f"{len(bad)} differ, first: {bad[0]}" if bad else "identical")


# ---------------------------------------------------------------- mechanical tests
def check_needle():
    mine, pub = load(os.path.join(MINE, "needle_mechanical.json")), load(os.path.join(PUB, "needle_mechanical.json"))
    if not mine:
        return
    print(f"\n{BOLD}Needle test, mechanical grader{END}")
    for c, x in mine.items():
        p = pub.get(c)
        if p is None:
            continue
        same_acc = all(close(x[k], p[k]) for k in ("path_acc", "line_acc", "text_acc"))
        if x["n"] == p["n"]:
            ok = same_acc and close(x["mean_prompt_tokens"], p["mean_prompt_tokens"])
            report("PASS" if ok else "FAIL", f"{c:15s} accuracy and prompt size",
                   f"file {x['path_acc']:.0%}, line {x['line_acc']:.0%}, text {x['text_acc']:.0%}")
        else:
            report("PASS" if same_acc else "FAIL", f"{c:15s} accuracy ({x['n']} cases; published used {p['n']})",
                   f"file {x['path_acc']:.0%}, line {x['line_acc']:.0%}, text {x['text_acc']:.0%}")


def check_chain_mechanical():
    for name in sorted(os.listdir(MINE)) if os.path.isdir(MINE) else []:
        if not (name.startswith("chain_mechanical") and name.endswith(".json")):
            continue
        r = load(os.path.join(MINE, name))
        print(f"\n{BOLD}Diff-chain harness, mechanical grader ({name}){END}")
        for c in ("conv", "hc", "lazy"):
            if c in r:
                report("PASS" if r[c]["accuracy"] == 1.0 else "FAIL", f"{c:5s} reader recovers the right line",
                       f"{r[c]['correct']}/{r[c]['n']}" + (f", {r[c]['diffs_to_apply_mean']:.0f} diffs to apply"
                                                          if "diffs_to_apply_mean" in r[c] else ""))


def check_paper_math():
    mine, pub = load(os.path.join(MINE, "paper_math.json")), load(os.path.join(PUB, "paper_math.json"))
    if not mine:
        return
    print(f"\n{BOLD}Re-derived paper arithmetic (E5){END}")
    report("PASS" if mine == pub else "FAIL", "every recomputed number", "identical" if mine == pub else "differs")


def check_micro():
    mine, pub = load(os.path.join(MINE, "micro.json")), load(os.path.join(PUB, "micro.json"))
    if not mine:
        return
    print(f"\n{BOLD}Micro-benchmarks (E4){END}")
    sm, sp = mine.get("sigma", {}), pub.get("sigma", {})
    ok = all(close(sm[t][k], sp[t][k]) for t in sp for k in ("paper_mean", "compact_mean", "paper_listing1") if t in sm)
    report("PASS" if ok else "FAIL", "stub sizes in tokens (the paper's example stub and real paths)",
           f"paper's example stub = {sm.get('cl100k_base', {}).get('paper_listing1')} tokens")
    try:
        a, b = mine["sha256"]["1048576"]["us_per_MB"], pub["sha256"]["1048576"]["us_per_MB"]
        report("INFO", "SHA-256 speed (depends on your CPU)", f"yours {a / 1000:.2f} ms/MB, published {b / 1000:.2f} ms/MB")
    except KeyError:
        pass


def check_gateway():
    mine, pub = load(os.path.join(MINE, "gateway_loadtest.json")), load(os.path.join(PUB, "gateway_loadtest.json"))
    if not mine:
        return
    print(f"\n{BOLD}HTTP gateway load test (E3){END}")
    for mode in ("baseline", "gateway", "edge"):
        if mode not in mine:
            continue
        tm = sum(x["prompt_tokens"] for x in mine[mode]["rows"])
        tp = sum(x["prompt_tokens"] for x in pub[mode]["rows"])
        report("PASS" if tm == tp else "FAIL", f"{mode:8s} backend prompt tokens", f"{tm:,}")
        report("INFO", f"{mode:8s} requests per second (depends on your machine)",
               f"yours {mine[mode]['rps']:.0f}, published {pub[mode]['rps']:.0f}")
    v = max(x["verified"] for x in mine["gateway"]["rows"]) if "gateway" in mine else None
    if v is not None:
        report("PASS" if v == mine["gateway"]["requests"] else "FAIL", "every gateway request verified lossless",
               f"{v}/{mine['gateway']['requests']}")


def check_live():
    for name in ("chain_llm_anchor.json", "chain_llm_anchor_rebase5.json", "chain_llm.json"):
        m = load(os.path.join(MINE, name))
        if not m:
            continue
        p = load(os.path.join(PUB, name)) or {}
        print(f"\n{BOLD}Live-LLM diff-chain test ({name}, model {m.get('grader')}){END}")
        for c in ("conv", "hc", "lazy"):
            if c in m:
                pp = p.get(c)
                report("INFO", f"{c:5s} accuracy", f"yours {m[c]['correct']}/{m[c]['n']}" +
                       (f", published {pp['correct']}/{pp['n']} ({p.get('grader')})" if pp else ""))


def main():
    if not os.path.isdir(MINE) or not os.listdir(MINE):
        print("No results/ yet. Run `make quick` or `make full` first (see docs/03-reproduce.md).")
        return 1
    print(f"{BOLD}Comparing results/ with published-results/{END}")
    check_fleet()
    check_serving()
    check_needle()
    check_chain_mechanical()
    check_paper_math()
    check_micro()
    check_gateway()
    check_live()
    print()
    if failures:
        print(f"{RED}{BOLD}{failures} of {checks} exact checks differ.{END} See docs/03-reproduce.md, 'If a check fails'.")
        return 1
    print(f"{GREEN}{BOLD}All {checks} exact checks match the published results.{END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
