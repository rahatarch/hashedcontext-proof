"""Dual-mode trade-off tables (Mode A: Authoritative Tail, Mode B: Lazy Tail + line diff).
Reads results/fleet_turns.csv.gz, fleet_meta.json, serving_all.json, needle_mechanical.json.
Writes results/tradeoff_tables.md and results/tradeoff_tables.tex.
usage: python -m experiments.tradeoff_tables"""
import json
import os

import numpy as np

from experiments.analyze import load_fleet

R = os.path.join(os.path.dirname(__file__), "..", "results")
KV = 163_840
POOL_GB = 19073 * 16 * KV / 1e9


def pct(x):
    return f"{100 * x:+.1f}%"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def _load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def live_results():
    """Rows for the live diff-chain table, from results/chain_llm*.json (None if not run)."""
    anc, anc5, line = _load("chain_llm_anchor.json"), _load("chain_llm_anchor_rebase5.json"), _load("chain_llm.json")
    if not anc:
        return None
    rows = []
    for q, r in (("Line after anchor", anc), ("Line number", line)):
        if not r:
            continue
        rows += [(q, "Append-only", "0", r["conv"]["correct"], r["conv"]["n"]),
                 (q, "Mode A", "0", r["hc"]["correct"], r["hc"]["n"]),
                 (q, "Mode B, N=10", f"{r['lazy']['diffs_to_apply_mean']:.0f}", r["lazy"]["correct"], r["lazy"]["n"])]
        if q == "Line after anchor" and anc5:
            rows.append((q, "Mode B, N=5", f"{anc5['lazy']['diffs_to_apply_mean']:.0f}",
                         anc5["lazy"]["correct"], anc5["lazy"]["n"]))
    return dict(model=anc["grader"], rows=rows)


def live_evidence():
    live = live_results()
    if not live:
        return "A always shows one complete current copy; B needs up to 10 diffs applied, not yet tested with a live model"
    got = {lab: (k, n) for q, lab, _, k, n in live["rows"] if q == "Line after anchor"}
    f = lambda lab: f"{got[lab][0] / got[lab][1]:.1%}" if lab in got else "n/a"
    return (f"Live test, line after anchor (Table G): A {f('Mode A')}, B {f('Mode B, N=10')} with 10 diffs, "
            f"{f('Mode B, N=5')} with N=5; differences of 1--2 of 30 cases, within noise")


def fleet_rows():
    d = load_fleet()
    prof = d["profile"]
    meta = json.load(open(os.path.join(R, "fleet_meta.json")))
    out = {}
    for name, m in [("all", np.ones(len(prof), bool))] + [(p, prof == p) for p in ("paper", "realistic", "runaway")]:
        n = m.sum()
        tot = {c: d[c][m].sum() for c in ("conv", "conv_with_compaction", "hc_paper_stub", "lazy", "lazy_inf")}
        new = dict(conv=(d["conv"][m] - d["conv_cache_hit"][m]).sum(),
                   hc_paper_stub=(d["hc_paper_stub"][m] - d["hc_cache_hit"][m]).sum(),
                   lazy=(d["lazy"][m] - d["lazy_cache_hit"][m]).sum(),
                   lazy_inf=(d["lazy_inf"][m] - d["lazy_inf_cache_hit"][m]).sum(),
                   # compaction baseline: its own cache misses plus the summarization calls' input
                   conv_with_compaction=(d["conv_with_compaction"][m] - d["eff_cache_hit"][m]).sum() + d["aux_in"][m].sum())
        out[name] = dict(n=int(n), tot=tot, new=new,
                         kv={c: v / n * KV / 1e9 for c, v in tot.items()})
    return out, meta


def main():
    F, meta = fleet_rows()
    sv = json.load(open(os.path.join(R, "serving_all.json")))
    nd = json.load(open(os.path.join(R, "needle_mechanical.json")))
    a = F["all"]
    md, tex = [], []

    # ---------------- Table 1: fleet tokens / memory / compute
    rows = [("Append-only", "conv"), ("Append-only + compaction", "conv_with_compaction"),
            ("Mode A: Authoritative Tail", "hc_paper_stub"), ("Mode B: Lazy Tail, rebase after 10 diffs", "lazy"),
            ("Mode B without rebasing", "lazy_inf")]
    md.append("### Table A. Memory and compute across 20,000 simulated sessions (824,728 turns)\n")
    md.append("| Mode | Mean KV per request | KV vs append-only | KV vs compaction | Computed tokens, warm cache, vs append-only | Computed tokens, no cache, vs append-only | Prefix-cache hit rate |")
    md.append("|---|---|---|---|---|---|---|")
    tex.append(r"\begin{table*}[t]\centering\small\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lrrrrrr@{}}\toprule")
    tex.append(r"\textbf{Mode} & \textbf{KV / request} & \textbf{vs append} & \textbf{vs compaction} & \textbf{Compute, warm cache} & \textbf{Compute, no cache} & \textbf{Cache hit} \\ \midrule")
    for label, c in rows:
        kv = a["kv"][c]
        r_app = 1 - a["tot"][c] / a["tot"]["conv"]
        r_cmp = 1 - a["tot"][c] / a["tot"]["conv_with_compaction"]
        if c in a["new"]:
            warm = a["new"][c] / a["new"]["conv"] - 1
            hit = 1 - a["new"][c] / a["tot"][c]
            ws, hs = pct(warm), f"{100 * hit:.1f}%"
        else:
            ws, hs = "n/a", "n/a"
        nc = a["tot"][c] / a["tot"]["conv"] - 1
        cells = [label, f"{kv:.2f} GB", "—" if c == "conv" else pct(-r_app),
                 "—" if c == "conv_with_compaction" else pct(-r_cmp), "—" if c == "conv" else ws,
                 "—" if c == "conv" else pct(nc), hs]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(x.replace("%", r"\%").replace("—", "---") for x in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}}")
    tex.append(r"\caption{Memory and compute across 20,000 simulated sessions (824,728 turns). Compute = prompt tokens the server must process. Warm cache = ideal per-session prefix cache that never evicts.}")
    tex.append(r"\label{tab:dual_fleet}\end{table*}")
    md.append(f"\nMode B diff chains (most diffs applied to one file in a session): with rebasing p50/p90/p99/max = "
              f"{'/'.join(str(int(x)) for x in meta['lazy_max_diff_chain'])}; without rebasing = "
              f"{'/'.join(str(int(x)) for x in meta['lazy_max_diff_chain_no_rebase'])}. "
              f"Rebases: {meta['lazy_rebases']:,}. Diffs verified byte-exact: {meta['lazy_diffs_verified']:,} of {meta['lazy_diffs']:,}. "
              f"Prompt-only reconstruction: {meta['lazy_ok']:,} of {meta['lazy_total']:,}.\n")

    # ---------------- Table 2: per profile
    md.append("### Table B. By workload profile (change vs append-only)\n")
    md.append("| Profile | Mode A KV | Mode B KV | Mode A compute, warm cache | Mode B compute, warm cache | Mode A compute, no cache | Mode B compute, no cache |")
    md.append("|---|---|---|---|---|---|---|")
    tex.append(r"\begin{table}[htbp]\centering\small\resizebox{\columnwidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lrrrrrr@{}}\toprule")
    tex.append(r"& \multicolumn{2}{c}{\textbf{KV}} & \multicolumn{2}{c}{\textbf{Compute, warm}} & \multicolumn{2}{c}{\textbf{Compute, no cache}} \\")
    tex.append(r"\textbf{Profile} & A & B & A & B & A & B \\ \midrule")
    for p in ("paper", "realistic", "runaway", "all"):
        f = F[p]
        cells = [p.capitalize() if p != "all" else "All",
                 pct(f["tot"]["hc_paper_stub"] / f["tot"]["conv"] - 1), pct(f["tot"]["lazy"] / f["tot"]["conv"] - 1),
                 pct(f["new"]["hc_paper_stub"] / f["new"]["conv"] - 1), pct(f["new"]["lazy"] / f["new"]["conv"] - 1),
                 pct(f["tot"]["hc_paper_stub"] / f["tot"]["conv"] - 1), pct(f["tot"]["lazy"] / f["tot"]["conv"] - 1)]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(x.replace("%", r"\%") for x in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}}")
    tex.append(r"\caption{Change vs append-only by workload profile. KV = mean KV-cache per request; with no cache, compute equals prompt length, so it matches the KV column.}")
    tex.append(r"\label{tab:dual_profile}\end{table}")

    # ---------------- Table 3: 16-GPU cluster
    cl = [r for r in sv if r["kind"] == "cluster"]
    names = {"conv": "Append-only + compaction", "hc": "Mode A", "lazy": "Mode B"}
    agg = {}
    for m in ("conv", "hc", "lazy"):
        xs = [r for r in cl if r["mode"] == m]
        if not xs:
            continue
        util = np.mean([r["mean_kv_util"] for r in xs])
        batch = np.mean([r["mean_batch"] for r in xs])
        agg[m] = dict(tph=sum(r["turns_per_hour"] for r in xs), p95=float(np.median([r["lat_p95"] for r in xs])),
                      ttft=float(np.median([r["ttft_p95"] for r in xs])), batch=batch,
                      kv_seq=util * POOL_GB / batch, hit=np.mean([r["apc_hit_rate"] for r in xs]),
                      pre=sum(r["prefill_tokens"] for r in xs) / sum(r["turns"] for r in xs),
                      gen=sum(r["gen_tok_per_s"] for r in xs), aux=sum(r["aux_calls"] for r in xs), util=util)
    md.append("\n### Table C. Simulated 16-GPU H100 cluster, 512 agents, prefix caching on, 1 hour\n")
    hdr = ["Metric"] + [names[m] for m in agg]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    metrics = [("Agent turns per hour", "tph", "{:,.0f}"), ("p95 turn latency (s)", "p95", "{:.0f}"),
               ("p95 time to first token (s)", "ttft", "{:.0f}"), ("Requests running at once", "batch", "{:.2f}"),
               ("KV per running request (GB)", "kv_seq", "{:.2f}"), ("KV pool utilisation", "util", "{:.1%}"),
               ("Prefix-cache hit rate", "hit", "{:.1%}"), ("Prompt tokens computed per turn", "pre", "{:,.0f}"),
               ("Generated tokens/s (cluster)", "gen", "{:,.0f}"), ("Compaction calls", "aux", "{:,.0f}")]
    tex.append(r"\begin{table}[htbp]\centering\small\resizebox{\columnwidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}l" + "r" * len(agg) + r"@{}}\toprule")
    tex.append(" & ".join(r"\textbf{" + h + "}" for h in hdr) + r" \\ \midrule")
    for lab, key, fmt in metrics:
        cells = [lab] + [fmt.format(agg[m][key]) for m in agg]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(x.replace("%", r"\%") for x in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}}")
    tex.append(r"\caption{Simulated 16-GPU H100 cluster serving 512 agents with prefix caching on (1 hour, paper's H100 constants).}")
    tex.append(r"\label{tab:dual_cluster}\end{table}")
    if "conv" in agg:
        md.append("\nRelative to append-only + compaction: " + "; ".join(
            f"{names[m]} {agg[m]['tph'] / agg['conv']['tph']:.2f}× turns/h, p95 {agg[m]['p95'] / agg['conv']['p95'] - 1:+.0%}"
            for m in agg if m != "conv") + ".\n")

    # ---------------- Table 4: scenario sweep
    sw = [r for r in sv if r["kind"] == "sweep"]
    md.append("### Table D. Agent turns per GPU-hour by serving scenario (single simulated H100)\n")
    md.append("| Prefix cache | Agents / GPU | Append-only + compaction | Mode A | Mode B | B vs A |")
    md.append("|---|---|---|---|---|---|")
    tex.append(r"\begin{table}[htbp]\centering\small\resizebox{\columnwidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lrrrrr@{}}\toprule")
    tex.append(r"\textbf{Prefix cache} & \textbf{Agents/GPU} & \textbf{Baseline} & \textbf{Mode A} & \textbf{Mode B} & \textbf{B vs A} \\ \midrule")
    for apc in (True, False):
        for n in (1, 2, 4, 8, 16, 32, 64):
            g = {m: next((r for r in sw if r["apc"] == apc and r["agents"] == n and r["mode"] == m), None)
                 for m in ("conv", "hc", "lazy")}
            if not all(g.values()):
                continue
            cells = ["on" if apc else "off", str(n)] + [f"{g[m]['turns_per_hour']:,.0f}" for m in ("conv", "hc", "lazy")] + \
                    [f"{g['lazy']['turns_per_hour'] / g['hc']['turns_per_hour']:.2f}×"]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("×", r"$\times$") for c in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}}")
    tex.append(r"\caption{Agent turns completed per GPU-hour on one simulated H100 by serving scenario.}")
    tex.append(r"\label{tab:dual_sweep}\end{table}")

    # ---------------- Table 5: needle
    md.append("\n### Table E. Mechanical needle test (300 cases, one external one-line change)\n")
    md.append("| Prompt | Mean prompt tokens | Correct file | Correct line | Correct text | Diffs to apply (mean / max) |")
    md.append("|---|---|---|---|---|---|")
    tex.append(r"\begin{table}[htbp]\centering\small\resizebox{\columnwidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lrrrr@{}}\toprule")
    tex.append(r"\textbf{Prompt} & \textbf{Tokens} & \textbf{File} & \textbf{Line} & \textbf{Text} \\ \midrule")
    lab = {"conv": "Append-only", "hc": "Mode A (stubs + live tail)", "hc_resolve": r"Mode A + R(h)",
           "lazy": "Mode B (rebase after 10)", "lazy_no_rebase": "Mode B (no rebase)"}
    for c in ("conv", "hc", "hc_resolve", "lazy", "lazy_no_rebase"):
        x = nd.get(c)
        if not x:
            continue
        ch = f"{x['diffs_to_apply_mean']:.1f} / {x['diffs_to_apply_max']}" if "diffs_to_apply_mean" in x else "—"
        cells = [lab[c], f"{x['mean_prompt_tokens']:,.0f}", f"{x['path_acc']:.0%}", f"{x['line_acc']:.0%}", f"{x['text_acc']:.0%}"]
        md.append("| " + " | ".join(cells + [ch]) + " |")
        tex.append(" & ".join(t.replace("%", r"\%").replace("R(h)", r"$\mathcal{R}(h)$") for t in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}}")
    tex.append(r"\caption{Mechanical needle test: what each prompt contains (300 cases). The grader is a program, so it does not measure whether a live model can apply diff chains.}")
    tex.append(r"\label{tab:dual_needle}\end{table}")

    # ---------------- Table 6: scenario guidance (derived from Tables C and D)
    def tph(apc, n, m):
        return next(r["turns_per_hour"] for r in sw if r["apc"] == apc and r["agents"] == n and r["mode"] == m)
    on_low = [tph(True, n, "lazy") / tph(True, n, "hc") for n in (1, 2, 4, 8)]
    on_high = [tph(True, n, "lazy") / tph(True, n, "hc") for n in (16, 32, 64)]
    off = [tph(False, n, "lazy") / tph(False, n, "hc") for n in (1, 2, 4, 8, 16, 32, 64)]
    clr = agg["lazy"]["tph"] / agg["hc"]["tph"]
    scen = [
        ("Prefix cache on, light load (1--8 agents per GPU, cache stays warm)", "Mode B",
         f"B completes {min(on_low):.2f}--{max(on_low):.2f}x the turns of A (Table D)"),
        ("Prefix cache on, heavy load (16+ agents per GPU, cache evicted)", "Mode A",
         f"B completes {min(on_high):.2f}--{max(on_high):.2f}x the turns of A; cluster: {clr:.2f}x (Table C)"),
        ("No prefix cache, or requests not routed to the same cache", "Mode A",
         f"B completes {min(off):.2f}--{max(off):.2f}x the turns of A (Table D)"),
        ("Billed API with discounted cached input tokens", "Mode B",
         f"B computes {abs(a['new']['lazy'] / a['new']['conv'] - 1):.1%} fewer prompt tokens than append-only with a warm cache; A computes {a['new']['hc_paper_stub'] / a['new']['conv'] - 1:+.1%} (Table A)"),
        ("Lowest KV memory per request", "Mode A",
         f"A {a['kv']['hc_paper_stub']:.2f} GB vs B {a['kv']['lazy']:.2f} GB per request (Table A)"),
        ("Long edit loops, accuracy-critical work", "Mode A, or Mode B with N=5", live_evidence()),
        ("Model must see exactly what changed, without retrieval", "Mode B",
         "B shows the changed lines directly; A needs R(h) for that (Table E)"),
    ]
    md.append("\n### Table F. Recommended mode by scenario\n")
    md.append("| Scenario | Recommended | Evidence |")
    md.append("|---|---|---|")
    tex.append(r"\begin{table*}[t]\centering\small")
    tex.append(r"\begin{tabular}{@{}p{5.2cm}p{2.5cm}p{8.7cm}@{}}\toprule")
    tex.append(r"\textbf{Scenario} & \textbf{Mode} & \textbf{Evidence} \\ \midrule")
    for sc, mode, ev in scen:
        md.append(f"| {sc.replace('--', '–')} | {mode} | {ev.replace('--', '–').replace('x ', '× ').replace('x;', '×;').replace('x (', '× (')} |")
        ev_tex = ev
        for t_md, lab_tex in (("Table A", "tab:dual_fleet"), ("Table C", "tab:dual_cluster"),
                              ("Table D", "tab:dual_sweep"), ("Table E", "tab:dual_needle"),
                              ("Table G", "tab:dual_live")):
            ev_tex = ev_tex.replace(t_md, r"Table~\ref{" + lab_tex + "}")
        ev_tex = ev_tex.replace("R(h)", r"$\mathcal{R}(h)$")
        tex.append(f"{sc} & {mode} & " + ev_tex.replace("%", r"\%").replace("x ", r"$\times$ ").replace("x;", r"$\times$;").replace("x (", r"$\times$ (") + r" \\")
    tex.append(r"\bottomrule\end{tabular}")
    tex.append(r"\caption{Recommended deployment mode by serving scenario, derived from the simulated measurements and the live diff-chain test.}")
    tex.append(r"\label{tab:dual_scenarios}\end{table*}")

    # ---------------- Table 7: live-LLM diff-chain test
    live = live_results()
    if live:
        md.append("\n### Table G. Live-LLM diff-chain test (" + live["model"] + ", 30 cases, file has 10 edits)\n")
        md.append("| Question | Prompt | Diffs to apply | Correct | 95% CI |")
        md.append("|---|---|---|---|---|")
        tex.append(r"\begin{table}[htbp]\centering\small\resizebox{\columnwidth}{!}{%")
        tex.append(r"\begin{tabular}{@{}llrrr@{}}\toprule")
        tex.append(r"\textbf{Question} & \textbf{Prompt} & \textbf{Diffs} & \textbf{Correct} & \textbf{95\% CI} \\ \midrule")
        for q, lab, diffs, k, n in live["rows"]:
            lo, hi = wilson(k, n)
            cells = [q, lab, diffs, f"{k}/{n} ({k / n:.1%})", f"{lo:.0%}–{hi:.0%}"]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("%", r"\%").replace("–", "--") for c in cells) + r" \\")
        tex.append(r"\bottomrule\end{tabular}}")
        tex.append(r"\caption{Live-LLM diff-chain test on " + live["model"] + r" (30 cases; the target file has 10 edits). "
                   r"\emph{Line after anchor}: report the line that follows a given unique line. "
                   r"\emph{Line number}: report line $N$, which mostly measures line counting. 95\% Wilson intervals.}")
        tex.append(r"\label{tab:dual_live}\end{table}")

    open(os.path.join(R, "tradeoff_tables.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(R, "tradeoff_tables.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
