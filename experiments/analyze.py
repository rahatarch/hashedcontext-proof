"""Aggregate all results -> results/summary.json + figures/*.png
usage: python -m experiments.analyze"""
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)
C_CONV, C_HC, C_HCC, C_AUX = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
KV = 163_840
A_IN, A_OUT = 2.5e-6, 10e-6

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
                     "lines.linewidth": 2})


def load_fleet():
    cols = None
    data = {}
    with gzip.open(os.path.join(R, "fleet_turns.csv.gz"), "rt") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split(",")
            if i == 0:
                cols = parts
                data = {c: [] for c in cols}
                continue
            for c, v in zip(cols, parts):
                data[c].append(v)
    out = {}
    for c, v in data.items():
        out[c] = np.array(v) if c == "profile" else np.array(v, dtype=np.int64)
    return out


def pct(a, b):
    return 100 * (1 - a / b)


def main():
    S = {}
    d = load_fleet()
    meta = json.load(open(os.path.join(R, "fleet_meta.json")))
    S["fleet_meta"] = {k: v for k, v in meta.items() if k != "tok_err"}
    te = np.array(meta["tok_err"], dtype=float)
    S["fleet_meta"]["tok_segment_vs_exact_max_rel_err"] = float(np.max(np.abs(te[:, 0] - te[:, 1]) / te[:, 1]))
    prof = d["profile"]
    tot = {}
    for name, mask in [("all", np.ones(len(prof), bool))] + [(p, prof == p) for p in ("paper", "realistic", "runaway")]:
        conv, hc, hcc, eff = (d[c][mask].sum() for c in ("conv", "hc_paper_stub", "hc_compact_stub", "conv_with_compaction"))
        cf, hf = d["conv_file"][mask].sum(), d["hc_file"][mask].sum()
        aux_in, aux_out, gen = d["aux_in"][mask].sum(), d["aux_out"][mask].sum(), d["generated"][mask].sum()
        # Regime A: per-session prefix cache (no eviction) -> tokens that must actually be prefilled
        conv_new = (d["conv"][mask] - d["conv_cache_hit"][mask]).sum()
        hc_new = (d["hc_paper_stub"][mask] - d["hc_cache_hit"][mask]).sum()
        eff_new = (d["conv_with_compaction"][mask] - d["eff_cache_hit"][mask]).sum() + aux_in
        tot[name] = dict(
            turns=int(mask.sum()), conv_tokens=int(conv), hc_tokens=int(hc), hc_compact_tokens=int(hcc),
            conv_compaction_tokens=int(eff + aux_in),
            reduction_vs_append_only=pct(hc, conv), reduction_compact_stub=pct(hcc, conv),
            reduction_vs_compaction_baseline=pct(hc, eff + aux_in),
            file_token_reduction=pct(hf, cf),
            prefill_tokens_regimeA=dict(conv=int(conv_new), hc=int(hc_new), conv_compaction=int(eff_new),
                                        hc_over_conv=hc_new / conv_new),
            cost_usd_paper_pricing=dict(conv=conv * A_IN + gen * A_OUT, hc=hc * A_IN + gen * A_OUT,
                                        conv_compaction=(eff + aux_in) * A_IN + (gen + aux_out) * A_OUT),
            mean_prompt=dict(conv=conv / mask.sum(), hc=hc / mask.sum(), conv_compaction=eff / mask.sum()),
            kv_GB_resident_mean=dict(conv=conv / mask.sum() * KV / 1e9, hc=hc / mask.sum() * KV / 1e9),
        )
    S["totals"] = tot
    # ---------- by turn depth (paper profile & realistic)
    depth = {}
    for p in ("paper", "realistic", "runaway"):
        m = prof == p
        ks = d["k"][m]
        r = d["hc_paper_stub"][m] / d["conv"][m]
        rc = d["hc_compact_stub"][m] / d["conv"][m]
        rows = []
        for k in range(1, ks.max() + 1):
            mk = ks == k
            if mk.sum() < 20:
                continue
            rows.append(dict(k=k, n=int(mk.sum()), red_median=float(100 * (1 - np.median(r[mk]))),
                             red_p10=float(100 * (1 - np.percentile(r[mk], 90))),
                             red_p90=float(100 * (1 - np.percentile(r[mk], 10))),
                             red_compact_median=float(100 * (1 - np.median(rc[mk]))),
                             conv_median=float(np.median(d["conv"][m][mk])),
                             hc_median=float(np.median(d["hc_paper_stub"][m][mk])),
                             hcc_median=float(np.median(d["hc_compact_stub"][m][mk])),
                             eff_median=float(np.median(d["conv_with_compaction"][m][mk]))))
        depth[p] = rows
    S["by_depth"] = depth
    # ---------- runaway loop costs (cumulative, per session, P=100)
    m = prof == "runaway"
    sess = d["session"][m]
    run = {}
    for c in ("conv", "hc_paper_stub", "conv_with_compaction", "aux_in", "generated", "aux_out"):
        v = d[c][m].reshape(-1, 100)
        run[c] = v
    cum_conv = np.cumsum(run["conv"] * A_IN + run["generated"] * A_OUT, axis=1)
    cum_hc = np.cumsum(run["hc_paper_stub"] * A_IN + run["generated"] * A_OUT, axis=1)
    cum_eff = np.cumsum((run["conv_with_compaction"] + run["aux_in"]) * A_IN + (run["generated"] + run["aux_out"]) * A_OUT, axis=1)
    S["runaway"] = dict(sessions=int(len(np.unique(sess))),
                        cost_P30=dict(conv=float(np.median(cum_conv[:, 29])), hc=float(np.median(cum_hc[:, 29])),
                                      conv_compaction=float(np.median(cum_eff[:, 29]))),
                        cost_P100=dict(conv=float(np.median(cum_conv[:, 99])), hc=float(np.median(cum_hc[:, 99])),
                                       conv_compaction=float(np.median(cum_eff[:, 99]))),
                        Pmax_10usd=dict(conv=float(np.median((cum_conv <= 10).sum(1))),
                                        hc=float(np.median((cum_hc <= 10).sum(1)))))
    # ---------- figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, p, title in [(axes[0], "paper", "Paper profile (M=5, re-read every turn)"),
                         (axes[1], "realistic", "Realistic profile (lazy, partial re-reads)")]:
        rows = depth[p]
        k = [x["k"] for x in rows]
        ax.plot(k, [x["conv_median"] / 1e3 for x in rows], color=C_CONV, label="Append-only")
        ax.plot(k, [x["eff_median"] / 1e3 for x in rows], color=C_AUX, ls="--", lw=1.5, label="Append-only + compaction @128K")
        ax.plot(k, [x["hc_median"] / 1e3 for x in rows], color=C_HC, label="HashedContext (paper stub)")
        ax.plot(k, [x["hcc_median"] / 1e3 for x in rows], color=C_HCC, lw=1.5, label="HashedContext (compact stub)")
        ax.axhline(128, color=MUTED, lw=1, ls=":")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Agent turn k")
        ax.set_ylabel("Prompt tokens (thousands, median)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Prompt length per turn - measured on the agent fleet (cl100k BPE)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig1_prompt_per_turn.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(1, 101)
    ax.plot(x, np.median(cum_conv, 0), color=C_CONV, label="Append-only (measured)")
    ax.plot(x, np.median(cum_eff, 0), color=C_AUX, ls="--", lw=1.5, label="Append-only + compaction (measured)")
    ax.plot(x, np.median(cum_hc, 0), color=C_HC, label="HashedContext (measured)")
    ax.set_yscale("log")
    ax.axhline(10, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Loop depth P (turns)")
    ax.set_ylabel("Cumulative API cost, USD (log)")
    ax.set_title(f"Runaway loops: median of {S['runaway']['sessions']} stuck agents", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig2_runaway_cost.png"), dpi=160)
    plt.close(fig)

    # ---------- serving
    sp = os.path.join(R, "serving_all.json")
    if os.path.exists(sp):
        sv = json.load(open(sp))
        sweep = [r for r in sv if r["kind"] == "sweep"]
        S["serving_sweep"] = sweep
        cl = [r for r in sv if r["kind"] == "cluster"]
        if cl:
            S["serving_cluster"] = {m: dict(
                gpus=len([r for r in cl if r["mode"] == m]),
                agents=sum(r["agents"] for r in cl if r["mode"] == m),
                turns_per_hour=sum(r["turns_per_hour"] for r in cl if r["mode"] == m),
                gen_tok_per_s=sum(r["gen_tok_per_s"] for r in cl if r["mode"] == m),
                lat_p50=float(np.median([r["lat_p50"] for r in cl if r["mode"] == m])),
                lat_p95=float(np.median([r["lat_p95"] for r in cl if r["mode"] == m])),
                mean_batch=float(np.mean([r["mean_batch"] for r in cl if r["mode"] == m])),
                apc_hit=float(np.mean([r["apc_hit_rate"] for r in cl if r["mode"] == m])),
                aux_calls=sum(r["aux_calls"] for r in cl if r["mode"] == m),
                prefill_tokens=sum(r["prefill_tokens"] for r in cl if r["mode"] == m)) for m in ("conv", "hc")}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for apc, ax in ((True, axes[0]), (False, axes[1])):
            for mode, col, lab in (("conv", C_CONV, "Append-only + compaction"), ("hc", C_HC, "HashedContext"),
                                   ("hc_compact", C_HCC, "HashedContext (compact stub)")):
                rr = sorted([r for r in sweep if r["apc"] == apc and r["mode"] == mode], key=lambda r: r["agents"])
                ax.plot([r["agents"] for r in rr], [r["turns_per_hour"] for r in rr], color=col, marker="o", ms=5, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Concurrent agents per GPU")
            ax.set_ylabel("Agent turns completed / GPU-hour")
            ax.set_title("Prefix caching ON (Regime A)" if apc else "Prefix caching OFF (Regime B)", loc="left", fontsize=10)
        axes[0].legend(fontsize=8)
        ax = axes[2]
        for mode, col, lab in (("conv", C_CONV, "Append-only + compaction"), ("hc", C_HC, "HashedContext")):
            rr = sorted([r for r in sweep if r["apc"] and r["mode"] == mode], key=lambda r: r["agents"])
            ax.plot([r["agents"] for r in rr], [r["lat_p95"] for r in rr], color=col, marker="o", ms=5, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Concurrent agents per GPU")
        ax.set_ylabel("p95 turn latency, s (log)")
        ax.set_title("Turn latency (prefix caching ON)", loc="left", fontsize=10)
        fig.suptitle("Simulated H100 replica, 70B-class FP8 model (paper Sec. 5 constants), 1 h per point",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(F, "fig3_serving.png"), dpi=160)
        plt.close(fig)

    gp = os.path.join(R, "gateway_loadtest.json")
    if os.path.exists(gp):
        g = json.load(open(gp))
        S["gateway"] = {}
        for mode, r in g.items():
            rows = r["rows"]
            S["gateway"][mode] = dict(requests=r["requests"], clients=r["clients"], wall_s=r["wall_s"], rps=r["rps"],
                                      ingress_GB=sum(x["in_bytes"] for x in rows) / 1e9,
                                      to_backend_GB=sum(x["out_bytes"] for x in rows) / 1e9,
                                      backend_prompt_tokens=sum(x["prompt_tokens"] for x in rows),
                                      verified=max(x["verified"] for x in rows),
                                      gw_cpu_ms_total=sum(x["gw_us"] for x in rows) / 1e3,
                                      parse_ms_total=sum(x["parse_us"] for x in rows) / 1e3,
                                      vault_objects=max(x["vault_objects"] for x in rows),
                                      vault_MB=max(x["vault_bytes"] for x in rows) / 1e6)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for mode, col, lab in (("baseline", C_CONV, "Unmodified client (ingress)"), ("edge", C_HC, "Edge-native HashedContext (ingress)")):
            rows = g[mode]["rows"]
            ks = sorted(set(x["k"] for x in rows))
            ax.plot(ks, [np.median([x["in_bytes"] for x in rows if x["k"] == k]) / 1e6 for k in ks], color=col, label=lab)
        rows = g["gateway"]["rows"]
        ks = sorted(set(x["k"] for x in rows))
        ax.plot(ks, [np.median([x["out_bytes"] for x in rows if x["k"] == k]) / 1e6 for k in ks], color=C_HCC, ls="--",
                label="Gateway-native HashedContext (to backend)")
        ax.set_xlabel("Agent turn k")
        ax.set_ylabel("Request body, MB (median)")
        ax.set_title("Real HTTP load test: JSON payload per request", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(F, "fig4_gateway_payload.png"), dpi=160)
        plt.close(fig)
    for name in ("micro.json", "paper_math.json", "needle_mechanical.json"):
        p = os.path.join(R, name)
        if os.path.exists(p):
            S[name.split(".")[0]] = json.load(open(p))
    json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: S[k] for k in ("fleet_meta", "totals", "runaway") if k in S}, indent=1, default=float))


if __name__ == "__main__":
    main()
