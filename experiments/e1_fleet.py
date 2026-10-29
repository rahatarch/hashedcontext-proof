"""E1 - fleet-scale token accounting & verification.
usage: python -m experiments.e1_fleet [n_sessions] [workers]"""
import gzip
import json
import os
import pickle
import sys

import numpy as np

from sim.fleet import run_fleet

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
# default: leave two hardware threads free (14 on a 16-thread CPU)
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

out, dt = run_fleet(N, procs=WORKERS)
out.sort(key=lambda r: r["spec"]["id"])

# per-turn table
rows = ["session,profile,repo,M,P,k,conv,hc_paper_stub,hc_compact_stub,conv_with_compaction,"
        "conv_file,hc_file,dialogue,generated,conv_cache_hit,hc_cache_hit,eff_cache_hit,aux_in,aux_out,"
        "lazy,lazy_cache_hit,lazy_file,lazy_inf,lazy_inf_cache_hit"]
for r in out:
    s, rec = r["spec"], r["rec"]
    for i in range(len(rec["conv"])):
        rows.append(",".join(map(str, [s["id"], s["profile"], s["repo"], s["M"], s["P"], i + 1,
                                       rec["conv"][i], rec["hc"][i], rec["hcc"][i], rec["eff"][i],
                                       rec["conv_file"][i], rec["hc_file"][i], rec["dlg"][i], rec["gen"][i],
                                       rec["conv_hit"][i], rec["hc_hit"][i], rec["eff_hit"][i],
                                       rec["aux_in"][i], rec["aux_out"][i],
                                       rec["lazy"][i], rec["lazy_hit"][i], rec["lazy_file"][i],
                                       rec["lazy_inf"][i], rec["lazy_inf_hit"][i]])))
with gzip.open(os.path.join(OUT, "fleet_turns.csv.gz"), "wt") as f:
    f.write("\n".join(rows) + "\n")

# session-level extras
uniq = {}
for r in out:
    for h, n in r["hashes"]:
        uniq[h] = n
meta = dict(
    sessions=len(out), wall_seconds=dt,
    turns=sum(len(r["rec"]["conv"]) for r in out),
    file_observations=sum(len(r["hashes"]) for r in out),
    observed_bytes=sum(n for r in out for _, n in r["hashes"]),
    fleet_vault_unique_objects=len(uniq), fleet_vault_unique_bytes=sum(uniq.values()),
    per_session_vault_unique_bytes=sum(r["vault_unique"] for r in out),
    edits=sum(r["edits"] for r in out), check_failures=sum(r["check_failures"] for r in out),
    live_ok=sum(r["checks"]["live_ok"] for r in out), live_total=sum(r["checks"]["live_total"] for r in out),
    expand_ok=sum(r["checks"]["expand_ok"] for r in out), expand_total=sum(r["checks"]["expand_total"] for r in out),
    tok_err=[[a, b] for r in out for a, b in r["checks"]["tok_err"]],
    sigma_paper=np.percentile([x for r in out for x in r["sigma_paper"]], [0, 5, 50, 95, 100]).tolist(),
    sigma_paper_mean=float(np.mean([x for r in out for x in r["sigma_paper"]])),
    sigma_compact_mean=float(np.mean([x for r in out for x in r["sigma_compact"]])),
    live_header_mean=float(np.mean([x for r in out for x in r["live_hdr"]])),
    compactions=sum(r["compactions"] for r in out), dropped_tokens=sum(r["dropped"] for r in out),
    workers=WORKERS,
    lazy_ok=sum(r["checks"]["lazy_ok"] for r in out), lazy_total=sum(r["checks"]["lazy_total"] for r in out),
    lazy_full_blocks=sum(r["lazy"]["n_full"] for r in out), lazy_stubs=sum(r["lazy"]["n_stub"] for r in out),
    lazy_diffs=sum(r["lazy"]["n_diff"] for r in out), lazy_diffs_verified=sum(r["lazy"]["diff_verified"] for r in out),
    lazy_rebases=sum(r["lazy"]["n_rebase"] for r in out),
    lazy_max_diff_chain_no_rebase=np.percentile([r["lazy"]["max_chain_inf"] for r in out], [50, 90, 99, 100]).tolist(),
    lazy_max_diff_chain=np.percentile([r["lazy"]["max_chain"] for r in out], [50, 90, 99, 100]).tolist(),
    sigma_lazy=np.percentile([x for r in out for x in r["lazy"]["sigma"]], [0, 5, 50, 95, 100]).tolist(),
    sigma_lazy_mean=float(np.mean([x for r in out for x in r["lazy"]["sigma"]])),
    diff_block_tokens=np.percentile([x for r in out for x in r["lazy"]["diff_tok"]], [5, 50, 95, 99]).tolist(),
    N_mean_by_profile={p: float(np.mean([r["N_mean"] for r in out if r["spec"]["profile"] == p]))
                       for p in ("paper", "realistic", "runaway")},
)
json.dump(meta, open(os.path.join(OUT, "fleet_meta.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "tok_err"}, indent=1))
