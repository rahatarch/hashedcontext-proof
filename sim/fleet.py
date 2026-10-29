"""
Fleet-scale token accounting + correctness verification.

For every agent session and every turn we build BOTH prompts (conventional
append-only and HashedContext) and record exact BPE token counts.  On top we
verify, per request:
  * live tail == agent's actual workspace bytes      (authoritative tail)
  * expand(HC prompt, vault) == conventional prompt  (losslessness)   [sampled]
  * segment-sum token count == exact full-string BPE count             [sampled]
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from hashedcontext.core import (FILE_CLOSE, FILE_OPEN, LIVE_CLOSE, LIVE_OPEN, HashVault,
                                assemble_conventional, assemble_hashed, expand, oracle_read,
                                render_stub, sha256_hex)
from hashedcontext.lazy import LazyContext, assemble_lazy, replay
from hashedcontext.tokens import Tokenizer
from sim.agent import SYSTEM_PROMPT, CodingAgent
from sim.corpus import load_repos

CTX_LIMIT = 128_000
COMPACT_TO = 0.30 * CTX_LIMIT
SUMMARY_OUT = 2_000
_G: Dict = {}


def _init(tok_name: str = "cl100k_base"):
    tok = Tokenizer(tok_name)
    _G["tok"] = tok
    _G["repos"] = load_repos(tok)


def make_specs(n: int, seed: int = 7) -> List[dict]:
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        u = rng.random()
        if u < 0.40:
            s = dict(profile="realistic", M=rng.randint(3, 8), P=rng.randint(10, 60), p_edit=0.45)
        elif u < 0.80:
            s = dict(profile="paper", M=5, P=rng.randint(5, 30), p_edit=0.5, min_file_tok=2000)
        else:
            s = dict(profile="runaway", M=2, P=100, p_edit=1.0)
        s.update(id=i, seed=rng.randrange(1 << 30), repo=rng.randrange(4), verify=(i % 25 == 0))
        specs.append(s)
    return specs


def run_session(spec: dict) -> dict:
    tok: Tokenizer = _G["tok"]
    repo = _G["repos"][spec["repo"]]
    rng = random.Random(spec["seed"])
    agent = CodingAgent(spec["id"], repo, spec["profile"], rng, M=spec["M"], P=spec["P"],
                        p_edit=spec["p_edit"], min_file_tok=spec.get("min_file_tok", 0), tok=tok)
    tS = tok.count(SYSTEM_PROMPT)
    vault = HashVault()
    turns = []
    rec = {k: [] for k in ("conv", "hc", "hcc", "conv_file", "hc_file", "dlg", "gen",
                           "conv_hit", "hc_hit", "eff", "eff_hit", "aux_in", "aux_out",
                           "lazy", "lazy_hit", "lazy_file", "lazy_inf", "lazy_inf_hit")}
    lazy = LazyContext()            # Mode B, default max_diffs_before_rebase = 10
    lazy_inf = LazyContext(None)    # Mode B without rebasing (unbounded diff chains)
    lazy_hist = lazy_file_hist = 0  # history is never rewritten: whole prompt is next turn's prefix
    lazy_inf_hist = prev_lazy_inf = 0
    prev_lazy = 0
    sigma_lazy, diff_tok = [], []
    conv_hist = 0          # sum over i<=k-1 of (U_i + files_i)
    hc_hist = hcc_hist = 0 # sum over i<=k-1 of (U_i + stubs_i)
    conv_file_hist = hc_file_hist = 0
    prev_conv = prev_hc_shared = 0
    eff_base = tS          # append-only with compaction at CTX_LIMIT (realistic baseline)
    prev_eff = 0
    compactions = dropped = 0
    sigma_paper, sigma_compact, live_hdr = [], [], []
    checks = dict(live_ok=0, live_total=0, expand_ok=0, expand_total=0, tok_err=[],
                  lazy_ok=0, lazy_total=0)
    hashes = []
    for k in range(1, spec["P"] + 1):
        t = agent.step(k)
        turns.append(t)
        u = tok.count(t.text)
        g = tok.count(agent.generated_text)
        fb = lb = sp = sc = zb = zi = 0
        for f in t.files:
            zi += tok.count(lazy_inf.render(k, f.path, f.content)[1])
            kind, seg = lazy.render(k, f.path, f.content)
            n_seg = tok.count(seg)
            zb += n_seg
            if kind == "stub":
                sigma_lazy.append(n_seg)
            elif kind == "diff":
                diff_tok.append(n_seg)
            h = sha256_hex(f.content)
            hashes.append((h, len(f.content)))
            vault.put(f.content)
            c = tok.count(f.content)
            fb += tok.count(FILE_OPEN.format(path=f.path)) + c + tok.count(FILE_CLOSE)
            lh = tok.count(LIVE_OPEN.format(path=f.path, h=h)) + tok.count(LIVE_CLOSE)
            live_hdr.append(lh)
            lb += lh + c
            s1, s2 = tok.count(render_stub(f.path, h)), tok.count(render_stub(f.path, h, True))
            sigma_paper.append(s1)
            sigma_compact.append(s2)
            sp += s1
            sc += s2
            # authoritative-tail check: live bytes == workspace bytes at request time
            checks["live_total"] += 1
            checks["live_ok"] += int(f.content == agent.file(f.path))
        conv = tS + conv_hist + u + fb
        hc = tS + hc_hist + u + lb
        hcc = tS + hcc_hist + u + lb
        lz = tS + lazy_hist + u + zb
        rec["lazy"].append(lz)
        rec["lazy_hit"].append((prev_lazy // 16) * 16)
        rec["lazy_file"].append(lazy_file_hist + zb)
        prev_lazy = lz
        lazy_hist += u + zb
        li = tS + lazy_inf_hist + u + zi
        rec["lazy_inf"].append(li)
        rec["lazy_inf_hit"].append((prev_lazy_inf // 16) * 16)
        prev_lazy_inf = li
        lazy_inf_hist += u + zi
        lazy_file_hist += zb
        rec["conv"].append(conv)
        rec["hc"].append(hc)
        rec["hcc"].append(hcc)
        rec["conv_file"].append(conv_file_hist + fb)
        rec["hc_file"].append(hc_file_hist + lb)
        rec["dlg"].append(u)
        rec["gen"].append(g)
        # Regime A (single-session prefix cache, 16-token blocks, no eviction)
        rec["conv_hit"].append((prev_conv // 16) * 16)
        rec["hc_hit"].append((prev_hc_shared // 16) * 16)
        prev_conv = conv
        prev_hc_shared = tS + hc_hist + u  # prefix the NEXT request will share
        # realistic append-only baseline with post-hoc compaction
        eff = eff_base + u + fb
        aux_i = aux_o = 0
        hit = (prev_eff // 16) * 16
        if eff > CTX_LIMIT:
            compactions += 1
            aux_i, aux_o = eff - u - fb, SUMMARY_OUT
            dropped += eff - u - fb - COMPACT_TO
            eff_base = int(COMPACT_TO)
            eff = eff_base + u + fb
            hit = 0
        rec["eff"].append(eff)
        rec["eff_hit"].append(hit)
        rec["aux_in"].append(aux_i)
        rec["aux_out"].append(aux_o)
        prev_eff = eff
        eff_base = eff
        conv_hist += u + fb
        hc_hist += u + sp
        hcc_hist += u + sc
        conv_file_hist += fb
        hc_file_hist += sp
        # sampled heavy verification
        if spec["verify"] and (k == spec["P"] or k % 7 == 0):
            vv = HashVault()
            conv_s = "".join(assemble_conventional(SYSTEM_PROMPT, turns))
            for compact in (False, True):
                hc_s = "".join(assemble_hashed(SYSTEM_PROMPT, turns, vv, compact=compact))
                checks["expand_total"] += 1
                checks["expand_ok"] += int(expand(hc_s, vv, compact=compact) == conv_s)
                if not compact:
                    live, _ = oracle_read(hc_s)
                    assert all(live[p] == agent.file(p) for p in live)
                    exact_hc = len(tok.enc.encode_ordinary(hc_s))
                    checks["tok_err"].append((hc, exact_hc))
            # lazy prompt: rebuild every file from the prompt text alone, compare to workspace
            lz_s = assemble_lazy(SYSTEM_PROMPT, turns)
            st = replay(lz_s)
            checks["lazy_total"] += 1
            checks["lazy_ok"] += int(all(st[f.path] == agent.file(f.path) for f in t.files)
                                     and all(p in st for tt in turns for p in (f.path for f in tt.files)))
            checks["tok_err"].append((lz, len(tok.enc.encode_ordinary(lz_s))))
            if k == spec["P"]:
                checks["tok_err"].append((conv, len(tok.enc.encode_ordinary(conv_s))))
    files0 = [tok.count(repo.files[p]) for p in agent.working_set]
    return dict(spec=spec, rec=rec, N_mean=sum(files0) / len(files0), compactions=compactions,
                dropped=dropped, sigma_paper=sigma_paper, sigma_compact=sigma_compact,
                live_hdr=live_hdr, checks=checks, hashes=hashes, edits=agent.edits,
                check_failures=agent.check_failures, vault_unique=vault.unique_bytes,
                vault_offered=vault.bytes_offered,
                lazy=dict(n_full=lazy.n_full, n_stub=lazy.n_stub, n_diff=lazy.n_diff,
                          n_rebase=lazy.n_rebase, diff_verified=lazy.diff_verified,
                          max_chain=lazy.max_chain, max_chain_inf=lazy_inf.max_chain,
                          sigma=sigma_lazy, diff_tok=diff_tok))


def run_fleet(n: int, procs: int = 14, seed: int = 7, tok_name: str = "cl100k_base"):
    specs = make_specs(n, seed)
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=procs, initializer=_init, initargs=(tok_name,)) as ex:
        for i, r in enumerate(ex.map(run_session, specs, chunksize=8)):
            out.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n} sessions  {time.time()-t0:.0f}s", flush=True)
    return out, time.time() - t0
