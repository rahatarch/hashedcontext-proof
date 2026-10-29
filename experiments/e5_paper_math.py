"""E5 - re-derive every closed-form number printed in the paper, with the
paper's own parameters, and again with the measured stub size sigma.
usage: python -m experiments.e5_paper_math"""
import json
import math
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def conv_cost(P, M=5, N=4000, Cd=500, G=1000, a_in=2.5e-6, a_out=10e-6):
    return a_in * P * (P + 1) / 2 * (Cd + M * N) + a_out * P * G


def hc_cost(P, M=5, N=4000, Cd=500, G=1000, sigma=35, D=100, a_in=2.5e-6, a_out=10e-6):
    return a_in * (P * (P + 1) / 2 * Cd + P * (P - 1) / 2 * M * sigma + P * (M * N + D)) + a_out * P * G


def first_P_over(budget, f, **kw):
    P = 1
    while f(P, **kw) <= budget:
        P += 1
    return P - 1


def S_conv(k, M=5, N=4000, Cd=500):
    return k * Cd + k * M * N


def S_hc(k, M=5, N=4000, Cd=500, sigma=35, D=0):
    return k * Cd + (k - 1) * M * sigma + M * N + D


def main(sigma_measured=None):
    out = {}
    kv = 2 * 80 * 8 * 128 * 1
    out["kv_bytes_per_token"] = dict(paper="~160 KB", computed=kv)
    out["kv_100k_GB"] = dict(paper=16.4, computed=kv * 100_000 / 1e9)
    out["decode_ms_100k"] = dict(paper=4.9, computed=kv * 100_000 / 3.35e12 * 1e3)
    out["decode_ms_20k"] = dict(paper=0.98, computed=kv * 20_000 / 3.35e12 * 1e3)
    out["Bmax_100k"] = dict(paper=3, computed=math.floor(50e9 / (kv * 100_000)))
    out["Bmax_20k"] = dict(paper=15, computed=math.floor(50e9 / (kv * 20_000)))
    out["R_inf_file"] = dict(paper=0.99125, computed=1 - 35 / 4000)
    out["prefill_k10_conv"] = dict(paper="10*Cd+200,000", computed=S_conv(10) - 10 * 500)
    out["prefill_k10_hc"] = dict(paper="10*Cd+21,750", computed=S_hc(10) - 10 * 500,
                                 note="paper's 21,750 = M*N + k*M*sigma (uses k not k-1)")
    pc = 30 * (500 + 5 * 4000 + 150)
    ph = 30 * (500 + 5 * 35 + 150) + 5 * 4000
    out["prefix_total"] = dict(paper=[619_500, 44_750, 0.928], computed=[pc, ph, 1 - ph / pc],
                               mem_GB=[pc * 160e3 / 1e9, ph * 160e3 / 1e9])
    for P in (30, 100):
        out[f"cost_P{P}"] = dict(paper={30: [24.13, 2.58], 100: [259.81, 14.50]}[P],
                                 computed=[round(conv_cost(P), 2), round(hc_cost(P), 2)])
    out["cost_ratio_P100"] = dict(paper=17.9, computed=conv_cost(100) / hc_cost(100))
    # P_max at $10 budget
    a = 2.5e-6
    closed = math.floor((10 - a * 5 * 4000) / (a * (35 + 500)))
    out["Pmax_budget10"] = dict(paper=dict(conv=19, hc=78, closed_form_eq="(B - a*M*N)/(a*(sigma+Cd))"),
                                computed=dict(conv=first_P_over(10, conv_cost), hc=first_P_over(10, hc_cost),
                                              closed_form_value=closed,
                                              note="closed form omits the quadratic k*Cd and Mσ terms "
                                                   "and the per-turn M*N tail"))
    out["payload"] = dict(paper=dict(conv30=2_460_000, hc30=160_900, ratio=15.3),
                          computed=dict(conv30=4 * 30 * 20_500, hc30=4 * (15_000 + 29 * 175 + 20_000 + 150),
                                        ratio=(4 * 30 * 20_500) / (4 * (15_000 + 29 * 175 + 20_000 + 150))))
    out["telemetry_table"] = dict(turn4=(27747 - 11746) / 27747, turn5=(31747 - 11746) / 31747,
                                  file=(20000 - 4140) / 20000, cache=(27136 - 10752) / 27136,
                                  abstract_claims=[0.577, 0.672], contributions_claim=0.6719)
    # Figure 2 dwell means
    Cd, M, N, s = 500, 5, 4000, 35
    hc_series = [k * Cd + (k - 1) * M * s + M * N for k in range(1, 21)]
    out["fig2_hc_series"] = dict(paper_turn1=20_500, paper_step=675, computed_turn1=hc_series[0],
                                 computed_step=hc_series[1] - hc_series[0],
                                 mean_paper=27_000, mean_computed=sum(hc_series) / 20)
    if sigma_measured:
        sm = sigma_measured
        out["with_measured_sigma"] = dict(
            sigma=sm, R_inf_file=1 - sm / 4000,
            cost_P30=[conv_cost(30), hc_cost(30, sigma=sm)], cost_P100=[conv_cost(100), hc_cost(100, sigma=sm)],
            prefill_k10_hc=S_hc(10, sigma=sm) - 5000)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "paper_math.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 67.0)
