"""Cache-aware analysis: what did each GPU fetch buy?

For every fetched expert we follow it (OFFLINE, using future rows) until it is evicted or
the phase ends, and record
  future_hits     times it was routed again while resident (CPU executions avoided)
  residency_tok   tokens it stayed resident (censored at phase end)
  first_reuse     tokens until first reuse
and the PRE-fetch signals a runtime rule could use: its frequency in the last 32 tokens of
that layer (p = f/window), tokens since last use, the LRU victim age, free slots, m.

Then: immediate fetch cost vs realized future saving, and how well simple predictors of
reuse (p * residency estimate) track the realized reuse.
"""

from __future__ import annotations

import json
from collections import Counter, deque

import numpy as np
import pandas as pd

from experiments.runtime_q.qlib import RQ

OUT = RQ / "results" / "cache_analysis"
E = 32


def ids(s) -> list[int]:
    return [int(v) for v in str(s).split(";") if v not in ("", "nan")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(RQ / "data" / "clean" / "dataset.pkl")
    uni = df[df["policy"].str.startswith("uniform")]
    wall = uni.groupby(["m", "n_fetch"])["wall_net_ms"].mean()
    t_c = float((uni[uni.n_cpu > 0].cpu_task_ms / uni[uni.n_cpu > 0].n_cpu).median())
    t_p = float((uni[uni.n_fetch > 0].pcie_ms / uni[uni.n_fetch > 0].n_fetch).median())
    marginal = {f"m{int(m)}_q{int(q)}to{int(q) + 1}": round(float(wall[(m, q + 1)] - wall[(m, q)]), 4)
                for (m, q) in wall.index if (m, q + 1) in wall.index}
    save_per_reuse = {int(m): round(float(wall[(m, 0)] - wall[(m - 1, 0)]), 4) for m in (1, 2, 3, 4)}

    fetches = []
    for (ep, phase), g in df.groupby(["episode", "phase"], sort=False):
        g = g.sort_values(["tok", "layer_id"])
        pol, wl = g["policy"].iloc[0], g["workload"].iloc[0]
        tok_end = int(g["tok"].max())
        routed_hist = {}
        last_seen = {}
        live = {}  # gid -> record
        for r in g.itertuples(index=False):
            L, tok = int(r.layer_id), int(r.tok)
            routed = ids(r.routed_ids)
            for gid in ids(r.evicted_gids):
                rec = live.pop(gid, None)
                if rec:
                    rec["residency_tok"], rec["censored"] = tok - rec["tok"], False
            for e in routed:
                rec = live.get(L * E + e)
                if rec and rec["tok"] != tok:
                    rec["future_hits"] += 1
                    if rec["first_reuse"] is None:
                        rec["first_reuse"] = tok - rec["tok"]
            hist = routed_hist.setdefault(L, deque(maxlen=32))
            freq = Counter(e for t in hist for e in t)
            for e in ids(r.fetched_ids):
                rec = {"episode": ep, "phase": phase, "policy": pol, "workload": wl,
                       "layer_id": L, "tok": tok, "expert": e, "m": int(r.m), "q": int(r.n_fetch),
                       "f32": freq.get(e, 0), "window": len(hist),
                       "rec": tok - last_seen[(L, e)] if (L, e) in last_seen else 64,
                       "lru_age_tok": float(r.lru_age_tok), "free_slots": float(r.free_slots),
                       "cap": float(r.cap), "future_hits": 0, "first_reuse": None,
                       "residency_tok": tok_end - tok, "censored": True, "remaining_tok": tok_end - tok}
                live[L * E + e] = rec
                fetches.append(rec)
            hist.append(routed)
            for e in routed:
                last_seen[(L, e)] = tok
    f = pd.DataFrame(fetches)
    f["p_hat"] = f["f32"] / f["window"].clip(lower=1)
    f["saving_ms"] = f["future_hits"] * t_c
    f["marg_cost_ms"] = [marginal.get(f"m{m}_q{q - 1}to{q}", t_p) for m, q in zip(f["m"], f["q"])]
    f["net_ms"] = f["saving_ms"] - f["marg_cost_ms"]
    f["paid_off"] = f["net_ms"] > 0
    f["hit_rate_resident"] = f["future_hits"] / f["residency_tok"].clip(lower=1)
    f.to_csv(OUT / "fetch_events.csv.gz", index=False)

    by_pol = f.groupby(["workload", "policy"]).agg(
        fetches=("expert", "size"), mean_future_hits=("future_hits", "mean"),
        median_future_hits=("future_hits", "median"), mean_residency_tok=("residency_tok", "mean"),
        frac_never_reused=("future_hits", lambda s: float((s == 0).mean())),
        frac_paid_off=("paid_off", "mean"), saving_ms=("saving_ms", "sum"),
        fetch_cost_ms=("marg_cost_ms", "sum"), net_ms=("net_ms", "sum")).round(3)
    by_pol.to_csv(OUT / "fetch_value_by_policy.csv")

    # how reuse depends on the pre-fetch signals (uniform + exploration rows = unbiased-ish)
    r = f[~f["policy"].isin(["q0", "f25", "f50", "f75", "qm"])].copy()
    r["f32_bin"] = pd.cut(r["f32"], [-1, 0, 2, 4, 8, 12, 16, 32])
    by_freq = r.groupby("f32_bin", observed=True).agg(
        n=("expert", "size"), hit_rate_resident=("hit_rate_resident", "mean"),
        p_hat=("p_hat", "mean"), future_hits=("future_hits", "mean"),
        residency=("residency_tok", "mean"), frac_paid_off=("paid_off", "mean")).round(3)
    by_freq.to_csv(OUT / "reuse_by_recent_frequency.csv")
    r["full"] = r["free_slots"] <= 0
    by_press = r.groupby(["workload", "full"]).agg(
        n=("expert", "size"), residency=("residency_tok", "mean"), lru_age=("lru_age_tok", "mean"),
        future_hits=("future_hits", "mean"), frac_paid_off=("paid_off", "mean")).round(3)
    by_press.to_csv(OUT / "reuse_by_cache_pressure.csv")

    # predictors of realized reuse (pre-action only)
    full = r[r["full"] & (r["window"] >= 8)]
    preds = {
        "p_hat * lru_age": full["p_hat"] * full["lru_age_tok"],
        "p_hat * remaining (oracle horizon)": full["p_hat"] * full["remaining_tok"],
        "p_hat": full["p_hat"], "lru_age": full["lru_age_tok"], "1/recency": 1 / full["rec"].clip(lower=1),
    }
    pred_rows = [{"predictor": k, "spearman_with_future_hits": round(float(v.corr(full["future_hits"], method="spearman")), 3),
                  "pearson": round(float(v.corr(full["future_hits"])), 3)} for k, v in preds.items()]
    pd.DataFrame(pred_rows).to_csv(OUT / "reuse_predictors.csv", index=False)

    summary = {"t_cpu_per_expert_ms": round(t_c, 4), "t_pcie_per_expert_ms": round(t_p, 4),
               "rho_tc_over_tp": round(t_c / t_p, 4),
               "breakeven_reuses_unhidden": round(t_p / t_c, 2),
               "marginal_fetch_cost_ms": marginal,
               "cpu_saving_per_avoided_expert_ms": save_per_reuse,
               "breakeven_reuses_first_fetch_by_m": {
                   m: round(marginal[f"m{m}_q0to1"] / save_per_reuse[m], 2) for m in (1, 2, 3, 4)},
               "fetch_events": int(len(f)),
               "resident_hit_rate_vs_p_hat_spearman": round(float(r["p_hat"].corr(r["hit_rate_resident"], method="spearman")), 3)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(by_freq.to_string())
    print(by_press.to_string())
    print(pd.DataFrame(pred_rows).to_string(index=False))
    print(by_pol.loc[["med_miss"]].to_string())


if __name__ == "__main__":
    main()
