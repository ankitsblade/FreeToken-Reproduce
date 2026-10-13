"""Aggregate every end-to-end run into tables (CSV + markdown).

Latency = client-measured decode ms/token through the real server (mean of the rotated
repeats, sample std, 95% t-interval). Speedups vs q* are PAIRED: ratio within each repeat
round, then averaged. End-to-end regret = latency - best evaluated policy in that workload.
Single-step regret uses the measured wall(m, q) table from uniform-random logging: how much
slower the chosen q is than the single-step oracle (q=0) for that miss count.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.runtime_q.qlib import RQ

REP = RQ / "reports"
W = ["low_miss", "med_miss", "high_miss"]
LABEL = {"qstar": "q* (FreeToken)", "q0": "q=0", "f25": "q=25%", "f50": "q=50%", "f75": "q=75%",
         "qm": "q=m", "t0011": "m-table (q=1 if m>=3)", "ra_k1": "Reuse-aware rule (kappa=1)",
         "ra_k4": "Reuse-aware rule (kappa=4)", "ra_k1_uniformp": "Reuse-aware, uniform reuse prior",
         "ra_k1_nocache": "Reuse-aware, no cache state", "linear_h16": "Linear (H=16)",
         "ridge_h8": "Ridge (H=8)", "tree_h16": "Tree (H=16)", "gbm_h16": "GBM (H=16)",
         "mlp_h16": "ANN-GELU 2x64 (H=16)", "linucb_h0": "LinUCB myopic (H=0)",
         "linucb_h16": "LinUCB (H=16)", "ts_h16": "Thompson sampling (H=16)"}


def episodes(path: str) -> pd.DataFrame:
    p = RQ / path
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])


def ci95(v: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def summarize(ep: pd.DataFrame, ref: str = "qstar") -> pd.DataFrame:
    out = []
    for wl, g in ep.groupby("workload"):
        base = g[g.policy == ref].set_index("rep")["ms_per_token"]
        best = g.groupby("policy")["ms_per_token"].mean().min()
        for pol, d in g.groupby("policy"):
            v = d["ms_per_token"].to_numpy()
            ratio = (base.reindex(d["rep"]).to_numpy() / v) if len(base) else np.array([np.nan])
            ratio = ratio[~np.isnan(ratio)]
            tt = stats.ttest_rel(base.reindex(d["rep"]).to_numpy(), v) if len(ratio) > 1 and pol != ref else None
            out.append({"workload": wl, "policy": pol, "n": len(v), "ms_per_token": v.mean(),
                        "ms_std": v.std(ddof=1) if len(v) > 1 else 0.0, "ms_ci95": ci95(v),
                        "tok_s": d["decode_tok_s"].mean(), "p95_ms": d["p95_ms"].mean(),
                        "speedup_vs_qstar": ratio.mean() if len(ratio) else np.nan,
                        "speedup_ci95": ci95(ratio) if len(ratio) > 1 else np.nan,
                        "paired_p": float(tt.pvalue) if tt else np.nan,
                        "e2e_regret_ms": v.mean() - best})
    return pd.DataFrame(out)


def row_metrics(pattern: str) -> pd.DataFrame:
    """Per-policy decision-level metrics from the logged rows (measure phase, rep 0)."""
    uni = pd.read_pickle(RQ / "data" / "clean" / "dataset.pkl")
    uni = uni[uni["policy"].str.startswith("uniform")]
    wall = uni.groupby(["m", "n_fetch"])["wall_net_ms"].mean().to_dict()
    eb = 13.23648
    out = []
    for f in sorted(glob.glob(str(RQ / pattern))):
        d = pd.read_csv(f, low_memory=False)
        d = d[d["phase"] == "measure"]
        for (wl, pol), g in d.groupby(["workload", "policy"]):
            ntok = g["tok"].nunique()
            tc, tg = g["cpu_task_ms"].fillna(0), g["pcie_ms"] + g["gemm_ms"]
            ss = [wall.get((m, q), np.nan) - wall.get((m, 0), np.nan) for m, q in zip(g["m"], g["n_fetch"])]
            out.append({"workload": wl, "policy": pol,
                        "cache_hit_rate": 1 - g["m"].sum() / (4 * len(g)),
                        "mean_m": g["m"].mean(), "mean_q": g["n_fetch"].mean(),
                        "fetches_per_token": g["n_fetch"].sum() / ntok,
                        "cpu_mb_per_token": (g["m"] - g["n_fetch"]).sum() * eb / ntok,
                        "pcie_mb_per_token": g["n_fetch"].sum() * eb / ntok,
                        "t_cpu_ms_per_token": tc.sum() / ntok, "t_gpu_ms_per_token": tg.sum() / ntok,
                        "pcie_ms_per_token": g["pcie_ms"].sum() / ntok,
                        "imbalance": float((tc - tg).abs().sum() / np.maximum(tc, tg).sum()),
                        "max_branch_ms_per_token": float(np.maximum(tc, tg).sum() / ntok),
                        "moe_wall_ms_per_token": g["wall_net_ms"].sum() / ntok,
                        "decide_us": g["decide_us"].median(),
                        "single_step_regret_ms_per_token": float(np.nansum(ss) / ntok)})
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "-" for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    md = ["# Runtime-aware q: result tables\n",
          "All numbers: gpt-oss-20b MXFP4, RTX 4070, hybrid backend, eager decode through the "
          "runtime-q hook, greedy, 128 measured tokens, same prompt. ms/token is client-side.\n"]

    # ---- main matrix
    ep = episodes("results/eval/main_episodes.jsonl")
    s = summarize(ep)
    rm = row_metrics("results/eval/main_rows_*.csv")
    s = s.merge(rm, on=["workload", "policy"], how="left")
    s.to_csv(RQ / "results" / "eval" / "main_summary.csv", index=False)
    piv = s.pivot(index="policy", columns="workload", values="ms_per_token")[W]
    geo = np.exp(np.log(piv.loc["qstar"] / piv).mean(axis=1))
    order = geo.sort_values(ascending=False).index
    final = pd.DataFrame({"Policy": [LABEL.get(p, p) for p in order]})
    for w in W:
        sw = s[s.workload == w].set_index("policy")
        final[w + " ms/tok"] = [f"{sw.loc[p, 'ms_per_token']:.1f} ± {sw.loc[p, 'ms_ci95']:.1f}" for p in order]
    g = s.groupby("policy")
    final["Avg ms/tok"] = [piv.loc[p].mean() for p in order]
    final["Avg tok/s"] = [g["tok_s"].mean()[p] for p in order]
    final["Geo speedup vs q*"] = [geo[p] for p in order]
    final["Worst speedup"] = [(piv.loc["qstar"] / piv.loc[p]).min() for p in order]
    final["E2E regret ms"] = [g["e2e_regret_ms"].mean()[p] for p in order]
    final["Hit rate"] = [g["cache_hit_rate"].mean()[p] for p in order]
    final["CPU MB/tok"] = [g["cpu_mb_per_token"].mean()[p] for p in order]
    final["PCIe MB/tok"] = [g["pcie_mb_per_token"].mean()[p] for p in order]
    final["Decide us"] = [g["decide_us"].mean()[p] for p in order]
    final["1-step regret ms/tok"] = [g["single_step_regret_ms_per_token"].mean()[p] for p in order]
    final.to_csv(RQ / "results" / "eval" / "final_comparison.csv", index=False)
    fm = {"Avg ms/tok": "{:.1f}", "Avg tok/s": "{:.1f}", "Geo speedup vs q*": "{:.3f}", "Worst speedup": "{:.2f}",
          "E2E regret ms": "{:.1f}", "Hit rate": "{:.2f}", "CPU MB/tok": "{:.0f}", "PCIe MB/tok": "{:.0f}",
          "Decide us": "{:.0f}", "1-step regret ms/tok": "{:.2f}"}
    md += ["## 1. Final comparison (3 rotated repeats; ± = 95% CI)\n", md_table(final, list(final.columns), fm), ""]
    sig = s[s.policy.isin(["ra_k1", "ridge_h8", "f25", "t0011", "linucb_h16"])][
        ["workload", "policy", "speedup_vs_qstar", "speedup_ci95", "paired_p"]]
    md += ["### Paired speedup vs q* (same round), 95% CI, paired t-test p\n",
           md_table(sig, list(sig.columns), {"speedup_vs_qstar": "{:.3f}", "speedup_ci95": "±{:.3f}", "paired_p": "{:.4f}"}), ""]

    # ---- ablations (end-to-end, 1 run each) + offline metrics
    ab = episodes("results/ablations/e2e_episodes.jsonl")
    if len(ab):
        ab["family"] = ab["policy"].str.split(":").str[0]
        ab["config"] = ab["policy"].str.split(":").str[-1]
        piv = ab.pivot_table(index=["family", "config"], columns="workload", values="ms_per_token")[W]
        q = piv.loc[("qstar", "qstar")]
        piv["avg_ms"] = piv[W].mean(axis=1)
        piv["geo_speedup_vs_qstar"] = np.exp(np.log(q[W] / piv[W]).mean(axis=1))
        best = piv[W].min()
        piv["e2e_regret_ms"] = (piv[W] - best).mean(axis=1)
        off = pd.concat([pd.read_csv(f) for f in glob.glob(str(RQ / "results" / "ablations" / "offline_*.csv")) +
                         glob.glob(str(RQ / "results" / "ann" / "offline_ann.csv"))])
        off = off.set_index("name")[["n_features", "mae", "rmse", "r2", "mean_q", "infer_us", "n_params"]]
        piv = piv.reset_index().merge(off, left_on="config", right_index=True, how="left")
        piv.to_csv(RQ / "results" / "ablations" / "ablation_summary.csv", index=False)
        fm = {w: "{:.1f}" for w in W} | {"avg_ms": "{:.1f}", "geo_speedup_vs_qstar": "{:.3f}", "e2e_regret_ms": "{:.1f}",
                                          "mae": "{:.2f}", "rmse": "{:.2f}", "r2": "{:.3f}", "mean_q": "{:.2f}",
                                          "infer_us": "{:.0f}", "n_features": "{:.0f}", "n_params": "{:.0f}"}
        cols = ["config", *W, "avg_ms", "geo_speedup_vs_qstar", "e2e_regret_ms", "n_features", "mae", "rmse", "infer_us"]
        titles = {"qstar": "Reference anchors measured inside the ablation run", "f25": None, "ra": None,
                  "chain": "2. Incremental feature-group addition (ridge, H=8)",
                  "logo": "3. Leave-one-group-out (ridge, H=8)",
                  "indiv": "4. Individual signal ablation (ridge, H=8)",
                  "cache": "5. Cache-information ladder (ridge, H=8)",
                  "history": "6. History-window ablation (ridge, H=8)",
                  "ann": "7. ANN architecture ablation (GELU MLP, H=16)"}
        anchors = piv[piv.family.isin(["qstar", "f25", "ra_k1"])]
        md += ["## Ablation anchors (same servers as the ablation runs)\n", md_table(anchors, cols[:7], fm), ""]
        for fam in ("chain", "logo", "indiv", "cache", "history", "ann"):
            d = piv[piv.family == fam].copy()
            if fam == "ann":
                d = d.assign(config=d["config"]).sort_values("config")
                c = cols + ["n_params"]
            else:
                c = cols
            md += [f"## {titles[fam]}\n", md_table(d, c, fm), ""]
        full = piv[(piv.family == "chain") & piv.config.str.contains("10_plus_History")].iloc[0]
        lo = piv[piv.family == "logo"].copy()
        lo["Removed Group"] = lo["config"].str.replace("ridge_minus_", "")
        lo["Performance Degradation (ms/tok)"] = lo["avg_ms"] - full["avg_ms"]
        lo["Degradation %"] = 100 * (lo["avg_ms"] / full["avg_ms"] - 1)
        lo["Regret Increase (ms/tok)"] = lo["e2e_regret_ms"] - full["e2e_regret_ms"]
        lo["Offline MAE increase"] = lo["mae"] - full["mae"]
        md += ["### Removed-group table (vs Full ridge)\n",
               md_table(lo, ["Removed Group", "Performance Degradation (ms/tok)", "Degradation %",
                             "Regret Increase (ms/tok)", "Offline MAE increase"],
                        {"Performance Degradation (ms/tok)": "{:+.1f}", "Degradation %": "{:+.1f}%",
                         "Regret Increase (ms/tok)": "{:+.1f}", "Offline MAE increase": "{:+.3f}"}), ""]

    # ---- generalization / overhead / stress / long
    for title, path in (("8. Generalization: unseen prompt (p1_*) and unseen cache sizes (rate*)", "results/eval/generalization_episodes.jsonl"),
                        ("9. Decision overhead: full feature logging vs lean", "results/overhead/overhead_episodes.jsonl"),
                        ("10. Runtime adaptation: 4 CPU busy-loops during decode", "results/eval/stress_episodes.jsonl"),
                        ("11. Longer decode (384 tokens)", "results/eval/long_episodes.jsonl")):
        e = episodes(path)
        if not len(e):
            continue
        ref = "qstar" if "qstar" in set(e.policy) else "qstar_full"
        t = summarize(e, ref)
        t.to_csv(RQ / path.replace("_episodes.jsonl", "_summary.csv"), index=False)
        p = t.pivot(index="policy", columns="workload", values="ms_per_token")
        sp = (p.loc[ref] / p)
        p.columns = [f"{c} ms/tok" for c in p.columns]
        p["geo speedup"] = np.exp(np.log(sp).mean(axis=1))
        p["worst speedup"] = sp.min(axis=1)
        p = p.sort_values("geo speedup", ascending=False).reset_index()
        md += [f"## {title}\n", md_table(p, list(p.columns), {c: "{:.1f}" for c in p.columns if "ms/tok" in c} |
                                         {"geo speedup": "{:.3f}", "worst speedup": "{:.2f}"}), ""]
    target = RQ / "reports/tables.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
