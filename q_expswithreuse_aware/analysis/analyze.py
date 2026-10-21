"""Analysis of the q* / Reuse-Aware / Runtime-Variable-Aware ablation (one run directory).

Metric definitions are the driver's (experiments/runtime_q/driver.py), which are the upstream
decode benchmark's: ms/token = (t_last - t_first) / (completion_tokens - 1) of the measured
streamed request; TTFT = first token - request start; end-to-end = last token - request start.

Timing rows: hooked arms, rounds 0-2 (round 3 writes the decision log and is excluded); unhooked
arms, all rounds. Every hooked arm of one task ran on the same server, rotated each round, so arms
are compared as paired ratios per (task, round): speedup = ms(reference) / ms(arm), geometric mean,
95% t-interval and one-sample t-test on the log ratios. Unhooked arms ran on separate servers and
are used only for the instrumentation overhead.

Ablation (B and the seven C arms form a 2^3 factorial over V1, V2, V3): least squares on
log ms/token with one fixed effect per (task, round) block; exp(coefficient) is the factor a
switch applies (main effects relative to B, interactions on top of the main effects).
Usage: analyze.py --root RUN_ROOT [--plan PLAN_STEM] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

A, B = "A_qstar_paper_min1", "B_reuse_aware"
DIAG = ["A0_qstar_paper", "E_qstar_engine"]
C = ["C_V1", "C_V2", "C_V3", "C_V1V2", "C_V1V3", "C_V2V3", "C_V1V2V3"]
NOHOOK = {"A0_qstar_paper_nohook": "A0_qstar_paper", "E_qstar_engine_nohook": "E_qstar_engine"}
ORDER = [A, B] + C + DIAG


def vflags(arm: str) -> tuple[int, int, int]:
    s = arm.removeprefix("C_") if arm.startswith("C_") else ""
    return int("V1" in s), int("V2" in s), int("V3" in s)


def ci(logs: np.ndarray) -> tuple[float, float, float, float]:
    n = len(logs)
    gm = float(np.exp(logs.mean()))
    if n < 2:
        return gm, float("nan"), float("nan"), float("nan")
    se = logs.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(logs.mean() / se), n - 1)) if se > 0 else 0.0
    return gm, float(np.exp(logs.mean() - t * se)), float(np.exp(logs.mean() + t * se)), p


def paired(perf: pd.DataFrame, ref: str, arms: list[str], metric: str, by: list[str]) -> pd.DataFrame:
    piv = perf.pivot_table(index=["problem", "rep"], columns="policy", values=metric)
    rows = []
    for arm in arms:
        if arm not in piv or ref not in piv:
            continue
        r = np.log(piv[ref] / piv[arm]).dropna()
        groups = [((), r)] if not by else list(r.groupby(level=by))
        for key, g in groups:
            gm, lo, hi, p = ci(g.to_numpy())
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by, key)), "arm": arm, "reference": ref, "metric": metric, "pairs": len(g),
                         "ratio": round(gm, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 5)})
    return pd.DataFrame(rows)


def factorial(perf: pd.DataFrame) -> pd.DataFrame:
    d = perf[perf.policy.isin([B] + C)].copy()
    if d.policy.nunique() < 8:
        return pd.DataFrame()
    terms = ["V1", "V2", "V3", "V1:V2", "V1:V3", "V2:V3", "V1:V2:V3"]
    f = np.array([vflags(p) for p in d.policy])
    cols = {"V1": f[:, 0], "V2": f[:, 1], "V3": f[:, 2]}
    for t in terms[3:]:
        cols[t] = np.prod([cols[v] for v in t.split(":")], axis=0)
    blocks = pd.get_dummies(d.problem.astype(str) + "_" + d.rep.astype(str), dtype=float).to_numpy()
    X = np.column_stack([blocks] + [cols[t] for t in terms])
    y = np.log(d.ms_per_token.to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    k0 = blocks.shape[1]
    t = stats.t.ppf(0.975, dof)
    rows = []
    for i, name in enumerate(terms):
        b, se = beta[k0 + i], np.sqrt(cov[k0 + i, k0 + i])
        # factor on ms/token; < 1 means the switch makes decode faster. speedup = 1 / factor
        rows.append({"term": name, "ms_factor": round(float(np.exp(b)), 4),
                     "ci_lo": round(float(np.exp(b - t * se)), 4), "ci_hi": round(float(np.exp(b + t * se)), 4),
                     "p": round(float(2 * stats.t.sf(abs(b / se), dof)), 5), "dof": int(dof), "n": len(y)})
    return pd.DataFrame(rows)


def ids(s) -> list[int]:
    return [int(float(v)) for v in str(s).split(";") if v not in ("", "nan")]


def diagnostics(lean: Path) -> pd.DataFrame:
    cols = ["episode", "phase", "policy", "problem", "tok", "layer_id", "m", "n_hits", "q", "n_fetch", "n_evict",
            "wall_ms", "pre_ms", "pcie_ms", "gemm_ms", "exposed_ms", "cpu_task_ms", "decide_us", "alpha",
            "v_hit_ms", "g0_ms", "tp_ms", "c0_ms", "tc_ms", "free_slots", "occ", "victim_gids", "evicted_gids",
            "fit_g_prior", "fit_c_prior", "c_pred_ms", "calib_n", "n_remiss", "victim_fix"]
    d = pd.read_csv(lean, usecols=lambda c: c in cols, low_memory=False)
    d = d[d.phase == "measure"]
    d["moe_obs"] = d.pcie_ms + d.gemm_ms + d.exposed_ms
    d["n_cpu"] = d.m - d.n_fetch
    rows = []
    for (pol, prob), g in d.groupby(["policy", "problem"]):
        toks = int(g.groupby("episode").tok.nunique().sum())  # tokens over the logged episodes
        acc = g.m + g.n_hits
        fill = g[g.n_fetch > 0]
        vm = [sorted(x for x in ids(v)[: int(n)] if x >= 0) == sorted(ids(e))
              for n, v, e in zip(fill.n_fetch, fill.victim_gids, fill.evicted_gids)]
        mm = g[g.m > 0]
        err = (mm.c_pred_ms.astype(float) - mm.moe_obs) if "c_pred_ms" in mm else pd.Series(dtype=float)
        last = g.groupby("episode").tail(1)
        rows.append({
            "policy": pol, "problem": prob, "tokens": toks,
            "hit_rate": round(float(g.n_hits.sum() / acc.sum()), 4),
            "misses_per_tok": round(float(g.m.sum() / toks), 3),
            "fills_per_tok": round(float(g.n_fetch.sum() / toks), 3),
            "cpu_experts_per_tok": round(float(g.n_cpu.sum() / toks), 3),
            "evictions_per_tok": round(float(g.n_evict.sum() / toks), 3),
            "remiss_share_of_misses": round(float(g.n_remiss.sum() / max(1, g.m.sum())), 4) if "n_remiss" in g else None,
            "cpu_task_ms_per_tok": round(float(g.cpu_task_ms.fillna(0).sum() / toks), 3),
            "pcie_ms_per_tok": round(float(g.pcie_ms.sum() / toks), 3),
            "gemm_ms_per_tok": round(float(g.gemm_ms.sum() / toks), 3),
            "exposed_ms_per_tok": round(float(g.exposed_ms.sum() / toks), 3),
            "moe_layer_ms_per_tok": round(float(g.wall_ms.sum() / toks), 3),
            "decide_us_median": round(float(g.decide_us.median()), 1),
            "decide_ms_per_tok": round(float(g.decide_us.sum() / 1000 / toks), 3),
            "v1_pred_mae_ms": round(float(err.abs().mean()), 4) if len(err) else None,
            "v1_pred_bias_ms": round(float(err.mean()), 4) if len(err) else None,
            "gpu_fit_prior_share": round(float(g.fit_g_prior.mean()), 4) if "fit_g_prior" in g else None,
            "cpu_fit_prior_share": round(float(g.fit_c_prior.mean()), 4) if "fit_c_prior" in g else None,
            "alpha_final": round(float(last.alpha.mean()), 4),
            "calib_residencies_final": float(last.calib_n.mean()) if "calib_n" in g else None,
            "victim_match_rate": round(float(np.mean(vm)), 5) if vm else None,
            "victim_fix_rows": int((g.victim_fix > 0).sum()) if "victim_fix" in g else None,
            "cap": int((g.occ + g.free_slots).max()),
        })
    return pd.DataFrame(rows)


def plots(out: Path, perf: pd.DataFrame, vs_b: pd.DataFrame) -> list[str]:
    files = []
    if not vs_b.empty:
        v = vs_b.set_index("arm").reindex([a for a in [A] + C + DIAG if a in set(vs_b.arm)])
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
        y = np.arange(len(v))
        ax.errorbar(v.ratio, y, xerr=[v.ratio - v.ci_lo, v.ci_hi - v.ratio], fmt="o", color="#3b6ea5", capsize=3)
        ax.axvline(1.0, color="#888", lw=1, ls="--")
        ax.set_yticks(y, v.index)
        ax.invert_yaxis()
        ax.set_xlabel("speedup over B (ms/token B / arm), 95% CI")
        ax.set_title("Paired speedup over original Reuse-Aware (all tasks)")
        fig.tight_layout()
        p = out / "speedup_vs_B.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p.name)
    arms = [a for a in ORDER if a in set(perf.policy)]
    probs = sorted(perf.problem.unique())
    fig, axes = plt.subplots(1, len(probs), figsize=(3.6 * len(probs), 3.8), dpi=160, sharey=True, squeeze=False)
    for ax, pr in zip(axes[0], probs):
        g = perf[perf.problem == pr].groupby("policy").ms_per_token.agg(["mean", "std"]).reindex(arms)
        ax.barh(np.arange(len(arms)), g["mean"], xerr=g["std"], color="#7aa6c2")
        ax.set_yticks(np.arange(len(arms)), arms)
        ax.set_title(f"AIME-25 #{pr}")
        ax.set_xlabel("ms/token (mean, sd over rounds)")
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    p = out / "ms_per_token.png"
    fig.savefig(p)
    plt.close(fig)
    files.append(p.name)
    return files


def md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(no data)\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="EXPERIMENT_OUTPUT_ROOT of the run")
    ap.add_argument("--plan", default="ablation_qwen_auto")
    ap.add_argument("--lean", default="ablation_lean_qwen.csv")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = a.root / "runtime_q" / "results"
    out = a.out or a.root / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    ep = pd.DataFrame([json.loads(l) for l in (res / f"{a.plan}_episodes.jsonl").read_text().splitlines() if l.strip()])
    servers = [json.loads(l) for l in (res / f"{a.plan}_episodes_servers.jsonl").read_text().splitlines() if l.strip()]
    ep.drop(columns=[c for c in ("text", "gpu_before") if c in ep]).to_csv(out / "episodes.csv", index=False)
    hooked = ~ep.policy.isin(list(NOHOOK))
    perf = ep[(hooked & (ep.rep <= 2) & ~ep.logged.astype(bool)) | ~hooked].copy()
    perf.to_csv(out / "timing_rows.csv", index=False)
    hp = perf[~perf.policy.isin(list(NOHOOK))]

    tables = {}
    arms_all = [B] + C + DIAG
    tables["vs_A"] = paired(hp, A, arms_all, "ms_per_token", [])
    tables["vs_A_task"] = paired(hp, A, arms_all, "ms_per_token", ["problem"])
    tables["vs_B"] = paired(hp, B, [A] + C + DIAG, "ms_per_token", [])
    tables["vs_B_task"] = paired(hp, B, [A] + C + DIAG, "ms_per_token", ["problem"])
    tables["ttft_vs_B"] = paired(hp, B, [A] + C + DIAG, "ttft_ms", [])
    tables["e2e_vs_B"] = paired(hp, B, [A] + C + DIAG, "e2e_ms", [])
    tables["e2e_vs_A"] = paired(hp, A, arms_all, "e2e_ms", [])
    loo = []
    for drop, arm in (("V1", "C_V2V3"), ("V2", "C_V1V3"), ("V3", "C_V1V2")):
        t = paired(hp, arm, ["C_V1V2V3"], "ms_per_token", [])
        if not t.empty:
            loo.append({"removed": drop, **t.iloc[0].to_dict()})
    tables["leave_one_out"] = pd.DataFrame(loo)
    tables["factorial"] = factorial(hp)
    cell = perf.groupby(["problem", "policy"]).agg(
        n=("ms_per_token", "size"), ms_per_token=("ms_per_token", "mean"), ms_sd=("ms_per_token", "std"),
        tok_s=("decode_tok_s", "mean"), ttft_ms=("ttft_ms", "mean"), ttft_sd=("ttft_ms", "std"),
        e2e_ms=("e2e_ms", "mean"), p95_ms=("p95_ms", "mean"), outputs=("output_sha1", "nunique"),
        cache_slots=("moe_cache_size", "first")).reset_index()
    cell["cv_pct"] = (100 * cell.ms_sd / cell.ms_per_token).round(2)
    tables["cells"] = cell.round(3)
    ov = []
    for nh, arm in NOHOOK.items():
        x, y = perf[perf.policy == nh], hp[(hp.policy == arm) & (hp.problem.isin(perf[perf.policy == nh].problem))]
        if len(x) and len(y):
            ov.append({"arm": arm, "problem": int(x.problem.iloc[0]), "hooked_ms": round(y.ms_per_token.mean(), 3),
                       "unhooked_ms": round(x.ms_per_token.mean(), 3),
                       "overhead_ms": round(y.ms_per_token.mean() - x.ms_per_token.mean(), 3),
                       "hooked_over_unhooked": round(y.ms_per_token.mean() / x.ms_per_token.mean(), 4),
                       "hooked_ttft": round(y.ttft_ms.mean(), 1), "unhooked_ttft": round(x.ttft_ms.mean(), 1)})
    tables["overhead"] = pd.DataFrame(ov)
    lean = res / a.lean
    if lean.exists():
        tables["diagnostics"] = diagnostics(lean)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([{k: s.get(k) for k in ("server", "problem", "hook", "ready_s", "time")} |
                  {"moe_cache_size": s["geometry"].get("moe_cache_size"), "kv_pages": s["geometry"].get("num_pages")}
                  for s in servers]).to_csv(out / "servers.csv", index=False)
    figs = plots(out, hp, tables["vs_B"])

    lines = [f"# Ablation results (generated)\n\nRun root `{a.root}`; {len(ep)} episodes, {len(perf)} timing rows. "
             "Ratios > 1 mean the arm is faster than the reference. Generated by "
             "`q_expswithreuse_aware/analysis/analyze.py`; interpretation is in docs/FINAL_REPORT.md.\n"]
    titles = {"vs_A": "Speedup over A (paper q*, min one fill)", "vs_A_task": "Speedup over A per task",
              "vs_B": "Speedup over B (original Reuse-Aware)", "vs_B_task": "Speedup over B per task",
              "ttft_vs_B": "TTFT ratio over B", "e2e_vs_B": "End-to-end ratio over B", "e2e_vs_A": "End-to-end ratio over A",
              "leave_one_out": "Leave one out: V1+V2+V3 over the pair without one switch (> 1: the switch helps)",
              "factorial": "Factorial effects on ms/token (factor < 1: faster; main effects relative to B)",
              "cells": "Per cell", "overhead": "Instrumentation overhead (hooked vs unhooked, separate servers)",
              "diagnostics": "Decision-log diagnostics (logged round 3)"}
    for name, title in titles.items():
        if name in tables:
            lines += [f"\n## {title}\n\n", md(tables[name])]
    lines += ["\n## Servers\n\n", md(pd.read_csv(out / "servers.csv"))]
    lines += [f"\n![{f}]({f})\n" for f in figs]
    (out / "REPORT.md").write_text("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
