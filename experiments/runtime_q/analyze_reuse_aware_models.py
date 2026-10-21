"""Reuse-aware rule vs q* across models: paired comparison tables and figures.

Reads one study root written by the reuse_aware_models plans
(`configs/runtime_q/reuse_aware_models/*.json`):
  runtime_q/results/<model>_episodes.jsonl   client ms/token per episode (3 rotated repeats)
  runtime_q/results/decisions_<model>.csv    per-decision rows of repeat 0
  calibration/rho_by_model.json              B_P, B_H, rho and the engine's native fraction

Speedups are paired within a repeat round of the same server (cache size x task) and
summarized as geometric means with 95% t-intervals on the log ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODELS = {"gpt_oss": "gpt-oss-20b (MXFP4)", "qwen": "Qwen3.6-35B-A3B (NVFP4)", "gemma": "Gemma-4-26B-A4B (NVFP4)"}
EXPERT_BYTES = {"gpt_oss": 13236480, "qwen": 1775616, "gemma": 3353856}
TASKS = {0: "code", 1: "summarization", 2: "math"}
POLICIES = ["qstar_paper", "qstar_engine", "reuse_aware", "reuse_aware_uniform_p"]
LABEL = {"qstar_paper": "q* (paper: m*B_P/B_H)", "qstar_engine": "q* (FreeToken built-in)",
         "reuse_aware": "Reuse-aware (per-expert p)", "reuse_aware_uniform_p": "Reuse-aware (p = top_k/E)"}
REFS = ["qstar_paper", "qstar_engine"]
RULES = ["reuse_aware", "reuse_aware_uniform_p"]
COLOR = {"qstar_paper": "#898781", "qstar_engine": "#52514e", "reuse_aware": "#2a78d6", "reuse_aware_uniform_p": "#1baf7a"}


def geo_ci(ratios: np.ndarray) -> tuple[float, float, float, float]:
    """Geometric mean of ratios, 95% CI, and two-sided p for log-ratio = 0."""
    lr = np.log(ratios)
    g = float(np.exp(lr.mean()))
    if len(lr) < 2 or lr.std(ddof=1) == 0:
        return g, g, g, float("nan")
    half = stats.t.ppf(0.975, len(lr) - 1) * lr.std(ddof=1) / np.sqrt(len(lr))
    p = float(stats.ttest_1samp(lr, 0.0).pvalue)
    return g, float(np.exp(lr.mean() - half)), float(np.exp(lr.mean() + half)), p


def load(study: Path) -> pd.DataFrame:
    frames = []
    for key in MODELS:
        path = study / "runtime_q" / "results" / f"{key}_episodes.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            frames.append(pd.DataFrame(rows).assign(model=key))
    ep = pd.concat(frames, ignore_index=True)
    ep["task"] = ep["problem"].map(TASKS)
    return ep


def paired(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, cache, task, rep, rule, reference): speedup = ref_ms / rule_ms."""
    piv = ep.pivot_table(index=["model", "cache_rate", "task", "rep"], columns="policy", values="ms_per_token")
    out = []
    for idx, r in piv.iterrows():
        for rule in RULES:
            for ref in REFS:
                if pd.notna(r.get(rule)) and pd.notna(r.get(ref)):
                    out.append(dict(zip(["model", "cache_rate", "task", "rep"], idx), rule=rule, ref=ref,
                                    speedup=r[ref] / r[rule]))
    return pd.DataFrame(out)


def decisions(study: Path) -> pd.DataFrame:
    out = []
    for key in MODELS:
        path = study / "runtime_q" / "results" / f"decisions_{key}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, low_memory=False)
        d = d[d["phase"] == "measure"]
        for (rate, prob, pol), g in d.groupby(["cache_rate", "problem", "policy"]):
            ntok = g["tok"].nunique()
            out.append({"model": key, "cache_rate": rate, "task": TASKS[int(prob)], "policy": pol,
                        "decisions": len(g), "mean_m": g["m"].mean(), "max_m": int(g["m"].max()),
                        "mean_q": g["n_fetch"].mean(),
                        "hit_rate": 1 - g["m"].sum() / max(1, g["n_unique"].sum()),
                        "fetches_per_token": g["n_fetch"].sum() / ntok,
                        "cpu_mb_per_token": (g["m"] - g["n_fetch"]).sum() * EXPERT_BYTES[key] / 1e6 / ntok,
                        "pcie_mb_per_token": g["n_fetch"].sum() * EXPERT_BYTES[key] / 1e6 / ntok,
                        "decide_us_median": g["decide_us"].median()})
    return pd.DataFrame(out)


def md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "-" for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, type=Path)
    a = ap.parse_args()
    study = a.study.resolve()
    out = study / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rho = json.loads((study / "calibration" / "rho_by_model.json").read_text())
    ep = load(study)
    ep.to_csv(out / "episodes.csv", index=False)

    # per-cell latency (mean of repeats) and per-cell paired speedups
    cell = ep.groupby(["model", "cache_rate", "task", "policy"]).agg(
        n=("ms_per_token", "size"), ms_per_token=("ms_per_token", "mean"), ms_std=("ms_per_token", "std"),
        p95_ms=("p95_ms", "mean"), output_hashes=("output_sha1", "nunique")).reset_index()
    cell.to_csv(out / "cells.csv", index=False)
    pr = paired(ep)
    pr.to_csv(out / "paired_speedups.csv", index=False)

    rows = []
    for (model, rule, ref), g in pr.groupby(["model", "rule", "ref"]):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        cells = g.groupby(["cache_rate", "task"])["speedup"].apply(lambda s: float(np.exp(np.log(s).mean())))
        rows.append({"model": model, "rule": rule, "reference": ref, "pairs": len(g),
                     "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi, "p": p,
                     "cells_faster": int((cells > 1).sum()), "cells": len(cells),
                     "worst_cell": float(cells.min()), "best_cell": float(cells.max())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)

    by_cache = []
    for (model, rate, rule, ref), g in pr.groupby(["model", "cache_rate", "rule", "ref"]):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        by_cache.append({"model": model, "cache_rate": rate, "rule": rule, "reference": ref,
                         "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi})
    by_task = []
    for (model, task, rule, ref), g in pr.groupby(["model", "task", "rule", "ref"]):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        by_task.append({"model": model, "task": task, "rule": rule, "reference": ref,
                        "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(by_cache).to_csv(out / "by_cache.csv", index=False)
    pd.DataFrame(by_task).to_csv(out / "by_task.csv", index=False)
    dec = decisions(study)
    if len(dec):
        dec.to_csv(out / "decisions_summary.csv", index=False)

    # latency figure: one panel per model, x = cache size, grouped by policy, averaged over tasks
    present = [m for m in MODELS if m in set(ep["model"])]
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 3.8), squeeze=False)
    for ax, model in zip(axes[0], present):
        c = cell[cell.model == model].groupby(["cache_rate", "policy"])["ms_per_token"].mean().unstack()
        x = np.arange(len(c.index))
        for i, pol in enumerate(POLICIES):
            if pol in c:
                ax.bar(x + (i - 1.5) * 0.2, c[pol], 0.18, color=COLOR[pol], label=LABEL[pol])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in c.index])
        ax.set_xlabel("expert-cache rate (fraction of layers x experts)")
        ax.set_ylabel("decode ms / token (mean of 3 tasks x 3 repeats)")
        ax.set_title(MODELS[model], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "latency_by_model.png", dpi=140)
    plt.close(fig)

    # report
    fm = {"geo_speedup": "{:.3f}x", "ci_lo": "{:.3f}", "ci_hi": "{:.3f}", "p": "{:.4f}", "worst_cell": "{:.3f}x",
          "best_cell": "{:.3f}x", "mean_m": "{:.2f}", "mean_q": "{:.2f}", "hit_rate": "{:.2f}",
          "fetches_per_token": "{:.1f}", "cpu_mb_per_token": "{:.0f}", "pcie_mb_per_token": "{:.0f}",
          "decide_us_median": "{:.0f}", "ms_per_token": "{:.1f}", "ms_std": "{:.1f}", "B_P": "{:.2f}",
          "B_H": "{:.2f}", "rho": "{:.3f}", "engine_fraction": "{:.3f}"}
    rt = pd.DataFrame([{"model": MODELS[k], "B_P": v["B_P_overlap"], "B_H": v["B_H_overlap"], "rho": v["rho"],
                        "engine_fraction": v["engine_fraction"],
                        "cache rates": ", ".join(f"{r:.2f}" for r in v["cache_rates"])} for k, v in rho.items()])
    s2 = summ.assign(model=summ.model.map(MODELS), rule=summ.rule.map(LABEL), reference=summ.reference.map(LABEL),
                     **{"cells faster": summ.cells_faster.astype(str) + "/" + summ.cells.astype(str)})
    s2 = s2[["model", "rule", "reference", "pairs", "geo_speedup", "ci_lo", "ci_hi", "p", "cells faster",
             "worst_cell", "best_cell"]]
    lat = cell.groupby(["model", "policy"])["ms_per_token"].mean().unstack()[POLICIES].reset_index()
    lat["model"] = lat["model"].map(MODELS)
    lat.columns = ["model"] + [LABEL[p] for p in POLICIES]
    text = ["# Reuse-aware rule vs q* across models (RTX 4070)\n",
            "Speedup = q* ms/token / rule ms/token, paired within a repeat round of the same server; "
            "geometric mean over 3 cache sizes x 3 tasks x 3 repeats with a 95% interval. Above 1 means "
            "the reuse-aware rule is faster.\n",
            "## Calibration\n", md(rt, fm), "",
            "## Mean decode latency (ms/token, mean over cache sizes, tasks and repeats)\n",
            md(lat, {c: "{:.1f}" for c in lat.columns if c != "model"}), "",
            "## Paired speedup of the rule over q*\n", md(s2, fm), ""]
    bc = pd.DataFrame(by_cache)
    bc = bc[bc.reference == "qstar_paper"].assign(model=lambda d: d.model.map(MODELS), rule=lambda d: d.rule.map(LABEL))
    text += ["## By cache size (vs paper q*)\n", md(bc.drop(columns="reference"), fm), ""]
    bt = pd.DataFrame(by_task)
    bt = bt[bt.reference == "qstar_paper"].assign(model=lambda d: d.model.map(MODELS), rule=lambda d: d.rule.map(LABEL))
    text += ["## By task (vs paper q*)\n", md(bt.drop(columns="reference"), fm), ""]
    if len(dec):
        dd = dec.groupby(["model", "policy"])[["mean_m", "max_m", "mean_q", "hit_rate", "fetches_per_token",
                                               "cpu_mb_per_token", "pcie_mb_per_token", "decide_us_median"]].mean().reset_index()
        dd = dd.assign(model=dd.model.map(MODELS), policy=dd.policy.map(LABEL))
        text += ["## What each policy did (repeat 0, measured phase, mean over cache sizes and tasks)\n", md(dd, fm), ""]
    hashes = cell.groupby(["model", "cache_rate", "task"])["output_hashes"].max()
    text += ["## Output consistency\n",
             f"Distinct output hashes per policy cell (across repeats): max {int(cell.output_hashes.max())}. "
             f"Distinct hashes across policies within a cell: " +
             ", ".join(f"{MODELS[m]} {int(ep[ep.model == m].groupby(['cache_rate', 'task']).output_sha1.nunique().max())}"
                       for m in present) + ".", ""]
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
