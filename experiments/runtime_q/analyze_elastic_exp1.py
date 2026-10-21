"""Experiment 1 analysis: original Reuse-Aware vs paper q* (and min-1-fill, built-in q*), 3 models.

Reads <root>/runtime_q/results/exp1_<model>_episodes.jsonl (+ _servers.jsonl, exp1_lean_<model>.csv)
and writes <root>/analysis/exp1/{REPORT.md, *.csv, *.png}.

Timing = repeats 0-2 of the hooked arms (repeat 3 writes the decision log and is excluded) and
all repeats of the unhooked arms. Speedup of arm X over reference R in one (model, cache rate,
task, repeat) round = ms/token(R) / ms/token(X) (decode inter-token latency, same server);
summaries are geometric means with 95% t-intervals on the log ratios. Instrumented (hooked) and
uninstrumented (unhooked, separate servers) results are never mixed in one paired ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runtime_q.analyze_reuse_aware_models import geo_ci, md

MODELS = ["gpt_oss", "qwen", "gemma"]
MNAME = {"gpt_oss": "gpt-oss-20b", "qwen": "Qwen3.6-35B-A3B", "gemma": "Gemma-4-26B-A4B"}
TASKS = {0: "code", 1: "summarization", 2: "math"}
HOOKED = ["qstar_paper", "qstar_paper_min1", "reuse_aware", "qstar_engine"]
NOHOOK = ["qstar_paper_nohook", "qstar_engine_nohook"]
PNAME = {"qstar_paper": "paper q*", "qstar_paper_min1": "paper q* (min 1 fill)",
         "reuse_aware": "Reuse-Aware (original)", "qstar_engine": "built-in q*",
         "qstar_paper_nohook": "paper q* (no hook)", "qstar_engine_nohook": "built-in q* (no hook)"}
PCOL = {"qstar_paper": "#2a78d6", "qstar_paper_min1": "#e87ba4", "reuse_aware": "#1baf7a",
        "qstar_engine": "#eb6834"}
OWN = {"reuse_aware": "q_orig"}
LIST_COLS = ("routed_ids", "missing_ids", "fetched_ids", "evicted_gids", "victim_gids", "p_miss",
             "pred_fetched")


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps, srv = [], []
    for m in MODELS:
        p = root / "runtime_q/results" / f"exp1_{m}_episodes.jsonl"
        if p.exists():
            d = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
            d["model"] = m
            eps.append(d)
        s = p.with_name(f"exp1_{m}_servers.jsonl")
        if s.exists():
            for l in s.read_text().splitlines():
                r = json.loads(l)
                g = r.get("geometry") or {}
                srv.append({"model": m, "server": r["server"], "cache_rate": r["cache_rate"],
                            "problem": r["problem"], "hook": r["hook"], "ready_s": r["ready_s"],
                            "moe_cache_size": g.get("moe_cache_size"), "kv_pages": g.get("num_pages"),
                            "page_size": g.get("page_size"),
                            "moe_per_expert_b": (g.get("unit_bytes") or {}).get("moe_per_expert"),
                            "kv_per_token_b": (g.get("unit_bytes") or {}).get("kv_per_token"),
                            "cache_budget_b": g.get("cache_budget_bytes"),
                            "gpu_mem_used_mib": (r.get("gpu") or {}).get("mem_used_mib")})
    ep = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    if len(ep):
        ep["task"] = ep["problem"].map(TASKS)
    return ep, pd.DataFrame(srv)


def timing(ep: pd.DataFrame) -> pd.DataFrame:
    hooked = ep.policy.isin(HOOKED) & (ep.rep <= 2) & ~ep["logged"].astype(bool)
    return ep[hooked | ep.policy.isin(NOHOOK)]


def paired(perf: pd.DataFrame, ref: str, arms: list[str], metric: str = "ms_per_token") -> pd.DataFrame:
    piv = perf[perf.policy.isin(arms + [ref])].pivot_table(
        index=["model", "cache_rate", "task", "rep"], columns="policy", values=metric)
    rows = []
    for arm in arms:
        if arm == ref or arm not in piv or ref not in piv:
            continue
        for key, v in (piv[ref] / piv[arm]).dropna().items():
            rows.append({"policy": arm, "model": key[0], "cache_rate": key[1], "task": key[2],
                         "rep": key[3], "speedup": v})
    return pd.DataFrame(rows)


def summarize(pr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pr.groupby(by):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "pairs": len(g),
                     "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi, "p": p})
    return pd.DataFrame(rows)


def ids(v) -> list[int]:
    if not isinstance(v, str) or v == "":
        return []
    return [int(float(t)) for t in v.split(";") if t != ""]


def diagnostics(root: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        p = root / "runtime_q/results" / f"exp1_lean_{m}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False, dtype={c: str for c in LIST_COLS})
        d = d[d.phase == "measure"]
        for (pol, rate, prob), g in d.groupby(["policy", "cache_rate", "problem"]):
            ntok = g["tok"].nunique()
            miss = g[g.m > 0]
            own = OWN.get(pol)
            rows.append({
                "model": m, "policy": pol, "cache_rate": rate, "task": TASKS.get(prob, prob),
                "decide_ms_per_token": g["decide_us"].sum() / 1e3 / ntok,
                "moe_ms_per_token": g["wall_ms"].sum() / ntok,
                "misses_per_token": g["m"].sum() / ntok, "fetches_per_token": g["n_fetch"].sum() / ntok,
                "q_over_m": miss["n_fetch"].sum() / max(1, miss["m"].sum()),
                "hit_rate": g["n_hits"].sum() / max(1, (g["n_hits"] + g["m"]).sum()),
                "evictions_per_token": g["n_evict"].sum() / ntok,
                "occ_mean": g["occ"].mean(), "pcie_ms_per_token": g["pcie_ms"].sum() / ntok,
                "cpu_task_ms_per_token": g["cpu_task_ms"].sum() / ntok,
                "own_rule_match": (miss[own] == miss["q"]).mean() if own else np.nan,
            })
    return pd.DataFrame(rows)


def plots(out: Path, perf: pd.DataFrame, s_rate: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.6})
    files = []
    present = [m for m in MODELS if m in set(perf.model)]
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.2), dpi=180, squeeze=False)
    for ax, m in zip(axes[0], present):
        d = perf[(perf.model == m) & perf.policy.isin(HOOKED)]
        rates = sorted(d.cache_rate.unique())
        w = 0.8 / len(HOOKED)
        for i, pol in enumerate(HOOKED):
            v = [d[(d.cache_rate == r) & (d.policy == pol)].ms_per_token.mean() for r in rates]
            xs = [j + (i - 1.5) * w for j in range(len(rates))]
            ax.bar(xs, v, w * 0.92, color=PCOL[pol], label=PNAME[pol], zorder=3)
            for x, y in zip(xs, v):
                ax.text(x, y + 0.5, f"{y:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(rates)), [f"cache {r:.2f}" for r in rates])
        ax.set_title(MNAME[m], fontsize=9)
        ax.set_ylabel("decode ms/token (lower is better)")
        ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("Experiment 1: mean decode latency (instrumented, 3 tasks x 3 rounds)", fontsize=9)
    fig.tight_layout()
    f = out / "exp1_latency.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)

    s = s_rate[s_rate.policy != "qstar_paper"]
    fig, ax = plt.subplots(figsize=(7.4, 0.28 * len(s) + 1.0), dpi=180)
    y, labels = 0, []
    for m in present:
        for pol in ["qstar_paper_min1", "reuse_aware", "qstar_engine"]:
            for _, r in s[(s.model == m) & (s.policy == pol)].sort_values("cache_rate").iterrows():
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=PCOL[pol], lw=2)
                ax.plot(r.geo_speedup, y, "o", color=PCOL[pol], mec="white", ms=6)
                ax.text(max(r.ci_hi, r.geo_speedup) + 0.01, y, f"{r.geo_speedup:.3f}x", va="center", fontsize=6.5)
                labels.append(f"{MNAME[m]} | {PNAME[pol]} | {r.cache_rate:.2f}")
                y -= 1
        y -= 0.6
    ticks = [t for t in np.arange(0, y, -1)][:len(labels)]
    ax.set_yticks([i for i in range(0, -len(labels) - 100, -1)][:0])
    ax.axvline(1, color="#52514e", ls="--", lw=0.8)
    yy = []
    y = 0
    for m in present:
        for pol in ["qstar_paper_min1", "reuse_aware", "qstar_engine"]:
            for _ in range(len(s[(s.model == m) & (s.policy == pol)])):
                yy.append(y)
                y -= 1
        y -= 0.6
    ax.set_yticks(yy, labels, fontsize=6.5)
    ax.set_xlabel("speedup over paper q* = ms/token(paper q*) / ms/token(policy); bar = 95% CI")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    f = out / "exp1_speedup_vs_paper_qstar.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / "analysis/exp1"
    out.mkdir(parents=True, exist_ok=True)
    ep, srv = load(root)
    ep.drop(columns=[c for c in ("text",) if c in ep]).to_csv(out / "episodes.csv", index=False)
    srv.to_csv(out / "servers.csv", index=False)
    perf = timing(ep)
    cells = perf.groupby(["model", "cache_rate", "task", "policy"]).agg(
        n=("ms_per_token", "size"), ms_per_token=("ms_per_token", "mean"), ms_std=("ms_per_token", "std"),
        tok_s=("decode_tok_s", "mean"), ttft_ms=("ttft_ms", "mean"), ttft_std=("ttft_ms", "std"),
        e2e_ms=("e2e_ms", "mean"), p95_ms=("p95_ms", "mean")).reset_index()
    cells.to_csv(out / "cells.csv", index=False)

    pr = paired(perf, "qstar_paper", HOOKED)
    pr.to_csv(out / "paired_vs_qstar_paper.csv", index=False)
    s_model = summarize(pr, ["model", "policy"])
    s_rate = summarize(pr, ["model", "policy", "cache_rate"])
    s_cell = summarize(pr, ["model", "policy", "cache_rate", "task"])
    pr_ra = paired(perf, "reuse_aware", ["qstar_paper_min1", "qstar_engine"])
    s_ra = summarize(pr_ra, ["model", "policy", "cache_rate"]) if len(pr_ra) else pd.DataFrame()
    pr_ttft = paired(perf, "qstar_paper", HOOKED, "ttft_ms")
    s_ttft = summarize(pr_ttft, ["model", "policy"]) if len(pr_ttft) else pd.DataFrame()
    pr_e2e = paired(perf, "qstar_paper", HOOKED, "e2e_ms")
    s_e2e = summarize(pr_e2e, ["model", "policy", "cache_rate"]) if len(pr_e2e) else pd.DataFrame()
    for name, s in (("speedup_by_model", s_model), ("speedup_by_rate", s_rate), ("speedup_by_cell", s_cell),
                    ("vs_reuse_aware_by_rate", s_ra), ("ttft_speedup_by_model", s_ttft),
                    ("e2e_speedup_by_rate", s_e2e)):
        s.to_csv(out / f"{name}.csv", index=False)

    # uninstrumented: paper q* vs built-in q* (code task; separate servers, rep-indexed pairs)
    nh = perf[perf.policy.isin(NOHOOK) & (perf.task == "code")]
    pr_nh = paired(nh, "qstar_paper_nohook", NOHOOK)
    s_nh = summarize(pr_nh, ["model", "policy", "cache_rate"]) if len(pr_nh) else pd.DataFrame()
    # instrumentation overhead: hooked vs unhooked paper q* in the same cell (different servers)
    ov = []
    for (m, r, t), g in perf.groupby(["model", "cache_rate", "task"]):
        h = g[g.policy == "qstar_paper"].ms_per_token
        u = g[g.policy == "qstar_paper_nohook"].ms_per_token
        if len(h) and len(u):
            ov.append({"model": m, "cache_rate": r, "task": t, "hooked_ms": h.mean(), "nohook_ms": u.mean(),
                       "overhead_ms": h.mean() - u.mean(), "hooked_over_nohook": h.mean() / u.mean()})
    ov = pd.DataFrame(ov)
    ov.to_csv(out / "instrumentation_overhead.csv", index=False)

    diag = diagnostics(root)
    diag.to_csv(out / "diagnostics.csv", index=False)
    hashes = ep[ep.rep >= 0].groupby(["model", "cache_rate", "task"]).agg(
        distinct_outputs=("output_sha1", "nunique"), min_steps=("decode_steps", "min"),
        max_steps=("decode_steps", "max"), episodes=("output_sha1", "size")).reset_index()
    hashes.to_csv(out / "quality.csv", index=False)
    figs = plots(out, perf, s_rate) if len(perf) else []

    fm = {"geo_speedup": "{:.3f}x", "ci_lo": "{:.3f}", "ci_hi": "{:.3f}", "p": "{:.4f}",
          "ms_per_token": "{:.2f}", "ms_std": "{:.2f}", "tok_s": "{:.2f}", "ttft_ms": "{:.0f}",
          "ttft_std": "{:.0f}", "e2e_ms": "{:.0f}", "p95_ms": "{:.1f}", "hooked_ms": "{:.2f}",
          "nohook_ms": "{:.2f}", "overhead_ms": "{:+.2f}", "hooked_over_nohook": "{:.3f}",
          "decide_ms_per_token": "{:.2f}", "moe_ms_per_token": "{:.1f}", "misses_per_token": "{:.1f}",
          "fetches_per_token": "{:.2f}", "q_over_m": "{:.3f}", "hit_rate": "{:.3f}",
          "evictions_per_token": "{:.2f}", "occ_mean": "{:.0f}", "pcie_ms_per_token": "{:.1f}",
          "cpu_task_ms_per_token": "{:.1f}", "own_rule_match": "{:.3f}", "ready_s": "{:.0f}"}
    sv = srv.groupby(["model", "cache_rate", "hook"]).agg(
        servers=("server", "size"), moe_cache_size=("moe_cache_size", "first"), kv_pages=("kv_pages", "first"),
        moe_per_expert_b=("moe_per_expert_b", "first"), cache_budget_b=("cache_budget_b", "first"),
        ready_s=("ready_s", "mean")).reset_index() if len(srv) else srv
    lines = [
        "# Experiment 1 — original Reuse-Aware vs paper q* (RTX 4070)\n",
        f"Root `{root}`. Episodes {len(ep)} (timing rows {len(perf)}). Speedup = ms/token(paper q*) / "
        "ms/token(policy), paired per (model, cache rate, task, round) on the same hooked server; "
        "geometric mean, 95% t-CI. Hooked = lean runtime-q harness (instrumented); no-hook rows are "
        "uninstrumented servers and are compared only with each other.\n",
        "## Speedup over paper q* — per model\n", md(s_model, fm),
        "\n## Speedup over paper q* — per model and cache rate\n", md(s_rate, fm),
        "\n## Speedup over paper q* — per cell\n", md(s_cell, fm),
        "\n## Original Reuse-Aware as reference (speedup of the other arm over Reuse-Aware)\n",
        md(s_ra, fm) if len(s_ra) else "(none)",
        "\n## TTFT ratio (paper q* TTFT / policy TTFT) per model\n", md(s_ttft, fm) if len(s_ttft) else "(none)",
        "\n## End-to-end ratio (paper q* e2e / policy e2e) per model and rate\n", md(s_e2e, fm) if len(s_e2e) else "(none)",
        "\n## Uninstrumented: built-in q* vs paper q* (code task)\n", md(s_nh, fm) if len(s_nh) else "(none)",
        "\n## Instrumentation overhead (hooked vs unhooked paper q*, same cell, different servers)\n",
        md(ov, fm) if len(ov) else "(none)",
        "\n## Per-cell latency, TTFT, throughput\n", md(cells, fm),
        "\n## Decision-log diagnostics (logged round 3, measure phase)\n", md(diag, fm) if len(diag) else "(none)",
        "\n## Server geometry (expert slots, KV pages, budget) and load time\n", md(sv, fm) if len(sv) else "(none)",
        "\n## Output consistency (distinct greedy outputs per cell, all arms and rounds)\n", md(hashes, fm),
        "\n## Figures\n", *[f"- `{f}`" for f in figs],
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out / "REPORT.md").read_text()[:4000])


if __name__ == "__main__":
    main()
