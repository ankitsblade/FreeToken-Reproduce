"""Policy study analysis: paper q* (A) vs original Reuse-Aware (B) vs runtime-variable-aware (C).

Reads <root>/runtime_q/results/stage{1,2}_<model>_episodes.jsonl, *_servers.jsonl and the lean
decision logs stage{1,2}_lean_<model>.csv; writes <root>/analysis/{REPORT.md, *.csv, *.png}.

Timing rows: hooked arms, repeats 0-2 (repeat 3 writes the decision log and is excluded);
unhooked arms, all repeats. Speedup of X over reference R in one (model, cache rate, task,
repeat) round on the same server = ms/token(R) / ms/token(X) (decode inter-token latency);
summaries are geometric means with 95% t-intervals on the log ratios. Hooked and unhooked arms
are never mixed inside one paired ratio.
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
A, A0, E, B = "A_qstar_paper_min1", "A0_qstar_paper", "E_qstar_engine", "B_reuse_aware"
S1 = [A, A0, E, B]
C_INC = ["C_V1", "C_V2", "C_V3", "C_V1V2V3"]          # each runtime variant alone, then all three
C_LOO = ["C_V2V3", "C_V1V3", "C_V1V2"]                # all three minus V1 / V2 / V3
S2 = [A, B] + C_INC + C_LOO
NOHOOK = ["A0_qstar_paper_nohook", "E_qstar_engine_nohook"]
LABEL = {A: "A paper q* (min 1 fill)", A0: "A0 paper q* (no minimum)", E: "E built-in q*",
         B: "B Reuse-Aware (original)", "C_V1": "B + V1 measured costs", "C_V2": "B + V2 calibrated reuse",
         "C_V3": "B + V3 eviction-aware", "C_V1V2V3": "C full (V1+V2+V3)", "C_V2V3": "C full - V1",
         "C_V1V3": "C full - V2", "C_V1V2": "C full - V3",
         "A0_qstar_paper_nohook": "A0 paper q* (no hook)", "E_qstar_engine_nohook": "E built-in q* (no hook)"}
COL = {A: "#2a78d6", A0: "#e87ba4", E: "#eb6834", B: "#1baf7a", "C_V1V2V3": "#eda100"}
OWN = {B: "q_orig", "C_V1": "q_v1", "C_V2": "q_v2", "C_V3": "q_v3"}
LIST_COLS = ("routed_ids", "missing_ids", "fetched_ids", "evicted_gids", "victim_gids", "p_miss", "pred_fetched")


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps, srv = [], []
    for stage in ("stage1", "stage2"):
        for m in MODELS:
            p = root / "runtime_q/results" / f"{stage}_{m}_episodes.jsonl"
            if p.exists():
                d = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
                d["model"], d["stage"] = m, stage
                eps.append(d)
            s = p.with_name(f"{stage}_{m}_servers.jsonl")
            if s.exists():
                for l in s.read_text().splitlines():
                    r = json.loads(l)
                    g = r.get("geometry") or {}
                    srv.append({"stage": stage, "model": m, "cache_rate": r["cache_rate"], "problem": r["problem"],
                                "hook": r["hook"], "ready_s": r["ready_s"], "moe_cache_size": g.get("moe_cache_size"),
                                "kv_pages": g.get("num_pages"),
                                "moe_per_expert_b": (g.get("unit_bytes") or {}).get("moe_per_expert"),
                                "cache_budget_b": g.get("cache_budget_bytes"),
                                "gpu_mem_used_mib": (r.get("gpu") or {}).get("mem_used_mib"),
                                "gpu_temp_c": (r.get("gpu") or {}).get("temp_c")})
    ep = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    if len(ep):
        ep["task"] = ep["problem"].map(TASKS)
    return ep, pd.DataFrame(srv)


def timing(ep: pd.DataFrame) -> pd.DataFrame:
    hooked = ~ep.policy.isin(NOHOOK) & (ep.rep <= 2) & ~ep["logged"].astype(bool)
    return ep[hooked | ep.policy.isin(NOHOOK)]


def paired(perf: pd.DataFrame, ref: str, arms: list[str], metric: str = "ms_per_token") -> pd.DataFrame:
    piv = perf[perf.policy.isin(arms + [ref])].pivot_table(
        index=["model", "cache_rate", "task", "rep"], columns="policy", values=metric)
    rows = []
    for arm in arms:
        if arm == ref or arm not in piv or ref not in piv:
            continue
        for key, v in (piv[ref] / piv[arm]).dropna().items():
            rows.append({"policy": arm, "model": key[0], "cache_rate": key[1], "task": key[2], "rep": key[3],
                         "speedup": v})
    return pd.DataFrame(rows)


def summarize(pr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pr.groupby(by):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "pairs": len(g),
                     "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi, "p": p})
    return pd.DataFrame(rows)


def diagnostics(root: Path) -> pd.DataFrame:
    rows = []
    for stage in ("stage1", "stage2"):
        for m in MODELS:
            p = root / "runtime_q/results" / f"{stage}_lean_{m}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p, low_memory=False, dtype={c: str for c in LIST_COLS})
            d = d[d.phase == "measure"]
            for (pol, rate, prob), g in d.groupby(["policy", "cache_rate", "problem"]):
                ntok = g["tok"].nunique()
                miss = g[g.m > 0]
                own = OWN.get(pol)
                rows.append({
                    "stage": stage, "model": m, "policy": pol, "cache_rate": rate, "task": TASKS.get(prob, prob),
                    "hit_rate": g["n_hits"].sum() / max(1, (g["n_hits"] + g["m"]).sum()),
                    "misses_per_token": g["m"].sum() / ntok, "fetches_per_token": g["n_fetch"].sum() / ntok,
                    "q_over_m": miss["n_fetch"].sum() / max(1, miss["m"].sum()),
                    "evictions_per_token": g["n_evict"].sum() / ntok, "occ_mean": g["occ"].mean(),
                    "pcie_ms_per_token": g["pcie_ms"].sum() / ntok, "gemm_ms_per_token": g["gemm_ms"].sum() / ntok,
                    "cpu_task_ms_per_token": pd.to_numeric(g["cpu_task_ms"], errors="coerce").sum() / ntok,
                    "moe_ms_per_token": g["wall_ms"].sum() / ntok,
                    "decide_ms_per_token": g["decide_us"].sum() / 1e3 / ntok,
                    "own_rule_match": (miss[own] == miss["q"]).mean() if own and own in miss else np.nan,
                })
    return pd.DataFrame(rows)


def plots(out: Path, perf: pd.DataFrame, s1_rate: pd.DataFrame, s2_model: pd.DataFrame, ov: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.6})
    files = []
    present = [m for m in MODELS if m in set(perf.model)]
    # 1. stage 1 latency per model
    st1 = perf[perf.stage == "stage1"]
    if len(st1):
        fig, axes = plt.subplots(1, len(present), figsize=(3.7 * len(present), 3.3), dpi=180, squeeze=False)
        for ax, m in zip(axes[0], present):
            d = st1[(st1.model == m) & st1.policy.isin(S1)]
            rates = sorted(d.cache_rate.unique())
            w = 0.8 / len(S1)
            for i, pol in enumerate(S1):
                v = [d[(d.cache_rate == r) & (d.policy == pol)].ms_per_token.mean() for r in rates]
                xs = [j + (i - 1.5) * w for j in range(len(rates))]
                ax.bar(xs, v, w * 0.92, color=COL[pol], label=LABEL[pol], zorder=3)
                for x, y in zip(xs, v):
                    if y == y:
                        ax.text(x, y + 0.4, f"{y:.0f}", ha="center", va="bottom", fontsize=5.8)
            ax.set_xticks(range(len(rates)), [f"cache {r:.2f}" for r in rates])
            ax.set_title(MNAME[m], fontsize=9)
            ax.set_ylabel("decode ms/token (lower is better)")
            ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].legend(fontsize=6.3, frameon=False, loc="upper right")
        fig.suptitle("Stage 1: mean decode latency, instrumented (3 tasks x 3 rounds per bar)", fontsize=9)
        fig.tight_layout()
        f = out / "stage1_latency.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)
    # 2. stage 1 forest: speedup over A
    if len(s1_rate):
        rows = s1_rate.sort_values(["model", "policy", "cache_rate"])
        fig, ax = plt.subplots(figsize=(7.6, 0.24 * len(rows) + 1.1), dpi=180)
        ys, labels = [], []
        y = 0
        for m in present:
            for pol in (A0, E, B):
                for _, r in rows[(rows.model == m) & (rows.policy == pol)].iterrows():
                    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=COL[pol], lw=2)
                    ax.plot(r.geo_speedup, y, "o", color=COL[pol], mec="white", ms=5.5)
                    ax.text(r.ci_hi + 0.01, y, f"{r.geo_speedup:.3f}x", va="center", fontsize=6)
                    ys.append(y)
                    labels.append(f"{MNAME[m]} | {LABEL[pol]} | {r.cache_rate:.2f}")
                    y -= 1
            y -= 0.7
        ax.axvline(1, color="#52514e", ls="--", lw=0.8)
        ax.set_yticks(ys, labels, fontsize=6)
        ax.set_xlabel("speedup over A = ms/token(A) / ms/token(policy); bar = 95% CI")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        f = out / "stage1_speedup_vs_A.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)
    # 3. stage 2 ablation vs B
    if len(s2_model):
        arms = C_INC + C_LOO + [A]
        fig, axes = plt.subplots(1, len(present), figsize=(3.7 * len(present), 3.6), dpi=180, squeeze=False, sharey=True)
        for ax, m in zip(axes[0], present):
            d = s2_model[s2_model.model == m].set_index("policy")
            ys = list(range(len(arms)))[::-1]
            for yy, pol in zip(ys, arms):
                if pol not in d.index:
                    continue
                r = d.loc[pol]
                c = "#2a78d6" if pol == A else ("#eda100" if pol in C_INC else "#8a8984")
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=c, lw=2)
                ax.plot(r.geo_speedup, yy, "o", color=c, mec="white", ms=5.5)
                ax.text(r.ci_hi + 0.005, yy, f"{r.geo_speedup:.3f}", va="center", fontsize=6)
            ax.axvline(1, color="#52514e", ls="--", lw=0.8)
            ax.set_yticks(ys, [LABEL[a] for a in arms], fontsize=6.3)
            ax.set_title(MNAME[m], fontsize=9)
            ax.set_xlabel("speedup over B (95% CI)", fontsize=7)
            ax.spines[["top", "right", "left"]].set_visible(False)
        fig.suptitle("Stage 2: Policy C feature ablations vs original Reuse-Aware (B)", fontsize=9)
        fig.tight_layout()
        f = out / "stage2_ablation_vs_B.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)
    # 4. instrumentation overhead
    if len(ov):
        fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=180)
        lab = [f"{MNAME[r.model]}\n{r.arm.split('_')[0]} @ {r.cache_rate:.2f}" for r in ov.itertuples()]
        x = np.arange(len(ov))
        ax.bar(x - 0.2, ov.nohook_ms, 0.38, color="#8a8984", label="no hook (uninstrumented)", zorder=3)
        ax.bar(x + 0.2, ov.hooked_ms, 0.38, color="#2a78d6", label="hooked (lean harness)", zorder=3)
        ax.set_xticks(x, lab, fontsize=5.5)
        ax.set_ylabel("ms/token, code task")
        ax.legend(fontsize=6.5, frameon=False)
        ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        f = out / "instrumentation_overhead.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ep, srv = load(root)
    ep.drop(columns=[c for c in ("text",) if c in ep]).to_csv(out / "episodes.csv", index=False)
    srv.to_csv(out / "servers.csv", index=False)
    perf = timing(ep)
    cells = perf.groupby(["stage", "model", "cache_rate", "task", "policy"]).agg(
        n=("ms_per_token", "size"), ms_per_token=("ms_per_token", "mean"), ms_std=("ms_per_token", "std"),
        tok_s=("decode_tok_s", "mean"), ttft_ms=("ttft_ms", "mean"), ttft_std=("ttft_ms", "std"),
        e2e_ms=("e2e_ms", "mean"), e2e_std=("e2e_ms", "std"), p95_ms=("p95_ms", "mean")).reset_index()
    cells.to_csv(out / "cells.csv", index=False)
    p1 = perf[perf.stage == "stage1"]
    p2 = perf[perf.stage == "stage2"]
    res = {}
    pr = paired(p1, A, S1)
    res["s1_model"] = summarize(pr, ["model", "policy"]) if len(pr) else pd.DataFrame()
    res["s1_rate"] = summarize(pr, ["model", "policy", "cache_rate"]) if len(pr) else pd.DataFrame()
    res["s1_cell"] = summarize(pr, ["model", "policy", "cache_rate", "task"]) if len(pr) else pd.DataFrame()
    pr_b = paired(p1, A0, [B])
    res["s1_B_vs_A0_rate"] = summarize(pr_b, ["model", "policy", "cache_rate"]) if len(pr_b) else pd.DataFrame()
    for metric in ("ttft_ms", "e2e_ms"):
        q = paired(p1, A, S1, metric)
        res[f"s1_{metric}_model"] = summarize(q, ["model", "policy"]) if len(q) else pd.DataFrame()
    pr2 = paired(p2, B, S2)
    res["s2_model"] = summarize(pr2, ["model", "policy"]) if len(pr2) else pd.DataFrame()
    res["s2_rate"] = summarize(pr2, ["model", "policy", "cache_rate"]) if len(pr2) else pd.DataFrame()
    res["s2_all"] = summarize(pr2, ["policy"]) if len(pr2) else pd.DataFrame()
    pr2a = paired(p2, A, S2)
    res["s2_vs_A_model"] = summarize(pr2a, ["model", "policy"]) if len(pr2a) else pd.DataFrame()
    loo = []
    if len(p2):
        for drop, arm in (("V1", "C_V2V3"), ("V2", "C_V1V3"), ("V3", "C_V1V2")):
            q = paired(p2, arm, ["C_V1V2V3"])
            if len(q):
                for m, g in q.groupby("model"):
                    gm, lo, hi, pv = geo_ci(g.speedup.to_numpy())
                    loo.append({"model": m, "group_removed": drop, "full_over_without": gm, "ci_lo": lo,
                                "ci_hi": hi, "p": pv, "pairs": len(g)})
    res["s2_loo"] = pd.DataFrame(loo)
    # uninstrumented comparison and instrumentation overhead (code task)
    nh = p1[p1.policy.isin(NOHOOK)]
    q = paired(nh, "A0_qstar_paper_nohook", NOHOOK)
    res["nohook"] = summarize(q, ["model", "policy", "cache_rate"]) if len(q) else pd.DataFrame()
    ov = []
    for (m, r), g in p1[p1.task == "code"].groupby(["model", "cache_rate"]):
        for hooked, unhooked in ((A0, "A0_qstar_paper_nohook"), (E, "E_qstar_engine_nohook")):
            h, u = g[g.policy == hooked].ms_per_token, g[g.policy == unhooked].ms_per_token
            if len(h) and len(u):
                ov.append({"model": m, "cache_rate": r, "arm": hooked, "hooked_ms": h.mean(), "nohook_ms": u.mean(),
                           "overhead_ms": h.mean() - u.mean(), "hooked_over_nohook": h.mean() / u.mean(),
                           "hooked_ttft": g[g.policy == hooked].ttft_ms.mean(),
                           "nohook_ttft": g[g.policy == unhooked].ttft_ms.mean()})
    ov = pd.DataFrame(ov)
    diag = diagnostics(root)
    quality = ep[ep.rep >= 0].groupby(["stage", "model", "cache_rate", "task"]).agg(
        episodes=("output_sha1", "size"), distinct_outputs=("output_sha1", "nunique"),
        min_steps=("decode_steps", "min"), max_steps=("decode_steps", "max")).reset_index()
    for name, d in {**res, "overhead": ov, "diagnostics": diag, "quality": quality}.items():
        d.to_csv(out / f"{name}.csv", index=False)
    figs = plots(out, perf, res["s1_rate"], res["s2_model"], ov) if len(perf) else []
    fm = {"geo_speedup": "{:.3f}x", "ci_lo": "{:.3f}", "ci_hi": "{:.3f}", "p": "{:.4f}", "ms_per_token": "{:.2f}",
          "ms_std": "{:.2f}", "tok_s": "{:.2f}", "ttft_ms": "{:.0f}", "ttft_std": "{:.0f}", "e2e_ms": "{:.0f}",
          "e2e_std": "{:.0f}", "p95_ms": "{:.1f}", "hooked_ms": "{:.2f}", "nohook_ms": "{:.2f}",
          "overhead_ms": "{:+.2f}", "hooked_over_nohook": "{:.3f}", "hooked_ttft": "{:.0f}", "nohook_ttft": "{:.0f}",
          "hit_rate": "{:.3f}", "misses_per_token": "{:.1f}", "fetches_per_token": "{:.2f}", "q_over_m": "{:.3f}",
          "evictions_per_token": "{:.2f}", "occ_mean": "{:.0f}", "pcie_ms_per_token": "{:.1f}",
          "gemm_ms_per_token": "{:.1f}", "cpu_task_ms_per_token": "{:.1f}", "moe_ms_per_token": "{:.1f}",
          "decide_ms_per_token": "{:.2f}", "own_rule_match": "{:.3f}", "full_over_without": "{:.3f}x"}

    def sec(title, d):
        return [f"\n## {title}\n", md(d, fm) if len(d) else "(no data yet)"]
    lines = [f"# Policy study — A paper q* vs B Reuse-Aware vs C runtime-variable-aware (RTX 4070)\n",
             f"Root `{root}`. Episodes {len(ep)} (timing rows {len(perf)}). Speedup = ms/token(reference) / "
             "ms/token(arm), paired per (model, cache rate, task, round) on the same server; geometric mean, "
             "95% t-CI. All hooked arms share the lean harness; no-hook arms are uninstrumented servers.\n"]
    lines += sec("Stage 1 — speedup over A (paper q*, min 1 fill), per model", res["s1_model"])
    lines += sec("Stage 1 — per model and cache rate", res["s1_rate"])
    lines += sec("Stage 1 — per cell", res["s1_cell"])
    lines += sec("Stage 1 — B over A0 (paper q* without minimum, earlier implementation)", res["s1_B_vs_A0_rate"])
    lines += sec("Stage 1 — TTFT ratio over A (TTFT(A)/TTFT(arm))", res["s1_ttft_ms_model"])
    lines += sec("Stage 1 — end-to-end ratio over A", res["s1_e2e_ms_model"])
    lines += sec("Stage 2 — speedup over B, all models pooled", res["s2_all"])
    lines += sec("Stage 2 — speedup over B per model", res["s2_model"])
    lines += sec("Stage 2 — speedup over B per model and cache rate", res["s2_rate"])
    lines += sec("Stage 2 — speedup over A per model", res["s2_vs_A_model"])
    lines += sec("Stage 2 — leave-one-group-out (V1+V2+V3 / without one variant; >1 = that variant helps)", res["s2_loo"])
    lines += sec("Uninstrumented: built-in q* over paper q* (code task, separate servers)", res["nohook"])
    lines += sec("Instrumentation overhead (hooked vs unhooked, code task, different servers)", ov)
    lines += sec("Per-cell latency, TTFT, throughput (timing rows)", cells)
    lines += sec("Decision-log diagnostics (logged round, measure phase)", diag)
    lines += sec("Output consistency per cell (all arms and rounds)", quality)
    lines += ["\n## Figures\n"] + [f"- `{f}`" for f in figs]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out / "REPORT.md").read_text()[:3000])


if __name__ == "__main__":
    main()
