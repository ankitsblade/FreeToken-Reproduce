"""Reuse-aware v2 pilot (gpt-oss-20b, RTX 4070): harness overhead ladder and variant comparison.

Reads one pilot root written by `configs/runtime_q/reuse_aware_v2/gpt_oss_pilot.json`:
  runtime_q/results/gpt_oss_pilot_episodes.jsonl  client ms/token per episode
  runtime_q/results/lean_gpt_oss.csv              lean-harness decision rows (diagnostic rep 3)
  runtime_q/results/study_gpt_oss.csv             study-harness decision rows (diagnostic rep 3)
  runtime_q/results/study_csv_gpt_oss.csv         study harness with CSV logging on every rep

Timing uses repeats 0-2 only (no decision logging); repeat 3 is diagnostic. Speedups are
paired within a repeat round of one server (cache size x task) and summarized as geometric
means with 95% t-intervals on the log ratio. Pre-registered test: a variant advances only if
it is not slower than reuse_aware_lean at either cache size (CI upper bound >= 1) and is
faster at one (CI lower bound > 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runtime_q.analyze_reuse_aware_models import geo_ci, md

TASKS = {0: "code", 2: "math"}
REF = "reuse_aware_lean"
ARMS = ["qstar_paper_lean", "qstar_engine_lean", "reuse_aware_lean", "reuse_aware_study",
        "ra_v1_cost", "ra_v2_alpha", "ra_v3_evict"]
LADDER = ["qstar_engine_nohook", "qstar_engine_bare", "qstar_engine_study", "qstar_engine_study_csv",
          "qstar_engine_lean"]
VARIANTS = {"ra_v1_cost": "q_v1", "ra_v2_alpha": "q_v2", "ra_v3_evict": "q_v3", "reuse_aware_lean": "q_orig"}
NUM_LAYERS = 24


def ids(v) -> list[int]:
    if not isinstance(v, str) or v == "":
        return []
    return [int(float(t)) for t in v.split(";") if t != ""]


def load(root: Path) -> pd.DataFrame:
    path = root / "runtime_q" / "results" / "gpt_oss_pilot_episodes.jsonl"
    ep = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    ep["task"] = ep["problem"].map(TASKS)
    return ep


def paired(perf: pd.DataFrame, ref: str) -> pd.DataFrame:
    piv = perf.pivot_table(index=["cache_rate", "task", "rep"], columns="policy", values="ms_per_token")
    rows = []
    for arm in piv.columns:
        if arm == ref:
            continue
        s = (piv[ref] / piv[arm]).dropna()
        for (rate, task, rep), v in s.items():
            rows.append({"policy": arm, "cache_rate": rate, "task": task, "rep": rep, "speedup": v})
    return pd.DataFrame(rows)


def summarize(pr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pr.groupby(by):
        gm, lo, hi, p = geo_ci(g["speedup"].to_numpy())
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "pairs": len(g),
                     "geo_speedup": gm, "ci_lo": lo, "ci_hi": hi, "p": p})
    return pd.DataFrame(rows)


def lean_diagnostics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path, low_memory=False, dtype={c: str for c in (
        "routed_ids", "missing_ids", "fetched_ids", "evicted_gids", "victim_gids", "p_miss", "pred_fetched")})
    d = d[d.phase == "measure"]
    rows, vic = [], []
    for (pol, rate, prob), g in d.groupby(["policy", "cache_rate", "problem"]):
        ntok = g["tok"].nunique()
        last = g.sort_values(["tok", "layer_id"]).iloc[-1]
        own = VARIANTS.get(pol)
        miss = g[g.m > 0]
        rows.append({
            "policy": pol, "cache_rate": rate, "task": TASKS.get(prob, prob),
            "decide_us_median": g["decide_us"].median(), "decide_ms_per_token": g["decide_us"].sum() / 1e3 / ntok,
            "moe_ms_per_token": g["wall_ms"].sum() / ntok,
            "misses_per_token": g["m"].sum() / ntok, "fetches_per_token": g["n_fetch"].sum() / ntok,
            "q_over_m": miss["n_fetch"].sum() / max(1, miss["m"].sum()),
            "hit_rate": g["n_hits"].sum() / max(1, (g["n_hits"] + g["m"]).sum()),
            "evictions_per_token": g["n_evict"].sum() / ntok,
            "alpha_end": last["alpha"], "v_hit_ms_end": last["v_hit_ms"],
            "g0_ms": last["g0_ms"], "tp_ms": last["tp_ms"], "c0_ms": last["c0_ms"], "tc_ms": last["tc_ms"],
            "q_orig_shadow": miss["q_orig"].mean(), "q_v1_shadow": miss["q_v1"].mean(),
            "q_v2_shadow": miss["q_v2"].mean(), "q_v3_shadow": miss["q_v3"].mean(), "q_taken": miss["q"].mean(),
            "own_rule_match": (miss[own] == miss["q"]).mean() if own else np.nan,
        })
        for r in g[g.n_evict > 0].itertuples(index=False):
            ev = ids(r.evicted_gids)
            pred = ids(r.victim_gids)[:r.n_fetch]
            vic.append({"policy": pol, "cache_rate": rate,
                        "victims_correct": len(set(ev) & set(pred)) / len(ev)})
    v = pd.DataFrame(vic)
    vs = v.groupby(["policy", "cache_rate"]).victims_correct.mean().reset_index() if len(v) else v
    return pd.DataFrame(rows), vs


def study_decide(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, usecols=["phase", "policy", "cache_rate", "decide_us", "wall_ms"], low_memory=False)
    d = d[d.phase == "measure"]
    g = d.groupby(["policy", "cache_rate"])
    return pd.DataFrame({"decide_us_median": g.decide_us.median(),
                         "decide_ms_per_token": g.decide_us.sum() / 1e3 / (g.size() / NUM_LAYERS),
                         "moe_ms_per_token": g.wall_ms.sum() / (g.size() / NUM_LAYERS)}).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    ep = load(root)
    ep.to_csv(out / "episodes.csv", index=False)
    perf = ep[(ep.rep <= 2) & ~ep["logged"].astype(bool)]

    cells = perf.groupby(["cache_rate", "task", "policy"]).ms_per_token.agg(["mean", "std", "size"]).reset_index()
    cells.to_csv(out / "cells.csv", index=False)

    pr = paired(perf[perf.policy.isin(ARMS)], REF)
    pr.to_csv(out / "paired_vs_reuse_aware_lean.csv", index=False)
    s_all = summarize(pr, ["policy"])
    s_rate = summarize(pr, ["policy", "cache_rate"])
    s_cell = summarize(pr, ["policy", "cache_rate", "task"])
    for name, s in (("speedup_vs_ra_lean.csv", s_all), ("speedup_vs_ra_lean_by_rate.csv", s_rate),
                    ("speedup_vs_ra_lean_by_cell.csv", s_cell)):
        s.to_csv(out / name, index=False)
    pr_paper = paired(perf[perf.policy.isin(ARMS)], "qstar_paper_lean")
    s_paper = summarize(pr_paper, ["policy", "cache_rate"])
    s_paper.to_csv(out / "speedup_vs_qstar_paper_lean_by_rate.csv", index=False)

    verdict = []
    for pol in ("ra_v1_cost", "ra_v2_alpha", "ra_v3_evict"):
        s = s_rate[s_rate.policy == pol]
        if len(s) == 0:
            continue
        slower = bool((s.ci_hi < 1).any())
        faster = bool((s.ci_lo > 1).any())
        verdict.append({"policy": pol, "slower_at_a_rate": slower, "faster_at_a_rate": faster,
                        "advances": faster and not slower})
    verdict = pd.DataFrame(verdict)

    lad = ep[ep.policy.isin(LADDER) & (ep.task == "code") & ((ep.rep <= 2) & (~ep["logged"].astype(bool) |
                                                                              (ep.policy == "qstar_engine_study_csv")))]
    ladder = lad.groupby(["cache_rate", "policy"]).ms_per_token.agg(["mean", "std", "size"]).reset_index()
    base = ladder[ladder.policy == "qstar_engine_nohook"].set_index("cache_rate")["mean"]
    ladder["minus_nohook_ms"] = ladder["mean"] - ladder.cache_rate.map(base)
    ladder["policy"] = pd.Categorical(ladder.policy, LADDER)
    ladder = ladder.sort_values(["cache_rate", "policy"])
    ladder.to_csv(out / "overhead_ladder.csv", index=False)

    hashes = ep[ep.rep >= 0].groupby(["cache_rate", "task"]).output_sha1.nunique().reset_index()

    lean_path = root / "runtime_q" / "results" / "lean_gpt_oss.csv"
    diag, vic = (lean_diagnostics(lean_path) if lean_path.exists() else (pd.DataFrame(), pd.DataFrame()))
    diag.to_csv(out / "lean_diagnostics.csv", index=False)
    vic.to_csv(out / "victim_prediction.csv", index=False)
    sd = study_decide(root / "runtime_q" / "results" / "study_gpt_oss.csv")
    sd.to_csv(out / "study_decide.csv", index=False)

    fm = {"geo_speedup": "{:.3f}x", "ci_lo": "{:.3f}", "ci_hi": "{:.3f}", "p": "{:.4f}", "mean": "{:.2f}",
          "std": "{:.2f}", "minus_nohook_ms": "{:+.2f}", "decide_us_median": "{:.0f}",
          "decide_ms_per_token": "{:.2f}", "moe_ms_per_token": "{:.1f}", "misses_per_token": "{:.1f}",
          "fetches_per_token": "{:.1f}", "q_over_m": "{:.3f}", "hit_rate": "{:.3f}",
          "evictions_per_token": "{:.1f}", "alpha_end": "{:.2f}", "v_hit_ms_end": "{:.3f}", "g0_ms": "{:.3f}",
          "tp_ms": "{:.3f}", "c0_ms": "{:.3f}", "tc_ms": "{:.3f}", "q_orig_shadow": "{:.2f}",
          "q_v1_shadow": "{:.2f}", "q_v2_shadow": "{:.2f}", "q_v3_shadow": "{:.2f}", "q_taken": "{:.2f}",
          "own_rule_match": "{:.3f}", "victims_correct": "{:.3f}"}
    lines = [
        f"# Reuse-aware v2 pilot — gpt-oss-20b, RTX 4070\n\nRoot: `{root}`. Timing = repeats 0–2 "
        "(no decision logging); repeat 3 is diagnostic only. Episodes: "
        f"{len(ep)} ({len(perf)} timing).\n",
        "## Speedup vs reuse_aware_lean (paired; >1 = arm faster)\n", md(s_all, fm),
        "\n### By cache rate\n", md(s_rate, fm), "\n### By cell\n", md(s_cell, fm),
        "\n### Pre-registered rule\n", md(verdict, fm) if len(verdict) else "(no variants)",
        "\n## Speedup vs qstar_paper_lean by cache rate\n", md(s_paper, fm),
        "\n## Mean ms/token per cell (repeats 0–2)\n", md(cells, fm),
        "\n## Harness overhead ladder (code task; built-in q*)\n"
        "nohook = separate unhooked server (expert cache not reset between episodes; approximate). "
        "study_csv includes every repeat with CSV logging.\n", md(ladder, fm),
        "\n## Distinct output hashes per cell (all arms, all reps)\n", md(hashes, fm),
        "\n## Lean-harness diagnostics (repeat 3, measure phase)\n", md(diag, fm) if len(diag) else "(missing)",
        "\n### Victim prediction (share of actual evictions predicted)\n", md(vic, fm) if len(vic) else "(missing)",
        "\n### Study-harness decide time (repeat 3)\n", md(sd, fm) if len(sd) else "(missing)",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
