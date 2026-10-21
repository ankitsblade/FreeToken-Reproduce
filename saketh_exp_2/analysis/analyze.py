"""Aggregate a saketh_exp_2 run: paired policy comparisons, 2^3 ablation, diagnostics, overhead.

Inputs (under --root, i.e. results/raw/<stamp>):
  runtime_q/results/main_<model>_episodes.jsonl   driver rows (one per measured episode)
  runtime_q/results/main_<model>_episodes_servers.jsonl   server geometry (auto cache size)
  runtime_q/results/sx2_stats_*.jsonl             hook counters per (episode, phase)
  reference/<model>_{graph,eager}.jsonl           unhooked official-benchmark runs
Timing uses rounds 0-4 only; round 5 writes the decision log and is excluded (as in the
policy study). Every comparison is paired within (problem, round): same server, same round.
Primary metric: decode ms/token = (t_last - t_first)/(completion_tokens - 1), the official
bench_decode_moe definition; also tok/s, TTFT and end-to-end time.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

MODELS = ("qwen", "gpt_oss", "gemma")
PAPER_MODEL = {"qwen": True, "gpt_oss": False, "gemma": False}
TIMED_REPS = range(5)
ARMS = ["A_qstar_paper", "A0_qstar_paper_nomin", "E_qstar_engine", "B_reuse_aware",
        "C_V1", "C_V2", "C_V3", "C_V1V2", "C_V1V3", "C_V2V3", "C_V1V2V3"]
FACTORS = {"B_reuse_aware": (0, 0, 0), "C_V1": (1, 0, 0), "C_V2": (0, 1, 0), "C_V3": (0, 0, 1),
           "C_V1V2": (1, 1, 0), "C_V1V3": (1, 0, 1), "C_V2V3": (0, 1, 1), "C_V1V2V3": (1, 1, 1)}


def ci(logs: np.ndarray) -> dict:
    """Geometric-mean ratio with a 95% t interval over paired log ratios."""
    n = len(logs)
    m = float(np.mean(logs)) if n else float("nan")
    if n < 2:
        return {"n": n, "ratio": math.exp(m) if n else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    se = float(np.std(logs, ddof=1)) / math.sqrt(n)
    t = st.t.ppf(0.975, n - 1)
    p = float(2 * st.t.sf(abs(m / se), n - 1)) if se > 0 else 0.0
    return {"n": n, "ratio": math.exp(m), "lo": math.exp(m - t * se), "hi": math.exp(m + t * se), "p": p}


def load_jsonl(path) -> list:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def paired(df: pd.DataFrame, arm: str, ref: str) -> dict:
    """Speedup of arm over ref: ms/token(ref) / ms/token(arm), paired per (problem, rep)."""
    a = df[df.policy == arm].set_index(["problem", "rep"]).ms_per_token
    r = df[df.policy == ref].set_index(["problem", "rep"]).ms_per_token
    j = pd.concat([a, r], axis=1, keys=["a", "r"]).dropna()
    return ci(np.log(j.r.values / j.a.values))


def factorial(df: pd.DataFrame) -> list:
    """Main effects and interactions of V1/V2/V3 on log ms/token, estimated per round from the
    complete 2^3 block (B = all off) and summarised over rounds. Effect > 1 = slower."""
    rows = []
    rounds = df[df.policy.isin(FACTORS)].pivot_table(index=["problem", "rep"], columns="policy",
                                                    values="ms_per_token").dropna()
    terms = {"V1": (0,), "V2": (1,), "V3": (2,), "V1xV2": (0, 1), "V1xV3": (0, 2),
             "V2xV3": (1, 2), "V1xV2xV3": (0, 1, 2)}
    for name, idx in terms.items():
        # Contrast coefficient of each cell: product of +-1 over the term's factors, / 4.
        coef = {arm: np.prod([1 if f[i] else -1 for i in idx]) / 4.0 for arm, f in FACTORS.items()}
        eff = sum(coef[a] * np.log(rounds[a].values) for a in FACTORS)
        r = ci(eff)
        rows.append({"term": name, "effect_ratio": r["ratio"], "lo": r["lo"], "hi": r["hi"],
                     "p": r["p"], "rounds": r["n"]})
    return rows


def diagnostics(stats_rows: list, eps: pd.DataFrame) -> pd.DataFrame:
    """Per-arm means of the hook counters over timed measured episodes."""
    if not stats_rows:
        return pd.DataFrame()
    s = pd.DataFrame(stats_rows)
    s = s[s.phase == "measure"].sort_values("time").groupby("episode").tail(1)
    s = s.merge(eps[["episode", "policy", "problem", "rep"]], on="episode", suffixes=("_hook", ""))
    s = s[s.rep.isin(TIMED_REPS)]
    d = s.decisions.clip(lower=1)
    out = pd.DataFrame({
        "policy": s.policy,
        "hit_rate": s.hits / s.unique.clip(lower=1),
        "misses_per_layer": s.m / d,
        "fills_per_layer": s.n_fetch / d,
        "cpu_experts_per_layer": s.n_cpu / d,
        "evictions_per_layer": s.evictions / d,
        "pcie_ms_per_layer": s.pcie_ms / d,
        "gemm_ms_per_layer": s.gemm_ms / d,
        "cpu_ms_per_cpu_layer": s.cpu_ms / s.cpu_n.clip(lower=1),
        "decide_us": s.decide_us / d,
        "victim_match_rate": s.victim_matches / s.victim_checks.clip(lower=1),
        "victim_checks": s.victim_checks,
        "v1_fit_g_fallback": s.fit_g_prior / d,
        "v1_fit_c_fallback": s.fit_c_prior / d,
        "v1_cost_mape": s.cost_abs_err_ms / s.cost_meas_ms.clip(lower=1e-9),
        "v2_alpha_final": s.alpha,
        "v2_cal_samples": s.cal_n,
        "tokens_covered": s.tokens,
    })
    return out.groupby("policy").mean(numeric_only=True).reindex([a for a in ARMS if a in set(out.policy)])


def model_report(root: Path, model: str, out: Path) -> dict:
    eps_all = pd.DataFrame(load_jsonl(root / f"runtime_q/results/main_{model}_episodes.jsonl"))
    if eps_all.empty:
        return {"model": model, "status": "no data"}
    servers = load_jsonl(root / f"runtime_q/results/main_{model}_episodes_servers.jsonl")
    eps = eps_all[eps_all.rep.isin(TIMED_REPS)]
    hooked = eps[eps.hook != "none"]
    res = {"model": model, "paper_model": PAPER_MODEL[model],
           "cache_slots": sorted({s["geometry"].get("moe_cache_size") for s in servers if "geometry" in s}),
           "episodes": int(len(eps_all)), "timed_hooked": int(len(hooked))}

    per_arm = hooked.groupby("policy").agg(
        n=("ms_per_token", "size"), ms_per_token=("ms_per_token", "mean"),
        tok_s=("decode_tok_s", "mean"), ttft_ms=("ttft_ms", "mean"), e2e_ms=("e2e_ms", "mean"),
        p95_ms=("p95_ms", "mean")).reindex([a for a in ARMS if a in set(hooked.policy)])
    per_arm.to_csv(out / f"{model}_per_arm.csv")

    comp = []
    for ref in ("A_qstar_paper", "E_qstar_engine", "B_reuse_aware"):
        for arm in ARMS:
            if arm != ref and arm in set(hooked.policy) and ref in set(hooked.policy):
                comp.append({"ref": ref, "arm": arm, **paired(hooked, arm, ref)})
                for p in sorted(hooked.problem.unique()):
                    comp.append({"ref": ref, "arm": arm, "problem": int(p),
                                 **paired(hooked[hooked.problem == p], arm, ref)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / f"{model}_paired.csv", index=False)

    # Sensitivity: the hook's decision time grows with the miss count, so it differs by policy.
    # Subtract each episode's measured decision time per token and repeat the paired tests.
    stats_rows = sum((load_jsonl(f) for f in glob.glob(str(root / "runtime_q/results/sx2_stats_*.jsonl"))), [])
    net = []
    if stats_rows:
        s = pd.DataFrame(stats_rows)
        s = s[s.phase == "measure"].sort_values("time").groupby("episode").tail(1)
        s["decide_ms_tok"] = s.decide_us / 1e3 / s.tokens.clip(lower=1)
        h = hooked.merge(s[["episode", "decide_ms_tok"]], on="episode", how="inner")
        h = h.assign(ms_per_token=h.ms_per_token - h.decide_ms_tok)
        for ref in ("A_qstar_paper", "E_qstar_engine", "B_reuse_aware"):
            for arm in ARMS:
                if arm != ref and arm in set(h.policy) and ref in set(h.policy):
                    net.append({"ref": ref, "arm": arm, **paired(h, arm, ref)})
        res["decide_ms_per_token"] = h.groupby("policy").decide_ms_tok.mean().to_dict()
    pd.DataFrame(net).to_csv(out / f"{model}_paired_net_of_decide.csv", index=False)

    fac = pd.DataFrame(factorial(hooked))
    fac.to_csv(out / f"{model}_factorial.csv", index=False)

    diag = diagnostics(sum((load_jsonl(f) for f in glob.glob(str(root / "runtime_q/results/sx2_stats_*.jsonl"))), []),
                       hooked)
    diag.to_csv(out / f"{model}_diagnostics.csv")

    # Output identity: greedy text per arm within each problem (CPU vs GPU numerics can diverge).
    ident = hooked.groupby("problem").output_sha1.nunique().to_dict()

    # Harness cost: hooked E vs unhooked E (same driver contract, separate server).
    nohook = eps[eps.policy == "E_qstar_engine_nohook"]
    oh = {}
    for p in sorted(nohook.problem.unique()):
        h = hooked[(hooked.policy == "E_qstar_engine") & (hooked.problem == p)].ms_per_token
        u = nohook[nohook.problem == p].ms_per_token
        oh[int(p)] = {"hooked_ms": float(h.mean()), "unhooked_eager_ms": float(u.mean()),
                      "ratio": float(h.mean() / u.mean())}
    refs = {}
    for mode in ("eager", "graph"):
        rows = load_jsonl(root / f"reference/{model}_{mode}.jsonl")
        if rows:
            r = pd.DataFrame(rows)
            refs[mode] = {"n": len(r), "ms_per_token": float(r.ms_per_token.mean()),
                          "tok_s": float(r.decode_tok_s.mean()), "ttft_ms": float(r.ttft_ms.mean()),
                          "per_problem_ms": r.groupby("problem").ms_per_token.mean().to_dict()}
    res.update({"distinct_outputs_per_problem": {int(k): int(v) for k, v in ident.items()},
                "harness_overhead": oh, "references": refs})
    return res


def fmt(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_markdown(out: Path, results: list) -> None:
    L = ["# saketh_exp_2 aggregated results", "",
         "Speedup = ms/token(ref) / ms/token(arm), paired per (problem, round); geometric mean "
         "with 95% t-interval. All hooked arms share one eager hook; numbers are conditional on it.", ""]
    for res in results:
        m = res["model"]
        L += [f"## {m} ({'paper model' if res.get('paper_model') else 'non-paper model'})", ""]
        if res.get("status") == "no data":
            L += ["No data.", ""]
            continue
        L += [f"Auto cache slots per server: {res['cache_slots']}; episodes {res['episodes']} "
              f"(timed hooked {res['timed_hooked']}). Distinct greedy outputs per problem across "
              f"arms: {res['distinct_outputs_per_problem']}.", ""]
        pa = pd.read_csv(out / f"{m}_per_arm.csv")
        L += ["| arm | n | ms/token | tok/s | TTFT ms | e2e ms |", "|---|---|---|---|---|---|"]
        for _, r in pa.iterrows():
            L.append(f"| {r.policy} | {int(r.n)} | {fmt(r.ms_per_token, 2)} | {fmt(r.tok_s, 2)} | "
                     f"{fmt(r.ttft_ms, 0)} | {fmt(r.e2e_ms, 0)} |")
        comp = pd.read_csv(out / f"{m}_paired.csv")
        allp = comp[comp.problem.isna()] if "problem" in comp else comp
        for ref in ("A_qstar_paper", "B_reuse_aware"):
            c = allp[allp.ref == ref]
            L += ["", f"Speedup over {ref} (all problems):", "",
                  "| arm | pairs | speedup | 95% CI | p |", "|---|---|---|---|---|"]
            for _, r in c.iterrows():
                L.append(f"| {r.arm} | {int(r.n)} | {fmt(r.ratio)} | [{fmt(r.lo)}, {fmt(r.hi)}] | {fmt(r.p, 4)} |")
        net_path = out / f"{m}_paired_net_of_decide.csv"
        net = pd.read_csv(net_path) if net_path.stat().st_size > 1 else pd.DataFrame()
        if not net.empty:
            L += ["", "Sensitivity, net of the hook's per-episode decision time "
                  f"(ms/token per arm: {json.dumps({k: round(v, 2) for k, v in res.get('decide_ms_per_token', {}).items()})}):", "",
                  "| ref | arm | pairs | speedup | 95% CI | p |", "|---|---|---|---|---|---|"]
            for _, r in net[net.ref.isin(["A_qstar_paper", "B_reuse_aware"])].iterrows():
                L.append(f"| {r.ref} | {r.arm} | {int(r.n)} | {fmt(r.ratio)} | [{fmt(r.lo)}, {fmt(r.hi)}] | {fmt(r.p, 4)} |")
        fac = pd.read_csv(out / f"{m}_factorial.csv")
        L += ["", "2^3 ablation on log ms/token (effect ratio < 1 = faster):", "",
              "| term | effect | 95% CI | p | rounds |", "|---|---|---|---|---|"]
        for _, r in fac.iterrows():
            L.append(f"| {r.term} | {fmt(r.effect_ratio)} | [{fmt(r.lo)}, {fmt(r.hi)}] | {fmt(r.p, 4)} | {int(r.rounds)} |")
        diag = pd.read_csv(out / f"{m}_diagnostics.csv")
        if not diag.empty:
            cols = ["hit_rate", "fills_per_layer", "cpu_experts_per_layer", "evictions_per_layer", "decide_us",
                    "victim_match_rate", "v1_fit_g_fallback", "v1_fit_c_fallback", "v1_cost_mape",
                    "v2_alpha_final", "v2_cal_samples"]
            L += ["", "Hook diagnostics (timed measured episodes, per-layer means):", "",
                  "| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for _, r in diag.iterrows():
                L.append(f"| {r.policy} | " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L += ["", f"Harness cost (hooked E vs unhooked eager E, ms/token): {json.dumps(res['harness_overhead'])}",
              "", f"Unhooked official-benchmark references: {json.dumps(res['references'])}", ""]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = [model_report(a.root, m, a.out) for m in MODELS]
    (a.out / "summary.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    write_markdown(a.out, results)
    print((a.out / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
