"""Train every model / ablation configuration and score it offline.

Families
  main     linear, ridge, tree, gbm, mlp on the full kept feature set
  horizon  ridge + gbm with the cost horizon H in {0,1,4,8,16,32}  (H=0 is the myopic cost)
  chain    cumulative feature-group addition: BASE, +G1, ..., +G10
  logo     leave-one-group-out: Full - Gk
  indiv    Full - one signal family (m, T_CPU, T_GPU, T_transfer, T_overlap, ...)
  cache    cache-information ladder C1..C5
  history  current state, then history windows <= 1, 4, 8, 16, 32
  ann      1/2/3 hidden layers x 16/32/64/128 units

Each configuration writes a runtime artifact (models/<family>/<name>.json) and one row of
offline metrics. Offline metrics rank configurations; the end-to-end runs decide.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from experiments.runtime_q.qlib import RQ, evaluate, fit, group_features, load_dataset, matched_value, save_artifact, training_frames

GROUP_ORDER = ["G1_routing", "G2_cpu", "G3_gpu", "G4_transfer", "G5_overlap", "G6_cache",
               "G7_reuse", "G8_util", "G9_memory", "G10_history"]
SHORT = {"G1_routing": "Routing", "G2_cpu": "CPU", "G3_gpu": "GPU", "G4_transfer": "Transfer",
         "G5_overlap": "Overlap", "G6_cache": "Cache", "G7_reuse": "Reuse", "G8_util": "Utilization",
         "G9_memory": "Memory", "G10_history": "History"}
DEFAULT_H = 8


def configs(report: dict, family: str, kinds: list[str], h: int = DEFAULT_H) -> list[dict]:
    kept = report["kept"]
    full = [f for f in kept]
    G = lambda gs: group_features(report, gs)  # noqa: E731
    tgt = f"cost_h{h}"
    out = []
    add = lambda name, kind, feats, target=tgt, **hp: out.append(  # noqa: E731
        {"family": family, "name": name, "kind": kind, "feats": feats, "target": target, "hp": hp})
    if family == "main":
        for k in ("linear", "ridge", "tree", "gbm", "mlp"):
            add(f"{k}_full", k, full)
    elif family == "horizon":
        for h in (0, 1, 4, 8, 16, 32):
            for k in ("ridge", "gbm"):
                add(f"{k}_h{h}", k, full, f"cost_h{h}")
    elif family == "chain":
        for k in kinds:
            cur = ["BASE"]
            add(f"{k}_00_base", k, G(cur))
            for i, g in enumerate(GROUP_ORDER, 1):
                cur = cur + [g]
                add(f"{k}_{i:02d}_plus_{SHORT[g]}", k, G(cur))
    elif family == "logo":
        for k in kinds:
            for g in GROUP_ORDER:
                add(f"{k}_minus_{SHORT[g]}", k, G(["BASE"] + [x for x in GROUP_ORDER if x != g]))
    elif family == "indiv":
        pat = {"m": r"^m$", "T_CPU": r"cpu_ms|tcpu|_cpu$|bh_recent", "T_GPU": r"gemm|tgpu|_gpu$",
               "T_transfer": r"pcie|h2d|bp_recent", "T_overlap": None, "cache": None,
               "reuse": None, "utilization": None, "memory": None, "history": None,
               "B_P_recent": r"^bp_recent_gbps$", "B_H_recent": r"^bh_recent_gbps$"}
        grp = {"T_overlap": "G5_overlap", "cache": "G6_cache", "reuse": "G7_reuse",
               "utilization": "G8_util", "memory": "G9_memory", "history": "G10_history"}
        for k in kinds:
            for name, rx in pat.items():
                drop = set(report["groups_kept"][grp[name]]) if rx is None else {f for f in full if re.search(rx, f)}
                if drop:
                    add(f"{k}_minus_{name}", k, [f for f in full if f not in drop])
    elif family == "cache":
        core = [f for f in ("m", "layer_id", "tok") if f in kept]
        occ = [f for f in ("occ", "free_slots", "occ_frac", "layer_resident") if f in kept]
        hm = [f for f in ("n_hits", "miss_ratio", "hit_ewma") if f in kept]
        ladder = [("C1_none", core), ("C2_occupancy", core + occ), ("C3_occ_hitmiss", core + occ + hm),
                  ("C4_occ_hitmiss_reuse", core + occ + hm + report["groups_kept"]["G7_reuse"])]
        c5 = ladder[-1][1] + [f for f in report["groups_kept"]["G6_cache"] if f not in ladder[-1][1]]
        ladder.append(("C5_full_cache", c5))
        for k in kinds:
            for name, feats in ladder:
                add(f"{k}_{name}", k, feats)
    elif family == "history":
        now = G(["BASE", "G1_routing", "G6_cache", "G7_reuse", "G8_util", "G9_memory"])
        hist = report["groups_kept"]["G10_history"]
        for k in kinds:
            add(f"{k}_hist00_current", k, now)
            for n in (1, 4, 8, 16, 32):
                sel = [f for f in hist if int(re.match(r"hist(\d+)_", f).group(1)) <= n]
                add(f"{k}_hist{n:02d}", k, now + sel)
    elif family == "ann":
        for depth in (1, 2, 3):
            for width in (16, 32, 64, 128):
                add(f"mlp_{depth}x{width}", "mlp", full, hidden=[width] * depth)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="+", default=["main"])
    ap.add_argument("--kinds", nargs="+", default=["ridge", "gbm"])
    ap.add_argument("--max-train", type=int, default=120000)
    ap.add_argument("--h", type=int, default=8, help="cost horizon for all non-horizon families")
    ap.add_argument("--tag", default="", help="suffix for the model / result directory")
    a = ap.parse_args()
    df, report = load_dataset()
    rows = []
    for family in a.family:
        for c in configs(report, family, a.kinds, a.h):
            tr, va, te = training_frames(df, c["target"])
            if len(tr) > a.max_train:
                tr = tr.sample(a.max_train, random_state=0)
            uni = te[te["policy"].str.startswith("uniform")]
            art = fit(c["kind"], tr, va, c["feats"], c["target"], **c["hp"])
            art["name"], art["family"] = c["name"], family
            save_artifact(art, RQ / "models" / (family + a.tag) / f"{c['name']}.json")
            met = evaluate(art, te, uni, c["target"])
            ref, _ = matched_value(uni, np.zeros(len(uni), dtype=int), c["target"])
            row = {"family": family, "name": c["name"], "kind": c["kind"], "target": c["target"],
                   "n_features": len(c["feats"]), **met, "ips_value_q0": ref,
                   "ips_gain_vs_q0_ms": ref - met["ips_value"]}
            rows.append(row)
            print(f"{family:8s} {c['name']:32s} d={len(c['feats']):3d} mae={met['mae']:.3f} "
                  f"rmse={met['rmse']:.3f} r2={met['r2']:.3f} ips={met['ips_value']:.3f} "
                  f"(q0 {ref:.3f}) mean_q={met['mean_q']:.2f} infer={met['infer_us']:.0f}us", flush=True)
        out = RQ / "results" / ("ann" if family == "ann" else "ablations" if family not in ("main", "horizon") else "statistical")
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for r in rows if r["family"] == family]).to_csv(out / f"offline_{family}{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
