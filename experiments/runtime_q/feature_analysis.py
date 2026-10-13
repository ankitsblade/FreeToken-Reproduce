"""Feature quality report: statistics, correlations, MI, partial correlation, VIF,
permutation importance, per-workload / per-layer behavior and temporal stability.

Correlation is reported next to model-based importance on purpose: they answer different
questions and frequently disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

from experiments.runtime_q.qlib import RQ, group_features, load_dataset

OUT = RQ / "results" / "feature_analysis"
TARGET = "cost_h8"


def group_of(report: dict, f: str) -> str:
    return next(g for g, fs in report["groups_kept"].items() if f in fs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, report = load_dataset()
    feats = report["kept"]
    d = df[df["randomized"] & df[TARGET].notna()].copy()
    d["gpu_ms"] = d["pcie_ms"] + d["gemm_ms"]
    d["miss_reduction_h8"] = 8 * d["m"] - d["fut_m_h8"]
    outcomes = {"wall": "wall_net_ms", "T_CPU": "cpu_task_ms", "T_GPU": "gpu_ms",
                "q": "n_fetch", "cost_h8": TARGET, "future_hits_h8": "fut_hits_h8",
                "future_miss_reduction_h8": "miss_reduction_h8"}

    stats = d[feats].describe(percentiles=[.05, .5, .95]).T
    stats["missing"] = d[feats].isna().sum()
    stats["nunique"] = d[feats].nunique()
    stats.insert(0, "group", [group_of(report, f) for f in feats])
    for name, col in outcomes.items():
        stats[f"pearson_{name}"] = [d[f].corr(d[col]) for f in feats]
        stats[f"spearman_{name}"] = [d[f].corr(d[col], method="spearman") for f in feats]

    s = d.sample(min(25000, len(d)), random_state=0)
    stats["mutual_info_cost_h8"] = mutual_info_regression(s[feats], s[TARGET], random_state=0)
    # partial correlation with the target after removing what (m, q) already explain
    ctrl = np.column_stack([s["m"], s["n_fetch"], s["m"] * s["n_fetch"], np.ones(len(s))])
    resid = lambda v: v - ctrl @ np.linalg.lstsq(ctrl, v, rcond=None)[0]  # noqa: E731
    ry = resid(s[TARGET].to_numpy(float))
    pc = []
    for f in feats:
        rx = resid(s[f].to_numpy(float))
        pc.append(float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-9 else 0.0)
    stats["partial_corr_cost_h8_given_m_q"] = pc
    # VIF from the inverse correlation matrix (pinv: tolerate near-collinearity)
    z = (s[feats] - s[feats].mean()) / s[feats].std().replace(0, 1)
    stats["vif"] = np.diag(np.linalg.pinv(np.corrcoef(z.to_numpy().T)))

    # permutation importance of a GBM that sees x and q, on held-out episodes
    tr, te = d[d["split_run"] == "train"], d[d["split_run"] == "test"]
    cols = feats + ["n_fetch"]
    gbm = GradientBoostingRegressor(n_estimators=120, max_depth=3, subsample=0.5,
                                    random_state=0).fit(tr[cols], tr[TARGET])
    tes = te.sample(min(20000, len(te)), random_state=0)
    perm = permutation_importance(gbm, tes[cols], tes[TARGET], n_repeats=5, random_state=0,
                                  scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"feature": cols, "perm_importance_mae_ms": perm.importances_mean,
                        "perm_std": perm.importances_std,
                        "gbm_gain": gbm.feature_importances_}).sort_values(
        "perm_importance_mae_ms", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    stats["perm_importance_mae_ms"] = imp.set_index("feature")["perm_importance_mae_ms"].reindex(feats)
    stats.to_csv(OUT / "feature_stats.csv")

    d[feats + list(outcomes.values())].corr(method="spearman").to_csv(OUT / "spearman_matrix.csv")
    d.groupby("workload")[feats].agg(["mean", "std"]).T.to_csv(OUT / "per_workload.csv")
    # per-layer and temporal stability of the feature<->target relation
    key = [f for f in feats if f not in ("layer_id",)]
    per_layer = d.groupby("layer_id").apply(
        lambda g: g[key].corrwith(g[TARGET], method="spearman"), include_groups=False)
    per_layer.T.assign(std_across_layers=per_layer.std(0)).to_csv(OUT / "per_layer_spearman.csv")
    early = d[d["step_in_phase"] < d["phase_len"] / 2]
    late = d[d["step_in_phase"] >= d["phase_len"] / 2]
    pd.DataFrame({"spearman_early": early[key].corrwith(early[TARGET], method="spearman"),
                  "spearman_late": late[key].corrwith(late[TARGET], method="spearman")}).assign(
        abs_shift=lambda t: (t["spearman_early"] - t["spearman_late"]).abs()).to_csv(
        OUT / "temporal_stability.csv")

    top = imp.head(12)
    summary = {"target": TARGET, "rows": int(len(d)), "n_features": len(feats),
               "gbm_test_mae_ms": float(np.abs(gbm.predict(te[cols]) - te[TARGET]).mean()),
               "top_permutation": top["feature"].tolist(),
               "top_abs_spearman": stats[f"spearman_{'cost_h8'}"].abs().sort_values(
                   ascending=False).head(12).index.tolist(),
               "high_vif": stats.index[stats["vif"] > 10].tolist()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
