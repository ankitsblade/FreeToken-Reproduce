"""All figures for the runtime-q study (static PNGs for the report).

Colors: the validated default categorical order (blue, orange, aqua, ...) assigned to a
fixed entity (workload or policy), never to rank. One y-axis per panel; two measures of
different scale get two panels.
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runtime_q.qlib import RQ

FIG = RQ / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
W = ["low_miss", "med_miss", "high_miss"]
WC = dict(zip(W, C[:3]))
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
                     "lines.markersize": 6, "savefig.dpi": 140, "savefig.bbox": "tight"})
SHORT = {"qstar": "q*", "q0": "q=0", "f25": "25%", "f50": "50%", "f75": "75%", "qm": "q=m",
         "t0011": "m-table", "ra_k1": "reuse-aware", "ra_k4": "reuse-aware k=4",
         "ra_k1_uniformp": "RA uniform p", "ra_k1_nocache": "RA no cache",
         "linear_h16": "linear", "ridge_h8": "ridge", "tree_h16": "tree", "gbm_h16": "GBM",
         "mlp_h16": "ANN", "linucb_h0": "LinUCB H=0", "linucb_h16": "LinUCB", "ts_h16": "Thompson"}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def grouped(ax, df, policies, value, err=None):
    n = len(W)
    for i, w in enumerate(W):
        d = df[df.workload == w].set_index("policy").reindex(policies)
        x = np.arange(len(policies)) + (i - (n - 1) / 2) * 0.27
        ax.bar(x, d[value], 0.25, color=WC[w], label=w, yerr=d[err] if err else None,
               error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 1.5})
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([SHORT.get(p, p) for p in policies], rotation=30, ha="right")
    ax.grid(axis="x", visible=False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RQ / "results" / "eval" / "main_summary.csv")
    fixed = ["q0", "f25", "f50", "f75", "qm"]

    # 01 q* vs runtime-aware latency
    pol = ["qstar", "f25", "f50", "t0011", "ra_k1", "ridge_h8", "linear_h16", "linucb_h16", "ts_h16", "mlp_h16", "gbm_h16"]
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped(ax, s, pol, "ms_per_token", "ms_ci95")
    ax.set_ylabel("decode latency (ms / token)")
    ax.set_title("End-to-end latency: q* vs fixed vs runtime-aware policies (3 repeats, 95% CI)")
    ax.legend(title="workload", ncol=3)
    save(fig, "01_qstar_vs_runtime_aware_latency.png")

    # 02-05 q (fixed fraction) vs latency / hit rate / CPU traffic / PCIe traffic
    for name, col, ylabel, title in (
            ("02_q_vs_latency", "ms_per_token", "ms / token", "Latency vs fixed fetch fraction"),
            ("03_q_vs_cache_hit_rate", "cache_hit_rate", "cache hit rate", "Cache hit rate vs fixed fetch fraction"),
            ("04_cpu_traffic_vs_q", "cpu_mb_per_token", "CPU expert traffic (MB / token)", "CPU traffic vs fixed fetch fraction"),
            ("05_pcie_traffic_vs_q", "pcie_mb_per_token", "PCIe expert traffic (MB / token)", "PCIe traffic vs fixed fetch fraction")):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for w in W:
            d = s[s.workload == w].set_index("policy").reindex(fixed)
            ax.plot([0, 25, 50, 75, 100], d[col], "o-", color=WC[w], label=w)
            ra = s[(s.workload == w) & (s.policy == "ra_k1")][col].iloc[0]
            ax.axhline(ra, color=WC[w], lw=1, ls=":")
        ax.set_xlabel("fixed fetch fraction q/m (%)   (dotted = reuse-aware rule)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="workload")
        save(fig, name + ".png")

    # 06 branch balance
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    show = ["qstar", "f25", "f50", "f75", "qm", "ra_k1", "ridge_h8"]
    for w in W:
        d = s[(s.workload == w) & s.policy.isin(show)]
        ax.scatter(d.t_cpu_ms_per_token, d.t_gpu_ms_per_token, color=WC[w], s=42, edgecolor=SURF, linewidth=1.2, label=w, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(SHORT[r.policy], (r.t_cpu_ms_per_token, r.t_gpu_ms_per_token), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#52514e")
    lim = max(s[s.policy.isin(show)].t_cpu_ms_per_token.max(), 40)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    ax.set_xlim(0, lim * 1.05); ax.set_ylim(0, lim * 1.6)
    ax.set_xlabel("T_CPU: CPU expert time (ms / token)")
    ax.set_ylabel("T_GPU: PCIe + GEMM time (ms / token)")
    ax.set_title("Branch balance (dashed = balanced)")
    ax.legend(title="workload")
    save(fig, "06_branch_balance.png")

    # 07-12 ablations
    ab_path = RQ / "results" / "ablations" / "ablation_summary.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        q = ab[ab.family == "qstar"].iloc[0]
        ra = ab[ab.family == "ra_k1"].iloc[0]

        def ablation(fam, name, title, labeler, sort=None):
            d = ab[ab.family == fam].copy()
            d["label"] = d["config"].map(labeler)
            if sort:
                d = d.sort_values(sort)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
            x = np.arange(len(d))
            for i, w in enumerate(W):
                axes[0].bar(x + (i - 1) * 0.27, d[w], 0.25, color=WC[w], label=w)
                axes[0].axhline(ra[w], color=WC[w], lw=1, ls=":")
            axes[0].axhline(q["avg_ms"], color=INK, lw=1, ls="--")
            axes[0].text(len(d) - 0.5, q["avg_ms"], " q*", va="bottom", ha="right", fontsize=8)
            axes[0].set_xticks(x); axes[0].set_xticklabels(d["label"], rotation=35, ha="right")
            axes[0].set_ylabel("ms / token (end-to-end)"); axes[0].grid(axis="x", visible=False)
            axes[0].set_title(title + "  (dotted = reuse-aware rule)")
            axes[0].legend(title="workload", ncol=3, fontsize=7)
            axes[1].plot(x, d["mae"], "o-", color=C[6])
            axes[1].set_xticks(x); axes[1].set_xticklabels(d["label"], rotation=35, ha="right", fontsize=7)
            axes[1].set_ylabel("offline MAE (ms)"); axes[1].set_title("Prediction error")
            save(fig, name)

        strip = lambda p: (lambda c: c.replace(p, ""))  # noqa: E731
        ablation("chain", "07_incremental_feature_addition.png", "Incremental feature-group addition",
                 lambda c: c.split("_", 2)[2].replace("plus_", "+").replace("base", "m (+B_P,B_H)"), "config")
        ablation("logo", "08_leave_one_group_out.png", "Leave-one-group-out", strip("ridge_minus_"))
        ablation("indiv", "09_individual_feature_ablation.png", "Individual signal ablation (Full minus ...)", strip("ridge_minus_"))
        ablation("cache", "10_cache_ablation.png", "Cache-information ladder", strip("ridge_"), "config")
        ablation("history", "11_history_ablation.png", "History-window ablation",
                 lambda c: c.replace("ridge_hist", "N=").replace("N=00_current", "current"), "config")
        d = ab[ab.family == "ann"].copy()
        d[["depth", "width"]] = d["config"].str.extract(r"mlp_(\d)x(\d+)").astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for i, depth in enumerate((1, 2, 3)):
            g = d[d.depth == depth].sort_values("width")
            axes[0].plot(g.width, g.avg_ms, "o-", color=C[i], label=f"{depth} hidden")
            axes[1].plot(g.width, g.mae, "o-", color=C[i], label=f"{depth} hidden")
            axes[2].plot(g.width, g.infer_us, "o-", color=C[i], label=f"{depth} hidden")
        axes[0].axhline(q["avg_ms"], color=INK, lw=1, ls="--"); axes[0].axhline(ra["avg_ms"], color=MUTED, lw=1, ls=":")
        for a, yl, t in zip(axes, ("avg ms / token (e2e)", "offline MAE (ms)", "decision time (us)"),
                            ("End-to-end latency (dashed q*, dotted rule)", "Prediction error", "Inference cost")):
            a.set_xscale("log", base=2); a.set_xticks([16, 32, 64, 128]); a.set_xticklabels([16, 32, 64, 128])
            a.set_xlabel("hidden width"); a.set_ylabel(yl); a.set_title(t); a.legend(fontsize=7)
        save(fig, "12_ann_architecture_ablation.png")

    # 13 model complexity vs performance
    offs = []
    for f in glob.glob(str(RQ / "results" / "statistical" / "offline_main_r2_h*.csv")):
        offs.append(pd.read_csv(f).assign(h=int(f.split("_h")[-1].split(".")[0])))
    off = pd.concat(offs)
    geo = np.exp(np.log(s[s.policy == "qstar"].set_index("workload").ms_per_token /
                        s.pivot(index="policy", columns="workload", values="ms_per_token")).mean(axis=1))
    comp = {"qstar": 0, "f25": 0, "t0011": 5, "ra_k1": 3,
            "linear_h16": int(off[(off.kind == "linear") & (off.h == 16)].n_params.iloc[0]),
            "ridge_h8": int(off[(off.kind == "ridge") & (off.h == 8)].n_params.iloc[0]),
            "tree_h16": int(off[(off.kind == "tree") & (off.h == 16)].n_params.iloc[0]),
            "mlp_h16": int(off[(off.kind == "mlp") & (off.h == 16)].n_params.iloc[0]),
            "gbm_h16": int(off[(off.kind == "gbm") & (off.h == 16)].n_params.iloc[0]),
            "linucb_h16": 14, "ts_h16": 14}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for p, n in comp.items():
        ax.scatter(max(n, 1), geo[p], color=C[0] if p.startswith(("ra", "qstar", "f25", "t00")) else C[1], s=44, zorder=3, edgecolor=SURF)
        ax.annotate(SHORT[p], (max(n, 1), geo[p]), xytext=(5, 3), textcoords="offset points", fontsize=8, color="#52514e")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("model size (parameters / tree nodes; rules = thresholds)")
    ax.set_ylabel("geo-mean speedup vs q*")
    ax.set_title("Model complexity vs end-to-end performance  (blue = rules, orange = learned)")
    save(fig, "13_model_complexity_vs_performance.png")

    # 14-15 feature analysis
    fa = RQ / "results" / "feature_analysis"
    imp = pd.read_csv(fa / "feature_importance.csv").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.feature, imp.perm_importance_mae_ms, color=C[0], xerr=imp.perm_std, error_kw={"elinewidth": 0.8})
    ax.set_xlabel("permutation importance (increase in MAE of cost_h8, ms)")
    ax.set_title("Feature importance (GBM, held-out episodes)"); ax.grid(axis="y", visible=False)
    save(fig, "15_feature_importance.png")
    sm = pd.read_csv(fa / "spearman_matrix.csv", index_col=0)
    keep = list(pd.read_csv(fa / "feature_importance.csv").feature.head(18))
    keep = [k for k in keep if k in sm.index] + ["wall_net_ms", "cost_h8", "fut_hits_h8"]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(sm.loc[keep, keep], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=7); ax.grid(False)
    fig.colorbar(im, shrink=0.7, label="Spearman rho")
    ax.set_title("Feature correlation matrix (top features + outcomes)")
    save(fig, "14_feature_correlation_matrix.png")

    # 16 single-step cost vs q (CPU / PCIe / wall), from uniform-random logging
    df = pd.read_pickle(RQ / "data" / "clean" / "dataset.pkl")
    uni = df[df.policy.str.startswith("uniform")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for i, m in enumerate((1, 2, 3, 4)):
        g = uni[uni.m == m].groupby("n_fetch")[["wall_net_ms", "cpu_task_ms", "pcie_ms"]].mean()
        for a, col in zip(axes, ("cpu_task_ms", "pcie_ms", "wall_net_ms")):
            a.plot(g.index, g[col], "o-", color=C[i], label=f"m={m}")
    for a, t in zip(axes, ("CPU branch time vs q", "PCIe transfer time vs q", "Layer wall time vs q (single step)")):
        a.set_xlabel("q (experts fetched)"); a.set_ylabel("ms / layer call"); a.set_title(t); a.legend(fontsize=7)
    save(fig, "16_single_step_cost_vs_q.png")

    # 17 future cache benefit per fetch
    fv = pd.read_csv(RQ / "results" / "cache_analysis" / "fetch_value_by_policy.csv")
    fv = fv[fv.policy.isin(["f25", "f50", "f75", "qm"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for w in W:
        d = fv[fv.workload == w].set_index("policy").reindex(["f25", "f50", "f75", "qm"])
        axes[0].plot([25, 50, 75, 100], d.mean_future_hits, "o-", color=WC[w], label=w)
        axes[1].plot([25, 50, 75, 100], d.net_ms / 1000, "o-", color=WC[w], label=w)
    axes[0].set_yscale("log"); axes[0].set_ylabel("future hits per fetched expert"); axes[0].set_title("Future cache benefit per fetch")
    axes[1].axhline(0, color=MUTED, lw=1); axes[1].set_ylabel("net time saved by fetching (s, per 254 tokens)")
    axes[1].set_title("CPU time saved minus exposed fetch cost")
    for a in axes:
        a.set_xlabel("fixed fetch fraction (%)"); a.legend(title="workload", fontsize=7)
    save(fig, "17_future_cache_benefit_vs_q.png")

    # 18-19 selected q over time / vs state, from the logged main rows
    rows = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(str(RQ / "results" / "eval" / "main_rows_*.csv"))])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for a, w in zip(axes, W):
        for i, p in enumerate(("f25", "ra_k1", "ridge_h8", "linucb_h16")):
            g = rows[(rows.workload == w) & (rows.policy == p)].groupby("tok").n_fetch.sum().rolling(8, min_periods=1).mean()
            a.plot(g.index, g.values, color=C[i], lw=1.6, label=SHORT[p])
        a.axvline(127, color=MUTED, lw=1, ls="--")
        a.set_title(f"{w}: fetches per token (8-token mean)"); a.set_xlabel("token (dashed = measured prompt starts)")
    axes[0].set_ylabel("experts fetched per token (24 layers)"); axes[0].legend(fontsize=7)
    save(fig, "18_selected_q_over_time.png")
    ra = rows[rows.policy == "ra_k1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for w in W:
        g = ra[ra.workload == w]
        axes[0].plot(*zip(*g.groupby("m").n_fetch.mean().items()), "o-", color=WC[w], label=w)
        b = g[g.m > 0].assign(bin=lambda t: pd.cut(t.occ / t.cap, [-.01, .25, .5, .75, .99, 1.0])).groupby("bin", observed=True).n_fetch.mean()
        axes[1].plot(range(len(b)), b.values, "o-", color=WC[w], label=w)
        full = g[(g.m > 0) & (g.free_slots <= 0)]
        b2 = full.assign(bin=pd.cut(full.lru_age_tok, [0, 2, 4, 8, 16, 32, 200])).groupby("bin", observed=True).n_fetch.mean()
        axes[2].plot(range(len(b2)), b2.values, "o-", color=WC[w], label=w)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["<25%", "25-50", "50-75", "75-99", "full"])
    axes[2].set_xticks(range(6)); axes[2].set_xticklabels(["<2", "2-4", "4-8", "8-16", "16-32", ">32"])
    for a, xl, t in zip(axes, ("m (missing experts)", "cache occupancy", "LRU victim age (tokens), cache full"),
                        ("Selected q vs m", "Selected q vs cache occupancy", "Selected q vs LRU victim age")):
        a.set_xlabel(xl); a.set_ylabel("mean q (reuse-aware rule)"); a.set_title(t); a.legend(fontsize=7)
    save(fig, "19_selected_q_vs_state.png")

    # 20 selected q vs utilization (rule ignores it; learned ridge does not)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for i, p in enumerate(("ra_k1", "ridge_h8")):
        g = rows[(rows.policy == p) & (rows.m > 0)]
        for a, col, bins in zip(axes, ("cpu_util", "gpu_util"), ([0, 15, 25, 35, 45, 100], [0, 20, 35, 50, 65, 100])):
            b = g.assign(bin=pd.cut(g[col], bins)).groupby("bin", observed=True).n_fetch.mean()
            a.plot([str(k) for k in b.index], b.values, "o-", color=C[i], label=SHORT[p])
    for a, t in zip(axes, ("Selected q vs CPU utilization", "Selected q vs GPU utilization")):
        a.set_ylabel("mean q"); a.set_title(t); a.legend(); a.tick_params(axis="x", labelsize=7)
    save(fig, "20_selected_q_vs_utilization.png")

    # 21 predicted vs measured + error (held-out episodes)
    from policy_models import CostModel
    from experiments.runtime_q.qlib import training_frames
    _, _, te = training_frames(df, "cost_h16")
    te = te.sample(6000, random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for i, (kind, lab) in enumerate((("ridge_full", "ridge"), ("mlp_full", "ANN-GELU 2x64"))):
        mdl = CostModel.load(str(RQ / "models" / "main_r2_h16" / f"{kind}.json"))
        pred = mdl.predict(te[mdl.features].to_numpy(float), te.n_fetch.to_numpy(float), te.m.to_numpy(float))
        axes[i].scatter(te.cost_h16, pred, s=3, color=C[i], alpha=0.35)
        lim = [0, te.cost_h16.quantile(0.995)]
        axes[i].plot(lim, lim, color=INK, lw=1); axes[i].set_xlim(lim); axes[i].set_ylim(lim)
        axes[i].set_xlabel("measured cost_h16 (ms)"); axes[i].set_ylabel("predicted (ms)")
        axes[i].set_title(f"{lab}: predicted vs measured")
        axes[2].hist(pred - te.cost_h16, bins=80, range=(-30, 30), histtype="step", color=C[i], lw=1.6, label=lab)
    axes[2].set_xlabel("prediction error (ms)"); axes[2].set_ylabel("held-out decisions"); axes[2].set_title("Prediction error"); axes[2].legend()
    save(fig, "21_prediction_error.png")

    # 22 e2e regret, 23 latency CDF, 24 overhead, 25 horizon
    order = s.groupby("policy").e2e_regret_ms.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    grouped(ax, s, order, "e2e_regret_ms")
    ax.set_ylabel("regret vs best evaluated policy (ms / token)"); ax.set_title("End-to-end regret of each policy"); ax.legend(title="workload", ncol=3)
    save(fig, "22_regret.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for a, w in zip(axes, W):
        for i, p in enumerate(("qstar", "f25", "f50", "ra_k1", "ridge_h8")):
            v = np.sort(rows[(rows.workload == w) & (rows.policy == p) & (rows.phase == "measure") & (rows.layer_id == 0)].token_wall_ms.dropna().to_numpy()[2:])
            a.plot(v, np.linspace(0, 1, len(v)), color=C[i], lw=1.6, label=SHORT[p])
        a.set_xlim(0, 160); a.set_xlabel("token latency (ms, server side)"); a.set_title(f"{w}: latency CDF")
    axes[0].set_ylabel("fraction of tokens"); axes[0].legend(fontsize=7)
    save(fig, "23_latency_cdf.png")
    dec = s.groupby("policy").decide_us.mean().reindex(["qstar", "f25", "ra_k1", "ridge_h8", "linear_h16", "tree_h16", "mlp_h16", "linucb_h16", "ts_h16", "gbm_h16"])
    wallc = rows[rows.phase == "measure"].groupby("policy").wall_net_ms.mean().reindex(dec.index) * 1000
    fig, ax = plt.subplots(figsize=(8, 3.7))
    x = np.arange(len(dec))
    ax.bar(x - 0.2, dec.values, 0.38, color=C[0], label="decision (features + policy), us")
    ax.bar(x + 0.2, wallc.values, 0.38, color=C[1], label="layer execution, us")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in dec.index], rotation=30, ha="right"); ax.grid(axis="x", visible=False)
    ax.set_ylabel("microseconds per layer decision"); ax.set_title("Decision overhead vs layer execution time (full feature logging)"); ax.legend()
    save(fig, "24_decision_overhead.png")
    hz = pd.read_csv(RQ / "results" / "statistical" / "offline_horizon.csv")
    hz["h"] = hz.target.str.extract(r"h(\d+)").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for i, k in enumerate(("ridge", "gbm")):
        g = hz[hz.kind == k].sort_values("h")
        axes[0].plot(g.h, g.mean_q, "o-", color=C[i], label=k)
        axes[1].plot(g.h, g.r2, "o-", color=C[i], label=k)
    for a, yl, t in zip(axes, ("mean q chosen (uniform-log states)", "held-out R^2"), ("Chosen q vs cost horizon H", "Predictability vs horizon")):
        a.set_xscale("symlog", base=2, linthresh=1); a.set_xticks([0, 1, 4, 8, 16, 32]); a.set_xticklabels([0, 1, 4, 8, 16, 32])
        a.set_xlabel("cost horizon H (tokens)"); a.set_ylabel(yl); a.set_title(t); a.legend()
    save(fig, "25_horizon.png")

    # 26 generalization / stress
    for name, path, title in (("26_generalization.png", "generalization_summary.csv", "Unseen prompt (p1_*) and unseen cache sizes (rate*)"),
                              ("27_cpu_contention.png", "stress_summary.csv", "4 CPU busy-loops during decode"),
                              ("28_long_decode.png", "long_summary.csv", "384-token decode")):
        p = RQ / "results" / "eval" / path
        if not p.exists():
            continue
        g = pd.read_csv(p)
        wl = sorted(g.workload.unique())
        pols = [x for x in ("qstar", "f25", "f50", "t0011", "ra_k1", "ra_k1_online", "ridge_h8", "linear_h16") if x in set(g.policy)]
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(wl) + 3), 3.8))
        for i, pol in enumerate(pols):
            d = g[g.policy == pol].set_index("workload").reindex(wl)
            ax.bar(np.arange(len(wl)) + (i - (len(pols) - 1) / 2) * 0.1, d.ms_per_token, 0.09, color=C[i], label=SHORT.get(pol, pol),
                   yerr=d.ms_std, error_kw={"elinewidth": 0.7})
        ax.set_xticks(range(len(wl))); ax.set_xticklabels(wl); ax.grid(axis="x", visible=False)
        ax.set_ylabel("ms / token"); ax.set_title(title); ax.legend(ncol=4, fontsize=7)
        save(fig, name)


if __name__ == "__main__":
    main()
