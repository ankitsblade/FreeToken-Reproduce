"""Generate the 8 required figures from the summary CSVs. Missing sources are skipped."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
SUM = study_dir("split_policy") / "results" / "summary"
FIG = study_dir("split_policy") / "figures"
BRANCH = study_dir("split_policy") / "results" / "branch"

POLICY_ORDER = ["A_adaptive", "B_gpu", "C_cpu", "D_fixed25", "E_fixed50", "F_fixed75"]
POLICY_LABEL = {"A_adaptive": "q*", "B_gpu": "q=m", "C_cpu": "q=0",
                "D_fixed25": "25%", "E_fixed50": "50%", "F_fixed75": "75%"}
WL_ORDER = ["low_miss", "med_miss", "high_miss"]
WL_COLOR = {"low_miss": "#4C72B0", "med_miss": "#DD8452", "high_miss": "#C44E52"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def grouped_bar(rows, value_key, title, ylabel, out, ylog=False):
    if not rows:
        print(f"skip {out.name}: no data")
        return
    wls = [w for w in WL_ORDER if any(r["workload"] == w for r in rows)]
    x = range(len(POLICY_ORDER))
    width = 0.8 / max(1, len(wls))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, wl in enumerate(wls):
        vals = []
        for pol in POLICY_ORDER:
            match = [r for r in rows if r["workload"] == wl and r["policy"] == pol]
            vals.append(f(match[0][value_key]) if match else None)
        xs = [xi + i * width for xi in x]
        ax.bar(xs, [v if v is not None else 0 for v in vals], width,
               label=wl, color=WL_COLOR.get(wl))
    ax.set_xticks([xi + width * (len(wls) - 1) / 2 for xi in x])
    ax.set_xticklabels([POLICY_LABEL[p] for p in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylog:
        ax.set_yscale("log")
    ax.legend(title="workload")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    perf = read_csv(SUM / "summary_perf.csv")
    traffic = read_csv(SUM / "summary_traffic.csv")
    balance = read_csv(SUM / "summary_balance.csv")
    qbym = read_csv(SUM / "summary_q_by_m.csv")

    # Plot 1: q-policy vs CPU bandwidth demand (bytes routed to CPU per token)
    grouped_bar(traffic, "cpu_bytes_per_token",
                "CPU-side expert traffic per token", "CPU bytes / token",
                FIG / "plot1_cpu_bandwidth.png")

    # Plot 2: q-policy vs CPU execution time (isolated CPU branch, miss-weighted)
    grouped_bar(balance, "T_CPU_ms",
                "CPU branch time per MoE layer (miss-weighted)", "T_CPU (ms)",
                FIG / "plot2_cpu_exec_time.png")

    # Plot 3: q-policy vs GPU execution time (isolated PCIe/H2D branch, miss-weighted)
    grouped_bar(balance, "T_GPU_ms",
                "GPU/PCIe branch time per MoE layer (miss-weighted)", "T_GPU (ms)",
                FIG / "plot3_gpu_exec_time.png")

    # Plot 4: q-policy vs end-to-end decode latency
    grouped_bar(perf, "ms_per_token_mean",
                "End-to-end decode latency", "ms / token",
                FIG / "plot4_decode_latency.png")

    # Plot 5: q-policy vs tokens/sec
    grouped_bar(perf, "decode_tok_s_mean",
                "Decode throughput", "tokens / s",
                FIG / "plot5_tokens_per_sec.png")

    # Plot 6: T_CPU vs T_GPU scatter (one point per policy x workload)
    if balance:
        fig, ax = plt.subplots(figsize=(6, 6))
        for r in balance:
            tc, tg = f(r["T_CPU_ms"]), f(r["T_GPU_ms"])
            if tc is None or tg is None:
                continue
            ax.scatter(tc, tg, color=WL_COLOR.get(r["workload"], "#333"),
                       s=70, edgecolor="k", zorder=3)
            ax.annotate(POLICY_LABEL[r["policy"]], (tc, tg),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        lim = max([f(r["T_CPU_ms"]) for r in balance] +
                  [f(r["T_GPU_ms"]) for r in balance] + [0.01])
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="T_CPU = T_GPU (balanced)")
        ax.set_xlabel("T_CPU (ms)")
        ax.set_ylabel("T_GPU (ms)")
        ax.set_title("Branch balance: T_CPU vs T_GPU\n(closer to dashed line = better balanced)")
        handles = [plt.Line2D([], [], marker="o", ls="", color=WL_COLOR[w], label=w)
                   for w in WL_ORDER if any(r["workload"] == w for r in balance)]
        handles.append(plt.Line2D([], [], ls="--", color="k", label="balanced"))
        ax.legend(handles=handles)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "plot6_tcpu_vs_tgpu.png", dpi=130)
        plt.close(fig)
        print("wrote plot6_tcpu_vs_tgpu.png")
    else:
        print("skip plot6: no balance data")

    # Plot 7: predicted q* vs oracle q (from branch_oracle summary)
    bsum = read_csv(BRANCH / "branch_oracle_summary.csv")
    if bsum:
        seen = {}
        for r in bsum:
            m = int(r["m"])
            seen[m] = (int(r["q_predicted"]), int(r["q_oracle"]))
        ms = sorted(seen)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ms, [seen[m][0] for m in ms], "o-", label="q* predicted")
        ax.plot(ms, [seen[m][1] for m in ms], "s--", label="q oracle (min latency)")
        ax.set_xlabel("cache misses m")
        ax.set_ylabel("q (experts fetched to GPU)")
        ax.set_title("Predicted q* vs measured oracle q")
        ax.set_xticks(ms)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "plot7_qpred_vs_qoracle.png", dpi=130)
        plt.close(fig)
        print("wrote plot7_qpred_vs_qoracle.png")
    else:
        print("skip plot7: no branch oracle summary")

    # Plot 8: cache misses m vs selected q, per policy (med_miss workload)
    if qbym:
        wl = "med_miss" if any(r["workload"] == "med_miss" for r in qbym) else qbym[0]["workload"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in POLICY_ORDER:
            pts = sorted((int(r["m"]), f(r["mean_q"])) for r in qbym
                         if r["policy"] == pol and r["workload"] == wl)
            if pts:
                ax.plot([m for m, _ in pts], [q for _, q in pts], "o-",
                        label=POLICY_LABEL[pol])
        ax.set_xlabel("cache misses m (per layer per token)")
        ax.set_ylabel("selected q (mean)")
        ax.set_title(f"Selected q vs miss count m ({wl})")
        ax.legend(title="policy", ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "plot8_m_vs_q.png", dpi=130)
        plt.close(fig)
        print("wrote plot8_m_vs_q.png")
    else:
        print("skip plot8: no q-by-m data")


if __name__ == "__main__":
    main()
