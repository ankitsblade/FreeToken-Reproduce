"""Plots for Experiment 2 (routing locality vs LRU)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
RES = study_dir("routing_locality") / "results"
FIG = study_dir("routing_locality") / "figures"


def rd(name):
    return list(csv.DictReader((RES / name).open()))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    curve = rd("miss_curve.csv")
    C = [int(r["cache_size"]) for r in curve]
    mo = [float(r["miss_rate_original"]) for r in curve]
    ms = [float(r["miss_rate_shuffled_mean"]) for r in curve]
    msd = [float(r["miss_rate_shuffled_std"]) for r in curve]
    ben = [float(r["locality_benefit_missrate"]) for r in curve]

    # Plot 1: miss rate vs cache size, original vs shuffled
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(C, mo, "o-", color="#4C72B0", label="original routing order")
    ax.plot(C, ms, "s--", color="#C44E52", label="shuffled order (freq-preserved)")
    ax.fill_between(C, [a - b for a, b in zip(ms, msd)], [a + b for a, b in zip(ms, msd)],
                    color="#C44E52", alpha=0.15)
    ax.set_xlabel("LRU cache size (experts/layer, E=32)"); ax.set_ylabel("miss rate")
    ax.set_title("LRU miss rate vs cache size: original vs shuffled routing")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "locality1_missrate_vs_cache.png", dpi=130); plt.close(fig)

    # Plot 2: locality benefit (miss-rate reduction) vs cache size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(c) for c in C], ben, color="#55A868")
    for i, r in enumerate(curve):
        ax.text(i, ben[i], f"{float(r['locality_benefit_rel_pct']):.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("LRU cache size (experts/layer)")
    ax.set_ylabel("miss-rate reduction (shuffled - original)")
    ax.set_title("Temporal-locality benefit vs cache size (label = relative %)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "locality2_benefit_vs_cache.png", dpi=130); plt.close(fig)

    # Plot 3: reuse-distance CDF, original vs shuffled
    hist = rd("reuse_distance_hist.csv")
    ds, co, cs = [], [], []
    for r in hist:
        if r["reuse_distance"] == "cold":
            continue
        ds.append(int(r["reuse_distance"]))
        co.append(int(r["count_original"])); cs.append(int(r["count_shuffled"]))
    def cdf(counts):
        tot = sum(counts); acc = 0; out = []
        for c in counts:
            acc += c; out.append(acc / tot)
        return out
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ds, cdf(co), "-", color="#4C72B0", label="original")
    ax.plot(ds, cdf(cs), "--", color="#C44E52", label="shuffled")
    ax.set_xlabel("reuse distance (distinct experts since last use)")
    ax.set_ylabel("CDF of reused accesses")
    ax.set_title("Reuse-distance CDF: original order reuses experts sooner")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "locality3_reuse_distance_cdf.png", dpi=130); plt.close(fig)

    # Plot 4: per-layer locality benefit at C=8
    pl = rd("per_layer_missrate.csv")
    lids = [int(r["layer_id"]) for r in pl]
    lb = [float(r["locality_benefit"]) for r in pl]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lids, lb, color="#8172B3")
    ax.set_xlabel("MoE layer id"); ax.set_ylabel("miss-rate reduction (C=8)")
    ax.set_title("Per-layer temporal-locality benefit (cache = 8 experts)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "locality4_per_layer.png", dpi=130); plt.close(fig)
    print("wrote 4 figures to", FIG)


if __name__ == "__main__":
    main()
