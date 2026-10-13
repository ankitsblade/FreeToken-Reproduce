"""Analyze the prefill overlap ON/OFF sweep: summary CSV, decomposition, plots.

Decomposition (from the ON/OFF TTFT alone, no profiler needed):
  OFF(L) = overhead + compute(L) + transfer            (serial)
  ON(L)  = overhead + max(compute(L), transfer)         (perfect overlap)
  benefit(L) = OFF - ON = compute(L) + transfer - max(compute, transfer)
             = min(compute(L), transfer)                the hidden branch
So the overlap hides the *smaller* of expert-transfer and prefill-compute. Where ON is
flat while OFF rises, compute(L) < transfer for all tested L (prefill is transfer-bound),
and benefit(L) = compute(L); the ON floor approximates overhead + transfer.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
RES = study_dir("prefill_overlap") / "results"
FIG = study_dir("prefill_overlap") / "figures"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def monitor_stats(tag: str) -> dict:
    p = RES / f"monitor_overlap_{tag}.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    def agg(k, fn):
        v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
        return fn(v) if v else None
    return {
        "gpu_util_med": agg("utilization.gpu", statistics.median),
        "gpu_mem_util_med": agg("utilization.memory", statistics.median),
        "gpu_power_med_w": agg("power.draw", statistics.median),
        "gpu_power_max_w": agg("power.draw", max),
        "cpu_util_med": agg("cpu_utilization_pct", statistics.median),
        "cpu_util_max": agg("cpu_utilization_pct", max),
        "pcie_gen": rows[-1].get("pcie.link.gen.current"),
        "pcie_width": rows[-1].get("pcie.link.width.current"),
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load(RES / "prefill_ttft.jsonl")
    lengths = sorted({r["target_tokens"] for r in rows})
    def cell(overlap, L):
        ts = [r["ttft_ms"] for r in rows if r["overlap"] == overlap and r["target_tokens"] == L]
        pk = [r["prompt_tokens"] for r in rows if r["overlap"] == overlap and r["target_tokens"] == L]
        return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0), statistics.mean(pk)

    summary = []
    on_floor = min(cell(True, L)[0] for L in lengths)  # ON floor ~ overhead + transfer
    for L in lengths:
        on_m, on_s, ptok = cell(True, L)
        off_m, off_s, _ = cell(False, L)
        benefit = off_m - on_m
        summary.append({
            "target_tokens": L, "prompt_tokens": round(ptok),
            "ttft_on_ms": round(on_m, 1), "ttft_on_std": round(on_s, 1),
            "ttft_off_ms": round(off_m, 1), "ttft_off_std": round(off_s, 1),
            "overlap_benefit_ms": round(benefit, 1),
            "overlap_benefit_pct": round(100 * benefit / off_m, 1),
            # benefit(L) = min(compute, transfer); ON is transfer-bound here so benefit ~ compute(L)
            "compute_est_ms": round(benefit, 1),
            "transfer_lower_bound_ms": round(on_floor, 1),
            "exposed_ms_off": round(off_m - on_floor, 1),
        })
    with (RES / "summary_prefill.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    util = {"on": monitor_stats("on"), "off": monitor_stats("off")}
    (RES / "summary_util.json").write_text(json.dumps(util, indent=2) + "\n")

    print("prompt  TTFT_on  TTFT_off  benefit  benefit%  compute_est")
    for s in summary:
        print(f"{s['prompt_tokens']:6d}  {s['ttft_on_ms']:7.1f}  {s['ttft_off_ms']:8.1f}  "
              f"{s['overlap_benefit_ms']:7.1f}  {s['overlap_benefit_pct']:7.1f}  {s['compute_est_ms']:7.1f}")
    print("util ON :", util["on"])
    print("util OFF:", util["off"])

    xs = [s["prompt_tokens"] for s in summary]
    # Plot 1: TTFT vs prompt length, ON vs OFF
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [s["ttft_on_ms"] for s in summary], "o-", label="overlap ON", color="#4C72B0")
    ax.plot(xs, [s["ttft_off_ms"] for s in summary], "s--", label="overlap OFF", color="#C44E52")
    ax.fill_between(xs, [s["ttft_on_ms"] for s in summary], [s["ttft_off_ms"] for s in summary],
                    color="#4C72B0", alpha=0.12, label="hidden transfer/compute")
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefill TTFT vs prompt length: overlap ON vs OFF")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "prefill1_ttft_vs_length.png", dpi=130); plt.close(fig)

    # Plot 2: overlap benefit (ms and %) vs length
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(x) for x in xs], [s["overlap_benefit_ms"] for s in summary], color="#55A868")
    for i, s in enumerate(summary):
        ax.text(i, s["overlap_benefit_ms"], f"{s['overlap_benefit_pct']:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("overlap benefit = TTFT_off - TTFT_on (ms)")
    ax.set_title("Expert-transfer time hidden by overlap (label = % of TTFT_off)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "prefill2_benefit.png", dpi=130); plt.close(fig)

    # Plot 3: GPU power ON vs OFF (concurrency signal)
    if util["on"].get("gpu_power_max_w") and util["off"].get("gpu_power_max_w"):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        labels = ["overlap ON", "overlap OFF"]
        med = [util["on"]["gpu_power_med_w"], util["off"]["gpu_power_med_w"]]
        mx = [util["on"]["gpu_power_max_w"], util["off"]["gpu_power_max_w"]]
        x = range(2)
        ax.bar([i - 0.2 for i in x], med, 0.4, label="median", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], mx, 0.4, label="peak", color="#DD8452")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel("GPU power (W)")
        ax.set_title("GPU power: concurrent copy+compute (ON) draws higher peak")
        ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(FIG / "prefill3_gpu_power.png", dpi=130); plt.close(fig)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
