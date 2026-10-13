"""Emit the comparison tables from summary CSVs into the documentation folder.

Two tables per workload, as specified by the experiment brief:
  1. Policy | avg q | CPU bytes/tok | bytes H2D/tok | tok/s | ms/tok | p95 | GPU util | CPU util
  2. Policy | T_CPU | T_GPU | max(T_CPU,T_GPU) | imbalance | expected q
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
SUM = study_dir("split_policy") / "results" / "summary"

POLICY_ORDER = ["A_adaptive", "B_gpu", "C_cpu", "D_fixed25", "E_fixed50", "F_fixed75"]
LABEL = {"A_adaptive": "q* adaptive", "B_gpu": "q=m (GPU)", "C_cpu": "q=0 (CPU)",
         "D_fixed25": "q=25%", "E_fixed50": "q=50%", "F_fixed75": "q=75%"}
WL_ORDER = ["low_miss", "med_miss", "high_miss"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def idx(rows):
    return {(r["workload"], r["policy"]): r for r in rows}


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    perf = idx(read_csv(SUM / "summary_perf.csv"))
    traffic = idx(read_csv(SUM / "summary_traffic.csv"))
    balance = idx(read_csv(SUM / "summary_balance.csv"))
    out = ["# q* policy comparison tables\n"]
    for wl in WL_ORDER:
        if not any((wl, p) in perf for p in POLICY_ORDER):
            continue
        out.append(f"\n## Workload: {wl}\n")
        out.append("\n### Performance and traffic\n")
        out.append("| Policy | avg q | miss m | CPU MB/tok | H2D MB/tok | tok/s | ms/tok | "
                   "p95 ms | GPU util% | CPU util% |")
        out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for p in POLICY_ORDER:
            t = traffic.get((wl, p), {})
            pf = perf.get((wl, p), {})
            cpu_mb = (float(t["cpu_bytes_per_token"]) / 1e6) if t.get("cpu_bytes_per_token") else None
            h2d_mb = (float(t["bytes_h2d_per_token"]) / 1e6) if t.get("bytes_h2d_per_token") else None
            out.append(f"| {LABEL[p]} | {fmt(t.get('mean_q'))} | {fmt(t.get('mean_m'))} | "
                       f"{fmt(cpu_mb,1)} | {fmt(h2d_mb,1)} | "
                       f"{fmt(pf.get('decode_tok_s_mean'))} | {fmt(pf.get('ms_per_token_mean'))} | "
                       f"{fmt(pf.get('event_ms_p95_mean'))} | {fmt(t.get('gpu_util_pct_median'),0)} | "
                       f"{fmt(t.get('cpu_util_pct_median'),0)} |")
        out.append("\n### Branch balance (realized per-token traffic / measured bandwidths)\n")
        out.append("| Policy | avg q | miss rate | T_CPU ms | T_GPU ms | max(T_CPU,T_GPU) ms | imbalance |")
        out.append("|---|--:|--:|--:|--:|--:|--:|")
        for p in POLICY_ORDER:
            b = balance.get((wl, p), {})
            out.append(f"| {LABEL[p]} | {fmt(b.get('mean_q'))} | {fmt(b.get('miss_rate'),3)} | "
                       f"{fmt(b.get('T_CPU_ms'),2)} | {fmt(b.get('T_GPU_ms'),2)} | "
                       f"{fmt(b.get('max_branch_ms'),2)} | {fmt(b.get('imbalance'),3)} |")
    text = "\n".join(out) + "\n"
    target = study_dir("split_policy") / "reports/tables.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
