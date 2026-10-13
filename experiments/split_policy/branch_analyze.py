"""Per-policy CPU vs GPU branch balance from the *realized* decode traffic.

The hybrid GPU expert cache is populated only by GPU fetches, so the realized miss
distribution -- and therefore the CPU vs PCIe branch loads -- depend on the policy, not
just the workload. A static per-(m,q) model (branch_oracle) cannot capture that. So we take
each policy's own measured per-token traffic from the trace and divide by the measured
(contended) branch bandwidths:

    T_CPU = cpu_bytes_per_token / B_H          (CPU expert compute)
    T_GPU = h2d_bytes_per_token / B_P          (PCIe expert transfer, the GPU-side cost)
    branch = max(T_CPU, T_GPU)                 the two run concurrently per layer
    imbalance = |T_CPU - T_GPU| / max(T_CPU, T_GPU)

These are per generated token (summed over the 24 MoE layers). B_H, B_P are the overlapped
(both-branches-active) bandwidths from calibration -- the regime that actually applies when
the hybrid backend runs both branches at once.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
OUT = study_dir("split_policy") / "results"
SUM = OUT / "summary"

POLICY_ORDER = ["A_adaptive", "B_gpu", "C_cpu", "D_fixed25", "E_fixed50", "F_fixed75"]
WORKLOAD_ORDER = ["low_miss", "med_miss", "high_miss"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bandwidths() -> tuple[float, float]:
    """Overlapped PCIe gather (B_P) and CPU MoE (B_H) bandwidths in bytes/s."""
    bp = bh = None
    for p in list((OUT / "calibration").glob("*.json")) + \
             list(Path.home().glob(".cache/freetoken/benchbw/*.json")):
        try:
            d = json.load(open(p))
        except OSError:
            continue
        acc = {}
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("pcie_gather_overlap_gbs", "cpu_moe_overlap_gbs"):
                        acc[k] = v
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
        if "pcie_gather_overlap_gbs" in acc:
            bp = acc["pcie_gather_overlap_gbs"] * 1e9
        if "cpu_moe_overlap_gbs" in acc:
            bh = acc["cpu_moe_overlap_gbs"] * 1e9
    return bp or 6.71e9, bh or 35.68e9


def main() -> None:
    traffic = {(r["workload"], r["policy"]): r for r in read_csv(SUM / "summary_traffic.csv")}
    bp, bh = bandwidths()
    rows = []
    for wl in WORKLOAD_ORDER:
        for pol in POLICY_ORDER:
            t = traffic.get((wl, pol))
            if not t:
                continue
            cpu_bytes = float(t["cpu_bytes_per_token"])
            h2d_bytes = float(t["bytes_h2d_per_token"])
            t_cpu = cpu_bytes / bh * 1000
            t_gpu = h2d_bytes / bp * 1000
            denom = max(t_cpu, t_gpu)
            rows.append({
                "workload": wl, "policy": pol,
                "mean_q": t["mean_q"], "mean_m": t["mean_m"],
                "miss_rate": t["miss_rate"],
                "cpu_MB_per_tok": round(cpu_bytes / 1e6, 1),
                "h2d_MB_per_tok": round(h2d_bytes / 1e6, 1),
                "T_CPU_ms": round(t_cpu, 3), "T_GPU_ms": round(t_gpu, 3),
                "max_branch_ms": round(denom, 3),
                "imbalance": round(abs(t_cpu - t_gpu) / denom, 3) if denom else 0.0,
                "B_P_GBps": round(bp / 1e9, 2), "B_H_GBps": round(bh / 1e9, 2),
            })
    write_csv(SUM / "summary_balance.csv", rows)
    print(f"wrote {len(rows)} balance rows (B_P={bp/1e9:.2f} B_H={bh/1e9:.2f} GB/s)")
    for r in rows:
        print(f"  {r['workload']:9s} {r['policy']:11s} q~{float(r['mean_q']):.2f} "
              f"miss={float(r['miss_rate']):.2f}  T_CPU={r['T_CPU_ms']:6.2f} "
              f"T_GPU={r['T_GPU_ms']:6.2f} max={r['max_branch_ms']:6.2f} imb={r['imbalance']:.2f}")


if __name__ == "__main__":
    main()
