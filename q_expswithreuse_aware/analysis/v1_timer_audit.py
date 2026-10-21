"""V1 validation on real decision logs: what the branch timers measure and how well C(m, q) predicts.

Timers per hybrid layer (layers/moe.py::_decode_hybrid, hook events in runtime_q_hook):
  pre      layer start -> PCIe copy start (router done, host decision, ensure kernel)
  pcie     copy_missing on the stream; gemm: grouped GEMM of hits + fills
  exposed  GEMM end -> layer end (waiting for the CPU partial, merge)
  cpu_task the C++ worker's own timer for the CPU experts (excludes submit and copies)
Observed MoE time after the decision = pcie + gemm + exposed (the slower branch plus merge).
V1 predicts it with C(m, q) = max(g0 + q tp, [m > q](c0 + (m - q) tc)) from lagged online fits.
Usage: v1_timer_audit.py LEAN_CSV [--out JSON].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["phase", "policy", "m", "n_fetch", "wall_ms", "pre_ms", "pcie_ms", "gemm_ms", "exposed_ms",
        "cpu_task_ms", "g0_ms", "tp_ms", "c0_ms", "tc_ms", "v_hit_ms", "decide_us"]
NEW = ["fit_g_prior", "fit_c_prior", "c_pred_ms"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    head = pd.read_csv(a.csv, nrows=0).columns
    d = pd.read_csv(a.csv, usecols=COLS + [c for c in NEW if c in head])
    d = d[d.phase == "measure"]
    d["n_cpu"] = d.m - d.n_fetch
    d["moe_obs"] = d.pcie_ms + d.gemm_ms + d.exposed_ms
    d["host_gap"] = d.wall_ms - (d.pre_ms + d.moe_obs)
    cpu_bound = d[(d.n_cpu > 0) & d.cpu_task_ms.notna() & (d.exposed_ms > 0.2)]
    out = {"log": str(a.csv), "rows": int(len(d)), "median_ms": {
        k: round(float(d[k].median()), 4) for k in ("pre_ms", "pcie_ms", "gemm_ms", "exposed_ms", "wall_ms", "host_gap")},
        "decide_us_median": round(float(d.decide_us.median()), 1),
        "cpu_bound_layers": int(len(cpu_bound)),
        "cpu_bound_moe_obs_minus_cpu_task_ms": {
            q: round(float((cpu_bound.moe_obs - cpu_bound.cpu_task_ms).quantile(q)), 4) for q in (0.1, 0.5, 0.9)},
        "by_policy": {}}
    for pol, g in d.groupby("policy"):
        g = g[g.m > 0]
        if "c_pred_ms" in g:
            pred = g.c_pred_ms.astype(float)
        else:
            q, m = g.n_fetch, g.m
            pred = np.maximum(g.g0_ms + q * g.tp_ms, np.where(m > q, g.c0_ms + (m - q) * g.tc_ms, 0.0))
        err = pred - g.moe_obs
        row = {"rows": int(len(g)), "bias_ms": round(float(err.mean()), 4), "mae_ms": round(float(err.abs().mean()), 4),
               "median_ape_pct": round(float((err.abs() / g.moe_obs).median() * 100), 2),
               "fills_per_layer": round(float(g.n_fetch.mean()), 4),
               "fit_ms_median": {k: round(float(g[k].median()), 4) for k in ("g0_ms", "tp_ms", "c0_ms", "tc_ms", "v_hit_ms")}}
        if "fit_g_prior" in g:
            row["gpu_fit_prior_share"] = round(float(g.fit_g_prior.mean()), 4)
            row["cpu_fit_prior_share"] = round(float(g.fit_c_prior.mean()), 4)
        out["by_policy"][pol] = row
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
