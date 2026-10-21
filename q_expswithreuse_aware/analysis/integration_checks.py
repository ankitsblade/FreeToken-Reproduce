"""Checks on real decision logs (lean CSV written by the hook during inference).

  decisions   each arm's q equals its own rule's shadow decision on the same state
              (B: q_orig; single-variant arms: q_v1 / q_v2 / q_v3)
  victims     predicted victims of the fills equal the slots the engine actually evicted
  alpha       V2's alpha changes only right after a layer that evicted something (causality:
              realised hits enter the calibration only when a residency ends)
  fits        (new logs) V1 never uses a fitted line with < 8 samples
Usage: integration_checks.py LEAN_CSV [--out JSON]. Prints and optionally writes a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHADOW = {"B_reuse_aware": "q_orig", "C_V1": "q_v1", "C_V2": "q_v2", "C_V3": "q_v3",
          # arm names of the v2 pilot (results/reuse_aware_v2/pilot_20261020T140940Z)
          "reuse_aware_lean": "q_orig", "ra_v1_cost": "q_v1", "ra_v2_alpha": "q_v2", "ra_v3_evict": "q_v3"}


def ids(s) -> list[int]:
    return [int(float(v)) for v in str(s).split(";") if v not in ("", "nan")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = pd.read_csv(a.csv, low_memory=False)
    d = d[d.phase == "measure"].copy()
    out: dict = {"log": str(a.csv), "rows": int(len(d))}

    dec = {}
    for pol, col in SHADOW.items():
        g = d[d.policy == pol]
        if len(g):
            dec[pol] = {"rows": int(len(g)), "mismatch": int((g.q != g[col]).sum()),
                        "fetching_rows": int((g.q > 0).sum())}
    out["decisions_vs_own_rule"] = dec

    fills = d[(d.n_fetch > 0)]
    match = total = free = 0
    by_pol: dict = {}
    for pol, n, vic, ev in zip(fills.policy, fills.n_fetch, fills.victim_gids, fills.evicted_gids):
        pred = ids(vic)[: int(n)]
        occupied = [g for g in pred if g >= 0]
        free += len(pred) - len(occupied)
        ok = sorted(occupied) == sorted(ids(ev))
        match += ok
        total += 1
        s = by_pol.setdefault(pol, [0, 0])
        s[0] += ok
        s[1] += 1
    out["victims"] = {"layers_with_fills": total, "exact_match": match,
                      "match_rate": round(match / total, 6) if total else None, "free_slot_fills": free,
                      "by_policy": {p: round(v[0] / v[1], 6) for p, v in by_pol.items()}}

    viol = checked = 0
    for _, g in d.groupby("episode", sort=False):
        al, ne = g.alpha.to_numpy(), g.n_evict.to_numpy()
        ch = np.flatnonzero(al[1:] != al[:-1]) + 1
        checked += len(ch)
        viol += int((ne[ch - 1] == 0).sum())
    out["alpha_changes"] = {"changes": int(checked), "without_preceding_eviction": viol}

    if "fit_g_prior" in d:
        out["fits"] = {
            "gpu_fit_below_8_samples": int(((d.fit_g_prior == 0) & (d.fit_g_n < 8)).sum()),
            "cpu_fit_below_8_samples": int(((d.fit_c_prior == 0) & (d.fit_c_n < 8)).sum()),
            "victim_prediction_changed_by_fix_rows": int((d.victim_fix > 0).sum()),
        }
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
