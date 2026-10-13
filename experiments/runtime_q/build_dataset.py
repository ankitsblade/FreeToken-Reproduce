"""Raw per-decision logs -> cleaned dataset with OFFLINE labels and splits.

One row = one q decision (episode, phase, token, layer).
  x_t        pre-action features (schema.PRE_ACTION)
  q_t        action actually executed (n_fetch)
  outcomes   wall / cpu / pcie / gemm of that layer call
  labels     cost_h{H} = this layer's net wall time now + over its next H visits (same
             episode+phase), fut_hits_h{H}, fut_m_h{H}. Built from FUTURE rows: labels only.

`randomized` marks rows from episodes whose action had a random component (uniform or
eps-exploration). Only those are used to fit L_hat(x, q): in a fixed-policy episode q is a
deterministic function of m and is confounded with the whole episode's future, so it cannot
tell what a different q would have cost.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paths import ROOT, study_dir
RQ = study_dir("runtime_q")
sys.path.insert(0, str(ROOT / "experiments/runtime_q/instrumentation"))
from schema import GROUPS, PRE_ACTION, UNAVAILABLE  # noqa: E402

HORIZONS = (0, 1, 4, 8, 16, 32)
TEST_POLICIES = {"uniform_b", "x0_e15", "t0111_e10"}
VAL_POLICIES = {"x25_e20", "t0012_e10"}
FIXED = {"q0", "f25", "f50", "f75", "qm", "native", "qstar_host"}


def forward_sum(a: np.ndarray, h: int, include_self: bool) -> np.ndarray:
    """sum a[i..i+h] (include_self) or a[i+1..i+h]; NaN where the future is not observed."""
    n = len(a)
    out = np.full(n, np.nan)
    if n > h:
        c = np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])
        i = np.arange(n - h)
        out[i] = c[i + h + 1] - c[i + (0 if include_self else 1)]
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["episode", "phase", "layer_id", "tok"]).reset_index(drop=True)
    g = df.groupby(["episode", "phase", "layer_id"], sort=False)
    df["step_in_phase"] = g.cumcount()
    df["phase_len"] = g["tok"].transform("size")
    for col, src, inc in (("cost", "wall_net_ms", True), ("fut_hits", "n_hits", False),
                          ("fut_m", "m", False), ("fut_fetch", "n_fetch", False)):
        for h in HORIZONS:
            if inc or h > 0:
                df[f"{col}_h{h}"] = g[src].transform(
                    lambda s, h=h, inc=inc: forward_sum(s.to_numpy(dtype=np.float64), h, inc))
    return df


def main() -> None:
    files = sorted(glob.glob(str(RQ / "data" / "raw" / "explore_*.csv")) +
                   glob.glob(str(RQ / "data" / "raw" / "round*_*.csv")))
    raw = pd.concat([pd.read_csv(f, low_memory=False).assign(source=Path(f).stem) for f in files],
                    ignore_index=True)
    n_raw = len(raw)
    df = add_labels(raw)
    df["cpu_task_ms"] = df["cpu_task_ms"].fillna(0.0)
    df["n_cpu"] = df["m"] - df["n_fetch"]
    df["randomized"] = ~df["policy"].isin(FIXED)
    df["split_run"] = np.where(df["policy"].isin(TEST_POLICIES), "test",
                               np.where(df["policy"].isin(VAL_POLICIES), "val", "train"))
    half = df["phase_len"] // 2
    df["split_time"] = np.where(df["phase"] == "warmup", "train",
                                np.where(df["step_in_phase"] < half, "val", "test"))

    report = {"raw_rows": n_raw, "files": [Path(f).name for f in files],
              "episodes": int(df["episode"].nunique()), "removed": {}, "kept": []}
    for f in PRE_ACTION:
        col = df[f]
        if col.isna().any():
            report["removed"][f] = "missing values"
        elif col.nunique() <= 1:
            report["removed"][f] = f"constant ({col.iloc[0]}) on this machine/model"
    # redundancy: |spearman| > 0.98 with an earlier kept feature
    kept: list[str] = []
    cand = [f for f in PRE_ACTION if f not in report["removed"]]
    sample = df.loc[df["randomized"], cand].sample(min(40000, int(df["randomized"].sum())), random_state=0)
    corr = sample.corr(method="spearman").abs()
    for f in cand:
        dup = next((k for k in kept if corr.loc[f, k] > 0.98), None)
        if dup:
            report["removed"][f] = f"redundant with {dup} (|rho|={corr.loc[f, dup]:.3f})"
        else:
            kept.append(f)
    report["kept"] = kept
    report["groups_kept"] = {g: [f for f in fs if f in kept] for g, fs in GROUPS.items()}
    report["unavailable"] = UNAVAILABLE
    report["rows"] = int(len(df))
    report["randomized_rows"] = int(df["randomized"].sum())
    report["split_run"] = df["split_run"].value_counts().to_dict()
    report["split_time"] = df["split_time"].value_counts().to_dict()

    out = RQ / "data" / "clean"
    out.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out / "dataset.pkl")
    df.drop(columns=["routed_ids", "missing_ids", "fetched_ids", "evicted_gids"]).to_csv(
        out / "dataset.csv.gz", index=False)
    (out / "feature_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"rows {len(df)} (randomized {report['randomized_rows']}), episodes {report['episodes']}")
    print(f"kept {len(kept)} / {len(PRE_ACTION)} pre-action features")
    for f, why in report["removed"].items():
        print(f"  removed {f:24s} {why}")


if __name__ == "__main__":
    main()
