"""P7 candidate teacher-forcing summary on D0 (plan sections 9.3, 10.1 and 10.2).

Reads accepted candidate batches (arm 1, R75, arm 2 per outer fold, requantization-only
control) and reports, per arm: propagated teacher-forced m against the paired native
capture, full-vocabulary KL(P_baseline || P_candidate), top-1 agreement and NLL, each per
workload with request bootstrap intervals; the KL (upper 95% bound <= 0.02 nats/token per
workload) and top-1 (lower bound >= 98%) gates; and the G2 output-changing m criteria.
For arm 2 it also reports incremental merge error over the requantization-only control.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import atomic_json
from experiments.pruning.m_reduction_stats import WORKLOADS, g2_output_changing, paired_bootstrap

ARMS = {"arm1": ["teacher_candidate_arm1"], "r75": ["teacher_candidate_r75"],
        "arm2": [f"teacher_candidate_arm2_fold{f}" for f in range(5)], "requant": ["teacher_candidate_requant"]}


def accepted(root):
    for batch in sorted(root.glob("*"), reverse=True):
        done = batch / "completion.json"
        if done.is_file():
            record = json.loads(done.read_text())
            if record.get("status") == "complete" and not record.get("limit"):
                return batch
    return None


def rows_for(study, model, arm):
    rows, batches = [], []
    for name in ARMS[arm]:
        batch = accepted(study / "captures" / model / name / "all")
        if batch is None:
            return None, None
        batches.append(str(batch))
        for manifest in sorted(batch.glob("*/manifest.json")):
            data = json.loads(manifest.read_text())
            v, request = data["validation"], data["request"]
            rows.append({"request_id": request["request_id"], "workload": request["workload"],
                         "fold": v.get("outer_fold"), "mean_m": v["sum_m"] / v["decode_events"],
                         "baseline_mean_m": v["baseline_sum_m"] / v["baseline_decode_events"],
                         "mean_kl": v["mean_kl"], "top1": v["top1_agreement"], "nll": v["mean_nll"],
                         "baseline_nll": v["mean_baseline_nll"]})
    return rows, batches


def bounds(values, seed=20261014):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    summary = {"model": args.model, "utc": datetime.now(timezone.utc).isoformat(), "arms": {}}
    per_arm = {}
    for arm in ARMS:
        rows, batches = rows_for(study, args.model, arm)
        if rows is None:
            summary["arms"][arm] = {"status": "missing"}
            continue
        per_arm[arm] = rows
        base = [{**r, "mean_m": r["baseline_mean_m"]} for r in rows]
        m = paired_bootstrap(rows, base, "mean_m")
        quality = {}
        for workload in WORKLOADS:
            sub = [r for r in rows if r["workload"] == workload]
            kl, top1 = bounds([r["mean_kl"] for r in sub]), bounds([r["top1"] for r in sub])
            nll = bounds([r["nll"] - r["baseline_nll"] for r in sub])
            quality[workload] = {"kl": kl, "kl_gate_upper_le_0_02": kl["ci95"][1] <= 0.02,
                                 "top1_agreement": top1, "top1_gate_lower_ge_0_98": top1["ci95"][0] >= 0.98,
                                 "nll_increase": nll,
                                 "perplexity_ratio": {"point": float(np.exp(nll["mean"])),
                                                      "ci95": [float(np.exp(v)) for v in nll["ci95"]]}}
        passed = all(q["kl_gate_upper_le_0_02"] and q["top1_gate_lower_ge_0_98"] for q in quality.values())
        summary["arms"][arm] = {"batches": batches, "requests": len(rows), "teacher_forced_m": m,
                                "quality": quality, "kl_top1_gates_passed": passed,
                                "g2_output_changing": g2_output_changing(m, quality_passed=passed if not passed else None)}
    if "arm2" in per_arm and "requant" in per_arm:
        requant = {r["request_id"]: r for r in per_arm["requant"]}
        summary["arm2_incremental_over_requant"] = {
            w: bounds([r["mean_kl"] - requant[r["request_id"]]["mean_kl"] for r in per_arm["arm2"] if r["workload"] == w])
            for w in WORKLOADS}
    out = study / "analysis" / args.model / f"p7_teacher_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, summary)
    for arm, result in summary["arms"].items():
        if "quality" in result:
            print(arm, "m", round(result["teacher_forced_m"]["reduction"], 4), result["teacher_forced_m"]["reduction_ci95"],
                  {w: (round(q["kl"]["mean"], 4), round(q["top1_agreement"]["mean"], 4)) for w, q in result["quality"].items()},
                  "gates", result["kl_top1_gates_passed"], result["g2_output_changing"]["decision"])
        else:
            print(arm, result["status"])
    print("out", out)


if __name__ == "__main__":
    main()
