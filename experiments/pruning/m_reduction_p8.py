"""P8: 2 x 2 routing x q interaction on fixed trajectories (plan section 10.3).

Holds the P1 seed and P5 allocation fixed. q0 = overlap_floor1, q1 = selected Reuse-Aware
variant; r0 = native routing, r1 = frozen qualified P6 finalist. A missing qualified
finalist is reported as a missing interaction, never replaced by a new policy.
Timing interaction uses paired live timings when supplied (JSONL with request_id,
repetition, arm, decode_ms_per_token).
"""

import json
from pathlib import Path

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_p6 import evaluate as routing_rows
from experiments.pruning.m_reduction_p6 import SETTINGS, label
from experiments.pruning.m_reduction_sim import uniform_quotas
from experiments.pruning.m_reduction_stats import SEED, WORKLOADS, equal_workload
from experiments.pruning.m_reduction_traces import calibration

ARMS = ("00", "01", "10", "11")


def interaction(rows, repeats=10000, seed=SEED):
    """I_m = M10 + M01 - M00 - M11 with request bootstrap within fold x workload cells."""
    keyed = {arm: {r["request_id"]: r for r in rows[arm]} for arm in ARMS}
    ids = sorted(keyed["00"])
    cells = {}
    for rid in ids:
        r = keyed["00"][rid]
        cells.setdefault((r["fold"], r["workload"]), []).append(rid)

    def value(sample):
        per = {w: [] for w in WORKLOADS}
        for rid in sample:
            w = keyed["00"][rid]["workload"]
            per[w].append(keyed["10"][rid]["mean_m"] + keyed["01"][rid]["mean_m"]
                          - keyed["00"][rid]["mean_m"] - keyed["11"][rid]["mean_m"])
        return float(np.mean([np.mean(per[w]) for w in WORKLOADS]))

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        sample = [cell[i] for cell in cells.values() for i in rng.integers(0, len(cell), len(cell))]
        boot.append(value(sample))
    return {"I_m": value(ids), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "interpretation": "Positive: combined miss reduction exceeds the sum of separate reductions"}


def timing_interaction(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    table = {}
    for r in rows:
        table.setdefault((r["request_id"], r["repetition"]), {})[r["arm"]] = r["decode_ms_per_token"]
    values = [np.log(t["10"]) + np.log(t["01"]) - np.log(t["00"]) - np.log(t["11"])
              for t in table.values() if all(a in t for a in ARMS)]
    rng = np.random.default_rng(SEED)
    boot = [np.mean(rng.choice(values, len(values))) for _ in range(10000)] if values else []
    return {"I_logT": float(np.mean(values)) if values else None, "pairs": len(values),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "interpretation": "Positive: improvement beyond multiplicative independent speedups"}


def main():
    p = parser(__doc__)
    p.add_argument("--q1", default="reuse_freq_floor1", help="Selected Reuse-Aware P4 variant")
    p.add_argument("--r1", help="Frozen qualified P6 finalist label, e.g. A_lam0.1_alphaNone")
    p.add_argument("--p5-quotas", type=Path, help="Directory with MODEL_quotas.json (P5 deployment allocation)")
    p.add_argument("--timing", type=Path, help="Paired live timing JSONL for the 2x2 cohort")
    args = p.parse_args()
    out, bindings, models, status = start(args, "p8_interaction", __file__,
                                          {"q0": "overlap_floor1", "q1": args.q1, "r1": args.r1,
                                           "p5_quotas": str(args.p5_quotas) if args.p5_quotas else None,
                                           "timing": str(args.timing) if args.timing else None})
    summary = {}
    settings = {label(s): s for s in SETTINGS}
    for model in models:
        if not args.r1 or args.r1 not in settings:
            summary[model] = {"status": "missing_interaction: no qualified P6 finalist supplied"}
            continue
        geometry, traces, source = load(bindings, model)
        table, _ = rules(geometry, calibration(bindings[model], model))
        quotas = uniform_quotas(geometry.layers, geometry.cache_slots)
        if args.p5_quotas and (args.p5_quotas / f"{model}_quotas.json").is_file():
            quotas = json.loads((args.p5_quotas / f"{model}_quotas.json").read_text())["deployment"]
        rows = {arm: [] for arm in ARMS}
        for fold in sorted({t.fold for t in traces}):
            train = [t for t in traces if t.fold != fold]
            held = [t for t in traces if t.fold == fold]
            for arm in ARMS:
                routing = settings[args.r1] if arm[0] == "1" else ("native", None, None)
                q_arm = args.q1 if arm[1] == "1" else "overlap_floor1"
                rows[arm].extend(routing_rows(held, train, geometry, table, q_arm, geometry.cache_slots, "native", routing))
        for arm, values in rows.items():
            write_lines(out / f"{model}_{arm}.jsonl", values)
        summary[model] = {"source": source, "quota_vector_recorded": quotas,
                          "mean_m": {arm: equal_workload(values, "mean_m") for arm, values in rows.items()},
                          "miss_interaction": interaction(rows),
                          "timing_interaction": timing_interaction(args.timing) if args.timing else None,
                          "note": ("Quota-partitioned replay of routed arms requires the live quota runtime; this "
                                   "fixed-trajectory factorial uses global LRU and records the P5 vector for the live arm.")}
    finish(out, summary, status)


if __name__ == "__main__":
    main()
