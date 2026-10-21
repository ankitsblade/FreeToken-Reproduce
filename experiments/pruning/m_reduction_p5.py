"""P5: static per-layer slot allocation at fixed total bytes (plan section 8.2).

Quota-partitioned layers are independent (each layer's q rule and victims use only its
own slots), so running every layer at quota c yields each layer's misses at c.
"""

import json
import math

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_sim import project_state, run, uniform_quotas
from experiments.pruning.m_reduction_stats import WORKLOADS, equal_workload, paired_bootstrap, request_metrics
from experiments.pruning.m_reduction_traces import calibration

POOLS = ("mixed", "math", "code", "incident")


def curve_range(geometry, spread):
    return geometry.top_k, min(geometry.experts, math.ceil(spread * geometry.cache_slots / geometry.layers))


def request_curves(trace, geometry, table, arm, quota_range):
    """Per-layer misses for every uniform quota in range under the selected q rule."""
    make, floor = table[arm]
    lo, hi = quota_range
    curves = np.zeros((geometry.layers, hi - lo + 1), dtype=np.int64)
    for j, c in enumerate(range(lo, hi + 1)):
        quotas = [c] * geometry.layers
        result = run(trace, geometry, make(), floor, c * geometry.layers, "native", "quota", quotas)
        np.add.at(curves[:, j], trace.layer, result["m"])
    return curves


def pooled_mean(curves, traces, pool):
    chosen = [t for t in traces if pool == "mixed" or t.workload == pool]
    by_workload = [np.mean([curves[t.capture_id] for t in chosen if t.workload == w], axis=0)
                   for w in WORKLOADS if any(t.workload == w for t in chosen)]
    return np.mean(by_workload, axis=0)


def decreasing_gains(curve):
    """Marginal gains made non-increasing by pooled adjacent violators, clipped at zero."""
    gains = np.maximum(curve[:-1] - curve[1:], 0.0).astype(float)
    blocks = [[g, 1] for g in gains]
    merged = []
    for value, weight in blocks:
        merged.append([value, weight])
        while len(merged) > 1 and merged[-2][0] < merged[-1][0]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            merged.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([[v] * w for v, w in merged]) if merged else gains


def allocate(mean_curves, geometry, quota_range):
    lo, hi = quota_range
    gains = [decreasing_gains(curve) for curve in mean_curves]
    quotas = [geometry.top_k] * geometry.layers
    for _ in range(geometry.cache_slots - sum(quotas)):
        best = max(range(geometry.layers), key=lambda l: (
            gains[l][quotas[l] - lo] if lo <= quotas[l] < hi and quotas[l] < geometry.experts else -1.0, -l))
        if quotas[best] >= min(hi, geometry.experts):
            # Every layer exhausted its measured range; extend the least-allocated layer by ID.
            best = min(range(geometry.layers), key=lambda l: (quotas[l], l))
        quotas[best] += 1
    if sum(quotas) != geometry.cache_slots or any(not geometry.top_k <= c <= geometry.experts for c in quotas):
        raise ValueError("Infeasible quota allocation")
    return quotas, [g.tolist() for g in gains]


def arm_rows(traces, geometry, table, arm, quotas, seed):
    make, floor = table[arm]
    rows = []
    for trace in traces:
        replacement = "lru" if quotas is None else "quota"
        out = run(trace, geometry, make(), floor, geometry.cache_slots, "native", replacement, quotas, seed_quotas=seed)
        row = request_metrics(trace, out["m"], out["q"], geometry.expert_bytes, geometry.layers)
        rows.append(row)
    return rows


def main():
    p = parser(__doc__)
    p.add_argument("--q-arm", default="overlap_floor1", help="P4-selected q/fetch rule (fixed for P5)")
    p.add_argument("--curve-spread", type=float, default=3.0, help="Upper curve quota as a multiple of C/L")
    args = p.parse_args()
    out, bindings, models, status = start(args, "p5_quotas", __file__,
                                          {"q_arm": args.q_arm, "curve_spread": args.curve_spread, "pools": POOLS,
                                           "allocation": "start at k, greedy next-slot fitted gain, tie lower layer",
                                           "fit": "pooled-adjacent-violators non-increasing marginal gains"})
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        table, derived = rules(geometry, calibration(bindings[model], model))
        qrange = curve_range(geometry, args.curve_spread)
        curves = {t.capture_id: request_curves(t, geometry, table, args.q_arm, qrange) for t in traces}
        np.savez_compressed(out / f"{model}_curves.npz", **curves)
        uniform = uniform_quotas(geometry.layers, geometry.cache_slots)
        result = {"source": source, "q_arm": args.q_arm, "quota_range": qrange, "uniform_quotas": uniform, "folds": {}}
        held_rows = {name: [] for name in ("global_seeded", "uniform", "utility", "global_native")}
        transfer = {}
        for fold in sorted({t.fold for t in traces}):
            train = [t for t in traces if t.fold != fold]
            held = [t for t in traces if t.fold == fold]
            fitted = {pool: allocate(pooled_mean(curves, train, pool), geometry, qrange) for pool in POOLS}
            utility = fitted["mixed"][0]
            seed = [min(a, b) for a, b in zip(uniform, utility)]
            rows = {"global_seeded": arm_rows(held, geometry, table, args.q_arm, None, seed),
                    "uniform": arm_rows(held, geometry, table, args.q_arm, uniform, seed),
                    "utility": arm_rows(held, geometry, table, args.q_arm, utility, seed),
                    "global_native": arm_rows(held, geometry, table, args.q_arm, None, None)}
            for name, values in rows.items():
                held_rows[name].extend(values)
            dropped = [len(project_state(t.initial, geometry.layers, geometry.experts, geometry.cache_slots, seed)[1])
                       for t in held]
            predicted = pooled_mean(curves, train, "mixed")
            actual = pooled_mean(curves, held, "mixed")
            index = [c - qrange[0] for c in utility]
            fit_error = float(np.mean([abs(predicted[l][min(index[l], len(predicted[l]) - 1)]
                                           - actual[l][min(index[l], len(actual[l]) - 1)])
                                       for l in range(geometry.layers)]))
            result["folds"][str(fold)] = {"quotas": {pool: fitted[pool][0] for pool in POOLS}, "seed": seed,
                                          "mean_dropped_residents": float(np.mean(dropped)),
                                          "dropped_bytes_per_request": float(np.mean(dropped)) * geometry.expert_bytes,
                                          "held_out_per_layer_fit_error_misses": fit_error}
            for pool in POOLS:
                for target in WORKLOADS:
                    sub = [t for t in held if t.workload == target]
                    pool_rows = arm_rows(sub, geometry, table, args.q_arm, fitted[pool][0], seed)
                    transfer.setdefault(f"{pool}->{target}", []).extend(pool_rows)
        for name, values in held_rows.items():
            write_lines(out / f"{model}_{name}.jsonl", values)
        base = held_rows["global_seeded"]
        result["comparisons"] = {name: {"vs_global_seeded_mean_m": paired_bootstrap(held_rows[name], base, "mean_m"),
                                        "vs_global_seeded_p95_m": paired_bootstrap(held_rows[name], base, "p95_m")}
                                 for name in ("uniform", "utility", "global_native")}
        result["transfer_cells"] = {cell: {"mean_m": float(np.mean([r["mean_m"] for r in rows])), "requests": len(rows)}
                                    for cell, rows in transfer.items()}
        deployment = allocate(pooled_mean(curves, traces, "mixed"), geometry, qrange)[0]
        result["deployment_quotas"] = deployment
        result["note"] = ("Quotas govern decode eviction from the common feasible seed; global staging is restored "
                          "before prefill. Adoption requires live confirmation against G2, not the offline curve.")
        (out / f"{model}_quotas.json").write_text(json.dumps({"deployment": deployment, "uniform": uniform,
                                                              "folds": result["folds"]}, indent=1))
        summary[model] = result
    finish(out, summary, status)


if __name__ == "__main__":
    main()
