"""P9: replacement protection (segmented LRU, thrash-targeted pinning) versus LRU (plan section 10.4)."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_p5 import POOLS
from experiments.pruning.m_reduction_sim import CacheSim, initial_state, run, uniform_quotas
from experiments.pruning.m_reduction_stats import (WORKLOADS, equal_workload, g2_exact, nested_select,
                                                    paired_bootstrap, request_metrics)
from experiments.pruning.m_reduction_traces import calibration

SLRU = (0.25, 0.50)
PINS = (0.10, 0.25)


def thrash_counts(trace, geometry, table, arm):
    """Per-expert thrash refetches and misses under the fixed q rule and global LRU."""
    make, floor = table[arm]
    E, C = geometry.experts, geometry.cache_slots
    sim = CacheSim(initial_state(trace, geometry, C, "native"), geometry.layers, E, make(), floor)
    stamp = np.full(geometry.layers * E, -1, dtype=np.int64)
    evicted, thrash, misses = set(), Counter(), Counter()
    high = math.ceil(1.10 * C)
    for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
        flat = [layer * E + e for e in ids]
        distance = {f: int((stamp > stamp[f]).sum()) if stamp[f] >= 0 else None for f in flat}
        out = sim.event(layer, ids)
        for e in out["missing"]:
            f = layer * E + e
            misses[f] += 1
            if f in evicted and distance[f] is not None and C < distance[f] <= high:
                thrash[f] += 1
        evicted.difference_update(flat)
        evicted.update(f for f in out["evicted"] if f >= 0)
        stamp[flat] = i
    return thrash, misses


def pin_set(train, counts, geometry, fraction, quotas):
    """Global pins floor(fraction x C), apportioned by the quota vector, ranked by thrash, misses, ID."""
    total = math.floor(fraction * geometry.cache_slots)
    share = [math.floor(total * q / geometry.cache_slots) for q in quotas]
    for layer in sorted(range(geometry.layers), key=lambda l: (-(total * quotas[l] / geometry.cache_slots - share[l]), l)):
        if sum(share) == total:
            break
        share[layer] += 1
    thrash, misses = Counter(), Counter()
    for trace in train:
        thrash.update(counts[trace.capture_id][0])
        misses.update(counts[trace.capture_id][1])
    pins = []
    for layer in range(geometry.layers):
        limit = min(share[layer], max(0, quotas[layer] - geometry.top_k))
        ranked = sorted(range(geometry.experts), key=lambda e: (-thrash[layer * geometry.experts + e],
                                                                -misses[layer * geometry.experts + e], e))
        pins.extend((layer, e) for e in ranked[:limit])
    return pins


def rows_for(traces, geometry, table, arm, replacement, slru=None, pins=None):
    make, floor = table[arm]
    out = []
    for trace in traces:
        result = run(trace, geometry, make(), floor, geometry.cache_slots, "native", replacement,
                     slru_fraction=slru, pins=pins)
        out.append(request_metrics(trace, result["m"], result["q"], geometry.expert_bytes, geometry.layers))
    return out


def main():
    p = parser(__doc__)
    p.add_argument("--q-arm", default="overlap_floor1", help="P4-selected q rule held fixed")
    p.add_argument("--p5-quotas", type=Path, help="P5 MODEL_quotas.json directory for pin apportionment")
    p.add_argument("--p3-summary", type=Path, help="P3 summary.json to record G1 trigger status")
    args = p.parse_args()
    out, bindings, models, status = start(args, "p9_replacement", __file__,
                                          {"q_arm": args.q_arm, "slru_protected_fractions": SLRU, "pin_fractions": PINS,
                                           "pins": "protection only; not preloaded; reset per request",
                                           "slru": "admit to probation; second hit promotes; overflow demotes to probation MRU"})
    triggers = json.loads(args.p3_summary.read_text()) if args.p3_summary else {}
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        table, _ = rules(geometry, calibration(bindings[model], model))
        quotas = uniform_quotas(geometry.layers, geometry.cache_slots)
        if args.p5_quotas and (args.p5_quotas / f"{model}_quotas.json").is_file():
            quotas = json.loads((args.p5_quotas / f"{model}_quotas.json").read_text())["deployment"]
        counts = {t.capture_id: thrash_counts(t, geometry, table, args.q_arm) for t in traces}
        held = {"lru": [], **{f"slru_{f}": [] for f in SLRU}, **{f"pin_{f}": [] for f in PINS}}
        transfer, pins_by_fold = {}, {}
        for fold in sorted({t.fold for t in traces}):
            train = [t for t in traces if t.fold != fold]
            test = [t for t in traces if t.fold == fold]
            held["lru"].extend(rows_for(test, geometry, table, args.q_arm, "lru"))
            for f in SLRU:
                held[f"slru_{f}"].extend(rows_for(test, geometry, table, args.q_arm, "slru", slru=f))
            pins_by_fold[str(fold)] = {}
            for f in PINS:
                pins = pin_set(train, counts, geometry, f, quotas)
                pins_by_fold[str(fold)][str(f)] = pins
                held[f"pin_{f}"].extend(rows_for(test, geometry, table, args.q_arm, "lru", pins=pins))
                for pool in POOLS:
                    pool_train = [t for t in train if pool == "mixed" or t.workload == pool]
                    pool_pins = pin_set(pool_train, counts, geometry, f, quotas)
                    for target in WORKLOADS:
                        sub = [t for t in test if t.workload == target]
                        transfer.setdefault(f"pin_{f}:{pool}->{target}", []).extend(
                            rows_for(sub, geometry, table, args.q_arm, "lru", pins=pool_pins))
        for name, values in held.items():
            write_lines(out / f"{model}_{name}.jsonl", values)
        comparisons = {}
        for name in held:
            if name == "lru":
                continue
            mean = paired_bootstrap(held[name], held["lru"], "mean_m")
            p95 = paired_bootstrap(held[name], held["lru"], "p95_m")
            comparisons[name] = {"vs_lru_mean_m": mean, "vs_lru_p95_m": p95, "g2_exact": g2_exact(mean, p95)}
        cache = {}

        def score(candidate, fit, evaluation):
            key = (candidate, tuple(sorted(t.capture_id for t in fit)), tuple(sorted(t.capture_id for t in evaluation)))
            if key not in cache:
                kind, value = candidate.split("_")
                pins = pin_set(fit, counts, geometry, float(value), quotas) if kind == "pin" else None
                rows = rows_for(evaluation, geometry, table, args.q_arm, "slru" if kind == "slru" else "lru",
                                slru=float(value) if kind == "slru" else None, pins=pins)
                cache[key] = equal_workload(rows, "mean_m")
            return cache[key]

        selection = {family: nested_select(traces, [f"{family}_{v}" for v in values], score, lambda c: c)
                     for family, values in (("slru", SLRU), ("pin", PINS))}
        trigger = triggers.get(model, {}).get("cells", {}).get("primary/native", {}).get("g1", {}).get("prioritize_p9")
        (out / f"{model}_pins.json").write_text(json.dumps(pins_by_fold))
        summary[model] = {"source": source, "q_arm": args.q_arm, "quota_vector": quotas, "comparisons": comparisons,
                          "selection": selection, "g1_p9_trigger": trigger,
                          "transfer_cells": {k: {"mean_m": float(np.mean([r["mean_m"] for r in v])), "requests": len(v)}
                                             for k, v in transfer.items()},
                          "note": "Run on every model regardless of trigger; trigger sets priority and adoption eligibility."}
    finish(out, summary, status)


if __name__ == "__main__":
    main()
