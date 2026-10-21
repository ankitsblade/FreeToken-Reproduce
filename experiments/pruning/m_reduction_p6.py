"""P6: resident-biased routing screen in closed-loop fixed-logit simulation (plan section 9.2).

Closed loop with respect to cache and routing only: upstream hidden states and logits stay
baseline, so a fixed-logit win cannot qualify quality. Finalists need teacher forcing.
"""

import math

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_routing import (PROTECT_J, RangeTracker, family_a, family_b, mass_retained,
                                                      native_select)
from experiments.pruning.m_reduction_sim import CacheSim, capacity_regimes, initial_state
from experiments.pruning.m_reduction_stats import equal_workload, nested_select, paired_bootstrap, request_metrics
from experiments.pruning.m_reduction_traces import calibration

SETTINGS = ([("A", lam, None) for lam in (0.1, 0.2, 0.4)] + [("B", None, a) for a in (0.1, 0.3)]
            + [("C", lam, a) for lam in (0.1, 0.2, 0.4) for a in (0.1, 0.3)])
CONTROLS = [("A", 0.0, None), ("native", None, None)]
MASS_MEAN, MASS_P5 = 0.98, 0.95


def label(setting):
    family, lam, alpha = setting
    return family if family == "native" else f"{family}_lam{lam}_alpha{alpha}"


def routed_run(trace, geometry, table, arm, capacity, start_state, select, tracker=None, quotas=None, seed=None):
    """Closed-loop replay: select(z, resident, layer) -> (ids, swaps) chooses each decode event's experts."""
    make, floor = table[arm]
    replacement = "quota" if quotas is not None else "lru"
    state = initial_state(trace, geometry, capacity, start_state, seed if seed is not None else quotas)
    sim = CacheSim(state, geometry.layers, geometry.experts, make(), floor, replacement, quotas)
    scores = trace.scores(geometry.experts)
    n = len(trace.ids)
    m, q = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    mass, changed, swaps, replacement_hits, replacement_misses = np.ones(n), 0, 0, 0, 0
    usage = np.zeros((geometry.layers, geometry.experts), dtype=np.int64)
    for i, layer in enumerate(trace.layer.tolist()):
        z = scores[i].astype(np.float64)
        resident = sim.slots[layer] >= 0
        original = native_select(z, geometry.top_k)
        ids, swapped = select(z, resident, layer)
        if tracker is not None:
            tracker.update(layer, z)
        if set(ids) != set(original):
            changed += 1
            for e in set(ids) - set(original):
                replacement_hits += int(resident[e])
                replacement_misses += int(not resident[e])
        swaps += swapped
        mass[i] = mass_retained(z, original, ids)
        usage[layer, ids] += 1
        out = sim.event(layer, ids)
        m[i], q[i] = out["m"], out["q"]
    row = request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers)
    shares = usage / np.maximum(usage.sum(axis=1, keepdims=True), 1)
    entropy = -(shares * np.log(np.where(shares > 0, shares, 1))).sum(axis=1)
    row.update({"mass_mean": float(mass.mean()), "mass_p5": float(np.quantile(mass, 0.05, method="lower")),
                "mass_events_below_0_95": int((mass < MASS_P5).sum()),
                "changed_event_rate": changed / n, "substitutions_per_event": swaps / n,
                "replacement_hit_rate": replacement_hits / max(1, replacement_hits + replacement_misses),
                "route_entropy_mean": float(entropy.mean()), "experts_used": int((usage > 0).sum())})
    return row


def selector(setting, geometry, tracker):
    family, lam, alpha = setting
    k, j = geometry.top_k, PROTECT_J[geometry.model]

    def choose(z, resident, layer):
        if family == "native":
            return native_select(z, k), 0
        if family == "A":
            return family_a(z, k, j, resident, lam, tracker.mean(layer)), 0
        if family == "B":
            return family_b(z, k, j, resident, alpha)
        start = family_a(z, k, j, resident, lam, tracker.mean(layer))
        return family_b(z, k, j, resident, alpha, start=start)
    return choose


def evaluate(traces, calibration_traces, geometry, table, arm, capacity, start_state, setting):
    seed = RangeTracker.calibrate(calibration_traces, geometry)
    rows = []
    for trace in traces:
        tracker = RangeTracker(seed.sums, seed.counts)
        rows.append(routed_run(trace, geometry, table, arm, capacity, start_state,
                               selector(setting, geometry, tracker), tracker))
    return rows


def main():
    p = parser(__doc__)
    p.add_argument("--q-arm", default="overlap_floor1", help="q rule held fixed during the routing screen")
    args = p.parse_args()
    out, bindings, models, status = start(args, "p6_routing_screen", __file__,
                                          {"settings": [label(s) for s in SETTINGS], "controls": [label(s) for s in CONTROLS],
                                           "protect_j": PROTECT_J, "mass_mean_min": MASS_MEAN, "mass_p5_min": MASS_P5,
                                           "q_arm": args.q_arm, "range_tracker": "calibration mean, causal per-request update"})
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        table, _ = rules(geometry, calibration(bindings[model], model))
        capacity = capacity_regimes(geometry)["primary"]
        result = {"source": source, "cells": {}}
        for start_state in ("native", "cold"):
            rows = {label(s): [] for s in SETTINGS + CONTROLS}
            for fold in sorted({t.fold for t in traces}):
                train = [t for t in traces if t.fold != fold]
                held = [t for t in traces if t.fold == fold]
                for setting in SETTINGS + CONTROLS:
                    rows[label(setting)].extend(evaluate(held, train, geometry, table, args.q_arm, capacity,
                                                         start_state, setting))
            base = rows["native"]
            identity = all(a["mean_m"] == b["mean_m"] and a["changed_event_rate"] == 0
                           for a, b in zip(rows["A_lam0.0_alphaNone"], base))
            for name, values in rows.items():
                write_lines(out / f"{model}_{start_state}_{name}.jsonl", values)
            cell = {"zero_change_identity": identity, "settings": {}}
            for setting in SETTINGS:
                name = label(setting)
                cell["settings"][name] = {
                    "vs_native_mean_m": paired_bootstrap(rows[name], base, "mean_m"),
                    "mass_mean": equal_workload(rows[name], "mass_mean"),
                    "mass_p5_mean": equal_workload(rows[name], "mass_p5"),
                    "changed_event_rate": equal_workload(rows[name], "changed_event_rate"),
                    "route_entropy_mean": equal_workload(rows[name], "route_entropy_mean")}
            result["cells"][start_state] = cell
        cache = {}

        def score(setting, fit, held):
            key = (setting, tuple(sorted(t.capture_id for t in fit)), tuple(sorted(t.capture_id for t in held)))
            if key not in cache:
                arm_rows = evaluate(held, fit, geometry, table, args.q_arm, capacity, "native", setting)
                base_rows = evaluate(held, fit, geometry, table, args.q_arm, capacity, "native", ("native", None, None))
                # Event-level 5th percentile >= 0.95 holds exactly when at most 5% of events fall below 0.95.
                below = sum(r["mass_events_below_0_95"] for r in arm_rows) / sum(r["events"] for r in arm_rows)
                eligible = equal_workload(arm_rows, "mass_mean") >= MASS_MEAN and below <= 0.05
                ratio = equal_workload(arm_rows, "mean_m") / equal_workload(base_rows, "mean_m")
                cache[key] = ratio if eligible else math.inf
            return cache[key]

        finalists = {}
        for family in ("A", "B", "C"):
            candidates = [s for s in SETTINGS if s[0] == family]
            chosen = nested_select(traces, candidates, score, lambda s: ((s[1] or 0) + (s[2] or 0), label(s)))
            eligible = any(v["inner_scores"][str(chosen["deployment"])] is not None
                           for v in chosen["per_outer_fold"].values())
            finalists[family] = {"selection": {**chosen, "deployment": label(chosen["deployment"])},
                                 "finalist": label(chosen["deployment"]) if eligible else None,
                                 "status": "screen_finalist" if eligible else "no_eligible_setting"}
        result["finalists"] = finalists
        result["note"] = ("Finalists still need candidate teacher forcing, D1/D2/D3 quality and live timing; "
                          "an ineligible family is a recorded rejected screen, not an omitted model.")
        summary[model] = result
    finish(out, summary, status)


if __name__ == "__main__":
    main()
