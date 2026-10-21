"""P3: diagnose decode misses before choosing levers (plan section 7) and apply G1."""

import json
import math
from collections import Counter, defaultdict

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_sim import (CacheSim, QRule, belady, capacity_regimes, initial_state,
                                                  layer_curves, run)
from experiments.pruning.m_reduction_stats import equal_workload, request_metrics

ALPHAS = (0.1, 0.3)
PROTECT_J = {"gpt_oss_mxfp4": 1, "qwen_nvfp4": 2, "gemma_nvfp4": 2}
REMOVE = {"gpt_oss_mxfp4": 8, "qwen_nvfp4": 64, "gemma_nvfp4": 32}


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def diagnose(trace, geometry, capacity, start_state, with_router):
    E, L = geometry.experts, geometry.layers
    rule = QRule("native", fraction=float(trace.initial["fetch_fraction"]))
    state = initial_state(trace, geometry, capacity, start_state)
    sim = CacheSim(state, L, E, rule, floor=1)
    initially = {f for f in state["id_of_slot"] if f >= 0}
    full = run(trace, geometry, QRule("full"), 1, capacity, start_state, keep_events=True)["events"]
    _, min_sets = belady(trace, geometry, capacity, start_state)
    scores = trace.scores(E) if with_router else None
    touched, evicted_flag, last_access = set(), set(), {}
    labels = Counter()
    disagreement = 0
    per_expert = defaultdict(Counter)
    gaps_tokens, router = [], defaultdict(list)
    thrash_low, thrash_high = capacity, math.ceil(1.10 * capacity)
    stamp = np.full(L * E, -1, dtype=np.int64)
    m = np.zeros(len(trace.ids), dtype=np.int32)
    q = np.zeros(len(trace.ids), dtype=np.int32)
    for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
        token = int(trace.token[i])
        resident = sim.slots[layer] >= 0
        if with_router:
            z = scores[i].astype(np.float64)
            s = softmax(z)
            order = np.argsort(-z, kind="stable")
            k = geometry.top_k
            s_next = s[order[k]]
            protected = set(order[:PROTECT_J[geometry.model]].tolist())
            selected = np.zeros(E, dtype=bool)
            selected[list(ids)] = True
            router["logit_gap_k_k1"].append(float(z[order[k - 1]] - z[order[k]]))
            router["prob_gap_k_k1"].append(float(s[order[k - 1]] - s_next))
            router["logit_range"].append(float(z.max() - z.min()))
            for alpha in ALPHAS:
                cached = int((resident & ~selected & (s >= (1 - alpha) * s_next)).sum())
                eligible = sum(1 for e in ids if not resident[e] and e not in protected and s[e] < (1 + alpha) * s_next)
                router[f"cached_candidates_alpha_{alpha}"].append(cached)
                router[f"eligible_missing_alpha_{alpha}"].append(eligible)
        flat = [layer * E + e for e in ids]
        distances = [int((stamp > stamp[f]).sum()) if stamp[f] >= 0 else None for f in flat]
        out = sim.event(layer, ids)
        m[i], q[i] = out["m"], out["q"]
        missing = set(out["missing"])
        lru_missing, min_missing = set(full[i]["missing"]), min_sets[i]
        for e, f, d in zip(ids, flat, distances):
            stats = per_expert[f]
            stats["accesses"] += 1
            if f in last_access:
                gaps_tokens.append(token - last_access[f])
            last_access[f] = token
            if e in missing:
                stats["misses"] += 1
                if f in evicted_flag:
                    stats["refetches"] += 1
                    if d is not None and thrash_low < d <= thrash_high:
                        stats["thrash_refetches"] += 1
                if f not in touched and f not in initially:
                    label = "compulsory"
                elif e not in lru_missing:
                    label = "admission_state_sensitive"
                elif e in min_missing:
                    label = "capacity_like"
                else:
                    label = "replacement_sensitive"
                labels[label] += 1
                stats[label] += 1
            if e in min_missing and e not in lru_missing:
                disagreement += 1
            touched.add(f)
            evicted_flag.discard(f)
        for e in out["admitted"]:
            per_expert[layer * E + e]["fills"] += 1
        for f in out["evicted"]:
            if f >= 0:
                evicted_flag.add(f)
                per_expert[f]["evictions"] += 1
        stamp[flat] = i
    full_m = np.asarray([len(ev["missing"]) for ev in full], dtype=np.int32)
    min_m = np.asarray([len(s) for s in min_sets], dtype=np.int32)
    return m, q, full_m, min_m, labels, disagreement, per_expert, gaps_tokens, router


def deletion_shares(rows, traces, geometry):
    """Plan section 7 deletion estimates: rankings from outer-training folds, shares on held-out misses."""
    remove, E, L = REMOVE[geometry.model], geometry.experts, geometry.layers
    def matrix(field):
        values = np.zeros((len(rows), L * E), dtype=np.int64)
        for i, r in enumerate(rows):
            for f, stats in r["per_expert"].items():
                values[i, int(f)] = stats.get(field, 0)
        return values
    accesses, misses = matrix("accesses"), matrix("misses")
    folds = np.asarray([r["fold"] for r in rows])
    out = {}
    for fold in sorted(set(folds.tolist())):
        train, held = folds != fold, folds == fold
        result = {}
        for name, source in (("bottom_frequency", accesses), ("largest_miss_count", misses)):
            totals = source[train].sum(axis=0).reshape(L, E)
            removed = np.zeros((L, E), dtype=bool)
            for layer in range(L):
                ranked = sorted(range(E), key=lambda e: (totals[layer, e], e) if name == "bottom_frequency"
                                else (-totals[layer, e], e))
                removed[layer, ranked[:remove]] = True
            held_misses = misses[held].sum(axis=0)
            total = held_misses.sum()
            result[name] = float(held_misses[removed.reshape(-1)].sum() / total) if total else None
        out[str(fold)] = result
    return {name: float(np.mean([v[name] for v in out.values() if v[name] is not None]))
            for name in ("bottom_frequency", "largest_miss_count")} | {"per_fold": out}


def g1(labels, per_expert_total, deletion, population):
    total = sum(labels.values())
    share = {k: (labels[k] / total if total else None) for k in
             ("compulsory", "admission_state_sensitive", "capacity_like", "replacement_sensitive")}
    thrash = sorted(per_expert_total.items(), key=lambda kv: (-kv[1].get("thrash_refetches", 0), kv[0]))
    top = thrash[:math.ceil(0.10 * population)]
    replacement = sum(s.get("replacement_sensitive", 0) for _, s in per_expert_total.items())
    top_share = (sum(s.get("replacement_sensitive", 0) for _, s in top) / replacement) if replacement else 0.0
    p9 = (share["replacement_sensitive"] or 0) >= 0.15 and top_share >= 0.50
    return {"miss_class_shares": share, "top10pct_thrash_share_of_replacement_misses": top_share,
            "prioritize_p9": p9, "prioritize_p6": (share["capacity_like"] or 0) >= 0.50,
            "run_p4_p5_first": (share["capacity_like"] or 0) < 0.50,
            "lower_p7_exclusion_priority": deletion["largest_miss_count"] < 0.05,
            "rule": "Scheduling priority only; every experiment still runs on all three models."}


def main():
    p = parser(__doc__)
    p.add_argument("--subset", default="primary", choices=("primary", "repeat"))
    args = p.parse_args()
    out, bindings, models, status = start(args, "p3_diagnostics", __file__,
                                          {"alphas": ALPHAS, "protect_j": PROTECT_J, "remove_per_layer": REMOVE,
                                           "subset": args.subset, "q_rule": "native fraction, floor 1"})
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model, args.subset)
        regimes = capacity_regimes(geometry)
        model_summary = {"source": source, "geometry": vars(geometry), "regimes": regimes, "cells": {}}
        curves = {"quota_range": [geometry.top_k, geometry.experts], "per_request": {}}
        for regime, capacity in regimes.items():
            for start_state in ("native", "cold"):
                cell = f"{regime}/{start_state}"
                state_label = "native" if start_state == "native" and capacity == geometry.cache_slots else \
                    ("projected_native" if start_state == "native" else "cold_control")
                rows, label_total, per_expert_total, gaps, router_all, disagree = [], Counter(), defaultdict(Counter), [], defaultdict(list), 0
                for trace in traces:
                    m, q, full_m, min_m, labels, dis, per_expert, gap, router = diagnose(
                        trace, geometry, capacity, start_state, with_router=(cell == "primary/native"))
                    row = request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers)
                    row.update({"full_lru_mean_m": float(full_m.mean()), "min_mean_m": float(min_m.mean()),
                                "miss_classes": dict(labels), "min_miss_lru_hit": dis,
                                "per_expert": {str(f): dict(s) for f, s in per_expert.items()}})
                    rows.append(row)
                    label_total.update(labels)
                    disagree += dis
                    for f, s in per_expert.items():
                        per_expert_total[f].update(s)
                    gaps.extend(gap)
                    for k, v in router.items():
                        router_all[k].extend(v)
                    if cell == "primary/native":
                        curves["per_request"][trace.capture_id] = layer_curves(trace, geometry, curves["quota_range"]).tolist()
                write_lines(out / f"{model}_{regime}_{start_state}_requests.jsonl",
                            [{k: v for k, v in r.items() if k != "per_expert"} for r in rows])
                write_lines(out / f"{model}_{regime}_{start_state}_experts.jsonl",
                            [{"flat_id": f, "layer": f // geometry.experts, "expert": f % geometry.experts, **dict(s)}
                             for f, s in sorted(per_expert_total.items())])
                deletion = deletion_shares(rows, traces, geometry)
                result = {"capacity": capacity, "state": state_label,
                          "mean_m": equal_workload(rows, "mean_m"), "p95_m": equal_workload(rows, "p95_m"),
                          "full_lru_mean_m": equal_workload(rows, "full_lru_mean_m"),
                          "min_mean_m": equal_workload(rows, "min_mean_m"),
                          "miss_classes": dict(label_total), "min_miss_lru_hit": disagree,
                          "lru_min_gap_note": "Diagnostic bound for the fully admitting problem, not achievable gain at fixed q",
                          "inter_touch_token_gap_quantiles": np.quantile(gaps, [0.5, 0.9, 0.99]).tolist() if gaps else None,
                          "deletion_estimates": deletion}
                if cell == "primary/native":
                    result["router"] = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v))} for k, v in router_all.items()}
                    result["g1"] = g1(label_total, per_expert_total, deletion, geometry.layers * geometry.experts)
                model_summary["cells"][cell] = result
        (out / f"{model}_layer_curves.json").write_text(json.dumps(curves))
        summary[model] = model_summary
    finish(out, summary, status)


if __name__ == "__main__":
    main()
