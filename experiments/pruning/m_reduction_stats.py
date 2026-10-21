"""Request-level m metrics, equal-workload summaries, paired bootstrap and G2 decisions."""

import numpy as np

WORKLOADS = ("math", "code", "incident")
SEED = 20261014
REPEATS = 10000


def request_metrics(trace, m, q, expert_bytes, layers):
    """Plan section 4.1 metrics for one request; m/q are per decode layer-event arrays."""
    token = trace.token
    early, steady = token < 64, token >= 64
    p95 = lambda values: float(np.quantile(values, 0.95, method="higher")) if len(values) else None
    tokens = len(np.unique(trace.position))
    return {"request_id": trace.request_id, "capture_id": trace.capture_id, "workload": trace.workload,
            "fold": trace.fold, "events": int(len(m)), "decode_tokens": tokens,
            "mean_m": float(m.mean()), "p95_m": p95(m),
            "mean_m_early": float(m[early].mean()) if early.any() else None,
            "mean_m_steady": float(m[steady].mean()) if steady.any() else None,
            "p95_m_early": p95(m[early]), "p95_m_steady": p95(m[steady]),
            "missing_per_token": float(m.sum() / tokens), "fills_per_token": float(q.sum() / tokens),
            "cpu_misses_per_token": float((m - q).sum() / tokens),
            "missing_bytes_per_token": float(m.sum() * expert_bytes / tokens),
            "fill_bytes_per_token": float(q.sum() * expert_bytes / tokens),
            "unique_demand_miss_rate": float(m.sum() / sum(len(ids) for ids in trace.ids)),
            "per_layer_mean_m": [float(m[trace.layer == layer].mean()) if (trace.layer == layer).any() else None
                                 for layer in range(layers)]}


def equal_workload(rows, key):
    means = [np.mean([r[key] for r in rows if r["workload"] == w and r[key] is not None]) for w in WORKLOADS
             if any(r["workload"] == w and r[key] is not None for r in rows)]
    return float(np.mean(means)) if len(means) == len(WORKLOADS) else None


def reduction(arm, base):
    return None if base in (None, 0) or arm is None else 1.0 - arm / base


def paired_bootstrap(arm_rows, base_rows, key="mean_m", repeats=REPEATS, seed=SEED):
    """Resample request groups within each outer-fold x workload cell; returns relative and absolute deltas."""
    base = {r["request_id"]: r for r in base_rows}
    pairs = [(a, base[a["request_id"]]) for a in arm_rows if a["request_id"] in base]
    if len(pairs) != len(arm_rows):
        raise ValueError("Arm and baseline request sets differ")
    cells = {}
    for a, b in pairs:
        cells.setdefault((a["fold"], a["workload"]), []).append((a[key], b[key]))
    rng = np.random.default_rng(seed)
    point_arm = equal_workload([a for a, _ in pairs], key)
    point_base = equal_workload([b for _, b in pairs], key)
    rel, delta = [], []
    keys = sorted(cells)
    for _ in range(repeats):
        by_workload = {w: ([], []) for w in WORKLOADS}
        for cell in keys:
            values = cells[cell]
            pick = rng.integers(0, len(values), len(values))
            for i in pick:
                by_workload[cell[1]][0].append(values[i][0])
                by_workload[cell[1]][1].append(values[i][1])
        a = np.mean([np.mean(by_workload[w][0]) for w in WORKLOADS])
        b = np.mean([np.mean(by_workload[w][1]) for w in WORKLOADS])
        rel.append(1.0 - a / b if b else np.nan)
        delta.append(a - b)
    rel, delta = np.asarray(rel), np.asarray(delta)
    folds = {}
    for fold in sorted({a["fold"] for a, _ in pairs}):
        sub = [(a, b) for a, b in pairs if a["fold"] == fold]
        folds[str(fold)] = reduction(equal_workload([a for a, _ in sub], key), equal_workload([b for _, b in sub], key))
    domains = {}
    for w in WORKLOADS:
        sub = [(a, b) for a, b in pairs if a["workload"] == w]
        values = np.asarray([a[key] for a, _ in sub], dtype=float), np.asarray([b[key] for _, b in sub], dtype=float)
        boot = []
        wrng = np.random.default_rng(seed)
        for _ in range(repeats):
            pick = wrng.integers(0, len(sub), len(sub))
            boot.append(1.0 - values[0][pick].mean() / values[1][pick].mean())
        domains[w] = {"reduction": reduction(values[0].mean(), values[1].mean()),
                      "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}
    return {"metric": key, "arm": point_arm, "baseline": point_base, "reduction": reduction(point_arm, point_base),
            "reduction_ci95": [float(np.nanpercentile(rel, 2.5)), float(np.nanpercentile(rel, 97.5))],
            "absolute_delta": None if point_arm is None else point_arm - point_base,
            "absolute_delta_ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "fold_reductions": folds, "domains": domains, "requests": len(pairs), "repeats": repeats, "seed": seed,
            "weighting": "requests equal within workload; workloads one-third each"}


def g2_exact(mean_result, p95_result, outputs_changed=False, timing_upper=None):
    """Exact cache component gate (plan section 10.2), timing bound evaluated when available."""
    folds = [v for v in mean_result["fold_reductions"].values() if v is not None]
    mean_ok = mean_result["reduction"] is not None and mean_result["reduction"] >= 0.05 and mean_result["reduction_ci95"][0] > 0
    p95_ok = p95_result is not None and p95_result["reduction"] is not None and p95_result["reduction"] >= 0.10 \
        and p95_result["reduction_ci95"][0] > 0
    folds_ok = sum(v > 0 for v in folds) >= 4
    domains_ok = not any(d["ci95"][1] < 0 for d in mean_result["domains"].values())
    timing_ok = None if timing_upper is None else timing_upper <= 1.02
    decision = "keep" if (mean_ok or p95_ok) and folds_ok and domains_ok and not outputs_changed and timing_ok else \
        ("defer_timing" if (mean_ok or p95_ok) and folds_ok and domains_ok and not outputs_changed and timing_ok is None
         else "reject")
    return {"mean_m_criterion": mean_ok, "p95_criterion": p95_ok, "four_of_five_folds": folds_ok,
            "no_confidently_negative_domain": domains_ok, "outputs_changed": outputs_changed,
            "timing_upper_le_1_02": timing_ok, "decision": decision}


def g2_output_changing(mean_result, quality_passed=None, timing_upper=None):
    folds = [v for v in mean_result["fold_reductions"].values() if v is not None]
    m_ok = mean_result["reduction"] is not None and mean_result["reduction"] >= 0.10 and mean_result["reduction_ci95"][0] > 0
    folds_ok = sum(v > 0 for v in folds) >= 4
    domains_ok = not any(d["ci95"][1] < 0 for d in mean_result["domains"].values())
    if not (m_ok and folds_ok and domains_ok) or quality_passed is False:
        decision = "reject"
    elif quality_passed is None or timing_upper is None:
        decision = "provisional_pending_quality_and_timing"
    else:
        decision = "keep" if quality_passed and timing_upper <= 1.02 else "reject"
    return {"mean_m_ge_10pct": m_ok, "four_of_five_folds": folds_ok, "no_confidently_negative_domain": domains_ok,
            "quality_passed": quality_passed, "timing_upper_le_1_02": None if timing_upper is None else timing_upper <= 1.02,
            "decision": decision}


def nested_select(traces, candidates, score, tie_key):
    """Plan section 3.1 nested selection.

    score(candidate, fit_traces, eval_traces) -> lower is better. For each outer fold, each
    of its four remaining folds is inner validation (fit on the other three); the candidate
    with best mean inner score is the fold's choice. Deployment uses the most frequent
    choice, ties broken by tie_key (smaller perturbation, then arm ID).
    """
    folds = sorted({t.fold for t in traces})
    per_outer = {}
    for outer in folds:
        train = [f for f in folds if f != outer]
        inner = {}
        for c in candidates:
            values = []
            for validation in train:
                fit = [t for t in traces if t.fold in train and t.fold != validation]
                held = [t for t in traces if t.fold == validation]
                values.append(score(c, fit, held))
            inner[c] = float(np.mean(values))
        best = min(candidates, key=lambda c: (inner[c], tie_key(c)))
        # Non-finite scores mark ineligible candidates; JSON stores them as null.
        per_outer[str(outer)] = {"selected": best, "inner_scores": {str(k): (v if np.isfinite(v) else None)
                                                                    for k, v in inner.items()}}
    counts = {}
    for row in per_outer.values():
        counts[row["selected"]] = counts.get(row["selected"], 0) + 1
    deployment = min(counts, key=lambda c: (-counts[c], tie_key(c)))
    return {"per_outer_fold": per_outer, "deployment": deployment, "selection_counts": {str(k): v for k, v in counts.items()}}
