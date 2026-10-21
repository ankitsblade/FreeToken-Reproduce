"""P4: q and fetch-selection policy arms in faithful access replay (plan section 8.1)."""

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_sim import QRule, capacity_regimes, run
from experiments.pruning.m_reduction_stats import equal_workload, g2_exact, nested_select, paired_bootstrap, request_metrics
from experiments.pruning.m_reduction_traces import calibration

ARMS = ("paper_floor1", "overlap_floor1", "fixed25_floor1", "reuse_uniform_floor1", "reuse_freq_floor1")
CONTROLS = ("reuse_freq_floor0",)
BASELINE = "overlap_floor1"


def rules(geometry, profile):
    rho = profile["pcie_gather_overlap_gbs"] / profile["cpu_moe_overlap_gbs"]
    paper = min(1.0, profile["pcie_gather_gbs"] / profile["cpu_moe_gbs"])
    overlap = profile["pcie_gather_overlap_gbs"] / (profile["pcie_gather_overlap_gbs"] + profile["cpu_moe_overlap_gbs"])
    reuse = dict(rho=rho, top_k=geometry.top_k, experts=geometry.experts)
    table = {"paper_floor1": (lambda: QRule("native", fraction=paper), 1),
             "overlap_floor1": (lambda: QRule("native", fraction=overlap), 1),
             "fixed25_floor1": (lambda: QRule("fixed", fraction=0.25), 1),
             "reuse_uniform_floor1": (lambda: QRule("reuse", per_expert=False, **reuse), 1),
             "reuse_freq_floor1": (lambda: QRule("reuse", per_expert=True, **reuse), 1),
             "reuse_freq_floor0": (lambda: QRule("reuse", per_expert=True, **reuse), 0)}
    return table, {"rho": rho, "paper_fraction": paper, "overlap_fraction": overlap}


def evaluate(traces, geometry, table, arm, capacity, start_state):
    make, floor = table[arm]
    rows = []
    for trace in traces:
        out = run(trace, geometry, make(), floor, capacity, start_state)
        row = request_metrics(trace, out["m"], out["q"], geometry.expert_bytes, geometry.layers)
        row["evictions_per_token"] = out["evictions"] / row["decode_tokens"]
        rows.append(row)
    return rows


def main():
    p = parser(__doc__)
    args = p.parse_args()
    out, bindings, models, status = start(args, "p4_admission", __file__,
                                          {"arms": ARMS, "controls": CONTROLS, "baseline": BASELINE,
                                           "reuse": {"kappa": 1, "t_free": 32, "prior_n": 8, "window": 32,
                                                     "use_cache": True, "rho_online": False},
                                           "fetch_order": {"native": "pre-action recency desc, expert ID",
                                                           "reuse": "expected reuse desc, recency desc, expert ID"}})
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        profile = calibration(bindings[model], model)
        table, derived = rules(geometry, profile)
        live = float(traces[0].initial["fetch_fraction"])
        result = {"source": source, "calibration": profile, "derived": derived, "live_native_fraction": live,
                  "overlap_matches_live_fraction": abs(derived["overlap_fraction"] - live) < 1e-6, "cells": {}}
        for regime, capacity in capacity_regimes(geometry).items():
            for start_state in ("native", "cold"):
                cell = f"{regime}/{start_state}"
                rows = {arm: evaluate(traces, geometry, table, arm, capacity, start_state) for arm in ARMS + CONTROLS}
                for arm, arm_rows in rows.items():
                    write_lines(out / f"{model}_{regime}_{start_state}_{arm}.jsonl", arm_rows)
                cell_result = {"capacity": capacity, "arms": {}}
                for arm in ARMS + CONTROLS:
                    mean = paired_bootstrap(rows[arm], rows[BASELINE], "mean_m")
                    p95 = paired_bootstrap(rows[arm], rows[BASELINE], "p95_m")
                    cell_result["arms"][arm] = {
                        "mean_m": mean["arm"], "fills_per_token": equal_workload(rows[arm], "fills_per_token"),
                        "cpu_misses_per_token": equal_workload(rows[arm], "cpu_misses_per_token"),
                        "vs_baseline_mean_m": mean, "vs_baseline_p95_m": p95,
                        "g2_exact": None if arm == BASELINE else g2_exact(mean, p95)}
                cell_result["frequency_vs_uniform"] = paired_bootstrap(rows["reuse_freq_floor1"], rows["reuse_uniform_floor1"])
                result["cells"][cell] = cell_result
        primary = capacity_regimes(geometry)["primary"]
        cache = {}

        def score(arm, fit, held):
            key = (arm, tuple(sorted(t.capture_id for t in held)))
            if key not in cache:
                cache[key] = equal_workload(evaluate(held, geometry, table, arm, primary, "native"), "mean_m")
            return cache[key]

        fills = {arm: result["cells"]["primary/native"]["arms"][arm]["fills_per_token"] for arm in ARMS}
        result["selection"] = nested_select(traces, ARMS, score, lambda arm: (arm != BASELINE, fills[arm], arm))
        result["selection"]["note"] = ("Inner-fold m selection on primary capacity, native state; adoption still "
                                       "requires the live latency gate. No arm parameters are fitted.")
        summary[model] = result
    finish(out, summary, status)


if __name__ == "__main__":
    main()
