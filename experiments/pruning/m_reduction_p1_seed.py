"""P1 prefill-informed seeding add-on, offline access replay (plan section 5).

Native decode starts from the captured post-prefill state, which is empty on this
runtime. Each seed arm instead fills the cache at the prefill->decode boundary with the
C most frequent prompt (layer, expert) routes, counted over the whole prompt or over its
last 64 tokens; ties by layer ID then expert ID. Every prompt-touched expert's recency
(and each seeded slot's LRU timestamp) is ordered by its last prompt touch, older than
any decode event. The same native floor-1 q rule and global LRU then replay the decode
accesses. Missing seed entries are real copies: their bytes are reported; seeding wall
time and TTFT need a live arm. Early (tokens 0-63) and steady-state m are separate.
"""

import json
from collections import Counter

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_sim import CacheSim, QRule, empty_state
from experiments.pruning.m_reduction_stats import equal_workload, paired_bootstrap, request_metrics
from experiments.pruning.m_reduction_traces import lines

ARMS = ("native", "seed_whole", "seed_last64")


def prompt_routes(trace):
    return [(row["input_position"], row["layer_id"], row["expert_ids"])
            for row in lines(trace.directory / "routes.jsonl.gz")
            if row["phase"] == "prefill" and not row.get("is_shared_expert")]


def seeded_state(trace, geometry, routes, window):
    counts, last = Counter(), {}
    for position, layer, ids in routes:
        flat = [layer * geometry.experts + e for e in ids]
        for f in flat:
            last[f] = max(last.get(f, -1), position * geometry.layers + layer)
        if window is None or position >= trace.prompt_tokens - window:
            counts.update(flat)
    chosen = sorted(counts, key=lambda f: (-counts[f], f // geometry.experts, f % geometry.experts))[:geometry.cache_slots]
    order = sorted(last, key=lambda f: (last[f], f))  # oldest prompt touch first
    rank = {f: i for i, f in enumerate(order)}
    state = empty_state(geometry.layers, geometry.experts, geometry.cache_slots)
    state["step"] = len(order)  # decode steps start after every prompt touch
    for f in order:
        state["expert_recency"][f // geometry.experts][f % geometry.experts] = rank[f]
    for slot, f in enumerate(chosen):
        state["id_of_slot"][slot] = f
        state["slot_for_id"][f // geometry.experts][f % geometry.experts] = slot
        state["usage"][slot] = rank[f]
    return state, len(chosen)


def replay(trace, geometry, state):
    sim = CacheSim(state, geometry.layers, geometry.experts,
                   QRule("native", fraction=float(trace.initial["fetch_fraction"])), 1)
    m = np.zeros(len(trace.ids), dtype=np.int32)
    q = np.zeros(len(trace.ids), dtype=np.int32)
    for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
        out = sim.event(int(layer), ids)
        m[i], q[i] = out["m"], out["q"]
    return m, q


def main():
    args = parser(__doc__).parse_args()
    out, bindings, models, status = start(args, "p1_seeding", __file__,
                                          {"arms": ARMS, "window_tokens": 64, "q_rule": "native fraction, floor 1",
                                           "selection": "top-C prompt frequency, ties layer then expert ID",
                                           "recency": "ordered by last prompt touch, older than every decode event"})
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        rows = {arm: [] for arm in ARMS}
        native_matches = True
        for trace in traces:
            routes = prompt_routes(trace)
            m, q = replay(trace, geometry, trace.initial)
            native_matches &= bool(np.array_equal(m, trace.live_m) and np.array_equal(q, trace.live_q))
            rows["native"].append({**request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers),
                                   "seed_entries": 0, "seed_bytes": 0})
            for arm, window in (("seed_whole", None), ("seed_last64", 64)):
                state, entries = seeded_state(trace, geometry, routes, window)
                m, q = replay(trace, geometry, state)
                rows[arm].append({**request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers),
                                  "seed_entries": entries, "seed_bytes": entries * geometry.expert_bytes})
        for arm, values in rows.items():
            write_lines(out / f"{model}_{arm}.jsonl", values)
        base = rows["native"]
        summary[model] = {
            "source": source, "native_replay_reproduces_live": native_matches,
            "arms": {arm: {"mean_m": equal_workload(values, "mean_m"),
                           "mean_m_early": equal_workload(values, "mean_m_early"),
                           "mean_m_steady": equal_workload(values, "mean_m_steady"),
                           "seed_bytes_mean": float(np.mean([r["seed_bytes"] for r in values])),
                           **({} if arm == "native" else {
                               "vs_native_mean_m": paired_bootstrap(values, base, "mean_m"),
                               "vs_native_mean_m_early": paired_bootstrap(values, base, "mean_m_early"),
                               "vs_native_mean_m_steady": paired_bootstrap(values, base, "mean_m_steady"),
                               "vs_native_p95_m": paired_bootstrap(values, base, "p95_m")})}
                     for arm, values in rows.items()},
            "note": ("Seeding copies are not free: seed bytes are real transfers at the prefill->decode boundary. "
                     "Adoption needs a timed, equal-budget live comparison meeting the exact-component gate.")}
    finish(out, summary, status)


if __name__ == "__main__":
    main()
