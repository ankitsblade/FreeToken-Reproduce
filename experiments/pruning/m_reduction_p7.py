"""P7: 75% decode exclusion and router aliasing in closed-loop fixed-logit simulation (plan section 9.3).

Arm 1 needs per-event pre-gate expert output L2 norms and arm 2 needs fold-specific
common-input expert output probes, both from m_reduction_experts over baseline teacher
forcing. Without them each arm is recorded as blocked; the norm-free controls (R75
frequency mask, random masks, F75 dropped demand, identity alias) still run.
Merged-weight requantization and live execution are runtime work outside this screen.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.pruning.m_reduction_offline import finish, load, parser, start, write_lines
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_p6 import routed_run
from experiments.pruning.m_reduction_routing import masked_select, native_select, physical
from experiments.pruning.m_reduction_sim import CacheSim, capacity_regimes, initial_state, run
from experiments.pruning.m_reduction_stats import equal_workload, paired_bootstrap, request_metrics
from experiments.pruning.m_reduction_traces import calibration, lines

REMOVE = {"gpt_oss_mxfp4": 8, "qwen_nvfp4": 64, "gemma_nvfp4": 32}
RANDOM_SEED = 20261013


def training_counts(train, geometry, table, arm):
    """Decode access and miss counts per (layer, expert) under the fixed q rule."""
    make, floor = table[arm]
    access, misses = Counter(), Counter()
    for trace in train:
        out = run(trace, geometry, make(), floor, geometry.cache_slots, keep_events=True)
        for layer, ids, event in zip(trace.layer.tolist(), trace.ids, out["events"]):
            access.update((layer, e) for e in ids)
            misses.update((layer, e) for e in event["missing"])
    return access, misses


def event_norms(trace, directory):
    """Per decode event pre-gate norms, in the capture's routes order (teacher forcing reproduced it)."""
    values = np.load(directory / "norms" / (trace.capture_id + ".npy"))
    keys = [(row["input_position"], row["layer_id"]) for row in lines(trace.directory / "routes.jsonl.gz")
            if row["phase"] == "decode"]
    if len(keys) != len(values):
        raise ValueError(trace.capture_id + ": expert norms do not align with decode routes")
    return dict(zip(keys, values))


def saliency_removal(train, geometry, counts, norms_dir):
    """Arm 1 ranking: miss_count x (1 - normalized mean(effective weight x output norm))."""
    access, misses = counts
    total, seen = np.zeros((geometry.layers, geometry.experts)), np.zeros((geometry.layers, geometry.experts))
    for trace in train:
        norms = event_norms(trace, norms_dir)
        for (position, layer), (ids, weights) in trace.routes().items():
            for e, w, n in zip(ids, weights, norms[(position, layer)]):
                total[layer, e] += w * n
                seen[layer, e] += 1
    removed = {}
    for layer in range(geometry.layers):
        mean = np.where(seen[layer] > 0, total[layer] / np.maximum(seen[layer], 1), 0.0)
        lo, hi = mean[seen[layer] > 0].min(initial=0.0), mean.max(initial=0.0)
        normalized = np.where((seen[layer] > 0) & (hi > lo), (mean - lo) / (hi - lo if hi > lo else 1), 0.0)
        score = {e: misses[(layer, e)] * (1 - normalized[e]) for e in range(geometry.experts)}
        ranked = sorted(range(geometry.experts), key=lambda e: (-score[e], -misses[(layer, e)], e))
        removed[layer] = ranked[:REMOVE[geometry.model]]
    return removed


def frequency_removal(counts, geometry):
    access, _ = counts
    return {layer: sorted(range(geometry.experts), key=lambda e: (access[(layer, e)], e))[:REMOVE[geometry.model]]
            for layer in range(geometry.layers)}


def random_removal(geometry, index):
    rng = np.random.default_rng(RANDOM_SEED + index)
    return {layer: sorted(rng.choice(geometry.experts, REMOVE[geometry.model], replace=False).tolist())
            for layer in range(geometry.layers)}


def masked_rows(traces, geometry, table, arm, removed):
    def choose(z, resident, layer):
        return masked_select(z, geometry.top_k, removed[layer]), 0
    return [routed_run(t, geometry, table, arm, geometry.cache_slots, "native", choose) for t in traces]


def dropped_rows(traces, geometry, table, arm, removed):
    """F75 diagnostic: drop removed experts after native top-k without rerouting (dropped demand)."""
    make, floor = table[arm]
    rows = []
    for trace in traces:
        sim = CacheSim(initial_state(trace, geometry, geometry.cache_slots, "native"), geometry.layers,
                       geometry.experts, make(), floor)
        m, q = np.zeros(len(trace.ids), dtype=np.int32), np.zeros(len(trace.ids), dtype=np.int32)
        dropped = 0
        for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
            kept = [e for e in ids if e not in set(removed[layer])]
            dropped += len(ids) - len(kept)
            out = sim.event(layer, kept) if kept else {"m": 0, "q": 0}
            m[i], q[i] = out["m"], out["q"]
        row = request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers)
        row["dropped_demand_per_event"] = dropped / len(trace.ids)
        rows.append(row)
    return rows


def cluster(vectors, keep):
    """Average-linkage cosine clustering to `keep` clusters (Lance-Williams updates).

    Zero-vector experts merge only when nothing else can; ties prefer smaller minimum original IDs.
    """
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.where(norms[:, None] > 0, norms[:, None], 1)
    distance = 1.0 - unit @ unit.T
    zero = norms == 0
    members = {e: [e] for e in range(len(vectors))}
    while len(members) > keep:
        ids = sorted(members)
        best = None
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                key = (bool(zero[a] or zero[b]), float(distance[a, b]), min(members[a]), min(members[b]))
                if best is None or key < best[0]:
                    best = (key, a, b)
        _, a, b = best
        na, nb = len(members[a]), len(members[b])
        for c in members:
            if c not in (a, b):
                distance[a, c] = distance[c, a] = (na * distance[a, c] + nb * distance[b, c]) / (na + nb)
        zero[a] = zero[a] and zero[b]
        members[a] = sorted(members[a] + members.pop(b))
    return sorted(members.values())


def alias_rows(traces, geometry, table, arm, alias):
    make, floor = table[arm]
    rows = []
    for trace in traces:
        sim = CacheSim(initial_state(trace, geometry, geometry.cache_slots, "native"), geometry.layers,
                       geometry.experts, make(), floor)
        m, q = np.zeros(len(trace.ids), dtype=np.int32), np.zeros(len(trace.ids), dtype=np.int32)
        logical = physical_k = 0
        for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
            mapped = physical(ids, alias[layer])
            logical += len(ids)
            physical_k += len(mapped)
            out = sim.event(layer, mapped)
            m[i], q[i] = out["m"], out["q"]
        row = request_metrics(trace, m, q, geometry.expert_bytes, geometry.layers)
        row["physical_k_per_logical_k"] = physical_k / logical
        rows.append(row)
    return rows


def main():
    p = parser(__doc__)
    p.add_argument("--q-arm", default="overlap_floor1")
    p.add_argument("--expert-outputs", action="append", default=[],
                   help="MODEL=DIR from m_reduction_experts (norms/ and probes.npz)")
    p.add_argument("--random-masks", type=int, default=200)
    args = p.parse_args()
    out, bindings, models, status = start(args, "p7_exclusion_alias", __file__,
                                          {"remove_per_layer": REMOVE, "random_masks": args.random_masks,
                                           "random_seed_base": RANDOM_SEED, "q_arm": args.q_arm,
                                           "masking": "decode only, minus infinity before native top-k; prefill native",
                                           "expert_outputs": args.expert_outputs, "probe_pool": "mixed"})
    outputs = {model: Path(path).resolve() for model, path in (entry.split("=", 1) for entry in args.expert_outputs)}
    summary = {}
    for model in models:
        geometry, traces, source = load(bindings, model)
        table, _ = rules(geometry, calibration(bindings[model], model))
        arms = {"all100": [], "r75_frequency": [], "f75_dropped": [], "identity_alias": []}
        probes = None
        if model in outputs:
            if json.loads((outputs[model] / "completion.json").read_text()).get("status") != "complete":
                raise ValueError(model + ": expert outputs are incomplete")
            arms["arm1_miss_targeted"] = []
            arms["arm2_alias"] = []
            probes = np.load(outputs[model] / "probes.npz")
        random_means = []
        artefacts = {}
        for fold in sorted({t.fold for t in traces}):
            train = [t for t in traces if t.fold != fold]
            held = [t for t in traces if t.fold == fold]
            counts = training_counts(train, geometry, table, args.q_arm)
            r75 = frequency_removal(counts, geometry)
            identity = {l: list(range(geometry.experts)) for l in range(geometry.layers)}
            arms["all100"].extend(masked_rows(held, geometry, table, args.q_arm, {l: [] for l in range(geometry.layers)}))
            arms["r75_frequency"].extend(masked_rows(held, geometry, table, args.q_arm, r75))
            arms["f75_dropped"].extend(dropped_rows(held, geometry, table, args.q_arm, r75))
            arms["identity_alias"].extend(alias_rows(held, geometry, table, args.q_arm, identity))
            artefacts[str(fold)] = {"r75_removed": r75}
            if "arm1_miss_targeted" in arms:
                removed = saliency_removal(train, geometry, counts, outputs[model])
                arms["arm1_miss_targeted"].extend(masked_rows(held, geometry, table, args.q_arm, removed))
                artefacts[str(fold)]["arm1_removed"] = removed
            if "arm2_alias" in arms:
                # Probes come only from this outer fold's training captures (mixed pool).
                alias = {layer: {e: min(g) for g in cluster(probes[f"{model}_fold{fold}_mixed_{layer}"],
                                                            geometry.experts - REMOVE[model]) for e in g}
                         for layer in range(geometry.layers)}
                artefacts[str(fold)]["arm2_alias"] = {str(l): {str(e): t for e, t in m.items()} for l, m in alias.items()}
                arms["arm2_alias"].extend(alias_rows(held, geometry, table, args.q_arm, alias))
        for index in range(args.random_masks):
            removed = random_removal(geometry, index)
            random_means.append(equal_workload(masked_rows(traces, geometry, table, args.q_arm, removed), "mean_m"))
        for name, values in arms.items():
            write_lines(out / f"{model}_{name}.jsonl", values)
        (out / f"{model}_artefacts.json").write_text(json.dumps(artefacts))
        base = arms["all100"]
        summary[model] = {
            "source": source,
            "comparisons": {name: paired_bootstrap(values, base, "mean_m") for name, values in arms.items() if name != "all100"},
            "identity_alias_reproduces_native": all(a["mean_m"] == b["mean_m"] for a, b in zip(arms["identity_alias"], base)),
            "random_mask_mean_m": {"values": random_means,
                                   "quantiles": np.quantile(random_means, [0.05, 0.5, 0.95]).tolist() if random_means else None},
            "arm1_status": "screened" if "arm1_miss_targeted" in arms else "blocked: P2 expert output norms unavailable",
            "arm2_status": ("routing screen only; merged-weight execution still blocked" if "arm2_alias" in arms
                            else "blocked: P2 common-input expert probes unavailable"),
            "note": ("F75 is a dropped-demand diagnostic, never native R75. Merging/requantization quality and live "
                     "execution require runtime support; no checkpoint or RAM saving is claimed.")}
    finish(out, summary, status)


if __name__ == "__main__":
    main()
