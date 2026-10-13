"""Experiment 2: how much of the LRU cache benefit comes from temporal locality in the
original expert-routing order.

For each MoE layer we take the real decode routing sequence (per token, its top_k expert
ids, in order) and compare LRU behavior on:
  A. the original token order
  B. the same tokens after shuffling their order (identical expert identities and per-expert
     frequencies; only the temporal order changes -> temporal locality destroyed)

Method: LRU miss/hit is determined exactly by the stack (reuse) distance -- the number of
DISTINCT experts touched since an expert was last used. Under LRU with capacity C, an access
is a hit iff its stack distance < C (a first-ever access is a cold miss). Computing stack
distances once yields the whole miss-rate-vs-C curve (Mattson's stack algorithm); we do it
for the original order and for several random shuffles. An explicit LRU simulator confirms
the counts and gives eviction totals and residency.

Only the routing order changes; the model, workload, and cache mechanism are untouched.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
RES = study_dir("routing_locality") / "results"

CACHE_SIZES = [1, 2, 4, 6, 8, 12, 16, 24, 32]
SHUFFLE_SEEDS = [1, 2, 3, 4, 5]


def load_layers() -> tuple[dict[int, list[list[int]]], int, int]:
    meta = [json.loads(l) for l in (RES / "routing_trace_meta.json").read_text().splitlines()][-1]
    s, e = int(meta["trace_start_row"]), int(meta["trace_end_row"])
    rows = list(csv.DictReader((RES / "routing_trace.csv").open()))[s:e]
    layers: dict[int, list[list[int]]] = {}
    E = int(rows[0]["E"])
    for r in rows:
        layers.setdefault(int(r["layer_id"]), []).append(
            [int(x) for x in r["routed_experts"].split(";")])
    return layers, E, len(layers)


def stack_distances(accesses: list[int]) -> list[int | None]:
    """Distinct-reference (LRU stack) distance per access; None = first (cold)."""
    recency: list[int] = []  # MRU at end
    out: list[int | None] = []
    for e in accesses:
        if e in recency:
            idx = recency.index(e)
            out.append(len(recency) - 1 - idx)  # distinct experts more recent than e
            recency.pop(idx)
        else:
            out.append(None)
        recency.append(e)
    return out


def miss_rate_from_distances(dists: list[int | None], C: int) -> float:
    misses = sum(1 for d in dists if d is None or d >= C)
    return misses / len(dists)


def lru_counts(accesses: list[int], C: int) -> dict:
    """Explicit LRU: hits, misses, evictions (validates the stack-distance curve)."""
    recency: list[int] = []
    hits = misses = evict = 0
    for e in accesses:
        if e in recency:
            recency.remove(e); recency.append(e); hits += 1
        else:
            misses += 1
            if len(recency) >= C:
                recency.pop(0); evict += 1
            recency.append(e)
    return {"hits": hits, "misses": misses, "evictions": evict, "n": len(accesses)}


def flatten(tokens: list[list[int]]) -> list[int]:
    return [e for tok in tokens for e in tok]


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    layers, E, n_layers = load_layers()
    rng = random.Random(42)

    # Per-(cache size) aggregate miss rate: original vs shuffled (mean over seeds & layers).
    curve_rows = []
    per_layer_rows = []
    reuse_hist = {"original": Counter(), "shuffled": Counter()}
    evict_rows = []

    # Precompute original stack distances per layer; shuffled distances per seed per layer.
    orig_dists = {lid: stack_distances(flatten(toks)) for lid, toks in layers.items()}
    shuf_dists = {seed: {} for seed in SHUFFLE_SEEDS}
    for seed in SHUFFLE_SEEDS:
        r = random.Random(seed)
        for lid, toks in layers.items():
            t = toks[:]
            r.shuffle(t)  # permute token order; identical expert multiset
            shuf_dists[seed][lid] = stack_distances(flatten(t))

    # reuse-distance histograms (finite distances only), aggregated across layers
    for lid in layers:
        for d in orig_dists[lid]:
            reuse_hist["original"][d if d is not None else -1] += 1
        for d in shuf_dists[SHUFFLE_SEEDS[0]][lid]:
            reuse_hist["shuffled"][d if d is not None else -1] += 1

    for C in CACHE_SIZES:
        # original: one deterministic value (mean over layers, access-weighted)
        o_acc = [(orig_dists[lid], sum(len(t) for t in layers[lid])) for lid in layers]
        o_miss = sum(miss_rate_from_distances(d, C) * n for d, n in o_acc) / sum(n for _, n in o_acc)
        # shuffled: mean +/- std over seeds (each seed access-weighted over layers)
        s_vals = []
        for seed in SHUFFLE_SEEDS:
            sm = sum(miss_rate_from_distances(shuf_dists[seed][lid], C) *
                     sum(len(t) for t in layers[lid]) for lid in layers)
            sm /= sum(sum(len(t) for t in layers[lid]) for lid in layers)
            s_vals.append(sm)
        curve_rows.append({
            "cache_size": C,
            "cache_pct_of_E": round(100 * C / E, 1),
            "miss_rate_original": round(o_miss, 4),
            "hit_rate_original": round(1 - o_miss, 4),
            "miss_rate_shuffled_mean": round(statistics.mean(s_vals), 4),
            "miss_rate_shuffled_std": round(statistics.stdev(s_vals) if len(s_vals) > 1 else 0, 4),
            "hit_rate_shuffled_mean": round(1 - statistics.mean(s_vals), 4),
            "locality_benefit_missrate": round(statistics.mean(s_vals) - o_miss, 4),
            "locality_benefit_rel_pct": round(100 * (statistics.mean(s_vals) - o_miss) /
                                              statistics.mean(s_vals), 1) if statistics.mean(s_vals) else 0,
        })
        # eviction counts (explicit sim) aggregated across layers, at this C
        o_ev = sum(lru_counts(flatten(layers[lid]), C)["evictions"] for lid in layers)
        s_ev = []
        for seed in SHUFFLE_SEEDS:
            r = random.Random(seed)
            tot = 0
            for lid in layers:
                t = layers[lid][:]; r.shuffle(t)
                tot += lru_counts(flatten(t), C)["evictions"]
            s_ev.append(tot)
        evict_rows.append({"cache_size": C, "evictions_original": o_ev,
                           "evictions_shuffled_mean": round(statistics.mean(s_ev), 1)})

    # per-layer miss rate at a representative small cache (C=8 = 2x top_k)
    Crep = 8
    for lid in sorted(layers):
        o = miss_rate_from_distances(orig_dists[lid], Crep)
        s = statistics.mean(miss_rate_from_distances(shuf_dists[seed][lid], Crep)
                            for seed in SHUFFLE_SEEDS)
        per_layer_rows.append({
            "layer_id": lid, "cache_size": Crep,
            "miss_rate_original": round(o, 4), "miss_rate_shuffled": round(s, 4),
            "locality_benefit": round(s - o, 4)})

    def write(name, rows):
        with (RES / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write("miss_curve.csv", curve_rows)
    write("evictions.csv", evict_rows)
    write("per_layer_missrate.csv", per_layer_rows)
    # reuse distance histogram
    maxd = max(max(reuse_hist["original"]), max(reuse_hist["shuffled"]))
    hist_rows = []
    for d in range(-1, maxd + 1):
        hist_rows.append({"reuse_distance": ("cold" if d == -1 else d),
                          "count_original": reuse_hist["original"].get(d, 0),
                          "count_shuffled": reuse_hist["shuffled"].get(d, 0)})
    write("reuse_distance_hist.csv", hist_rows)

    # reuse-distance summary (finite only)
    def finite_stats(hist):
        vals = [d for d, c in hist.items() if d >= 0 for _ in range(c)]
        return (statistics.mean(vals), statistics.median(vals)) if vals else (0, 0)
    om, omed = finite_stats(reuse_hist["original"])
    sm, smed = finite_stats(reuse_hist["shuffled"])
    (RES / "locality_summary.json").write_text(json.dumps({
        "n_layers": n_layers, "E": E, "top_k": len(next(iter(layers.values()))[0]),
        "tokens_per_layer": len(next(iter(layers.values()))),
        "total_accesses_per_order": sum(sum(len(t) for t in layers[lid]) for lid in layers),
        "shuffle_seeds": SHUFFLE_SEEDS,
        "reuse_distance_original_mean": round(om, 2), "reuse_distance_original_median": omed,
        "reuse_distance_shuffled_mean": round(sm, 2), "reuse_distance_shuffled_median": smed,
    }, indent=2) + "\n")

    print(f"layers={n_layers} E={E} tokens/layer={len(next(iter(layers.values())))}")
    print("reuse distance (distinct): original mean %.2f median %d | shuffled mean %.2f median %d"
          % (om, omed, sm, smed))
    print("\ncache  %E   miss_orig  miss_shuf  hit_orig  hit_shuf  locality_benefit")
    for r in curve_rows:
        print(f"{r['cache_size']:5d} {r['cache_pct_of_E']:5.0f}  {r['miss_rate_original']:9.3f}  "
              f"{r['miss_rate_shuffled_mean']:9.3f}  {r['hit_rate_original']:8.3f}  "
              f"{r['hit_rate_shuffled_mean']:8.3f}  {r['locality_benefit_missrate']:+.3f} "
              f"({r['locality_benefit_rel_pct']:.0f}%)")


if __name__ == "__main__":
    main()
