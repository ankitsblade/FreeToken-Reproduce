# Experiment 2 — Routing locality and LRU: original vs shuffled order (gpt-oss-20b)

**Attribution:** Yeshwanth; imported results reviewed against saved artifacts, not rerun during cleanup. Runner code is in `experiments/routing_locality/`; original data and figures are in `results/historical/routing_locality/`.

**Sampling correction:** the former narrative called this capture greedy, but `results/historical/routing_locality/results/routing_trace_meta.json` records temperature 1.0, top-p 0.95 and top-k 64. The canonical description uses that saved sampling metadata. This is one sampled trace and five shuffled orders, not a broad greedy workload study. The exact execution date is not recorded in that metadata.

Objective report. How much of the LRU expert-cache benefit comes from temporal locality in
the original decode routing order, versus the expert-frequency distribution alone.

## Setup

Routing sequence captured from a real decode on the `saketh_exps` workload (gpt-oss-20b,
`experiments/prompts/routing_workloads.jsonl#0`, sampled (temperature 1.0, top-p 0.95, top-k 64), 254 measured tokens), same model /
host / env / cache config. Geometry per layer: E=32 experts, top_k=4, 24 MoE layers, 255
tokens/layer, 24480 total expert accesses per order.

- **A. Original**: experts consumed in the real per-token routing order.
- **B. Shuffled**: the same tokens with their order permuted (5 seeds). This preserves each
  layer's exact expert multiset and per-expert frequency; it changes only temporal order.

LRU is analyzed exactly via stack (reuse) distance -- under LRU(C) an access hits iff its
stack distance < C -- giving the full miss-rate-vs-C curve. Verified against an explicit LRU
simulator (miss rates match to 1e-9). Cache mechanism, model, and workload are unchanged;
only the routing order changes.

## Reuse distance (distinct experts since an expert was last used)

| order | mean | median |
|---|--:|--:|
| original | 7.82 | 6 |
| shuffled | 9.51 | 8 |

The original order reuses an expert after ~6 distinct others; shuffling stretches that to
~8. Shorter reuse distance = more temporal locality, so more accesses hit at a given cache
size. The reuse-distance CDF (`results/historical/routing_locality/figures/locality3`) shows the original curve everywhere at or
above the shuffled curve.

## LRU miss rate vs cache size (access-weighted over layers; shuffled = mean of 5 seeds)

| cache (experts) | % of E | miss original | miss shuffled | locality benefit | rel. |
|--:|--:|--:|--:|--:|--:|
| 1 | 3 | 0.987 | 0.988 | +0.001 | 0% |
| 2 | 6 | 0.951 | 0.959 | +0.008 | 1% |
| 4 | 12 | 0.695 | 0.797 | +0.102 | 13% |
| 6 | 19 | 0.550 | 0.681 | +0.131 | 19% |
| 8 | 25 | 0.431 | 0.553 | +0.122 | 22% |
| 12 | 38 | 0.254 | 0.356 | +0.102 | 28% |
| 16 | 50 | 0.143 | 0.206 | +0.062 | 30% |
| 24 | 75 | 0.050 | 0.058 | +0.007 | 13% |
| 32 | 100 | 0.028 | 0.028 | +0.000 | 0% |

Evictions (explicit LRU, summed over layers) at C=6: original 13316 vs shuffled 16525
(+24% churn from shuffling).

## Patterns observed

1. **Temporal locality gives a real, substantial LRU benefit** at mid-range cache sizes:
   the original order misses less than a frequency-preserving shuffle by up to 0.13
   miss-rate points (C=6) and up to **30% relative** (C=16). At C=8 (2x top_k) locality cuts
   the miss rate from 0.55 to 0.43.

2. **The benefit is bounded to a cache-size window (~12-50% of E).** It vanishes at both
   extremes: at C<=2 the per-token working set (4) does not fit, so almost everything misses
   in either order; at C>=24 nearly the whole expert set is resident, so order no longer
   matters. Locality only pays when the cache is large enough to hold recently-used experts
   but too small to hold all of them -- exactly the regime an offload cache operates in.

3. **Absolute vs relative peak differ**: the largest absolute miss-rate reduction is at C=6
   (0.131), while the largest relative reduction is at C=16 (30%), because the baseline miss
   rate is already small there.

4. **Locality is layer-dependent.** At C=8 the per-layer benefit ranges from -0.04 (one
   layer whose routing is near-uniform in time -- shuffling barely changes it, and noise can
   make it marginally negative) to +0.25 (layer 13; layers 10, 11 also ~0.20+). A few layers
   carry most of the locality; others route almost memorylessly. (`results/historical/routing_locality/figures/locality4`.)

5. **Shuffling raises eviction churn** ~24% at C=6, the mechanism behind the higher miss
   rate: with locality destroyed, the LRU stack turns over faster.

## Quantifying the locality benefit

Of the LRU hits the cache achieves in the operating window, a meaningful share is
attributable to temporal ordering rather than frequency skew. At C=8, LRU with the original
order hits 56.9% of accesses; with frequency preserved but order destroyed it hits 44.7%.
So ~12 percentage points of the hit rate -- about a fifth of the hits at that size, and up
to ~30% relative at C=16 -- come specifically from temporal locality in the routing
sequence, concentrated in a subset of layers.

## Conclusion (objective)

For this model and workload the original expert-routing order carries genuine temporal
locality that LRU exploits: at cache sizes of roughly 12-50% of the per-layer expert set,
the real order misses 13-30% less (relative) than a frequency-matched shuffle, driven by
shorter reuse distances and lower eviction churn, and concentrated in a subset of MoE
layers. Outside that cache-size window the ordering is irrelevant. These are observations
for one routing trace on one model/host and are not claimed as novel.

## Artifacts

`results/historical/routing_locality/results/routing_trace.csv` (+ `_meta.json`) raw routing; `results/historical/routing_locality/results/miss_curve.csv`,
`evictions.csv`, `per_layer_missrate.csv`, `reuse_distance_hist.csv`,
`locality_summary.json`; `results/historical/routing_locality/figures/locality1..4.png`. Reproduction commands are below.


## Reproduction and artifact map

Execute commands from the repository root unless a `cd` is shown. Use `uv`; give reruns fresh output locations. Recorded results below are historical evidence.

Measures how much of FreeToken's LRU expert-cache benefit comes from temporal locality in
the original decode routing order, vs the expert-frequency distribution alone. Same model /
host / env / workload / cache config as saketh_exps; only the routing ORDER changes
(original vs a frequency-preserving shuffle).

## Reproduce
```bash
source scripts/env.sh

# 1. Capture a routing trace (already saved as results/routing_trace.csv). To regenerate:
uv run --no-sync python experiments/baselines/bench_decode_moe.py --model "$MODEL" --backend hybrid --hybrid-fetch -1 \
  --decode 256 --cache-rate 0.2 --no-graph \
  --trace-csv results/historical/routing_locality/results/routing_trace.csv \
  --aime experiments/prompts/routing_workloads.jsonl --problem 0 \
  --json results/historical/routing_locality/results/routing_trace_meta.json

# 2. LRU locality analysis (CPU only): original vs shuffled, stack-distance exact
uv run --no-sync python experiments/routing_locality/lru_locality.py

# 3. Plots
uv run --no-sync python experiments/routing_locality/plots_locality.py
```

## Method
LRU hit/miss is exact from the stack (reuse) distance: under LRU(C) an access hits iff its
distinct-reference distance < C. One pass yields the whole miss-rate-vs-C curve (validated
against an explicit LRU sim to 1e-9). The shuffle permutes token order per layer over 5
seeds, preserving each layer's expert multiset/frequency exactly.

## Outputs
- `results/historical/routing_locality/results/routing_trace.csv` raw per-layer routed experts; `_meta.json` measured window
- `results/historical/routing_locality/results/miss_curve.csv` miss/hit rate vs cache size, original vs shuffled + locality benefit
- `results/historical/routing_locality/results/evictions.csv`, `results/historical/routing_locality/results/per_layer_missrate.csv`, `results/historical/routing_locality/results/reuse_distance_hist.csv`
- `results/historical/routing_locality/results/locality_summary.json`
- `results/historical/routing_locality/figures/locality1..4.png`
- `report.md` interpretation

[Master experiment log](../../EXPERIMENT_LOG.md) · [Documentation index](../../README.md).
