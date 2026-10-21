# Reuse-Aware with runtime variables: V1, V2, V3

Status as of 2026-10-21. Hardware: RTX 4070 12 GB (PCIe gen4 x4), Ryzen 5 8500G, 30 GB RAM.
Runtime: `.runtime/freetoken` = upstream FreeToken `cab110e` (via base commit `4d0f849`) plus
`runtime/patches/experiment-hooks.patch` (hook plumbing only; cache sizing and policy code unchanged).

Contents: [the decision](#1-the-decision-every-policy-makes) ·
[baselines](#2-the-two-baselines) · [V1](#3-v1--measured-costs) · [V2](#4-v2--calibrated-reuse) ·
[V3](#5-v3--eviction-aware) · [combinations](#6-combinations) ·
[implementation](#7-implementation) · [what has been done](#8-what-has-been-done) ·
[results](#9-results-so-far) · [limitations](#10-limitations-and-open-items) · [files](#11-files)

---

## 1. The decision every policy makes

In FreeToken's hybrid decode, each MoE layer of each token routes to `k` experts (`k` = 4 for
gpt-oss-20b, 8 for Qwen3.6 and Gemma-4). Experts already in the GPU slot cache are **hits**. The
others, `m` of them, are **misses**. The policy chooses **q**, how many misses to copy to the GPU
over PCIe (they then stay cached); the other `m − q` run on the CPU. Both branches run concurrently
and the layer finishes when the slower one does. Which `q` misses are fetched (most recently active
first) and which slots they evict (global LRU) is decided by the engine, not the policy.

Notation used below:

| Symbol | Meaning |
|---|---|
| `m` | misses in this layer for this token |
| `h` | hits in this layer (`h = k − m`) |
| `q` | misses fetched to the GPU (decision) |
| `p_j` | reuse probability of the `j`-th best missed expert: its smoothed routing frequency in this layer over the last 32 tokens, `p = (count + 8·k/E) / (window + 8)` |
| `T_j` | expected residency in tokens: 32 if a free slot is available for the `j`-th fill, else the age (tokens) of the LRU slot it would evict |
| `ρ` | `B_P / B_H`, calibrated PCIe vs CPU-expert bandwidth (benchbw) |
| `S` | bytes of one expert |

## 2. The two baselines

**Policy A — paper q\*** (FreeToken §3.2, Eq. 4): `q* ≈ m · B_P / B_H`, rounded, "always retains at
least one fill". In code: `fraction_min1` (balance rounding of the engine, raised to 1 when `m > 0`).
The engine's own rule (`E`, built-in q\*) uses a different fraction (overlapped benchmark pair,
0.178 on gpt-oss) and no minimum; `A0` is `m · B_P/B_H` with the engine rounding and no minimum
(the implementation used in earlier studies).

**Policy B — original Reuse-Aware** (`runtime_q_hook.ReuseAware`, spec `{"kind": "reuse_aware",
"kappa": 1}`). Fetch candidates in decreasing `p` while

```
p_j · T_j  ≥  Δ(m, q)
Δ(m, q) = max((q+1)/ρ, m−q−1) − max(q/ρ, m−q)        (CPU-expert units; one future hit = 1)
```

i.e. fetch while the expected future hits of the fetched expert pay for the extra time the fetch
adds to this layer now, with both sides in units of "one expert computed on the CPU".

## 3. V1 — measured costs

**Question it answers:** are the paper's fixed bandwidth units the right cost model on this machine?

V1 replaces the dimensionless cost with milliseconds measured while the model runs:

```
C(m, j) = max( g0 + j · t_p ,  [m > j] · (c0 + (m − j) · t_c) )          (ms)
fetch while   p_j · T_j · v_hit  ≥  C(m, q+1) − C(m, q)
```

| Term | How it is obtained |
|---|---|
| `g0, t_p` | least-squares line of the GPU branch time (PCIe copy + GEMM, CUDA events) against the number fetched, over this episode's layer samples |
| `c0, t_c` | least-squares line of the CPU task timer against the number of CPU experts |
| fallback | until a fit has 8 samples at ≥ 2 distinct x with positive slope, slope = `S/B_P` (GPU) and `S/B_H` (CPU), intercept 0 |
| `v_hit` | ms one future hit saves: `Σ_m n_m [c(m) − c(m−1)] / Σ_m n_m` with `c(n) = max(g0, c0 + n·t_c)` and `n_m` the observed count of layers with `m` misses |

The key difference from B: B values one hit at a full CPU expert. V1 values it at the time it
actually saves. That is less when the CPU branch is not the slower one (few misses, large cache).

## 4. V2 — calibrated reuse

**Question it answers:** does the 32-token-window prediction `p · T` under- or over-estimate how often
fetched experts are actually reused?

```
R_j = α · p_j · T_j
α   = (realised hits + 8) / (predicted hits + 8)
```

over every expert fetched in this episode and already evicted: at fill time the hook records the
predicted `p · T`; while the expert stays resident it counts the hits it actually receives; when
it is evicted both totals are added. The `+8` makes `α` start at 1. Costs stay in B's units.

## 5. V3 — eviction-aware

**Question it answers:** a fetch into a full cache evicts an expert that might itself have been
reused. Should that lost reuse be charged?

```
R_j = (p_j − p_victim(j)) · T_j
```

The victim of the `j`-th fill is predicted as the engine will pick it: the slot with the smallest
LRU usage stamp among slots not owned by this layer's routed experts (free slots first, ties to
the lower slot). `p_victim` is that expert's smoothed 32-token frequency in its own layer
(0 for a free slot).

## 6. Combinations

The three switches are independent spec flags and compose in one rule:

```
fetch while   α^[V2] · (p_j − p_victim^[V3]) · T_j · v^[V1]  ≥  ΔC^[V1](m, q)
```

where `[V·]` means "only if that variant is on" (`α = 1`, `p_victim = 0`, and B's
`Δ`, `v = 1` otherwise). With all three off the rule is Policy B exactly: 0 decision mismatches in
60,000 randomised inputs per model (gpt-oss, Qwen, Gemma shapes), checked on the CPU.

## 7. Implementation

- File: `experiments/runtime_q/instrumentation/reuse_aware_v2_hook.py`.
  - `ReuseVariant(Policy)`, spec `{"kind": "reuse_aware_v", "cost": "paper"|"runtime",
    "reuse": "window"|"calibrated", "evict": false|true}`.
    V1 = `cost: runtime`, V2 = `reuse: calibrated`, V3 = `evict: true`.
  - `ReuseAwareV2(RuntimeQ)` subclass of the original hook (the original is not modified).
    It adds the **lean harness**, which computes, for every lean arm, only what the reuse rules need:
    - routing ids and this layer's slot row;
    - one device→host copy of slot ids, usage stamps and the step counter;
    - the 32-token windows;
    - the predicted LRU victims;
    - the online `α`;
    - the online branch-cost fits.

    Every lean arm pays the same harness cost, so arms are comparable.
  - `BranchFit`: the per-episode least-squares fit with the bandwidth prior.
- Decision log (lean CSV) records per layer, besides timers and cache state, the shadow
  decision of each rule on the same state: `q_orig`, `q_v1`, `q_v2`, `q_v3`. This is used to check
  that each arm took its own rule's decision.
- Everything a decision uses is measured before it (timer samples are lagged at least one
  layer). No future routing is used online.
- Fix made on 2026-10-21: the hook cached the slot-cache capacity at startup. After a runtime
  cache rebuild that value was stale (it reported 308 slots after shrinking to 77). The v2 hook
  now re-reads capacity at each episode and on any size change. This was validated on gpt-oss
  and Qwen. It does not affect fixed-capacity runs.

## 8. What has been done

| Date | Step | Outcome |
|---|---|---|
| 2026-10-20 | Cross-model study of B vs paper q\* (`results/reuse_aware_models/study_20261020T071921Z`, 324 episodes) | B vs paper q\*: gpt-oss 1.148×, Qwen 0.997×, Gemma 0.991× |
| 2026-10-20 | Stage A diagnosis of B's decision logs | Predicted V2 (more reuse) would help and V1, V3 would not |
| 2026-10-20 | **V1/V2/V3 pilot**, gpt-oss, cache 0.10 and 0.40, code and math tasks, 3 timed + 1 diagnostic round, paired against B on the same lean harness (`results/reuse_aware_v2/pilot_20261020T140940Z`, 144 episodes) | V1 passed the pre-set test; V2 failed; V3 inconclusive (section 9) |
| 2026-10-21 | CPU equivalence check of the V-rule (all off) vs B | 0 mismatches / 60,000 per model |
| 2026-10-21 | **Policy study** (`results/policy_study/20261021T090614Z`): Stage 1 = A, A0, built-in q\*, B on the established grid (3 models × 3 cache rates × 3 tasks, 3 timed + 1 logged round, plus unhooked A0 and built-in q\*); Stage 2 = A, B, V1, V2, V3, V1+V2, V1+V3, V2+V3, V1+V2+V3 (lowest and highest cache rate × code/math, 3 models) | Stage 1 gpt-oss done; Gemma running; Qwen and all of Stage 2 queued |

The pre-set pass rule for a variant was: not reliably slower than B at either cache size (95% CI
upper bound ≥ 1) and reliably faster at one (CI lower bound > 1).

## 9. Results so far

### Pilot (gpt-oss-20b, 12 pairs per variant; speedup = ms/token(B) / ms/token(variant))

| Variant | All | Cache 0.10 | Cache 0.40 | Pass? |
|---|---|---|---|---|
| V1 measured costs | **1.056×** [1.016, 1.096] | 1.026× [1.009, 1.044] | 1.086× [1.004, 1.175] | **yes** |
| V2 calibrated reuse | 0.952× [0.903, 1.003] | **0.905×** [0.879, 0.931] | 1.001× [0.911, 1.100] | no (slower at 0.10) |
| V3 eviction-aware | 1.037× [0.978, 1.098] | 1.025× [0.991, 1.060] | 1.048× [0.915, 1.201] | no (not significant) |

Against paper q\* (A0 implementation), same pilot:

| | Cache 0.10 | Cache 0.40 |
|---|---|---|
| B | 1.018× | 1.405× |
| V1 | 1.044× | **1.525×** |
| V2 | 0.920× | 1.407× |
| V3 | 1.043× | 1.472× |

What each rule did (decision logs, mean over the two tasks):

| | Fetches/token 0.10 | Hit rate 0.10 | Fetches/token 0.40 | Hit rate 0.40 |
|---|---|---|---|---|
| B | 5.2 | 0.366 | 1.9 | 0.830 |
| V1 | 3.0 | 0.373 | 0.9 | 0.827 |
| V2 | 10.5 | 0.373 | 1.8 | 0.837 |
| V3 | 1.6 | 0.370 | 1.2 | 0.831 |

Why:

- **V1** fetches 40–50% less than B at the same hit rate. Measured costs: one PCIe fetch
  `t_p` 1.74–1.92 ms, one CPU expert `t_c` 0.39–0.42 ms, GPU intercept `g0` 0.24–0.41 ms, and one
  future hit saves only `v_hit` 0.22–0.38 ms (B assumes a full CPU expert). A fetch has to be reused
  5–8 times to pay off, so marginal fetches are dropped.
- **V2**: `α` ended at 1.32–3.33, so fetched experts did get more hits than predicted. V2
  therefore fetched about twice as much, but the hit rate did not move. Each extra fill evicted
  another expert that would also have been reused. `α` counts gross hits of the fetched expert,
  not hits net of the evicted one, so Stage A's "fetch more" conclusion was refuted.
- **V3** cuts fetches sharply at small cache with the same hit rate. It was faster on average, but
  the confidence intervals include 1 with only 3 rounds per cell.

Checks:

- Every variant's actual decision matched its own rule's shadow decision in 100% of logged layers.
- The predicted eviction victim matched the engine's real eviction in 98.1–100% of cases.
- All episodes decoded the full 127 steps.

### Policy study, Stage 1 (gpt-oss; context for V1–V3)

The policy study's reference is Policy A = paper q\* *with* the paper's minimum-one-fill rule.

| | Cache 0.10 | Cache 0.20 | Cache 0.40 |
|---|---|---|---|
| B over A | 1.433× | 1.554× | 1.641× |
| B over A0 | 1.005× (n.s.) | 1.034× | 1.461× |

The minimum-one-fill rule makes A thrash on this GPU: about 23 fills and 23 evictions per token,
against 4–6 for A0 and B. V1–V3 Stage 2 results for all three models are pending.

## 10. Limitations and open items

- **Instrumentation.** All V1–V3 numbers are measured with the runtime-q hook attached. Its two GPU
  synchronisations per layer cost about 16–22 ms/token (×1.5) against an uninstrumented server.
  V1 needs those branch timers, so a deployable V1 would have to read timers one layer late without
  synchronising. That has not been built.
- **Scope.** The pilot covers one model, 2 tasks and 3 timed rounds per cell. Qwen and Gemma
  results for V1–V3 come from Stage 2, which has not run yet.
- **Cache sizing.** The pilot and the policy study use fixed `--moe-cache-rate` capacities (inherited
  from the earlier runtime-q studies, as a capacity sweep). FreeToken's default is `--moe-cache-auto`
  (experts take the VRAM left after a KV reserve). That is the configuration the paper's serving runs
  use (for example, "the RTX 5090 serving capacity, 37% of Qwen3.6's expert pool"); the paper's
  cache-size sweep (Fig. 4b) is an offline trace replay. An auto-sized run is not yet done.
- **Other deviations from the paper setup**, all inherited from the established experiments:
  - eager decode (`--no-graph`), required by the hook;
  - `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`;
  - one running request;
  - 5 CPU threads (paper: 6);
  - NVFP4 Qwen (paper: BF16 except on the laptop);
  - Gemma at memory ratio 0.8 with a 1280-token limit (an earlier startup-memory workaround);
  - three local prompts with 128 decode tokens instead of the paper's agentic traces.
- **Output text** differs between policies on NVFP4 models and on math/summarisation prompts. The
  CPU and GPU expert kernels differ in low bits, so greedy text depends on which experts run where.
  Code-task outputs on gpt-oss are identical across all policies.
- A feature-group alternative to V1–V3 (`reuse_aware_rv_hook.py`: exec / PCIe / cache / history
  groups) was built and smoke-tested but superseded by this V1–V3 design before it ran.

## 11. Files

| Path | Role |
|---|---|
| `experiments/runtime_q/instrumentation/runtime_q_hook.py` | original hook and Policy B (`ReuseAware`), unchanged |
| `experiments/runtime_q/instrumentation/reuse_aware_v2_hook.py` | lean harness, `ReuseVariant` (V1/V2/V3), `FractionMin1` (Policy A), `BranchFit`, capacity fix |
| `experiments/runtime_q/instrumentation/reuse_aware_rv_hook.py` | harness used by the policy study (subclass; adds bookkeeping, same decisions for A/B/V arms) |
| `experiments/runtime_q/check_policy_equivalence.py` | CPU-only equivalence check |
| `configs/runtime_q/reuse_aware_v2/gpt_oss_pilot.json` | pilot plan |
| `experiments/runtime_q/analyze_reuse_aware_v2.py` | pilot analysis → `results/reuse_aware_v2/pilot_20261020T140940Z/analysis/REPORT.md` |
| `configs/policy_study/stage{1,2}_{gpt_oss,qwen,gemma}.json` | policy-study plans (generator: `results/policy_study/20261021T090614Z/make_plans.py`) |
| `experiments/runtime_q/analyze_policy_study.py` | policy-study analysis → `results/policy_study/20261021T090614Z/analysis/REPORT.md` |
| `docs/reuse_aware_v2_pilot.pdf` | beginner-level PDF of the pilot |

Attribution: runtime-q harness and split policy, Saketh; reuse-aware variants and pilot, Ankit.
