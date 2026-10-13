# Runtime-aware q: findings and limitations

**Attribution:** Saketh. Executed 7 October after inspection on 6 October. The saved log set contains 889 episode records across exploration, pilot, scans, main matrix, ablations and follow-ups; this is not 889 independent prompts or fresh servers.

**Reading overhead:** 0.9 microseconds is the separate lean rule-decision measurement. The main result table's `Decide us` values include feature/logging work and are much larger. Eager execution adds substantial overhead relative to the earlier graph study. The rule has the best overall reported geometric speedup; fixed 25% is slightly faster in the main high-miss regime (54.4 versus 55.9 ms/token).

**Scope of every statement below:** gpt-oss-20b (MXFP4; 24 MoE layers, 32 experts/layer,
top_k 4, 13.2 MB/expert) on one RTX 4070 (PCIe gen4 x4, B_P = 6.71 GB/s) with a Ryzen 5
8500G (B_H = 35.7 GB/s), hybrid backend, batch 1, greedy, eager decode. No other hardware
or model was measured.

## 1. Conclusion

FreeToken's q\* is the right answer to the question it asks. On this host it returns q = 0
for every miss count, and the measured single-step optimum is also q = 0. The limitation is
the question: q\* balances one step and has no term for what a fetch does to later steps.
A fetched expert stays in the GPU cache and turns future misses into hits; a zero-fetch
policy never gets that.

Adding that one term gives a closed-form rule:

> fetch one more expert while its **expected reuses** are at least the **exposed cost of
> the fetch, in CPU-expert units**:
> `p · T_res ≥ max((j+1)/ρ, m−j−1) − max(j/ρ, m−j)`, with `ρ = B_P/B_H`,
> `p = top_k/E`, and `T_res` = the LRU victim's age (or a fixed 32 tokens when the cache has
> a free slot).

The uniform-prior variant uses `m`, free slots and LRU victim age with the calibration
ratio. The main variant additionally uses recent expert frequency. It had the best
overall geometric speedup in the evaluated matrix:

| | low_miss | med_miss | high_miss | geo-mean speedup | worst |
|---|---|---|---|---|---|
| q\* (ms/token) | 65.0 | 66.0 | 65.0 | 1.00× | — |
| reuse-aware rule | 31.6 | 45.0 | 55.9 | **1.52×** | 1.16× |
| best fixed split per workload | 34.9 (50%) | 45.8 (25%) | 54.4 (25%) | — | — |
| ridge regression | 33.4 | 46.7 | 58.2 | 1.45× | 1.12× |

On conditions it was never adjusted for (a second prompt at three cache sizes, and three
unseen cache sizes) it is 1.70× faster than q\* with a worst case of 1.28×. Its decision
costs 0.9 µs. No learned policy beat it, and the learned policies transferred worse.

The recommended policy is therefore the analytic rule, not a model. Whether it transfers to
other hardware and models is untested.

## 2. Why q\* and end-to-end behaviour differ

**Single step.** Measured layer time by miss count and fetch count (uniform-random q, real
serving):

| m \ q | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 1 | 0.64 | 2.34 | | | |
| 2 | 1.06 | 2.35 | 4.31 | | |
| 3 | 1.46 | 2.34 | 4.32 | 6.29 | |
| 4 | 1.86 | 2.33 | 4.31 | 6.29 | 8.26 |

One PCIe transfer (1.99 ms) costs as much as about 4.7 CPU expert evaluations (0.43 ms
each), so q = 0 wins every row. q\* predicts exactly this.

**Sustained decode.** The same table shows what the first fetch actually costs once the
overlap is counted: +0.48 ms at m = 4 (the transfer hides behind three CPU experts), +0.87
at m = 3, +1.29 at m = 2, +1.70 at m = 1. Each later reuse of that expert saves about
0.4 ms. So a fetch at m = 4 pays for itself after 1.2 reuses, and at m = 1 only after about
ten. Measured over 193k fetch events, a fetch under the 25% policy is reused 21 times; under
q = m, 0.8 times.

**Why more fetching is worse.** Cache capacity bounds the benefit: at med_miss the CPU time
saved saturates near 5 s per 254 tokens however much is fetched. 25% collects 3.0 s of it
with 332 fetches; q = m spends 13,435 fetches to collect 4.7 s and pays 26 s of exposed
transfer for it. Past the point where the cache is full, each fetch evicts something that
was still useful, and residency drops from 79 tokens to under 3. The quantity that tells
these regimes apart is how long an entry survives in the cache — the LRU victim's age —
which is the `T_res` in the rule.

**Consequence for learning.** A cost model trained on the immediate layer time, or on a
horizon of 8 tokens or less, chooses q ≈ 0 and reproduces q\*. The benefit only appears at
16–32 tokens. A myopic LinUCB converges to q = 0 for the same reason.

## 3. Answers to the research questions

1. **Is q\* sufficient?** For one step, yes. For sustained decode on this host, no: it is
   1.2–2.1× slower than the best policy at every cache size tested.
2. **When does q\* fail?** When PCIe is slow enough that q\* rounds to 0 and experts are
   reused across tokens. Then the GPU cache never fills: hit rate 0.00, 1,271 MB of expert
   weights through the CPU per token, at every cache size.
3. **Is q\* correct for its single-step objective?** Yes. Its choice equals the measured
   single-step optimum for m = 1..4, both in the isolated branch oracle of `saketh_exps` and
   in the real-serving table above.
4. **Why can end-to-end differ?** Section 2: a fetch has a delayed benefit that q\* has no
   term for, and a delayed cost (eviction) that fixed fractions ignore.
5. **Does cache state matter?** It is the only feature group that matters decisively.
   Removing cache state from the rule drops it from 1.52× to 1.01×. In the ridge ablation
   the speedup goes from 1.22× to 1.41× when the cache group is added, and removing it from
   the full model gives 0.59×. Occupancy and hit/miss counts alone are not enough (0.96×);
   the eviction-pressure features (LRU age, churn) are what carry it (1.46×).
6. **Does expert reuse matter?** Reuse as a concept is the whole mechanism. Per-expert reuse
   *features* are not needed: replacing each expert's recent frequency with the constant
   prior `top_k/E` changes nothing (1.505× vs 1.519×; in lean mode 1.62× vs 1.59×).
7. **Does CPU/GPU/PCIe utilization matter?** No, and it hurts learned policies. Adding the
   utilization, memory and history groups to the ridge model lowers its speedup from 1.45×
   to 1.39×, 1.09× and 0.93×. Under four CPU busy-loops the rule with a static ρ is 1.62×
   and with an online ρ 1.68× (two repeats; not distinguishable). PCIe utilization itself
   could not be measured.
8. **Does history matter?** Not for the decision. It gives the largest offline accuracy gain
   after cache state and makes the end-to-end policy worse.
9. **Is m + B_P + B_H insufficient?** Yes. A model on those alone is 0.96× (it fetches
   nothing and pays inference cost). B_P and B_H are constants on one machine, so on one
   machine that baseline is a function of m only.
10. **Smallest useful feature set?** `m`, free cache slots, LRU victim age, plus the
    calibration ratio B_P/B_H.
11. **Is a simple model enough?** A formula is enough. Ridge on m + cache features reaches
    1.41–1.46× in the conditions it was trained on, but the full ridge falls to 1.13× on
    unseen conditions (worst 0.78×) and to 0.91× under CPU contention.
12. **Does ANN complexity help?** No. Across 1–3 hidden layers and 16–128 units the MLPs
    have the lowest prediction error with GBM (R² 0.87–0.90) and nearly all choose q = 0.
    Ten of twelve architectures are slower than q\*; the two that are not (2×16, 3×64) are
    single runs and inconsistent across workloads.
13. **Do contextual bandits help?** Partly. LinUCB with a 16-token delayed cost learns
    within a single episode to 1.35×; Thompson sampling 1.29× with high variance; the
    myopic bandit equals q\* minus overhead. Warm-started variants were worse in the
    exploratory scan (single run, before burn-in was added) and were not carried into the
    main matrix. None reaches the rule.
14. **Is the runtime overhead justified?** For the rule, yes: 0.9 µs per decision against
    about 1,050 µs of layer execution. Linear models cost 11 µs, the MLP 43 µs, GBM
    1,200 µs — more than the layer itself, which is why GBM is 0.81× despite fetching
    sensibly. See section 6 for the cost of running any policy on the host.

## 4. Hypotheses

| | hypothesis | verdict on this setup |
|---|---|---|
| H1 | q\* is insufficient when cache dynamics matter | supported |
| H2 | cache state improves q selection beyond B_P/B_H | supported; the decisive group |
| H3 | recent expert reuse improves q selection | not supported for per-expert frequency features |
| H4 | branch imbalance is useful | the overlap *model* is needed (it sets the fetch cost); measured imbalance features are not |
| H5 | current utilization improves adaptation | not supported; harmful to learned policies |
| H6 | a runtime-aware policy can beat fixed q across miss regimes | supported overall: the rule has the best reported geometric speedup; fixed 25% is slightly faster at high_miss |
| H7 | a simple statistical model captures much of the benefit | in-distribution yes; it does not transfer |
| H8 | a GELU ANN can approximate Cost(x, q) | it predicts well and decides badly |
| H9 | a contextual bandit can adapt q online | it learns a useful policy within one episode; adaptation to changing conditions was not tested |
| H10 | future cache benefit is the missing term in q\* | supported by direct measurement |

## 5. What predicts versus what decides

Prediction accuracy and decision quality came apart throughout. The most accurate models
(MLP, GBM) made the worst decisions; ridge, with higher error, made better ones; and adding
feature groups kept lowering offline error while end-to-end latency rose. The rule has
positive single-step regret (2.35 ms/token worse than q = 0 by the single-step table) and
the lowest end-to-end latency. Offline error and single-step regret are therefore not
usable as selection criteria here; only end-to-end runs are.

## 6. Limitations

- **One machine, one model.** The rule is written in ratios (ρ, `top_k/E`, residency in
  tokens) so that it can transfer, and the model predicts it should fetch more when PCIe
  is faster and reduce to q\*'s behaviour when reuse is absent. None of that was measured.
  Cross-hardware and cross-model generalization is unvalidated.
- **Host-side policy.** Every policy here ran in eager mode with a sync per layer, which
  costs about 19 ms/token relative to CUDA-graph decode (q\* is 65 ms/token here and 41 in
  graph mode). The rule in this harness beats graph-mode q\* only at low_miss. In graph
  mode a fixed 25% split measured 25–34 ms/token in `saketh_exps`, which shows the benefit
  survives there, but the rule itself has not been implemented in the device-side kernel.
- **One free parameter is not derived.** `T_free = 32` tokens (assumed residency when the
  cache has room). 64 was worse at low_miss in the exploratory scan; it was not swept
  properly. κ was fixed at its derived value of 1; κ = 4 was within noise overall.
- **Two prompts, one decode length for most runs, batch 1.** The 384-token check (1.49×) is
  two repeats on one workload.
- **Statistics.** Main results have three repeats; several 95% intervals at high_miss are
  wide (rule vs q\*: 1.17× ± 0.23, p = 0.08). The low and medium regimes are clear
  (p ≤ 0.002). Individual-signal, history and ANN ablations are single runs.
- **Counterfactuals.** Cost models were fitted only to randomized actions, and their
  targets are "this action, then the logging policy afterwards". They are one step of
  policy improvement, not the value of running a policy for a whole decode.
- **Output equality.** Text is identical across policies for prompt 0 at 128 tokens. On
  prompt 1 and at 384 tokens it diverges between policies because the CPU and GPU expert
  kernels are not bit-identical; latency comparisons there are over slightly different
  token sequences.
- **Unmeasured signals.** PCIe utilization, memory-bandwidth utilization, CPU worker queue
  state, pinned-memory usage and KV-cache occupancy were not available
  (`docs/experiments/runtime_q/features.md`, section E).

## 7. Unmeasured extensions

The source study's GPU/RAM/NVMe hierarchy, elastic RAM, heterogeneous-byte scheduling,
device-side implementation and second-platform/model directions are maintained in
[the single experiment plan](../../PLAN.md). They are not completed measurements.

## 8. Where the evidence is

| claim | file |
|---|---|
| main comparison, paired statistics | `docs/experiments/runtime_q/tables.md` §1, `results/historical/runtime_q/results/eval/main_summary.csv`, fig 01, 22 |
| single-step table, fetch value | `results/historical/runtime_q/results/cache_analysis/`, fig 16, 17 |
| feature screening and importance | `docs/experiments/runtime_q/features.md`, `results/historical/runtime_q/results/feature_analysis/`, fig 14, 15 |
| group addition / leave-one-out / cache / history / ANN | `docs/experiments/runtime_q/tables.md` §2–7, fig 07–13 |
| horizon effect | `results/historical/runtime_q/results/statistical/offline_horizon.csv`, fig 25 |
| generalization, contention, long decode, overhead | `docs/experiments/runtime_q/tables.md` §8–11, fig 24, 26–28 |
| how the rule behaves | fig 18–20 |
| environment and versions | `results/historical/runtime_q/reports/reproducibility.json` |


## Maintained evidence links

[Complete result tables](tables.md) · [Feature inventory](features.md) · [Reproduction](reproduction.md) · [Single experiment log](../../EXPERIMENT_LOG.md).

The original progress/conclusion/README are recoverable from the documentation archive. Future directions are maintained only in [PLAN.md](../../PLAN.md); none of the unmeasured extensions is a completed result.
