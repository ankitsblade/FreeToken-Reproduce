# Findings: FreeToken q\* CPU/GPU-split policy vs fixed splits (gpt-oss-20b, RTX 4070)

**Attribution:** Saketh; imported results reviewed against saved artifacts, not rerun during cleanup. Runner code is in `experiments/split_policy/`; original data and figures are in `results/historical/split_policy/`.

**Configuration corrections:** saved runs/findings use high_miss cache rate 0.10 (77 slots), while the former README said 0.08. Use saved run configurations. Calibration gives `B_H/B_P` about 5.7 and `B_P/B_H` about 0.16-0.19; inverted ratio labels in the old narrative are corrected here. CUDA-graph performance and eager tracing/branch diagnostics remain distinct. Output identity is a result for the recorded prompt/settings, not a universal CPU/GPU bit-identity guarantee.

Objective report of what the experiment measured. No research contribution is claimed.

## 1. Setup (frozen for every run)

- Model: gpt-oss-20b, MXFP4 experts (`mxfp4_triton`), L=24 MoE layers, E=32 experts/layer,
  top_k=4, expert bank entry 12.62 MiB (13.24 MB decimal). FTW `~/.freetoken/models/gpt-oss-20b-ftw`.
- Host: RTX 4070 (12 GB), Ryzen 5 8500G (6c/12t), 30 GB RAM, CUDA 13 runtime.
- Backend `hybrid` for all policies. Decode bs=1 via the real `ft serve` path, greedy,
  one AIME-style prompt, 128 measured tokens (perf) / 64 (trace), 3 repeats, fresh server
  (cold cache warmed by one standardized warmup generate) per run.
- Only the q-selection policy changes. Workloads change one thing (GPU expert-cache size
  via `--cache-rate`): low_miss 0.40 (308 slots), med_miss 0.20 (154), high_miss 0.10 (77).

## 2. Calibration (this host)

Measured with `ft bench bw` on the deployed MXFP4 expert tensors:

| quantity | value |
|---|---|
| B_P PCIe expert-gather bandwidth | 6.72 GB/s (overlap 6.71) |
| B_H CPU MoE expert bandwidth | 38.1 GB/s (overlap 35.7) |
| B_H / B_P | 5.7x |
| adaptive fetch fraction B_P/B_H | ~0.16-0.19 |

Because PCIe is ~5.7x slower than CPU expert compute per byte, the adaptive policy sets
`q* = round(0.16..0.19 * m)`, which for top_k=4 (m in 1..4) rounds to **q\* = 0 for every
observed m** (occasionally 1 at m=4). On this host q\* is effectively "compute all misses
on the CPU."

## 3. Correctness (passed)

All six policies, across all three workloads (18 cells), produced **bit-identical greedy
output** (sha1 `f68a92a7123e`). Changing q does not change model output: CPU-side and
GPU-side expert execution are numerically equivalent. `results/historical/split_policy/results/summary/correctness.csv`.
(The per-layer miss count `m` differs between policies -- see section 5 -- but that is a
cache-occupancy effect, not a change in the computed result.)

## 4. Headline result: decode throughput (`results/historical/split_policy/figures/plot5`, `plot4`)

tokens/s (mean of 3; run-to-run std < 0.4):

| policy | avg q | low_miss | med_miss | high_miss |
|---|--:|--:|--:|--:|
| q\* adaptive | ~0 | 24.3 | 24.1 | 24.2 |
| q=0 (CPU) | 0 | 24.3 | 24.2 | 24.3 |
| q=25% | 0.05-0.23 | **40.3** | **37.1** | **29.4** |
| q=50% | 0.20-1.27 | **50.5** | 21.3 | 14.5 |
| q=75% | 0.9-2.3 | 21.0 | 11.1 | 8.3 |
| q=m (GPU) | 1.1-4.0 | 19.2 | 8.7 | 5.0 |

- q\* tracks q=0 exactly (it *is* q=0 here) and is **flat at ~24 tok/s regardless of cache
  size**.
- A fixed q=25% beats q\* by **1.2-1.7x** in every workload; q=50% beats it by 2.1x at
  low_miss. The best fixed split is 50% when the cache is large, 25% when it is small.
- q=m (fetch everything) is the worst and degrades sharply as misses rise (PCIe-bound).

## 5. Why: the GPU expert cache is populated only by GPU fetches

Per-policy realized traffic over the measured window (`results/historical/split_policy/results/summary/summary_traffic.csv`):

| workload | policy | mean m | mean hit | miss rate | CPU MB/tok | H2D MB/tok |
|---|---|--:|--:|--:|--:|--:|
| low_miss | q\* / q=0 | 4.00 | 0.00 | 1.00 | 1270 | 0 |
| low_miss | q=25% | 2.28 | 1.72 | 0.57 | 709 | 16 |
| low_miss | q=50% | 0.87 | 3.13 | 0.22 | 215 | 63 |
| high_miss | q\* / q=0 | 4.00 | 0.00 | 1.00 | 1270 | 0 |
| high_miss | q=25% | 2.93 | 1.07 | 0.73 | 858 | 74 |

The hybrid cache admits an expert only when it is fetched to the GPU. With q\*=0, nothing is
ever fetched, so the cache **never populates**: every token misses all 4 experts (miss rate
1.00, occupancy 0) and every expert is recomputed on the CPU, in every workload -- which is
why q\* and q=0 are insensitive to cache size. A fixed q=25-50% fetches a few experts each
step; those become cache hits on later tokens, which lowers both the CPU load *and* the
future miss count. The slow PCIe transfers are an investment that pays back as hits.

This is the mechanism q\*'s formula does not model: `q* = m B_P/B_H` is a per-step,
instantaneous bandwidth balance and is blind to the cross-step cache dynamics.

## 6. Branch balance (`results/historical/split_policy/figures/plot6`, `plot2`, `plot3`)

Per-token branch times from realized traffic / measured (overlapped) bandwidths
(`results/historical/split_policy/results/summary/summary_balance.csv`):

| workload | policy | T_CPU ms | T_GPU ms | max(T_CPU,T_GPU) | imbalance |
|---|---|--:|--:|--:|--:|
| low_miss | q\* / q=0 | 35.6 | 0.0 | 35.6 | 1.00 |
| low_miss | q=25% | 19.9 | 2.3 | 19.9 | 0.88 |
| low_miss | q=50% | 6.0 | 9.4 | **9.4** | **0.36** |
| med_miss | q=25% | 20.6 | 2.7 | **20.6** | 0.87 |
| high_miss | q=25% | 24.1 | 11.0 | **24.1** | 0.54 |

- **q\* does not balance the branches.** It runs everything on the CPU (T_GPU=0), so its
  imbalance is 1.00 -- the most imbalanced policy tested.
- The policy with the smallest realized `max(T_CPU,T_GPU)` is the fastest end-to-end in
  every workload (low_miss: q=50%, max 9.4 ms; med/high: q=25%). q\* never achieves the
  minimum max.

Note the caveat: the isolated per-(m,q) micro-benchmark (`branch_oracle`, section 7) uses a
*fixed* miss set and reports that q=0 minimizes the two-branch max and matches the oracle
exactly -- because per single step, with a slow PCIe, all-CPU is indeed cheapest. The
end-to-end picture reverses only once cache population across steps is included.

## 7. q\* vs the static oracle (`results/historical/split_policy/figures/plot7`)

`branch_oracle` swept q=0..m for each m with a fixed synthetic miss set (no cache dynamics),
40 repeats. Per-expert isolated costs: CPU 0.48 ms (~28 GB/s), PCIe 2.0 ms (~6.6 GB/s).
For every m the minimum-latency q is 0 and `q* = q_oracle = 0` (latency penalty 0.0). So
against the model q\* actually optimizes -- a single decode step's two offload branches --
q\* is exactly right. Its limitation is that this model is not the end-to-end objective.

## 8. Answers to the stated questions

- **Does q\* balance CPU/GPU execution?** No. On this host q\*=q=0; the GPU branch is idle
  (imbalance 1.00). It minimizes a single step's two-branch max, not the branch balance.
- **Does q\* improve CPU bandwidth utilization?** It maximizes CPU *use* (all misses on the
  CPU, 1270 MB/tok, ~41% host CPU utilization vs 18-34% for fetching policies), but that is
  not efficiency: the CPU repeats the same experts every token because the cache stays cold.
- **Do fixed q perform better?** Yes, on every workload here. q=25% is a robust winner
  (1.2-1.7x q\*); q=50% is best when the cache is large. q=75% and q=m are worse than q\*.
- **Sensitivity to B_P/B_H:** high. The 5.7x ratio is what drives q\* to ~0. The formula is
  a direct function of the ratio, so a faster PCIe (smaller ratio) would raise q\* and this
  gap would shrink; on this PCIe-starved host the ratio pins q\* at the CPU-only corner.
- **Does measured runtime agree with the theoretical q\* model?** For the per-step branch
  model it optimizes, yes (q\*=q_oracle, penalty 0). For end-to-end decode throughput, no --
  q\* is 1.2-2.1x slower than the best fixed split.
- **Where q\* fails / is suboptimal:** whenever GPU-cache population matters, i.e. any
  sustained decode where fetched experts would be reused. By choosing q~=0, q\* forfeits all
  cache warming and holds a permanent 100% miss rate.

## 9. Conclusion (objective)

On this hardware and workload, FreeToken's q\* policy is **correct for the sub-problem it
models** (a single decode step's PCIe-vs-CPU branch split, where it matches the measured
oracle) but is **not the throughput optimum for end-to-end bs=1 decode**: it is 1.2-2.1x
slower than a fixed q=25-50% split across low/medium/high miss regimes. The gap comes from a
factor outside the q\* formula -- the hybrid GPU expert cache is filled only by fetches, so a
zero-fetch policy never benefits from cache hits. q\* is useful where PCIe is fast relative
to CPU expert compute (larger B_P/B_H, so q\* is not driven to 0) and where cache reuse is
low; it is exposed here because B_H/B_P is 5.7x and reuse is high. These observations are
specific to this one machine, model, and decode setting and should not be generalized
without repeating the measurement on other hardware and geometries.

## 10. Reproduce / artifacts

See the reproduction section below for commands. Raw data: `results/historical/split_policy/results/e2e/{perf,trace}/`, `results/historical/split_policy/results/branch/`,
`results/historical/split_policy/results/calibration/`. Summaries: `results/summary/{summary_perf, summary_traffic,
summary_balance, summary_q_by_m, correctness}.csv` and `tables.md`. Figures:
`results/historical/split_policy/figures/plot1..8.png`. Single code change: the `FREETOKEN_MOE_FORCE_FETCH_FRACTION` hook in
`python/freetoken/engine/engine.py::_resolve_hybrid_fetch` (inactive unless the env var is
set; reuses the native `hybrid_fetch_fraction` path).


## Reproduction and artifact map

Execute commands from the repository root unless a `cd` is shown. Use `uv`; give reruns fresh output locations. Recorded results below are historical evidence.

Controlled study of FreeToken's bandwidth-adaptive `q*` decode policy against fixed
CPU/GPU expert-execution splits, on one MoE model and one machine. Only the q-selection
policy changes between runs; model, quantization, cache size, KV cache, prompt, decode
length, seed, and backend (`hybrid`) are held constant.

## Research question

FreeToken's hybrid decode splits each layer's `m` cache-missing experts into `q` fetched
to the GPU over PCIe and `m-q` computed on the CPU, concurrently. The adaptive policy sets
`q* = round(m * B_P / B_H)` where `B_P` is PCIe expert-transfer bandwidth and `B_H` is CPU
expert-processing bandwidth. We measure what happens to CPU/GPU branch balance, CPU-side
traffic, and decode performance when `q*` is replaced by fixed splits.

## Model and hardware

- Model: **gpt-oss-20b**, MXFP4 experts (`mxfp4_triton`), L=24 MoE layers, E=32 experts/layer,
  top_k=4, expert bank entry 12.62 MiB (13.24 MB decimal). FTW: `~/.freetoken/models/gpt-oss-20b-ftw`.
  The historical BF16 Qwen expert-bank configuration did not fit this host. The later local Qwen NVFP4 study qualified a different representation; this is not a blanket Qwen support blocker.
- Host: RTX 4070 (12 GB), AMD Ryzen 5 8500G (6c/12t), 30 GB RAM, PCIe gather ~6.7 GB/s,
  CPU MoE ~38 GB/s (see `results/historical/split_policy/results/calibration/`).

## Policies (backend = hybrid)

| Policy | q | how it is forced |
|---|---|---|
| A_adaptive | `round(m*B_P/B_H)` | `--hybrid-fetch -1` (engine reads the benched fraction) |
| B_gpu | `m` | `--hybrid-fetch -1` + `FREETOKEN_MOE_FORCE_FETCH_FRACTION=1.0` |
| C_cpu | `0` | `--hybrid-fetch 0` |
| D_fixed25 | `round(0.25 m)` | `--hybrid-fetch -1` + `FORCE_FETCH_FRACTION=0.25` |
| E_fixed50 | `round(0.50 m)` | `--hybrid-fetch -1` + `FORCE_FETCH_FRACTION=0.50` |
| F_fixed75 | `round(0.75 m)` | `--hybrid-fetch -1` + `FORCE_FETCH_FRACTION=0.75` |

The fixed-fraction policies reuse FreeToken's native `hybrid_fetch_fraction` execution path;
only the fraction value differs from `q*`. The single code change is an experiment hook in
`engine.py::_resolve_hybrid_fetch` that honors `FREETOKEN_MOE_FORCE_FETCH_FRACTION` when set
(inactive otherwise, so default behavior is unchanged).

Workloads vary miss pressure via the GPU expert-cache size (`--cache-rate`, slots as a
fraction of L*E): `low_miss`=0.40, `med_miss`=0.20, `high_miss`=0.10. Every policy sees the
same three cache sizes; each run starts from a cold cache warmed by one standardized warmup
generate (fresh server per run).

## Reproduce

```bash
source scripts/env.sh            # CUDA 13 environment; export MODEL separately

# 1. Calibrate B_P / B_H (also caches the profile the adaptive q* reads)
uv run --no-sync ft bench bw --model gpt-oss-20b --dtype mxfp4_triton --gpu 0 \
  -o results/historical/split_policy/results/calibration/benchbw_gpt-oss-20b.json

# 2. Run the full policy x workload matrix (fresh server per cell; resumable)
uv run --no-sync python experiments/split_policy/run_experiment.py \
  --perf-decode 128 --perf-repeats 3 --trace-decode 64 --gpu 0

# 3. Isolated CPU vs PCIe branch timings for the balance/oracle analysis (needs the GPU idle)
uv run --no-sync python experiments/bandwidth/branch_oracle.py --model gpt-oss-20b --fraction 0.159 \
  --misses 1 2 3 4 --repeats 40 --out results/historical/split_policy/results/branch/branch_oracle.csv

# 4. Aggregate + plot + tables
uv run --no-sync python -m experiments.split_policy.analyze            # -> results/summary/{perf,traffic,q_by_m,correctness}.csv
uv run --no-sync python -m experiments.split_policy.branch_analyze     # -> results/summary/summary_balance.csv
uv run --no-sync python -m experiments.split_policy.report             # -> results/summary/tables.md
uv run --no-sync python -m experiments.split_policy.plots              # -> figures/plot1..8.png
```

## Outputs

- `results/historical/split_policy/results/calibration/` benched B_P, B_H, adaptive fetch fraction.
- `results/historical/split_policy/results/e2e/perf/<workload>/<policy>.json` throughput/latency (CUDA graph, 3 repeats).
- `results/historical/split_policy/results/e2e/trace/<workload>/<policy>.trace.csv` per-layer routing/traffic over the
  measured decode window; `.monitor.csv` CPU/GPU utilization samples.
- `results/historical/split_policy/results/branch/` isolated per-(m,q) CPU and PCIe branch times + oracle q.
- `results/historical/split_policy/results/summary/` aggregated CSVs and `tables.md`.
- `results/historical/split_policy/figures/plot1..8.png` the eight required plots.
- `report.md` the written report.

## Runtime-aware extension

See [the separate runtime-q report](../runtime_q/report.md). Its eager-mode absolute timings differ from this study.


[Master experiment log](../../EXPERIMENT_LOG.md) · [Documentation index](../../README.md).
