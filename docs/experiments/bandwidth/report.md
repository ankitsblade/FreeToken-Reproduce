# Qwen/GPT-OSS bandwidth, traffic, geometry and q-oracle study

**Provenance:** local workspace reproduction; finite supported study closed by 30 September 2026. Artifact paths in code spans are repository-root relative. This report consolidates the former report, findings, completion and handoff documents. Historical test counts are recorded evidence; no tests were rerun during cleanup.

Report date: 2026-09-30.

This report consolidates the experiments, discoveries, unsuccessful attempts,
limitations, and recommended follow-up work from the
[experimental plan](../../PLAN.md).
It describes completed measurements, not new experiment runs.
Exact commands and preservation notes are in [reproduction.md](reproduction.md).
The [experiment log](../../EXPERIMENT_LOG.md) records the history; this report
contains the consolidated findings and final phase dispositions.

## 1. Executive summary

The study supports FreeToken's bandwidth-adaptive CPU/GPU split on the tested
host, but does not establish a new policy or a new end-to-end speedup.

The most important discoveries are:

1. Both quantized expert layouts reach approximately 6.72 GB/s on this host.
   The historical Qwen 22.04 GB/s versus GPT-OSS 8.87 GB/s comparison used
   different hardware and representations, so it cannot isolate expert-count
   effects. The same-host results support a platform-capacity explanation,
   without identifying every historical confounder.
2. Fewer routed experts do not imply less memory demand. GPT-OSS experts are
   much larger. In the sampled hybrid trace, GPT-OSS used no expert H2D fills
   but had approximately 1.27 GB/token of nominal CPU-executed expert weights,
   versus 239 MB/token for Qwen. These are accounting estimates, not DRAM
   hardware-counter measurements.
3. The existing split matches the branch-only oracle at every tested miss
   count. Controlled full-layer replay identifies Qwen's five-miss case as
   a state-dependent crossover, not a consistent model-wide policy failure.
4. Memory allocation matters beyond expert-cache capacity. Shrinking Qwen's
   expert cache did not fix long-prefill OOM because automatic KV allocation
   consumed the released space. Reserving more headroom did allow the tested
   approximately 8K-token prefill requests to complete.
5. Expert popularity does not generalize reliably across the sampled
   workloads. Offline access coverage does not justify actual expert pruning.

The supported finite experiment set is complete. Unsupported memory settings,
unmeasured hardware counters, model-quality evaluation, and broader workload
generalization are not claimed as completed.

## 2. Environment and model scope

| Item | Measured configuration |
|---|---|
| GPU | NVIDIA RTX 4070, 12,282 MiB VRAM |
| CPU | AMD Ryzen 5 8500G, six physical cores |
| System RAM | Approximately 32 GB installed |
| NVIDIA driver | 595.84 |
| PCIe under transfer load | Gen4 x4; approximate one-way signaling ceiling 7.876 GB/s |
| Workspace compiler | CUDA 13.0.88; system-default `nvcc` remained CUDA 12.0 |
| PyTorch | 2.11.0 with CUDA 13.0 runtime |
| Recorded repository HEAD | `4d0f84983700eba7dfb046fb4df3cbbcc488ecc2`, with local study changes |

The HEAD alone is insufficient to reproduce the study: the local scripts,
instrumentation, environment, and result artifacts must also be preserved.

| Model property | Qwen | GPT-OSS |
|---|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `openai/gpt-oss-20b` |
| Revision | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| Local directory | `results/checkpoints/qwen3.6-35b-nvfp4` | `results/checkpoints/gpt-oss-20b` |
| Expert representation | NVFP4 | MXFP4 |
| MoE layers | 40 | 24 |
| Experts per layer | 256 | 32 |
| Routed experts per layer, top-k | 8 | 4 |
| Complete expert row bytes across six banks | 1,775,616 | 13,236,480 |

Qwen BF16 was not the local benchmark target: its expert bank alone exceeded
host RAM. The quantized Qwen checkpoint enabled feasible runs on this machine.
Consequently, these results are not a BF16 reproduction of the historical setup.

## 3. Experiment inventory and plan disposition

| Phase | Experiments performed | Disposition |
|---|---|---|
| Setup | System snapshot, loaded PCIe probe, checkpoint validation, CUDA compile/import smoke | Completed for this host |
| 0: baseline | Five calibration repeats; real-model smoke; repeated backend/overlap decode; prompt-length and cache-capacity sweeps | Feasible configurations completed; memory failures retained |
| 1: transfer | Contiguous pinned copies, six-bank copies, production indexed gather, concurrent CPU/gather calibration | Completed |
| 2: traffic | Real decode traces with routed experts, misses, fills, CPU executions, bytes, and workload/cache comparisons | Completed for sampled requests |
| 3: geometry | Synthetic and measured-route counterfactuals varying expert count, top-k, locality, and cache normalization | Completed as replay, not altered-model inference |
| 4: split oracle | Forced-q diagnostics, branch-only sweeps, restored-state full-layer replay, independent replication, CPU timing, Nsight attribution | Completed for the stated finite coverage |
| 5: revised policy | Evaluate whether a stable policy error warrants implementation | Not activated: prerequisite not met |
| 6: optional pruning | Temporal held-out and cross-workload expert-popularity/access-coverage analysis | Offline exploration completed; no actual pruning |

Here, `m` is the number of unique expert misses at a layer and `q` is the
number of those misses selected for GPU filling/execution. `q*` denotes the
runtime's predicted split. An oracle is the fastest tested split under the
specified measurement procedure, not a proof of global optimality.

## 4. Baseline, calibration, and transfer discoveries

### 4.1 Same-host bandwidth capability

Five repeated production-kernel calibrations gave these medians:

| Calibration metric | Qwen NVFP4 | GPT-OSS MXFP4 |
|---|---:|---:|
| Production gather, GB/s | 6.72 | 6.72 |
| Isolated CPU MoE effective throughput, GB/s | 36.69 | 37.42 |
| Concurrent gather, GB/s | 6.71 | 6.71 |
| Concurrent CPU MoE effective throughput, GB/s | 33.09 | 34.29 |
| Runtime calibrated fetch fraction | 0.1674 | 0.1635 |

All five calibrations recommended hybrid execution for both layouts. CPU
throughput here uses logical expert bytes, not measured host-memory traffic.
Gather calibration values were rounded to two decimal places by the tool.

Contiguous pinned copies used ten repetitions per point. One-expert transfers
reached 6.617 GB/s for Qwen and 6.712 GB/s for GPT-OSS; sixteen-expert transfers
reached 6.720 and 6.723 GB/s respectively.

Splitting a transfer into six pinned banks imposed a modest small-transfer
cost: Qwen's one-expert bandwidth was 6.613 GB/s contiguous versus 6.450 GB/s
banked, a 2.5% latency increase. GPT-OSS's corresponding latency increase was
0.4%. At sixteen experts, these penalties fell to approximately 0.2% and 0.03%.
The banked copy proxy is not the production indexed-gather kernel.

Discovery: transfer granularity can affect small transfers, but both shapes
approach the same link capacity. Expert count alone does not explain the
historical raw-bandwidth difference.

Evidence: `results/transfer_microbench/` and
`results/baseline/calibration_repeat5_summary.csv`.

### 4.2 Repeated end-to-end decode baseline

At 20% expert cache, five repetitions of 256-token generation per cell gave
the following median decode throughput with prefill overlap enabled:

| Backend | Qwen tokens/s | GPT-OSS tokens/s |
|---|---:|---:|
| Auto | 67.18 | 23.77 |
| Hybrid | 66.24 | 23.83 |
| Offload | 29.65 | 10.67 |

With overlap disabled, auto/hybrid medians were 61.45/62.33 tokens/s for Qwen
and 24.92/24.65 for GPT-OSS. Overlap is a prefill feature; these decode
differences are not proof that it universally improves decode speed.

These runs used checkpoint-recommended sampling. Output hashes were stable
within cells, but not necessarily identical across backends or overlap
settings. The table is a baseline comparison, not a token-identical paired
policy A/B test. Earlier 64-token smoke runs are retained separately and
should not be pooled with the longer repeated baseline.

Evidence: `results/baseline/*cache20_decode256_repeat5_summary.csv`.

### 4.3 Prefill and the memory-budget interaction

Fresh-prompt sweeps used naive KV caching, three repetitions, and up to eight
output tokens. At approximately 7.9K prompt tokens:

| Model and allocation | Median TTFT, overlap on | Median TTFT, overlap off |
|---|---:|---:|
| Qwen, 80% GPU memory target, 20% expert cache | 3.504 s | 6.049 s |
| GPT-OSS, tested baseline allocation | 1.83 s | 3.17 s |

Qwen's original 90% memory-target setting failed at this prompt length. Even
reducing the expert cache from 20% to 10% and 5% did not fix the failure:
automatic KV allocation expanded from 3.78 GiB to 5.46 and 6.31 GiB, leaving
approximately 0.88 GiB free after initialization. The 80% memory target
provided enough headroom for the tested approximately 8K sweep.

Discovery: expert-cache, KV-cache, and temporary-allocation budgets must be
considered jointly. A smaller expert cache does not automatically mean more
scratch headroom.

TTFT means first visible text here, not necessarily first raw generated token.
Results from the 80% and 90% allocation settings remain separate.

Evidence: `results/baseline/*prefill*` and the phase-specific summaries cited here.

## 5. Real traffic and cache behavior

The core sampled 20%-cache hybrid traces reported:

| Metric per generated token | Qwen | GPT-OSS |
|---|---:|---:|
| Expert evaluations | 320 | 96 |
| Expert misses | 145.02 | 96 |
| GPU expert fills | 10.29 | 0 |
| CPU expert executions | 134.73 | 96 |
| Expert H2D bytes | 18.26 MB | 0 |
| Nominal CPU-executed expert bytes | 239 MB | 1.27 GB |

GPT-OSS's approximately 16.3% fetch fraction rounds to zero GPU fills when
there are four misses. Thus zero expert H2D traffic means work is placed on
the CPU, not that expert evaluation has no memory cost.

Separate matched 63-token greedy cache-capacity traces showed:

- Qwen at 10%, 20%, and 25% cache: 175.9, 143.7, and 143.7 misses/token;
  corresponding expert H2D traffic was 32.1, 16.9, and 16.9 MB/token.
- GPT-OSS at 10%, 20%, and 40% cache: 96 misses/token and no GPU fills at every
  setting. The observed hybrid cache remained empty.

Discovery: allocated capacity and useful residency are different quantities.
Qwen's 20%-to-25% plateau is specific to this short route sequence. GPT-OSS's
empty cache suggests a follow-up question about admission over multiple tokens,
but does not establish that forcing fills would improve performance.

Evidence: `results/baseline/model_comparison.csv` and `results/traffic_trace/`.

## 6. Controlled geometry experiments

Synthetic uniform-routing replay exercised the accounting path. Additional
measured-route replays varied total expert count, top-k, relative capacity,
raw cache slots, byte capacity, and locality-related demand.

Reduced expert counts were simulated by modulo-folding observed expert IDs.
These are controlled counterfactuals, not inference from retrained or genuinely
smaller checkpoints. Folding changes route collisions and locality, so its
miss-rate curves cannot establish the causal effect of reducing model experts.

The calibrated top-4 replay produced no fills, making its cache-capacity curves
uninformative about eviction. An explicitly counterfactual 40% fetch-fraction
replay produced fills and demonstrated that fixed raw slots and fixed relative
capacity are different comparisons.

Discovery: geometry comparisons require explicit byte and capacity accounting.
Expert count, expert size, top-k, route locality, and admission behavior cannot
be collapsed into a single "number of experts" explanation.

Evidence: `results/geometry_replay/` and its `counterfactual/` subdirectory.

## 7. Split-policy validation and attribution

### 7.1 Diagnostic sweeps and why restored-state replay was needed

Initial whole-request forced-q sweeps synchronized each layer. GPT-OSS's
observed four-miss case favored predicted q=0; Qwen's eight-miss case favored
predicted q=1. Larger forced fills were often slower.

However, changing q changed cache evolution. Warmup and post-hoc route matching
did not reliably produce identical states across runs, particularly for Qwen's
four-miss case. These diagnostic timings therefore could not provide a clean
counterfactual oracle for every miss count.

The subsequent production-layer replay captured actual activations, routing,
cache bookkeeping, and overwritten expert-bank rows. It restored the captured
state before each q trial, randomized trial order, and checked realized misses
and fills. Routing was frozen for the replay; it was not a whole-model serving
benchmark.

### 7.2 Branch-only and full-layer results

Branch-only sweeps tested every q at Qwen m=1..8 and GPT-OSS m=1..4 with
40 repetitions per cell. The predicted q matched the minimum-median split at
every tested miss count. These sweeps omitted cache admission and GPU expert
GEMM, so full-layer validation was still necessary.

The final audit consolidates 36 captured states and 8,000 measured q trials.
Full-layer coverage includes Qwen m=1..8 and GPT-OSS's observed m=4 states.
GPT-OSS m=1..3 were tested only in branch microbenchmarks. At m=0 there is no
split choice.

| CPU-untimed full-layer group | Result |
|---|---|
| Qwen m=1 | Prediction matched the one captured-state oracle |
| Qwen m=2 | Prediction matched both captured-state oracles |
| Qwen m=3 | Prediction matched both captured-state oracles |
| Qwen m=4 | Prediction matched all five captured-state oracles |
| Qwen m=5 | Prediction matched 7 of 15 captures across five cases and three runs |
| Qwen m=6, 7, 8 | Prediction matched the one captured-state oracle at each m |
| GPT-OSS m=4 | Prediction matched all three captured-state oracles |

Across the fifteen CPU-untimed Qwen m=5 captures, median predicted-q latency
penalty was approximately 0.45%, with a maximum of 7.53%. These are descriptive
state-level results, not traffic-weighted model-wide penalties.

Independent replication found that early coding consistently favored q=0,
but its penalty varied from 0.45% to 3.35%, and one paired interval included
zero. Later coding, early incident, and math changed median winners across
runs. Later incident consistently favored q=1.

Discovery: five misses lie near a CPU/gather crossover. The evidence does not
justify a blanket m=5 override or a revised geometry-wide policy. Phase 5 was
therefore not activated; this is a decision supported by the measurements,
not a missing implementation.

The oracle uses marginal median latency. Paired bootstrap intervals measure
within-repeat differences and need not agree in sign with the difference
between marginal medians. Matching case-selection metadata across independent
processes does not guarantee identical activations or full cache snapshots.

### 7.3 Native CPU timing

Five separately instrumented Qwen m=5 states contributed 1,200 trials.
Timing covered native CPU task preparation, dispatch, and worker computation;
it excluded coordinator waiting and activation/result transfer overhead.

- q=0 median CPU task durations: 0.304-0.355 ms.
- q=1 median CPU task durations: 0.246-0.294 ms.
- q=1 gather intervals: 0.276-0.290 ms.
- q=0 logical expert throughput: 25.0-29.2 GB/s, below isolated calibration.

These measurements place the branches close to the crossover. Timing was
opt-in and is off by default. Instrumented results remain separate from the
CPU-untimed oracle because instrumentation can affect latency.

### 7.4 Nsight Systems attribution

Nsight Systems 2025.3.2 successfully captured a real Qwen m=5 replay and
exported SQLite data after older tools failed with the CUDA environment.
The stage summary retained ten measured windows per q and excluded warmup.

Median active expert-GEMM kernel time stayed near 0.039 ms across q=0..5.
The gather kernel rose from approximately 0.0014 ms at q=0 to 0.270 ms at
q=1 and 1.328 ms at q=5. CUDA-kernel active fraction increased from
approximately 9.0% to 86.1%.

Discovery: higher CUDA activity did not itself mean a better split. The
production gather reads pinned host memory inside a CUDA kernel, so expert
traffic is not represented by ordinary H2D memcpy events.

These are profiled kernel timelines, not SM occupancy or DRAM counters.
Profiled latencies must not be pooled with unprofiled oracle timings, and
overlapping branch medians cannot be added to reconstruct layer latency.

Evidence: `results/final_study/`, `results/qstar_oracle/`, and
`results/qstar_oracle/nsys2025_qwen_m5_stages.csv`.

## 8. Optional popularity and pruning exploration

Temporal held-out analysis ranked experts on the first half of one short math
request and measured routed-access coverage on the second half:

| Expert retention | Qwen held-out coverage | GPT-OSS held-out coverage |
|---|---:|---:|
| 25% | 60.3% | 70.6% |
| 50% | 76.3% | 89.8% |

At 25% retention, Qwen's training access coverage was 96.4%, far above its
60.3% held-out coverage. Math-trained popularity also generalized poorly to
separate code and incident requests:

| Retention and evaluation workload | Qwen coverage | GPT-OSS coverage |
|---|---:|---:|
| 25%, code | 54.5% | 29.1% |
| 25%, incident | 51.0% | 23.2% |
| 50%, code | 76.3% | 63.4% |
| 50%, incident | 74.4% | 53.9% |

Discovery: popularity can be workload-specific, particularly in the sampled
GPT-OSS traces. Each workload was only one short request. No weights were
pruned, and coverage is not accuracy, perplexity, or evidence of safe removal.

Evidence: `results/pruning_optional/`.

## 9. Unsuccessful attempts and resolved blockers

| Attempt or limitation | Outcome and interpretation |
|---|---|
| Initial checkpoint transfer | A direct sample was approximately 0.113 MB/s; an Xet retry stalled. A later non-Xet path completed both pinned checkpoints and shard validation. |
| CUDA setup | Workspace-local CUDA 13 compiler/runtime and extensions were validated. The system compiler was not globally replaced. |
| Qwen 40% expert-cache baseline | All six backend/overlap cells failed before decode because the minimum KV pool could not fit on 12 GB. These are infeasible cells, not successful measurements. |
| Qwen approximately 8K prefill at 90% memory target | OOM, including smaller expert-cache retries. The 80% target completed the supported sweep. |
| Whole-request route matching | Did not consistently recover identical replay states. Controlled snapshot restoration replaced it for oracle evidence. |
| GPU unavailable to an earlier session | Device access was restored; subsequent GPU measurements and verification completed. |
| Older Nsight tools | CUDA trace/import failures; Nsight Systems 2025.3.2 later succeeded. |
| NVTX range-triggered profiling attempt | No report was generated; a full-capture retry succeeded. |
| Hardware DRAM/SM counters | Remain unmeasured under the available tools and profiling permissions. Timeline and logical-byte proxies are not substitutes. |

Failed runs and earlier diagnostic artifacts were retained. Failure counts
depend on which matrix or retry set is being discussed; this table does not
claim one global count of failures.

## 10. Validation, artifacts, and limits

The closeout audit validates complete q cells, repeat IDs, realized miss/fill
counts, 36 captured states, and 8,000 measured trials. All 36 included server
logs were archived. All 21 required plots are present, with supplementary
figures under `results/figures_final/`.

The recorded final CUDA-enabled MoE test run was:

```bash
uv run --no-sync pytest tests/moe -m 'not slow' -q
```

Result: 153 passed, 5 skipped, 6 deselected, and 11 dependency warnings in
18.11 seconds. The recorded closeout also passed `git diff --check`.
These are prior verification results, not tests rerun for this report.

Numerical comparisons found maximum absolute Qwen layer differences up to
0.001953125 in the historical controlled replays. Some GPT-OSS q>0 replays
had absolute differences up to 8.0. These values were not normalized by tensor
scale and are not evidence of either acceptable quality or a confirmed bug.
Stable request output hashes likewise do not replace model-quality evaluation.

Interpretation boundaries:

- One host and two quantized checkpoints; no BF16 or cross-hardware causal study.
- Small sampled math, code, and incident workloads; no population-wide claims.
- Primarily batch-one inference; no demonstrated multi-request scaling benefit.
- No actual host-DRAM traffic measurement or SM-occupancy measurement.
- No real pruning, held-out quality evaluation, or new runtime-policy A/B speedup.
- Restored-state layer replay is not uninstrumented end-to-end serving.

Artifact entry points:

- [Final audit](../../../results/final_study/artifact_audit.json).
- [Per-state summary](../../../results/final_study/state_summary.csv).
- [Per-case summary](../../../results/final_study/state_summary_by_case.csv).
- [Per-miss-count summary](../../../results/final_study/state_summary_by_m.csv).
- [Model comparison](../../../results/baseline/model_comparison.csv).
- [Reproduction commands](reproduction.md).
- [Session handoff](reproduction.md).

`results/` is ignored by Git. Preserve it separately, including raw data,
checkpoints, server logs, profiles, and figures, when transferring or archiving
the workspace. No runtime split-policy replacement or checkpoint modification
was made as a result of the study.

## 11. Unrun follow-up

Remaining directions are maintained in [the single experiment plan](../../PLAN.md). They are not additional completed measurements.

## 12. Overall verdict

This study is a successful bounded validation and bottleneck decomposition,
not an optimization breakthrough. It separates bandwidth capability from
traffic demand, explains the observed CPU/GPU placement, identifies the
five-miss crossover, and exposes practical memory-allocation constraints.

The next work should target joint memory budgeting, host execution efficiency,
and cache decisions across future tokens, rather than assume that fewer experts,
more GPU activity, forced fills, or pruning will improve inference.


## Canonical reading and evidence

[Reproduction and preservation notes](reproduction.md) · [Master experiment log](../../EXPERIMENT_LOG.md).

The final archive validates 36 controlled captures, 8,000 measured q trials and all 36 server logs. Preserve instrumented, uninstrumented, independent-run and profiler strata separately. Phase 5 was not activated, and Phase 6 measured offline access coverage only. Historical hardware-counter DRAM bandwidth and actual SM occupancy remain unmeasured. Exact superseded source text is in the documentation recovery archive.
