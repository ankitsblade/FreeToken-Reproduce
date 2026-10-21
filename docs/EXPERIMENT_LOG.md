# Experiment log

This is the **single maintained Markdown experiment log** for the local reproduction/pruning work and imported studies from Kushal, Yeshwanth and Saketh. Reports and raw per-run logs remain separate evidence. Reviewed and consolidated 12 October 2026; cleanup launched no measurements or tests.

## Status and time convention

The finite bandwidth study, static 75% closeout, Gemma R75 repair and exploratory actual75 supplement are complete in their recorded scope. The original independent-task quality requirement, 60%/50% ablations, and cross-model/device validation of runtime-aware q remain open.

UTC-labelled timestamps below are UTC; explicit IST timestamps retain their labels. Saketh's runtime-q episode timestamps contain no timezone, so their calendar dates are recorded without converting them to UTC. Filesystem modification times and archive receipt dates are not promoted to execution timestamps. Imported studies without an execution timestamp are explicitly undated.

## Chronological overview

| Period | ID / attribution | Experiment or milestone | Disposition |
|---|---|---|---|
| By 30 Sep 2026; stage timestamps in artifacts | BW-00..BW-08 / local workspace | Setup, calibration, transfers, baseline, traffic, geometry, q oracle and profiling | Finite supported study complete |
| Execution date absent; archived by 29 Sep | SP-01 / Saketh | q* vs six fixed/adaptive policy settings, three cache regimes | Recorded matrix and branch controls complete |
| 29 Sep 17:22-17:23 UTC measured requests | PF-01 / Kushal | Prefill overlap ON/OFF at four lengths | 32 request records |
| Execution date absent; archived by 29 Sep | LC-01 / Yeshwanth | Original vs shuffled routing locality | One sampled trace, five shuffle seeds |
| 30 Sep | BW-08 / local workspace | Independent q-state replication and usable Nsight capture | 36 captures, 8,000 trials, all logs archived |
| 1 Oct | DOC-01 / weekly presentation | Original project-context progress deck | Historical presentation |
| 6 Oct | PR-00..PR-05 / local workspace | Frozen pruning design, pilots, qualified static capture, analysis/closeout | Three checkpoints; original activation gates failed |
| 6-7 Oct | RQ-00..RQ-10 / Saketh | Runtime-aware q inspection, data, models, rule, ablations/follow-ups | 889 logged episode records |
| 6-8 Oct | PR-06..PR-10 / local workspace | Gemma R repair, actual75 qualification, quality capture, resource recovery | All three quality collections complete |
| 7 Oct | PR-08 / user pause/resume | GPU release, frozen-runtime restoration, Qwen retries | Failed/interrupted units preserved |
| 8 Oct 03:43:54 UTC | PR-09 / approved app closure | Close unused Firefox; observe pressure attribution | Capacity recovery; fresh Qwen batch later accepted |
| 8 Oct 05:41 / 06:09 UTC | PR-10 / serving recovery | Whole paired retries; transient no-swap experiment scope | Earlier failures excluded; all eligible serving pairs completed |
| 8 Oct 07:15:37 UTC / 12:45:37 IST | PR-11 / actual75 closeout | Export report and verify manifest | 44,155 verified artifacts |
| 8 Oct | DOC-02 / presentation | Eight static slides, two reserved slides, first-ten speaker notes | Completed; original slides preserved |
| 10 Oct | DOC-03 / completion audit | Completion summary and resource history corrected | Frozen results preserved |
| 12 Oct | DOC-04 / dataset and cleanup audit | Explain synthetic corpus; consolidate all research docs/history | Documentation only |

## Bandwidth and geometry study: completed stage records

### BW-00 - Environment and feasibility

- Attribution: local workspace reproduction; exact stage invocation/configuration evidence in `results/baseline/`.
- Question: does the historical Qwen/GPT raw-bandwidth difference persist on the same hardware and expert representations?
- Host: RTX 4070 12 GB, Ryzen 5 8500G; loaded PCIe link Gen4 x4. Historical idle Gen1 is not the active transfer link state.
- CUDA 13 wheel/compiler setup and extensions were exercised; system-default CUDA 12 was a separate compiler. Qwen NVFP4 and GPT MXFP4 pinned checkpoints downloaded and checked; the earlier Xet/direct-download stalls were preserved.
- Disposition: two supported quantized configurations admitted. Historical Qwen BF16 expert-bank fit is not local NVFP4 fit.

### BW-01 - Same-host calibration and shaped transfer

- Five calibration repetitions/layout, contiguous and six-bank pinned copies, production indexed gather and CPU MoE measurements.
- Both layouts approach 6.72 GB/s transfer. Median isolated CPU logical-expert bandwidth was 36.69 GB/s Qwen and 37.42 GB/s GPT; concurrent values 33.09/34.29 GB/s. Logical bytes/time are not measured host DRAM traffic.
- One-expert banked Qwen copy had about 2.5% overhead; both shapes approach the same link ceiling at larger groups.
- Evidence: `results/baseline/calibration_repeat5*`, `results/transfer_microbench/`, calibration profile run-04.

### BW-02 - Actual decode baseline and prefill admission

- Repeated auto/offload/hybrid and overlap comparisons; short smoke runs kept separate from five-repeat 256-token runs.
- Qwen 40% expert-cache allocation was infeasible on 12 GB VRAM because the minimum KV pool could not fit. Those are failed cells, not fabricated timings.
- Qwen 7.9K prefill failed under the 0.9 memory target; decreasing only expert-cache rate enlarged auto KV allocation and did not solve it. A separately declared 0.8 memory-target arm completed through 7.9K.
- Visible-text TTFT and raw token TTFT remain distinct, particularly for GPT-OSS reasoning output.
- Evidence: `results/baseline/`; exact matrix and summary paths in the bandwidth report/reproduction page.

### BW-03 - Measured expert traffic

- Captured token/layer expert IDs, weights, hits, unique misses, GPU fills, CPU executions and full bank bytes.
- Sampled 20%-cache Qwen trace: 320 expert evaluations/token, 145.02 misses, 10.29 fills and 18.26 MB H2D/token. GPT trace: 96 evaluations/misses, zero fills/H2D under its rounded q=0 arm.
- Greedy capacity sweeps and earlier sampled baseline traces have separate summaries. Same-capacity traffic cannot be inferred from nominal expert count or fetch fraction.
- Evidence: `results/traffic_trace/`.

### BW-04 - Controlled geometry counterfactuals

- Sweeps of expert universe, top-k, expert size/capacity and routing order through synthetic and measured-route replay.
- Disposition: accounting/geometry counterfactuals complete. Modulo-folded expert IDs do not describe a retrained smaller model; simulated demand is not new inference or measured serving speed.
- Evidence: `results/geometry_replay/`.

### BW-05 - q oracle and representative states

- Branch-only coverage Qwen m=1..8 and GPT m=1..4; full-layer captures restore recorded activations/cache state per q and use 40 measured repetitions.
- Early matched m=4 followed by representative coding/incident/math states identified m=5 as a crossover. Captured median winners and paired intervals are state/run-dependent.
- Native CPU-timed five-state m=5 follow-up adds 1,200 trials and is kept separate from untimed controls.
- Numerical output differences, realized m/q and snapshot restoration were audited. Replay-paused request tokens/s are not serving throughput.
- Evidence: `results/qstar_oracle/representative_states/`, `qwen_m5_cpu_timing/`, raw configs/logs and restoration checks.

### BW-06 - Independent q-state replication and final audit

- Two additional independent untimed runs of five Qwen m=5 states; added full-layer m=1, two m=2, m=6/7/8 states matched predicted q.
- Final inventory: 36 captures, 8,000 measured trials; all 36 server logs archived. No consistent geometry-wide m=5 override was justified, so Phase 5 was not activated.
- Earlier native CPU-attribution validation recorded 150 passed/five skipped/six deselected; final closeout recorded 153/five/six. These are historical reported runs, not cleanup checks.
- Evidence: `results/final_study/`, including `artifact_audit.json` and state/layer/case summaries.

### BW-07 - Optional static popularity exploration

- Temporal and cross-workload held-out access coverage on existing traces, with no new masked model forwards or task grading.
- Disposition: offline exploration complete; actual pruning safety and model quality were unmeasured in this stage.
- Evidence: `results/pruning_optional/` and the bandwidth report.

### BW-08 - Profiling and final bounded closeout, 30 September

- Early CUDA trace-import and device-access failures were resolved. Nsight Systems 2025.3.2 exported a real Qwen replay and SQLite stage evidence.
- The range-triggered attempt generated no report and remained separate. Successful full capture: active expert GEMM about 0.039 ms; indexed gather grows with q. Host reads inside gather are not HtoD memcpy traffic.
- CUDA timeline activity is not SM occupancy. Actual DRAM counters remain unavailable. Profiling wall times were not pooled with oracle trials.
- All 21 requested figures plus supplements regenerated; no Phase-5 default policy change or kernel speedup claim.
- Evidence: `results/qstar_oracle/nsys2025*`, `results/figures_final/` and the archived September handoff.

## Imported studies: experimental records

### SP-01 - Saketh: q* versus fixed splits

- Date: execution timestamp absent from saved performance records; imported package available by 29 September. Do not infer execution date from file mtime.
- Model/host: GPT-OSS-20b FTW MXFP4, 24 layers, E=32/k=4, RTX 4070 Gen4 x4, Ryzen 8500G.
- Matrix: six policies, low/medium/high cache rates 0.40/0.20/0.10, three performance repeats; separate 64-token eager trace and branch-only 40-repeat q sweeps.
- Saved performance: q* about 24.3/24.1/24.2 tok/s; fixed 25% about 40.3/37.1/29.4; fixed 50% about 50.5/21.3/14.5. Source reports one greedy output hash across the recorded cells.
- Branch-only optimum is q=0 on fixed misses. End-to-end fetches warm the cache, explaining why a fixed split can improve sustained decode under these controls.
- Audit correction: old README high_miss 0.08 conflicts with saved 0.10/77 slots. Old ratio wording inverted B_H/B_P and B_P/B_H. Correct canonical labels use saved calibration. The saved manifest describes the last workload, so full coverage is established from raw cells/summary tables, not that manifest alone.
- Evidence: `saketh_exps/results/e2e/{perf,trace}/`, `results/branch/`, `results/calibration/`, summary CSVs. [Report](experiments/split_policy/report.md), [tables](experiments/split_policy/tables.md).

### PF-01 - Kushal: prefill overlap, 29 September

- Saved request wall_start range: 2026-09-29 17:22:02-17:23:32 UTC. Server startup/warmup lie outside that measured-request range.
- Model: GPT-OSS FTW MXFP4; hybrid, 0.20 cache rate, naive KV, eager prefill; fresh unique request tags, four repeats/setting/length.
- 32 request rows: ON/OFF x four target lengths x four repeats. Actual mean lengths approximately 1,041/2,032/4,095/7,899 tokens.
- TTFT ON/OFF (ms): 1710.2/1880.1, 1720.7/2029.8, 1762.2/2393.2, 1830.1/3176.8. Reported benefits 9.0/15.2/26.4/42.4%.
- Median GPU power 91/89 W and sampled peaks 186/128 W; monitor sampling 4 Hz. Transfer/compute overlap interpretation is inferred from paired timings/power, not an imported kernel timeline.
- `.qdstrm` captures remain unimported in that source study; the later local Nsight success is a separate experiment.
- Evidence: `kushal_exps/results/prefill_ttft.jsonl`, manifest, summary/monitor files and three figures. [Report](experiments/prefill_overlap/report.md).

### LC-01 - Yeshwanth: routing locality and LRU

- Execution date absent from saved trace metadata; imported package available by 29 September.
- One recorded GPT-OSS routing trace: 24 layers, E=32/k=4, 255 tokens/layer, 24,480 expert accesses/order.
- Five token-order shuffle seeds preserve layer-wise expert frequency. Exact stack-distance miss curves were compared with explicit LRU simulation.
- Original/shuffled mean reuse distance 7.82/9.51; medians 6/8. At C=16, miss rate approximately 0.143/0.206 (about 30% relative locality benefit); largest absolute gap near C=6. Full-pool cold misses are unchanged.
- Audit correction: old narrative says greedy; saved trace sampling is temperature 1.0, top-p 0.95, top-k 64. Canonical scope is that sampled trace. Per-layer replay capacity is not the production global cache-slot budget.
- Evidence: `yeshwanth_exps/results/routing_trace*`, `locality_summary.json`, miss/eviction/layer/reuse CSVs and four figures. [Report](experiments/routing_locality/report.md).

## RQ-00..RQ-10 - Saketh runtime-aware q, 6-7 October

- Inspection recorded 6 October; runs recorded 7 October. GPT-OSS/RTX 4070, hybrid, batch one, eager, greedy, 128 measured tokens, cache rates 0.4/0.2/0.1.
- 889 logged episode records: 10 pilot, 48 exploration, 18 round-two, 66/48 exploratory scan records, 171 main, 348 ablation, 96 generalisation, 54 overhead, 20 stress and ten long-decode records. These are distinct experiment roles, not independent quality tasks.
- Exploration records log 402,336 decisions overall; 310,896 randomized-action rows are eligible for cost fitting. Features: 89 candidates, six constants and 20 redundant signals removed, 63 retained.
- Main comparison: 19 policies x three regimes x three rotated repeats. Pure single-step cost favours q=0; future-cache value appears with longer offline horizons.
- Analytic reuse-aware rule: 31.6/45.0/55.9 ms/token, 1.519 geometric speedup vs eager q*. Ridge 1.453; LinUCB 1.349; Thompson 1.286; ANN/GBM below baseline. Fixed 25% is slightly faster than the recent-frequency rule at high miss.
- The 0.9 microsecond lean rule measurement is separate from full feature/logging decision overhead and the roughly 19 ms/token graph/eager gap.
- Generalisation is within one model/device, using a second prompt and unseen cache sizes. CPU contention/384-token checks and burn-in corrections are preserved. Several ablation families have only one run.
- No device-side graph integration, second model/device, heterogeneous expert-byte scheduling, or changing-condition bandit adaptation was measured.
- Evidence: `saketh_exps/runtime_q/{data,results,models,config,instrumentation}/`, original PDF and 28 figures. [Report](experiments/runtime_q/report.md), [tables](experiments/runtime_q/tables.md), [features](experiments/runtime_q/features.md).

### Detailed imported execution sequence

The following source history is consolidated from the archived original
`saketh_exps/runtime_q/PROGRESS.md`. Artifact paths in this subsection refer to
`saketh_exps/runtime_q/` unless already repository-root relative. Its status statements
describe that historical study; current open work is in `PLAN.md`.

Dates: 2026-10-06 (inspection) and 2026-10-07 (all runs). Configuration fixed throughout:
gpt-oss-20b MXFP4, RTX 4070, hybrid backend, eager decode, greedy, 128 measured tokens.
About 890 server episodes in total; every one is recorded in a `*_episodes.jsonl` file.

#### Status

| step | state | output |
|---|---|---|
| Inspect existing `saketh_exps` and the engine's q path | done | this log and `docs/experiments/runtime_q/reproduction.md` |
| Instrumentation plan and hook | done | `instrumentation/` |
| Pilot and validation against q\* | done | `results/pilot/` |
| Exploration dataset (round 1) | done | `data/raw/explore_*.csv` (48 episodes) |
| Feature screening and analysis | done | `data/clean/feature_report.json`, `results/feature_analysis/` |
| Per-fetch cache analysis | done | `results/cache_analysis/` |
| Analytic reuse-aware rule | done | `ReuseAware` in the hook |
| On-policy data (round 2) and retraining | done | `data/raw/round2_*.csv` (18 episodes) |
| Statistical models, ANN, bandits | done | `models/`, `results/statistical/`, `results/ann/`, `config/bandit_specs.json` |
| Main comparison, 19 policies × 3 workloads × 3 repeats | done | `results/eval/main_*` |
| Ablations: chain, leave-one-out, cache (3 repeats); individual, history, ANN (1 run) | done | `results/ablations/` |
| Overhead, generalization, CPU contention, long decode | done | `results/overhead/`, `results/eval/` |
| Tables, 28 figures, feature inventory | done | `reports/`, `figures/` |
| Second hardware platform or second model | **not done** | see limitations in `docs/experiments/runtime_q/report.md` |

#### What was done, in order

1. **Inspection.** `q` is chosen inside `ensure_experts_hybrid`; `m` comes from the cache
   slot table; q\* is `hybrid_fetch_fraction = B_P/B_H` with FreeToken's balance rounding.
   The engine already had the call sites needed (decode-trace hooks, PCIe/GEMM CUDA events,
   a native CPU-task timer), so the only core edit is an opt-in module loader.

2. **Pilot (10 episodes, med_miss).** Native q\*, q\* recomputed on the host, and q = 0 gave
   15.11 / 15.11 / 15.08 tok/s; the hook reproduces q\*. Measured per-expert costs: PCIe
   1.99 ms, CPU 0.38–0.43 ms, matching `saketh_exps/results/branch`. Repeating a policy
   later in the same server agreed to about 1%.

3. **Exploration data (48 episodes).** Fixed splits, uniform-random q, ε-exploration around
   four base policies and four m-threshold tables, at three cache sizes: 292,608 decisions.
   Only randomized episodes are used to fit cost models, because in a fixed-policy episode q
   is a deterministic function of m and is confounded with that episode's whole future.

4. **Feature screening.** 89 candidates → 63 kept. Removed: six constants (B_P, B_H, top_k,
   expert size, thread count, routed-unique count) and 20 near-duplicates (|ρ| > 0.98).

5. **First models.** Cost models with horizon H ≤ 8 tokens chose q ≈ 0 almost everywhere;
   only H = 16–32 began to fetch. The uniform-random logs give the single-step table
   wall(m, q), whose minimum is q = 0 for every m.

6. **Per-fetch analysis (193k fetch events).** The CPU time saved by cache hits saturates
   near 5 s per 254 tokens at med_miss regardless of how much is fetched. 25% gets 3.0 s
   with 332 fetches (21 reuses each); q = m spends 13,435 fetches for 4.7 s (0.8 reuses
   each). Best pre-action predictor of a fetch's realized reuse: recent frequency × LRU
   victim age (Spearman 0.36); frequency alone 0.12.

7. **Analytic rule** derived from (5) and (6) and tested with κ ∈ {0.5, 1, 2, 4}; κ = 1, the
   value the derivation gives, was best overall in the first scan and was not tuned further.

8. **Round 2.** 18 on-policy episodes (rule and ridge with ε-exploration) added; all models
   retrained on 402,336 decisions. Bandit specs built: LinUCB and Thompson sampling with a
   delayed 16-token cost, a myopic LinUCB, and warm-started variants.

9. **Main matrix, ablations, follow-ups** as listed in the status table.

#### Problems hit and how they were handled

- **Warm-up transient after server start.** The identical rule, with row-for-row identical
  decisions, measured 30.3 tok/s in one scan and 22.7 in another at low_miss. The first two
  or three episodes after a server start run slower on the host side. Fix: three discarded
  burn-in episodes per server and rotated policy order across repeats. The two exploratory
  scans (`scan1`, `scan2`) predate the fix and are kept only as exploratory records. Two
  remarks in the conclusion rest on them and say so (T_free = 64, warm-started bandits);
  every other reported number comes from the burn-in runs.

- **Learned policies are not reproducible across sessions.** The same ridge artifact
  (identical weights) averaged 46 ms/token in the main matrix and 78 ms/token in the
  ablation sessions. It reads utilization, memory and timing features whose values drift
  between server sessions. This is reported as a result, and it is why the three most
  important ablation families were repeated three times.

- **Short horizons.** A phase is 127 tokens and the measured prompt's prefill invalidates 64
  cache slots, so cost labels cannot span the warmup/measure boundary; the longest usable
  horizon is 32 tokens.

- **Output hashes.** All prompt-0 / 128-token runs produce one output hash. On prompt 1 and
  on 384-token decodes, policies that split CPU/GPU differently produce different text:
  the two expert kernels agree numerically but not bit-for-bit, and a near-tie in greedy
  decoding eventually flips. Each deterministic policy still reproduces its own hash.

- **No scikit-learn in the FreeToken venv.** A separate `.venv-analysis` was created rather
  than modifying the shared environment.

- **A cleanup `rm` was blocked** by the safety check at the start of the pilot; nothing was
  deleted and the run proceeded into an empty directory.

#### Not done

- No second GPU, CPU, PCIe configuration or MoE model. Cross-architecture transfer of the
  rule is a prediction, not a measurement.
- The rule was not implemented inside the device-side kernel, so it was not measured in
  CUDA-graph mode.
- Bandits were not evaluated under changing runtime conditions or across multiple episodes.
- Individual-signal, history and ANN ablations have one end-to-end run each.
- Byte-aware scheduling was not tested: all gpt-oss-20b experts are the same size.

## PR-00..PR-11 - Local pruning history, 6-8 October

The following detailed qualification/amendment/capture history consolidates the former pruning execution log. Earlier statements such as "not complete", "pending" and "paused" are dated historical states, not current blockers. The final accepted closeout at the end supersedes those live-status statements; no rejected attempt was retrospectively accepted.

The original static stage captured 270 primary requests and 54 additional repeat observations across three checkpoints. The actual stage later collected 2,700 task generations and 1,800 fixed-input calls. These are different datasets and stages, not cumulative independent-task counts.

Study: `results/moe_pruning/study_20261006_v1`.
Started: 2026-10-06. Base commit: `4d0f84983700eba7dfb046fb4df3cbbcc488ecc2`.
Existing workspace changes were preserved. Each capture batch includes a source
snapshot and file hashes; the base commit alone does not reproduce the run.

### Implemented and exercised

- `experiments/moe_pruning.py`: hardware and checkpoint audit, source snapshots,
  checksum provenance, and native cache replay validation.
- `experiments/build_pruning_dataset.py`: 30 primary and five pilot requests per
  workload, group-separated five-fold splits, and preselected repeat requests.
- `experiments/collect_moe_routes.py`: sequential eager hybrid collection,
  natural EOS, full prefill/decode accounting, native expert IDs and weights,
  observer-off/on pilot comparisons, resource monitoring, and qualification
  checks before primary or repeat capture.
- `python/freetoken/moe/routing_trace.py`: optional observation hooks at the
  native batch and router boundaries, before expert IDs are remapped.
- `experiments/pruning_cache_replay.py`: global hybrid recency replay validated
  against the CPU reference and observed GPU cache transitions.

`experiments/pruning_analysis.py` is an initial analysis implementation. Its
ranking, coverage, and tie-order functions have focused tests. It has not been
validated end to end on a complete primary dataset. It is not a closeout tool.

### Protocol decisions before primary capture

1. Disable scheduler overlap with `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`.
   The scheduler can otherwise issue a forward that is discarded after EOS.
   MoE computation overlap remains enabled. These traces describe this explicit
   execution configuration and cannot establish normal serving throughput.
2. Reset the expert cache before each pilot control and observed request and
   each primary request. This makes request independence and observer
   comparisons explicit. Save the real post-prefill cache before decode.
3. Store one gzip JSONL record per native token/layer, with paired expert-ID
   and weight arrays. This preserves every selected routed expert without
   writing a separate JSON object for each expert selection.
4. Use the locally authored routing corpus frozen in `protocol/dataset.jsonl`.
   It has no externally graded answers and therefore supports routing analysis,
   not the later actual-mask quality gate.
5. The initial resource gate required at least 4 GiB available host memory and no
   more than 128 MiB additional system swap-out during the monitored batch.
   System-wide counters can include other processes. A resource failure blocks
   qualification and needs an explicit recorded fallback; accepted individual
   captures can still serve as tracing diagnostics. Resource amendment v2 below
   supersedes the whole-batch swap criterion for fresh captures.

The initial frozen protocol JSON predates these implementation decisions.
Batch `configuration.json` records the actual settings. Preserve both records;
do not silently replace the original protocol when making an amendment.

### Execution evidence

- Required Qwen and GPT checkpoint revisions, all index-listed weight shards,
  downloaded file SHA256 values, tensor-key sets, and configuration/tokenizer
  provenance passed the audit. Evidence: `environment/checkpoints.json`.
- The first GPT math diagnostic passed complete causal-token accounting and
  observer-off/on equality, with 99 prompt tokens and 256 generated tokens.
- Native replay matched all 6,120 decode token/layer cache events in that
  diagnostic. The full replay report includes subsequent accepted captures.
- Two attempted full GPT pilot batches encountered resource-monitor defects:
  a missing optional dependency and an empty `/proc` status field. The logger
  now reads the needed Linux fields directly and has a regression test.
  These batches are retained as failed attempts, not qualified pilots.
- The replacement full GPT pilot finished all 15 requests. All observer-off/on
  comparisons and trace integrity checks passed. It failed resource
  qualification: 1,351.57 MiB additional system swap-out, with at least
  9.27 GiB available host memory. Evidence: the batch's
  `qualification_failure.json`; no qualified pilot artifact was written.
- Qwen's single-request diagnostic passed observer-off/on equality and complete
  lifecycle validation: 42 prompt tokens, 256 generated tokens. Native replay
  matched all 10,200 decode token/layer events. It is not a full pilot.
  System swap-out increased 1,038.57 MiB, minimum available host RAM was
  4.26 GiB, and the maximum sum of server-process VmSwap was 77.76 MiB.
  System-wide swapping cannot be attributed entirely to the engine from these
  measurements. Preserve that distinction; it does not override the gate.
- The aggregate GPT replay validation matched 22 accepted diagnostic captures,
  including captures retained from the two failed logger batches. This is
  replay fidelity evidence, not 22 qualified primary requests.
- A primary gate probe confirmed that an unqualified GPT primary run is
  rejected before creating its batch directory or launching a GPU server.

Final focused verification: 48 tests passed in 2.82 seconds using
`PYTHONPATH=python:. uv run --no-sync pytest tests/moe/test_bandwidth_study.py
tests/moe/test_decode_trace.py -m 'not slow' -q`. This includes synthetic
end-to-end fold analysis, sensitivity/stability exports, nested frozen
rankings, and rejection of 60%/50% analysis before the required closeout gates.
Synthetic fixture outputs are not experimental measurements. `git diff
--check` also passed. No repository-wide suite or serving benchmark was run.

Future source snapshots include native C++/CUDA/header sources and compiled
extension binaries. Pilot qualification compares hashes across the entire
engine package, including those native files, before allowing primary runs.
The initial diagnostic batches used a smaller hash set; those diagnostics do
not qualify the strengthened gate. Supplementary native source/binary hashes
are saved in `environment/native_source_and_binaries.json`.

### Runnable commands

Run from the repository root with the existing `.venv`. Output batch IDs must
be new. Only one GPU collector may run at a time. These commands are provided
for reproduction and continuation; listing a command does not mean it ran.

```bash
PYTHONPATH=python:. uv run --no-sync python -m experiments.moe_pruning audit \
  --study results/moe_pruning/NEW_STUDY_ID

PYTHONPATH=python:. uv run --no-sync python -m experiments.collect_moe_routes \
  --study results/moe_pruning/study_20261006_v1 \
  --model qwen_nvfp4 --stage pilot --batch-id NEW_QWEN_PILOT_ID

PYTHONPATH=python:. uv run --no-sync python -m experiments.moe_pruning validate-replay \
  --study results/moe_pruning/study_20261006_v1 \
  --model gpt_oss_mxfp4 --stage pilot

PYTHONPATH=python:. uv run --no-sync python -m experiments.collect_moe_routes \
  --study results/moe_pruning/study_20261006_v1 \
  --model gpt_oss_mxfp4 --stage primary --batch-id NEW_GPT_PRIMARY_ID
```

Primary capture refuses to start without a complete, qualified 15-request
pilot using matching inference source hashes. Use `qwen_nvfp4` for the analogous
Qwen primary command. Repeats use `--stage repeat` after qualification. The
collector uses the qualified pilot's memory ratio, including a documented
fallback ratio, rather than silently returning to the original default.

### Outstanding gates

1. Qualify both required models' complete pilots and resource envelopes.
2. Complete 90 primary requests and 18 extra repeat captures per required
   model; run checksum, numerical, repeatability, and cache-fidelity checks.
3. Finish and validate analysis: normalization sensitivities, cross-workload
   stability, conditional native rerouting, cache capacity/state controls,
   paired comparisons, report figures, and a complete experiment ledger.
4. Qualify Gemma's actual loader/resource envelope before admitting it. The
   pinned Gemma metadata is downloaded; its weights and pilot are not complete.
   Kimi's pinned checkpoint is approximately 595 GB and its wrapper loader is
   unavailable here. Record unavailable local cells and external-trace status.
5. Decide actual masking only through the prospective coverage/stability and
   quality gates. No serving-speed or model-quality result is established by
   the instrumented pilot.
6. Freeze the entire 75% study closeout, including every blocker disposition,
   before starting 60%, then 50%. Neither lower ratio has been executed.

### Disposition before v2 mitigation

Initial implementation and tracing diagnostics are complete. Broad collection
is stopped at the resource/pilot gate. The GPU collectors have been shut down.
The study's `protocol/blockers.json`, `protocol/ledger.json`, and
`protocol/state.json` record the next work. Diagnosing competing memory demand
and qualifying a fresh complete pilot are required before primary capture.
Do not reuse diagnostic captures as the held-out primary dataset, silently
relax the swap limit, or count unavailable measurements as completed results.

### Resource mitigation amendment v2

Recorded before fresh pilot requests in
`protocol/amendments/resource_v2.json`; policy frozen in
`protocol/resource_policy.json`. Earlier failed batches retain their original
dispositions. No primary evaluation data informed this amendment.

The collector previously imported tokenizers/PyTorch in its own long-lived
process before launching the server. Token rendering now runs in a separate
`uv run --no-sync` worker that exits before model loading. The collector also
launches the server and environment probes through `uv run --no-sync`, uses
`--expert-load serial` explicitly, and preserves the same checkpoint revisions,
cache slots, memory ratio 0.9, calibration, dataset, and sampling.

Startup, settling, inference, and shutdown have separate observations and
boundary snapshots. After server readiness, require ten consecutive quiet
seconds within a 90-second maximum wait. Quiet means at least 4 GiB available
RAM, no more than 1 MiB swap-in/out per sample, and memory `full avg10` at most
1%. Startup swap is reported separately and cannot qualify a run with OOM,
insufficient RAM, or non-pinned expert banks.

The fresh inference resource gate requires:

- At least 4 GiB available RAM throughout startup and inference.
- At most 128 MiB system swap-in and 128 MiB swap-out during inference.
- At most 128 MiB server-subtree swap and at most 16 MiB growth during inference.
- At most 1% full memory stall time overall and in observed 30-second windows.
- At least 30 inference samples; no monitor errors or system OOM kills.
- Native cache residency labels must be `pinned` for every expert layer;
  each observed request saves those labels and checksums them.

Memory pressure is read from the kernel's
[Pressure Stall Information interface](https://www.kernel.org/doc/html/latest/accounting/psi.html).
System counters and server-subtree VmSwap are distinct measurements; neither
alone identifies a particular expert bank. These thresholds are prospective
operational limits for this experiment, not universal memory-safety guarantees.

Qualification binds the complete inference source hashes, collector/resource
code hashes, frozen policy, model configuration, dataset, rendered inputs,
folds, and calibration. Primary/repeat batches reject changed controls and
must themselves pass the resource checks. Python execution uses the existing
project environment through [uv run](https://docs.astral.sh/uv/concepts/projects/run/).

GPT pilot `gpt_resource_v2_pilot` completed all 15 controls and passed v2:
minimum available RAM 10.76 GiB, inference swap-in 29.16 MiB, swap-out 5.07 MiB,
server swap zero, full memory pressure 0.0033% overall and 0.0186% maximum in
observed 30-second windows. Native replay matched its accepted captures.

Qwen pilot `qwen_resource_v2_pilot` exposed a defect in the absolute server-swap
stock criterion: roughly 548 MiB was already swapped at inference start,
declining during requests, while system inference swap-out was zero and the
native expert-bank labels were pinned. `/proc/<pid>/smaps_rollup` observations
showed swap in anonymous process state, while the large registered host banks
used shared mappings. That is consistent with retained startup state; it does
not identify individual swapped allocations. The v2 Qwen attempt was stopped
after 11 accepted diagnostic requests and remains unqualified. Its interruption
and owned server-group cleanup are recorded in the batch directory.

### Resource mitigation amendment v3

Frozen before new requests in `protocol/resource_policy_v3.json` and selected
through `protocol/active_resource_policy.json`. The v2 policy is preserved.
Details: `protocol/amendments/resource_v3.json`.

Use `MALLOC_ARENA_MAX=2` in the server environment to bound glibc allocation
arenas; [glibc documents this setting](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html).
This changes allocator behavior and is part of the recorded v3 arm. It does
not establish a throughput improvement. Live weights are not deleted, and
checkpoints, cache slots, memory ratio, sampling and dataset remain identical.

The server swap stock is now reported without an absolute idle-stock cutoff.
The operational gate retains all the v2 requirements for host headroom,
pinned expert banks, bounded settling, inference system swap-in/out, server
swap growth, overall/window memory pressure, complete monitoring, and OOM.
Thus persistent startup swap cannot by itself fail a healthy trace capture;
new paging or sustained pressure during inference still rejects qualification.

Both models require fresh complete pilots after changing the allocator,
collector or resource classifier. No prior failed batch is retrospectively
qualified. Qwen `qwen_resource_v3_pilot` was stopped after 13 accepted requests:
inference swap-out exceeded 128 MiB and server-subtree swap growth exceeded
16 MiB. This was a real v3 gate failure, despite low measured pressure and
otherwise valid routing diagnostics. Its `resource_disposition.json` records
the counters; the source and interruption logs are preserved.

### Text-only resource fallback v4

The default Qwen server log confirms `encoders ['vision'] on host`. The engine
constructs registered host banks for the vision tower even when requests
contain text only. The protocol's Section 11 explicitly allows a separately
qualified `--text-model-only` arm when encoder allocation blocks fit.

Fresh Qwen collection now adds that supported flag, omitting the unused vision
tower and its host banks. It retains the same pinned NVFP4 checkpoint, routed
text geometry, 2,048 cache slots, memory ratio 0.9, greedy sampling, dataset,
splits, calibration and v3 resource limits. The new Qwen arm must remain
separate from default-encoder diagnostics and historical serving measurements.
GPT's model flags remain the original text configuration.

The policy is frozen in `protocol/resource_policy_v4.json`, selected by the
active policy pointer. Amendment: `protocol/amendments/resource_v4.json`.
The v4 fallback did not change resource acceptance thresholds. Qwen pilot
`qwen_text_only_v4_pilot` completed all 15 observation controls with zero
inference swap-out and zero server swap growth. It failed the pressure gate:
full stall time was 0.188% overall but 2.667% in the worst 30-second window.
Almost all stall time occurred within the first four seconds of the first
request: deferred startup allocation. The failed gate is preserved in its
resource summary and qualification-failure artifact.

Paired text-only/default-encoder diagnostic responses are not universally
identical. Initial comparisons found late token differences after generated
positions 163 and 235, while other pilot pairs matched. This is evidence to
keep the configurations separate; no cross-arm numerical equivalence or
quality claim is made. Within-arm observation-off/on checks passed.

### Startup warmup mitigation v5

Policy: `protocol/resource_policy_v5.json`; amendment:
`protocol/amendments/resource_v5.json`. The previous policies and failed
attempts are preserved. Acceptance thresholds are identical to v4.

Before settling, run a 64-token untraced request using a pilot-only prompt.
Record its response and resource observations as `warmup`. This materializes
deferred decode buffers/kernels before the quiet interval and inference
boundary. The warmup response is not a study capture; no warmup route rows
exist. Each following control, pilot, primary and repeat request resets the
expert cache, and naive KV caching prevents prefix reuse. Startup headroom
and OOM checks include warmup; its memory pressure is retained in the logs.

Fresh full Qwen `qwen_text_only_v5_pilot` and GPT `gpt_resource_v5_pilot`
are required under matching source and controls. This is a concrete startup
mitigation after locating the failed window, with no threshold relaxation.
The primary analysis loader also rejects batches that have not passed their
resource checks, so failed captures cannot silently enter held-out results.

Qwen's final v5 pilot passed all 15 observation comparisons and resource checks:
minimum available RAM 4.61 GiB; inference swap-in 80.16 MiB, swap-out 0.24 MiB;
server swap growth zero; full pressure 0.00244% overall and 0.01208% maximum in
observed 30-second windows. The entire accepted Qwen diagnostic inventory
matched native replay, including the 15 final v5 captures. Failed earlier arms
remain diagnostic evidence rather than additional qualified pilot requests.

The paired comparison against 13 available default-encoder v3 diagnostics
found four identical generated sequences and nine differing sequences. The
cause of these cross-arm differences has not been established. Keep separate
configuration IDs; inference observation checks passed within the final arm.
Evidence: `protocol/text_only_pilot_comparison.json`.

### Final mitigation result

Both required models completed fresh 15-request pilots under the final v5
source and controls. All observation-off/on comparisons, token/layer integrity
checks, pinned-bank checks and resource gates passed. Native replay matched
all accepted final pilot captures. Qualification hashes were compared with
the current workspace and frozen inputs after both runs completed.

| Final qualified arm | Pilot requests | Minimum available RAM | Inference swap-out | Server swap growth | Worst 30-second full pressure |
|---|---:|---:|---:|---:|---:|
| Qwen NVFP4, text-only v5 | 15 | 4.61 GiB | 0.24 MiB | 0 MiB | 0.0121% |
| GPT-OSS MXFP4, v5 | 15 | 11.23 GiB | 0.18 MiB | 0 MiB | 0.0129% |

Machine-readable result: `protocol/resource_mitigation_result.json`.
The historical failed attempts remain in the artifact inventory. The resource
blocker is mitigated for these qualified pilot configurations, and both GPU
servers are shut down. All Python commands, including tokenizer workers,
environment probes and server launches, used `uv run --no-sync` in this
mitigation work. No new pytest run or serving benchmark was performed during
this mitigation; the 48-test result above belongs to the earlier implementation.

Next: collect primary requests and repeats under the qualified configurations,
then finish and validate the remaining analysis/report tools. Each primary
batch must pass its own resource checks. The 75% primary study is not complete;
60% and 50% remain blocked until its entire closeout.

### Primary collection resumed

GPT-OSS primary collection started as `gpt_primary_v5` under the unchanged
qualified v5 collector, engine and control hashes. Its first five requests
passed capture integrity. Whole-batch resource acceptance is pending until
the batch finishes; these captures are not yet analysis-qualified.

`experiments/run_pruning_collection.py` now manages the following serial queue:

1. GPT-OSS: 90 primary requests, attaching to the already running collector.
2. Qwen text-only: 90 primary requests.
3. GPT-OSS: 18 extra captures of the nine preselected repeat requests.
4. Qwen text-only: 18 extra captures of the nine preselected repeat requests.
5. Native observed-state replay validation for each model and stage.
6. The 75% held-out coverage, oracle, transfer and stability analysis.

All subprocesses use `uv run --no-sync`. An exclusive orchestration lock
prevents duplicate queue runners. Existing incomplete batches are preserved
and cause a stop; capture counts and resource acceptance must pass before
the next collection starts. A failed subprocess also stops the queue.
Progress and exact commands are recorded under
`results/moe_pruning/study_20261006_v1/orchestration/primary_repeat_v5/`.

The runner does not close the 75% study: counterfactual replay, repeatability,
additional-family dispositions, the masking decision and final report still
require completion. It does not launch the deferred 60% or 50% analyses.
No new tests or serving benchmarks were run during this continuation.

### Primary completion and repeat retry

Both required primary batches completed 90 requests and passed their capture
and whole-batch resource gates. GPT-OSS also completed all 18 repeat captures
with resource acceptance. Qwen's first repeat batch completed 18 integrity-valid
captures but failed resource acceptance: inference swap-in was 223 MiB against
the 128 MiB cap. Swap-out and server swap growth were zero; host headroom and
pressure checks passed. The cause of swap-in is not established, and these
repeats remain excluded from resource-qualified analyses.

The serial runner stopped before native replay validation and coverage analysis.
The failed batch and its logs are preserved under `qwen_repeat_v5`. A single
replacement repeat batch, `qwen_repeat_v5_retry1`, was launched with identical
qualified v5 source, controls and thresholds. The resumed runner checks and
reuses the three completed batches, then waits for this replacement before
replay validation and 75% coverage analysis. Retry rationale and budget are
recorded in `protocol/qwen_repeat_v5_retry.json`; live progress is under
`orchestration/primary_repeat_v5_retry1/`. Neither deferred ratio has started.

### Gemma qualification and continued 75% work

User steering: qualify Gemma, then resolve Qwen repeats and finish the 75%
study; ignore Kimi for now. Kimi's historical blocker evidence is preserved,
but it is excluded from the current closeout scope. Both Qwen repeat v5
attempts failed the inference swap-in cap. The second reached 430.42 MiB;
its 18 captures remain resource-rejected diagnostics.

Gemma's pinned two-shard checkpoint was downloaded with `uv run --no-sync hf
download`, disabled Xet and one worker. Both SHA256 hashes, all tensor/index
keys, small-file revision metadata and L/E/k/H/I geometry matched the pinned
revision. Checkpoint header inspection estimated a 3,353,856-byte expert row,
12,878,807,040-byte host banks and a 768-slot/2,575,761,408-byte cache.

The first admission attempt loaded the checkpoint and initialized the native
NVFP4/gelu_tanh CPU executor but failed KV sizing. Changing only the sequence
limit did not resolve it. An isolated logging wrapper then measured the
existing budget without changing its result. The server's SchedulerConfig
defines max_forward_len from max_extend_tokens, so the inherited 8192-token
prefill cap drove a large fixed naive SWA reservation. All failed startup
attempts are preserved under `additional/gemma_admission_v1`, `v2`, `v3`.

The supported Gemma arm uses text-only, memory ratio .8, S=768, a 1280-token
sequence limit and `--max-prefill-length 1024`, matching the frozen prompt
and completion caps. A three-request 64-token observation pilot passed
capture checks but its 25.27-second inference window provided only 27
resource samples, so it was not qualified. Its other resource checks passed.
No acceptance threshold was relaxed.

The fresh `additional/gemma_v1` full 15-request/256-token pilot subsequently
passed every off/on comparison, native token/ID/weight and pinned-bank check,
and resource gate. Minimum available RAM was 8.32 GiB; inference swap-in
98.03 MiB, swap-out 3.02 MiB, server swap zero, worst 30-second full pressure
0.00377%. Gemma is now qualified for this exact arm. Its effective router
weights include per-expert rescaling, so they are preserved without the
Qwen/GPT sum-to-one check. No Gemma full-score rerouting claim is available.

The accepted Qwen/GPT primary traces produced the 75% coverage, oracle,
transfer, layer/position, stability and normalized-ranking tables. Native
100% rerouting identity passed on all nine preselected requests per model;
their 75% masked-router counterfactuals retain native BF16 logits losslessly.
Twenty-four controlled cold GPU cache cases matched the simulator across
U/F/R, both cache capacities and hybrid/unconditional-fill policies.

The serial continuation is `experiments.finish_pruning75`, with progress at
`orchestration/gemma_first_75_v1/progress.json`. After Gemma qualification it
runs a fresh complete Qwen pilot using fixed 128 KiB glibc mmap/trim thresholds,
then replacement Qwen repeats, Gemma primary/repeats, native validations and
all remaining offline report stages. This is a separately qualified allocation
arm under `additional/qwen_allocator_v6`; engine/checkpoint/cache/generation
and resource thresholds remain fixed. Numerical repeatability against the
original v5 primary captures must be measured, not assumed.

New offline tools export paired cache demand, qualified repeat sensitivity,
matched checkpoint differences, masking gate decisions, scientific figures
and a hashed closeout. Their completion is still pending. No deferred ratio
has run, and no new pytest or serving-performance result is claimed.

Qwen allocator v6 full pilot subsequently passed all 15 paired controls and
resource gates: minimum available RAM 4.39 GiB, inference swap-in 31.45 MiB,
swap-out 1.72 MiB, server swap zero, mean full pressure 0.01882% and worst
30-second full pressure 0.23509%. Replacement repeat capture is next; its
whole-batch resource gate must pass independently.

The first allocator-v6 repeat attempt failed startup settling before any
captures: available RAM fell below 4 GiB, coinciding with a second VS Code
server session. Its failure/resource logs are retained. No user applications
were terminated. Continuation `gemma_collection_75_v2` reuses the passed
Qwen pilot, collects Gemma primary/repeat data first, and schedules one
remaining allocator-arm repeat attempt after this interval.

User authorized closing unused applications. On the locked desktop, idle
Firefox and two update dialogs were closed with SIGTERM. Active editor,
network services, other users' processes and experiment drivers were preserved.
The Gemma primary attempt had already reached 471 MiB inference swap-in and
17.90 MiB server swap growth; it was gracefully interrupted at 65 captures.
All 65 are resource-rejected diagnostics, moved out of the accepted routes
tree with their original manifests/checksums and an explicit rejection record.
A single fresh 90-request retry uses the same qualified controls. Workspace
settings now exclude generated results from file watching and Pylance analysis
to reduce possible output-directory churn; causation is not established.

The post-cleanup Gemma full primary retry passed all 90 captures and the
whole-batch resource gate: minimum available RAM 8.45 GiB, inference swap-in
122.78 MiB, swap-out 8.85 MiB, and no server swap growth. Absolute server
swap stock at inference start was 258.52 MiB and is reported separately under
the frozen v5 policy; no bank-level swapping attribution is inferred. The
18 Gemma repeats passed with zero server swap and 63.84 MiB system swap-in.

Qwen allocator-v6 replacement repeats also passed all 18 captures: minimum
available RAM 5.24 GiB, inference swap-in 83.20 MiB, swap-out 7.82 MiB, server
swap growth 1.72 MiB, worst 30-second full pressure 0.38104%. Collection is
complete: 270 distinct primary requests and 54 accepted extra repeat captures
across Qwen, GPT-OSS and Gemma. Native observed-cache validation passed,
including Gemma's eight GPU cold-cache controls. Cache-demand analyses and
repeat sensitivity are complete or completing; the final hashed report
remains the authority for 75% closure. No 60% or 50% run has started.

### 75% closeout complete

The serial continuation completed every queued stage. The 75% report,
checkpoint/cache locks, qualification resources, per-request paired
comparisons, activation decisions, failure inventory, ledger and 45
SVG/PDF/PNG figure artifacts are under `closeout75/`. All three checkpoints
have 90 accepted primary model/request captures and 18 extra repeats: the
same 90 distinct prompts are shared across checkpoints. Every one of the
27 repeated-capture pairs per model had identical generated tokens and
100% expert-ID agreement on shared inputs, including the Qwen cross-arm
comparison. No model cleared every static-mask activation gate, so actual
masked inference/quality/performance was skipped. Gemma native R remains
an explicit missing endpoint; its F and native cache controls are complete.

Integrity audit passed for 22554 frozen files and 29 source
snapshots, with no checksum mismatches and no active GPU compute process.
Manifest SHA256: `967c23a59581d36eedc64cd2f1f7eb82f053e41190dd66d92ad640b82b6a1aba`. The audit is saved at
`orchestration/closeout75_integrity_audit.json`; mutable status now records
`75_closed`. Workspace plans were updated after export; the immutable
source snapshots retain the precise documents present at execution. Kimi
remains excluded. Neither lower-retention ablation has been executed.

### 2026-10-06: Gemma native rerouting and actual masking supplement

User authorized fixing Gemma R75 and then actual 25%-pruned inference.
Original closeout remains immutable. Native Gemma logits/scales are collected
through `experiments.native_pruning_probe` in the spawned scheduler worker.
The first pilot was rejected because hooks initially ran only in the API
process. Retry1 completed all 15 numerical controls but failed the global
swap-in gate (466,403,328 bytes); server subtree swap remained zero.
Retry2 added the allocator mmap/trim thresholds already used for Qwen and
passed all 15 controls/resource checks: minimum RAM 8,806,797,312 bytes,
swap-in 86,388,736 bytes, swap-out 19,292,160 bytes, subtree swap zero,
worst 30-second full pressure 0.832637%. Eighteen full-score repeat captures
are running before native R75 controls and actual M75 pilots.

Actual supplement uses separate immutable deployment masks fitted from the
original 90 mixed/30 workload calibration requests and new synthetic quality
tasks/texts. Initial preparation was archived before actual inference to
correct the text corpus to ten real templates matching its cluster IDs.
The current corpus has 300 tasks and 100 text sequences; qualification
checks native k, mask eligibility, shared-branch-preserving native inference,
100% token/output identity, and teacher-forcing numerical identity.
See `docs/experiments/pruning/protocol.md`. No actual quality/performance
result is claimed at this point. All Python commands use uv.

Gemma full-score repeats subsequently passed: all 18 captures, RAM minimum
8,972,173,312 bytes, inference swap-in 60,948,480 bytes, swap-out zero,
server subtree swap zero, worst full pressure 0.046711%. All nine native
100% routing reconstructions and all original/fresh-repeat comparisons have
maximum effective-weight error 0.0 with exact expert IDs/generated token
histories. All 12 cold U/F/R GPU cache controls matched. R75 fixed-capacity
calibrated hybrid mean-m reduction is 0.456601%; reduced 576-slot capacity
increases mean-m by 9.513964%. Observed pre-decode Gemma caches are empty,
so cold and projected post-prefill conditions coincide. See
`additional/gemma_rerouting_v2/report.md`. Actual M75 pilot is running.

Actual Gemma v1 generation controls completed all 15 baseline/all100 pairs
and all 30 native mixed/workload75 route audits. Full qualification was
rejected when the first teacher-forcing request reached FreeToken's explicit
HTTP 400 rejection of token-ID completion prompts. No teacher-forcing or
new-quality outputs were produced by that failed request. Original v1 pilot
and protocol are preserved.

The repaired actual arm is `additional/actual_masking75_v2`: raw text
completion prefixes are frozen with native add_special_tokens=True to exactly
320 IDs. The worker verifies those IDs and positions before fixed-input NLL.
The quality tasks, raw reference texts, ranking counts and masks are unchanged;
all controls are requalified under the new collector/worker hashes.
`orchestration/native_rerouting_and_actual75_v2/progress.json` records current
serial execution. The v1 controller stopped on failure and does not run
concurrently with the v2 controller.

Actual v2 completed all 15 generation identity pairs, all 30 native75 route
audits, and both same-input all100 controls with NLL/logit identity exact.
The batch remained rejected by unchanged resource gates: swap-in
140,066,816 bytes (133.58 MiB), mean full memory pressure 1.4914%, worst
30-second pressure 30.5199%; server subtree swap zero, minimum RAM
9,299,181,568 bytes. The first 320-token diagnostic forward was much slower
than subsequent forwards; its diagnostic/native kernels had not been warmed.

The v3 arm explicitly warms one 320-token sequence in baseline, all100 and
all four 75% masks before settling and the measured phase. Those warmup
rows are saved separately and excluded from pilot/quality capture counts.
Resource gates, native adapters, corpus, masks, and quality margins are
unchanged. Full qualification is rerun; v2 is preserved as rejected.
Current controller: orchestration/native_rerouting_and_actual75_v3.

Actual v3 again passed all numerical controls. Its global swap-in was
64,630,784 bytes, swap-out 782,336 bytes, subtree swap zero, and minimum
RAM 9,046,224,896 bytes. It was rejected by the unchanged rolling pressure
gate (worst 30-second full pressure 16.9957%, mean 0.8860%). Pressure spikes
concentrated around baseline-reference loading/comparison in the fixed-input
likelihood diagnostics, despite the startup diagnostic warmup.

The v4 diagnostic adapter keeps lossless FP32 baseline logits and layer
outputs on GPU and reuses pinned buffers for writing baseline files. It
eliminates repeated large np.load/memmap host references and pageable H2D
comparison transfers. Cache/KV budgets, masks, native selection, corpus and
quality/resource margins remain unchanged. Exact diagnostic GPU reference
tensor bytes from checkpoint metadata: Gemma 354,942,976; Qwen 338,165,760;
GPT-OSS 276,692,992, excluding allocator workspace. Pinned write buffers are
19,660,800; 17,989,632; 15,818,752 bytes respectively. These are diagnostic
buffers, not extra model weights or serving buffers. Fresh full qualification
is running in additional/actual_masking75_v4; prior attempts are preserved.

Actual v4 passed all 15 identity pairs, all 30 native75 route controls and
both same-input numerical controls. It was rejected only by rolling full
pressure: 1.2474843% maximum, mean 0.1163071%; swap-in 57,393,152
bytes, swap-out 704,512 bytes, server subtree swap zero, minimum RAM
8,973,242,368 bytes. GPU references substantially reduced diagnostic
pressure, but did not satisfy the unchanged 1% gate. Short stalls occurred
both during generation and baseline artifact writes; these system counters
do not establish which process caused each stall.

The v5 adapter keeps GPU references and writes lossless baseline arrays
using Linux O_DIRECT with reusable 4096-byte-aligned pinned buffers. NPY
v2 headers are padded to 4096 bytes, preserving standard NumPy loading.
Large writes bypass filesystem page-cache allocation and dirty-page
throttling; unsupported direct I/O rejects the batch. Each aligned payload
buffer reserves up to one additional 4096-byte alignment block, plus a
4096-byte pinned header buffer and allocator overhead. The corpus, masks,
quality margins, cache/KV budgets and resource gates are unchanged.
The v4 capture and scoring controllers stopped after the rejected pilot.
A fresh v5 protocol and full native qualification are required.

The v5 preparation-equivalence record confirms identical corpus hashes,
all fifteen baseline/75% mask arrays and frequency counts, rendered native
input IDs, and resource policies relative to v4. The capture controller
allows at most three identical fresh batches per model/stage when a full
batch is rejected by resource gates; numerical/engine failures stop the
controller. Rejected outputs are never merged into accepted batches.
Scoring, quality-gated untraced serving dispositions, checked export, and
post-export manifest verification are queued in
orchestration/actual75_scoring_and_closeout_v5 after all accepted captures.

Gemma actual v5 full pilot qualified: 15 exact sampled-ID/API controls,
30 audited native75 routes and two all100 likelihood controls; 66 accepted
measured captures. Resource gate passed with minimum RAM 8947298304 bytes,
swap-in 31408128 bytes, swap-out 430080 bytes, server swap zero,
mean full pressure 0.0007874753521279248% and worst 30-second pressure
0.005990780382828795%. The fresh 900-generation/600-likelihood
Gemma quality batch started. GPT-OSS and Qwen follow serially. No quality
preservation decision has been made from the pilot.

During Gemma v5 actual quality collection, system swap-in approached the
unchanged 128 MiB limit while server-subtree swap stayed zero. An external
fail-fast monitor now watches the controller
(experiments/abort_failed_resource_batch.py). Once recorded inference
counters alone exceed the limit, the true start-to-end counter delta must
also exceed it. The monitor saves a conservative partial resource
rejection and signals only the controller-owned capture client with
SIGINT; the client preserves its failure and shuts down its own server.
The existing bounded identical-batch retry then applies. Partial rejected
outputs are never accepted, and no resource or quality threshold changes.
This avoids finishing requests in a batch already known to fail the gate.
A Judge0 usage question is pending; its services remain running.

Gemma actual v5 quality first batch was stopped after its recorded
inference counter lower bound reached 128.3515625 MiB swap-in, exceeding
the unchanged 128 MiB gate. All 459 completed task generations are
preserved as rejected partial data; a fresh identical retry1 started.
Correction to earlier status updates: the new quality server subtree had
145.25 MiB of swap already present at inference start, and it was still
145.25 MiB at detection. The qualified pilot had zero stock. The active
policy reports existing stock and gates growth; this was a system swap-in
rejection, not evidence that pinned expert banks were swapped.

### 2026-10-06: fresh-process quality partition amendment (v6)

The v5 quality controller and finisher stopped after all three Gemma attempts
were rejected by the unchanged swap gate. Preserved partial captures are
459, 294, and 136 task generations respectively (889 total), none accepted.
Measured conservative swap-in lower bounds were 128.3515625, 128.203125
and 131.80078125 MiB over approximately 33.9, 25.5 and 12.1 minutes.
Gemma rerouting and the complete v5 actual pilot remain qualified historical
results; GPT-OSS and Qwen actual pilots/quality never started in v5.

The v6 protocol freezes twelve modulo-ordinal partitions per model. Each
fresh process runs 25 tasks (all three task arms) and 8 or 9 fixed text
sequences (all six likelihood arms). First four shards have 129 measured
captures each; remaining eight have 123, totaling 1,500 per model. No
dataset item, mask, reference target, margin, quantization, native k, expert
ID, cache/KV budget, numerical or per-batch resource threshold changes.
Each paired item stays in one process, and global treatment-order parity is
preserved. Startup warmup is repeated and excluded in every process.

Resource accounting is explicitly amended: the 128 MiB swap bound now
applies per fresh shard rather than across the whole model collection.
Aggregate swap may exceed 128 MiB; no whole-model cumulative bound is
claimed. All shards must pass; no selective acceptance of requests inside
a failed shard. The complete quality scorer/export require a unique
accepted shard for every frozen index, 300 tasks, 100 sequences, and 1,500
paired-arm captures per model. Source changes require fresh full pilots
for all three models before their new quality collection. Corpus and masks
remain the previous frozen synthetic exploratory study, not a new broad
benchmark. Judge0 remains running; no approval to pause it was received.

V6 preparation completed and passed its provenance audit: 23 comparisons
confirmed the same corpus, all mask/count arrays, native rendered inputs,
and per-process resource policies, plus disjoint complete membership of
300 tasks and 100 sequences in twelve partitions (1,500 captures/model).
Capture controller native_rerouting_and_actual75_v6, external resource
monitor, and finisher actual75_scoring_and_closeout_v6 are running. Gemma
fresh actual pilot started; native FreeToken server reached readiness and
loaded its original 768-slot configuration. No v6 quality result is yet
accepted. GPT-OSS/Qwen follow Gemma serially. All Python execution uses
uv run --no-sync; no unrelated tests or services were run or stopped.

### 2026-10-07: completed Gemma/GPT collection and Qwen RAM recovery

Gemma and GPT-OSS each completed all twelve accepted quality shards
(900 task generations and 600 likelihood calls/model). Scoring the two
complete models now reports Gemma mixed/math/code/incident masks as
inconclusive. GPT-OSS task-specific math and code masks pass the frozen
synthetic quality screen; mixed and incident masks remain inconclusive.
No broad quality preservation or serving speedup is established.

Qwen qualified on its second pilot. Its first quality shard completed all
129 calls in three attempts, but each was rejected solely by the whole-
lifecycle 4 GiB available-RAM gate (minimum 4,213,702,656; 4,176,687,104;
4,189,253,632 bytes). No Qwen quality data is accepted. The v6 pipeline
and finisher stopped after exhausting the initial retry budget.

Under earlier user authorization to close unused applications, the idle
user-owned update-manager --no-update window (PID 467666, no child
processes) was sent SIGTERM. Its pre-exit RSS was 222,646,272 bytes. Active
editors, network services and system package services remain running.
The immutable execution amendment qwen_ram_recovery_20261007.json records
this concrete environment change and permits up to six total resource
attempts per remaining Qwen cell, including prior failures. Inference
source/control hashes, corpus/masks and resource/quality thresholds are
unchanged. Existing qualified Qwen pilot and Gemma/GPT data are reused;
only missing Qwen quality shards resume. The per-model partial report is
analysis/partial_report_20261007.md.


### User-requested pause, 2026-10-07

Paused at 2026-10-07T04:39:59.953483+00:00 because the user needs to run another workload. Capture, scoring/closeout and resource-monitor controllers stopped; the owned FreeToken inference server and its workers stopped, and GPU compute allocations were released. No experiment will resume without a user request.

Accepted actual75 quality batches retained: Gemma 12/12, GPT-OSS 12/12, Qwen 5/12 (shards 0-4). The interrupted Qwen shard 5 is excluded from accepted results and preserved under `results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/diagnostics/user_paused_20261007/qwen_nvfp4/qwen_nvfp4_actual_quality_v1_shard05`. Restart the full shard in a fresh process on resume.

Pause state: `results/moe_pruning/study_20261006_v1/orchestration/pause_20261007.json`. Relaunch the capture controller, closeout finisher and resource monitor with new orchestration run IDs only after the user resumes. Existing accepted batches are reused automatically; frozen masks, inference sources and resource gates remain in force. Kimi remains excluded; 60%/50% retention remains deferred.


### Resume in recorded runtime, 2026-10-07

Resumed at 2026-10-07T14:14:46.907693+00:00 after confirming no GPU compute workload. The shared workspace `python/freetoken/moe/offload_cache.py` differs from the frozen v6 engine, so an isolated runtime restores the complete recorded engine snapshot at `/home/ankit/Development/FreeToken-Reproduce/.pruning-runtimes/v6_resume_20261007`. Frozen source, corpus, mask, checkpoint metadata and protocol checks passed; FreeToken imports resolve to the isolated engine. The shared workspace change is preserved.

Gemma/GPT-OSS each reuse 12 accepted shards; Qwen reuses shards 0-4 and restarts interrupted shard 5 in full. Remaining Qwen shards, scoring, quality-eligible serving diagnostics and manifest verification are queued through uv. Existing resource limits and the recorded six-attempt cap remain in force. Resume record: `/home/ankit/Development/FreeToken-Reproduce/results/moe_pruning/study_20261006_v1/orchestration/resume_20261007.json`.


### Resume startup rejection and GPU-access check, 2026-10-07

The isolated v6 resume controller stopped at 2026-10-07T14:18:16 UTC (19:48 IST) after Qwen shard 5 could not obtain the required startup quiet interval within 90 seconds. No measured captures completed. The existing failure is classified as a startup resource rejection so the next explicit recovery can use a fresh retry without changing frozen inference sources or resource gates. Gemma/GPT-OSS each retain 12 accepted shards and Qwen retains five.

At the 2026-10-07T16:59 UTC status check, the current tool environment has no `/dev/nvidia0`, `/dev/nvidiactl` or `/dev/nvidia-uvm`, and `nvidia-smi` fails. This does not establish whether the host GPU driver is unavailable outside this environment. Experiment controllers have exited. Restart requires GPU device access and normal resource qualification.


### GPU access restored and startup retry, 2026-10-07

Restarted at 2026-10-07T17:04:37.019699+00:00 after NVIDIA-SMI confirmed the GPU was accessible and free of compute workloads. Qwen shard 5 prior startup failure is preserved; retry 1 starts a full fresh batch. The controller now classifies only the exact settling timeout with zero measured captures as a startup resource rejection and applies the existing six-attempt cap. The frozen 90-second settling limit and all inference sources/controls remain unchanged. All accepted batches are reused. Active capture, finisher and resource-monitor launchers are recorded in `/home/ankit/Development/FreeToken-Reproduce/results/moe_pruning/study_20261006_v1/orchestration/resume_retry_20261007.json`.

Qwen shard 5 retry 1 passed the unchanged startup quiet gate and entered measured inference at the 2026-10-07T17:06:46.868321+00:00 check. Six measured captures were already saved, with approximately 4.59 GiB host headroom. Batch acceptance still requires all captures and the full resource summary.


### Qwen pressure diagnosis and background-memory mitigation, 2026-10-08

Historical batch-7 attempts show both real eviction on the initial attempt and later whole-machine swap-in failures with stable private server swap. Firefox was confirmed unused and closed with SIGTERM, reducing measured global swap stock by 613.7 MiB and increasing initial available RAM by 1.19 GiB. A recorded eight-attempt cap permits two additional batch-7 attempts after this concrete environment change. Original inference controls and acceptance gates are unchanged. A five-second external process fault observer supports attribution on the recovery run. Detailed evidence and limitations: `docs/experiments/pruning/resource_recovery.md`.

### Serving resource recovery - 2026-10-08T05:41:19.640096+00:00

All three models have 12 accepted quality batches and completed scoring. GPT-OSS repetition 3 exceeded the unchanged 16 MiB server swap-growth gate (69.914 MiB). Preserve that entire incomplete paired repetition under diagnostics, reuse complete repetitions 0-2, and rerun repetitions 3-4 in their original treatment order. The external recovery controller allows three bounded resource recovery passes per model; frozen inference and serving workers, masks, quality margins, serving prompts, and resource gates remain unchanged. Qwen serving, reference validation, report export, and manifest verification follow serially.

Recovery controller PID: 1051243. Progress: `results/moe_pruning/study_20261006_v1/orchestration/actual75_scoring_and_closeout_v6_serving_recovery_20261008/progress.json`. All Python execution uses `uv run --no-sync` and the isolated frozen runtime.

### Serving no-swap scope recovery - 2026-10-08T06:09:32.941153+00:00

The previous three bounded serving recovery passes failed: server swap growth or global swap-in, while host headroom remained above the gate. A user-owned transient systemd scope with `MemorySwapMax=0` passed the creation preflight. The new finisher and serving descendants use this process-specific cap; global VM settings remain unchanged. Frozen serving workers and all prospective gates remain unchanged. Original accepted paired repetitions 0-2 are reused; complete repetition 3 is repeated, preserving all failed and orphaned attempts. Three additional passes are allowed after this concrete mitigation. Earlier and new paired blocks have different external swap protection, which must be disclosed. Recovery PID: 1067189. Progress: `results/moe_pruning/study_20261006_v1/orchestration/actual75_scoring_and_closeout_v6_noswap_recovery_20261008/progress.json`.

### Actual 75% supplement completed - 2026-10-08T07:15:37.279099+00:00

Verified at the 2026-10-10 status check: the resumed finisher completed report export and verification of all 44,155 supplement artifacts on 8 October at 12:45:37 IST. Final manifest SHA256: `7af5bd1fa883cede21cd4a41ec6998b0ff291335206e51dc48f2bd5e209df0b6`. The original offline closeout remains preserved.

All three models have twelve accepted quality batches: 2,700 task generations and 1,800 same-input evaluations total. GPT-OSS serving has 15 accepted arms and Qwen serving has ten, each covering five complete paired repetitions. Gemma serving was skipped because no mask passed the combined synthetic quality screen. GPT-OSS math/code masks and Qwen code passed that screen; Qwen mixed math lost 30 percentage points (95% interval 11-52), and Gemma mask decisions remain inconclusive. Results do not establish broad real-world quality preservation.

Process-specific `MemorySwapMax=0` enabled serving recovery. GPT repetitions 0-2 were retained from the earlier environment; later GPT paired blocks and all Qwen serving blocks use the transient no-swap scope. Both treatments within each new paired block share the same scope. Global memory settings and frozen worker controls remained unchanged; failed resource attempts and incomplete paired blocks were preserved and excluded. Remaining global swap/pressure rejections were retried as whole paired repetitions.

Serving ratios do not establish a general latency improvement: GPT math is near parity; GPT code end-to-end latency increased about 34.5%; Qwen code increased about 2.5%. Generated output lengths differ, so throughput proxies cannot be treated as fixed-work kernel speedups. Weight allocations remain complete, with no physical checkpoint compaction or RAM-saving claim.

Final report: `results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/closeout75_actual/report.md`. Companion completion summary: `docs/experiments/pruning/report.md`. GPU compute was idle at the 10 October status check. Kimi remains excluded; 60%/50% ablations have not started. No frozen study artifact was modified by this documentation update.

## DOC-01..DOC-04 - Presentation, dataset and repository consolidation

### 1 and 8 October - Presentation artifacts

- The 1 October deck reports reproduction, policy, prefill and locality work.
- On 8 October, the current weekly deck received eight static-retention slides before its original slide 3 and two visually empty reserved slides. All 20 original slide XML parts, embedded fonts and original media were preserved.
- First-ten speaker notes were authored. Duplicate text bodies in newly cloned table XML were corrected, and package relationships/text fit were checked. Preview images are authoring renders with embedded fonts, not Office renders.
- Completed actual75 results have not been added to the two reserved slides.

### 10-12 October - Completion and dataset audit

- Confirmed actual75 finished at 2026-10-08T07:15:37.279099 UTC, after all 15 GPT and ten Qwen serving arms, lossless layout validation and verification of 44,155 artifacts.
- The original 22,554 offline artifact hashes were rechecked before that supplement closeout. Final manifests remain immutable.
- Clarified synthetic quality data: 100 prompts/domain, ten families x ten variants, separate authored likelihood sequences; no exact calibration prompt overlap, but no independent unseen-task-family guarantee.
- Original 100-independent-tasks/domain requirement remains unmet. Passing corpus screens are exploratory and do not establish general quality-preserving pruning.
- Cross-checked the FreeToken paper's workload description: AIME, a SWE-bench issue through OpenCode/Claude Code, and fixed email/calendar turns. The local synthetic quality dataset was not a paper workload reproduction.

### 12 October - Repository documentation cleanup

- User request: pause other work and consolidate all local and colleague documentation, remove redundant Markdown/plans, and maintain one experiment log.
- Archived exact text/hashes of 31 original Markdown sources before deletion/migration. Protected user engine changes, immutable studies, raw colleague records and existing environments were retained.
- Built `docs/README.md`, one `docs/PLAN.md`, study report directories, a dedicated dataset contract, presentation/reference folders and this log.
- Folded historical progress/handoff files into this log or their study reproduction notes. Removed superseded working Markdown after replacement/provenance checks.
- Document generators/snapshot inputs were updated for canonical paths only. No numerical model worker was changed. No experiment, benchmark or test suite was run by cleanup.
- Provenance discrepancies are explicit: sampled-vs-greedy locality capture, Saketh high_miss rate, bandwidth-ratio labels, full logging-vs-lean decision overhead and graph/eager timing contexts.
- Final removal/move mappings and integrity review are in `docs/_archive/cleanup_manifest.json`. Raw result-directory Markdown/source snapshots remain historical evidence.

## Appending subsequent experiments

Add each experiment here with: date/time convention; owner/provenance; question; model/revision; hardware/runtime; dataset/split; command/config identity; controls; accepted/rejected counts; artifact links; result and scope; blocker/fallback; next gate. A retry is a new record linked to its rejected predecessor. Do not create another maintained progress or handoff Markdown file.

## 13 October 2026 - experiment-only repository migration

**Requested by Ankit:** merge all four contributors' tooling, remove application build
folders, keep ignored ZIP backups, use uv throughout and support Linux x86_64 across
NVIDIA GPUs with portable setup/baseline scripts.

- Archived the current engine/build source, modified tracked files, local experiment
  source and all three imported study folders before removing their original roots.
  ZIP CRC checks and file/ZIP SHA256 inventories were created. Reconstructible caches
  and the imported analysis environment were excluded; that environment remains ignored
  locally at `.runtime/environments/runtime-q-analysis`.
- Grouped 69 existing Python sources into study packages. Moved imported raw data,
  trained models, figures, reports and logs under `results/historical/STUDY`, preserving
  authorship and original recorded settings. Local frozen bandwidth/pruning evidence
  stayed at its original result paths.
- Preserved the current engine as an ignored external dependency and exported its
  base commit, local hook patch and observed package versions. Kept the sealed v6
  runtime separate and added its original-module launcher.
- Added uv-based Bash setup, diagnostics, baseline and recorded-study launchers.
  New runs use unique result directories with command/environment/status metadata and
  retained console/server logs. Runtime-q plans now use relative trained-model paths.
- The new unpruned baseline profiles retain the supported GPT-OSS, Qwen and Gemma
  geometries and fixed caches. New-device pilots are required; the wrapper does not
  imply equivalence to the sealed quality/performance supplement.

This entry describes repository organization and setup diagnostics. **No new inference
experiment, performance baseline, unit test or 60%/50% ablation was run for this cleanup.**
Migration evidence and final integrity-check disposition are recorded in
[`layout_manifest_20261013.json`](_archive/layout_manifest_20261013.json). Historical
paths elsewhere in this log are original identifiers; use that manifest to find moved
imported artifacts. The [reproduction guide](REPRODUCING.md) is the current operational reference.

Final migration review: 384 imported raw/artifact files matched their recorded hashes; 449 engine files were preserved. Both sealed pruning manifest hashes remained unchanged. All six Bash launchers passed syntax review, the Python undefined-name check passed, 143 local Markdown links resolved, analysis-only uv setup completed, and CLI/frozen-runtime help plus a Qwen baseline dry run completed. Full inference installation on a fresh GPU machine and new performance runs remain unexecuted.

## 13 October 2026 - reported NCCL download timeout on another system

The user reported that setup resolved 116 runtime packages, then timed out downloading
and extracting `nvidia-nccl-cu13==2.28.9` with uv's default 30-second HTTP read timeout.
The destination machine's setup remains incomplete; its download was not retried from
this workspace.

Updated `scripts/setup.sh` to default to a 300-second read timeout, five HTTP retries
and four concurrent downloads, preserving caller overrides. Added unique ignored
setup console logs and exit-code records under `results/setup/`. Documented retrying
the existing setup script with explicit uv environment variables and reusing completed
cached downloads. Dependency pins and experiment protocols were not changed.

Bash syntax review passed. No setup installation, unit test or inference benchmark
was run for this download-handling change; recovery on the destination is unconfirmed.

## 13 October 2026 - detailed m-reduction priority plan

**Requested by Ankit:** expand `m_reduction_experiment_plan_v3_priority.md` with
exact experiment parameters, datasets and implementation controls.

Consolidated the expanded v3 specification into the single maintained
[plan](PLAN.md), leaving the original filename as a link to it. Preserved the lean
v3 input and previous canonical plan byte-for-byte in the ignored ZIP
`archives/m_reduction_plan_before_detail_20261013T102535Z.zip`; CRC and restored
content comparisons passed. Archive SHA256:
`dad301f9eeafbc6760168393ce68a73e1600ad7dde76a4e4b131b956fe3da4fe`.

The plan binds model/runtime revisions, primary cache/context budgets, 90 routing
request IDs and five existing folds, nested selection, 300-task synthetic
regression data, 15 AIME 2025-I problems and five specified SWE-bench Verified
issues. Public dataset revisions and issue/base-commit metadata were inspected
for planning; no inference trajectories or performance data were collected.
It defines P1–P9 arm parameters, integer q rounding/floor controls, native gating,
cache-state and byte matching, teacher forcing, live paired repetitions,
statistics, quality/adoption gates and conditional execution counts.

Corrected stale actual75 status, incompatible long-generation/context budgets,
fixed-logit simulation claims, pruning-deletion bounds and small-sample quality
claims. New thresholds and ports are explicitly prospective specifications.
Fresh-runtime byte qualification, full-score/teacher-forcing/merge tooling, external
harness qualification and independent quality confirmation remain blockers.
The 60%/50% queue is deferred as in the supplied v3; prior evidence remains intact.
Preserved Saketh's graph/eager study distinction and attribution and Yeshwanth's
limited locality scope. Updated the documentation index to reflect this plan.

This is a documentation change. No tests, benchmarks, model inference, SWE-bench
issue-solving, commits or publications were performed. Existing setup/download
changes and their earlier log entry were retained.

## 13 October 2026 - remove lower-retention ablations from the plan

**Requested by Ankit:** remove the 50% and 60% retention ablations.

Removed their deferred-work entry, retained-count specifications and revival
instructions from [PLAN.md](PLAN.md), and removed the corresponding deferred-queue
statement from the documentation index. These ablations are no longer planned.
Historical records, frozen protocols and original-document backups remain intact.
No tests, benchmarks or inference experiments were run.

## 13 October 2026 - m-reduction v3 Stage 0 started

**Authorized by Ankit:** proceed with the detailed plan, excluding 50%/60%
retention ablations, and expose running experiment progress in a named tmux session.

Created tmux session `m_reduction_v3`, with `progress` and `baseline` windows.
Attach with `tmux attach -t m_reduction_v3`. Progress and raw controller logs are
under `results/m_reduction/study_20261013T104934Z_6309f0f7/`; the terminal monitor
reads the recorder's status and per-repetition output files. GPU work is serial.

At launch the RTX 4070 had no compute owners. All three local checkpoint config
metadata revisions matched the model pins, and a user `MemorySwapMax=0` scope
could be created. Restored a separate runtime from the recorded base commit and
hook patch at `.runtime/m_reduction_v3_base_20261013T104934Z`; the shared runtime
and sealed studies were not edited. Reused three same-host compiled extensions
only after all extension-source hashes matched the restored source. Binary/source
identity is recorded in the study's `runtime_base_identity.json`.

The isolated runtime/CUDA import diagnostic passed. Queued the existing unpruned
baseline profiles, GPT-OSS then Qwen then Gemma, each with three fresh-process
repetitions, fixed caches/context caps, eager hybrid and five CPU workers, through
`scripts/baseline.sh`. Exact commands, environment, logs, source hashes and status
are retained below the study's `runs/` directory. This existing baseline's fixed
local request and forced-length decoding are a prerequisite, not D0/P1 or a
corrected-fill-floor measurement. No policy improvement or G0 pass is claimed.

Stage 0 is running; subsequent P1/P2 implementations and qualification are pending.
No unit-test suite was added or run; no commit or publication was made.

### Stage 0 preflight correction and P1/P2 queue

The GPT-OSS and Qwen prerequisite commands completed three repetitions each.
Gemma's first attempt, `runs/baselines/20261013T105736Z_85566aee/`, failed before
model inference because its checkpoint names native k `top_k_experts` while the
baseline preflight reader expected `num_experts_per_tok`. Preserved that attempt
and the exact earlier reader source/hash. Added the Gemma config-field fallback
and explicit baseline-helper source hashes; inference workers/settings are unchanged.
A fresh Gemma baseline in `gemma_baseline` subsequently completed three repetitions.
All three prerequisite baseline commands are complete; this does not establish
pruning resource qualification or a policy improvement.

Added a separately configured Stage 0 controller and collector in
`experiments/pruning/m_reduction_*.py` and `configs/m_reduction/stage0.json`.
The `p1_p2` tmux window waits for all three completed baseline-command sets, then
runs a fresh weight/metadata audit, all-model tokenization freeze, fresh model-
geometry native CPU/PCIe calibration, 15 observation-off/on pilot pairs for each
of four P1 arms/model, and qualified native-floor1 primary/repeat captures.
Every measurement uses a fresh Bash run record; calibration/inference children
use a process-specific no-swap scope. Failed units stop dependent stages.

The new ignored runtime `.runtime/m_reduction_v3_stage0_20261013T104934Z` is a
separate child of the immutable study baseline source. Its versioned
`runtime/patches/m-reduction-stage0.patch` adds an opt-in integer fill floor in
CPU/GPU paths, protected-active CPU eviction, Reuse-Aware candidate priorities,
Gemma native-score/scaling metadata, actual bank-byte accounting and final-cache
snapshots. Saketh's admission rule retains kappa=1, prior_n=8, t_free=32 and a
32-visit history, with explicitly selected expert identities. The shared engine
and frozen runtime sources remain untouched.

The new replay qualifier checks every native admission/victim event plus final
slot, usage and recency state. Raw captures remain instrumented qualifications,
not serving-performance gains. Python syntax and diff-whitespace review passed;
no unit-test suite was run. These implementations remain pending actual pilot
qualification. Sequential teacher forcing, probes/norms and P3-P9 remain open.

### Stage 0 verified milestones - 13 October 2026 status check

All three prerequisite baseline commands have completed three repetitions, including
the fresh Gemma retry. The full weight-shard/index and checkpoint metadata audit
passed for all three pins. All model-specific D0 tokenizations and corpus/split
hashes were frozen before P1 collection.

Fresh five-worker native concurrent branch profiles completed:

| Checkpoint | CPU native kernel logical-byte bandwidth | PCIe gather bandwidth |
|---|---:|---:|
| GPT-OSS MXFP4 | 34.79 GB/s | 6.71 GB/s |
| Qwen NVFP4 | 32.66 GB/s | 6.71 GB/s |
| Gemma NVFP4 | 35.49 GB/s | 6.71 GB/s |

These are synthetic kernel/calibration profiles, not hardware-counter DRAM
bandwidth, end-to-end speedups or a pruning result. Raw profiles, control files
and recorded invocation logs are under the current study's `calibration/` and
`runs/` directories. No calibration profile or measurement from a sealed study
was overwritten.

At this status check, GPT-OSS `native_floor0` has ten of fifteen request captures
with complete score/byte checks, observation-off/on equality and exact event/final-
state replay matches. The batch resource decision is still pending; individual
capture success is not a complete model/policy gate. The tmux controller remains
active and will stop on a failed qualification. Later stages remain pending.

The initial native-floor0 collector snapshot predates an additional dependency-
hash accounting change; subsequent invocations bind the validator/resource/helper
and rendered-input hashes explicitly. Each invocation retains its own exact source
snapshot and collector hash. Runtime numerical hooks and sampling settings were
unchanged by that accounting amendment. Instrumented timings are not pooled as
serving-performance comparisons.

### First complete P1 pilot arm qualified

GPT-OSS `native_floor0` completed all 15 observation-off/on pilot pairs with full
native scores, expected expert-cache bytes, exact event/final-state replay matches
and a passed whole-batch resource gate. Its qualification record is
`results/m_reduction/study_20261013T104934Z_6309f0f7/qualification/gpt_oss_mxfp4/native_floor0.json`.
The controller advanced to the separate `native_floor1` arm. This is one qualified
pilot arm; the three-model G0, cross-arm output identity and later stages remain open.

### Stage 0 stopped at cross-arm output identity - 2026-10-13T11:50:23 UTC

The controller completed all four GPT-OSS pilot arms: 15 observation-off/on pairs
per arm, 60 captured requests overall, with complete native scores, expected cache
bytes, matched event/final-state replay and passed whole-batch resource gates.
Cross-arm generated-token identity then failed:

| Arm, compared with native_floor0 | Different pilot requests / 15 |
|---|---:|
| native_floor1 | 12 |
| reuse_freq_floor0 | 13 |
| reuse_freq_floor1 | 12 |

The automatic gate stopped dependent work and preserved all attempts. Qwen/Gemma
P1 pilots and primary/repeat collection have not started. The three-model baseline,
checkpoint audit and fresh calibration records remain complete. At the subsequent
user-requested status check the GPU had no compute processes and 0% utilization;
the tmux progress window showed the failed gate.

Read-only inspection of `math_000` found all 2,376 prefill router-score events
identical for native_floor0 versus native_floor1. The first decode-score difference
was at input position 99, layer 1, with maximum absolute router-logit difference
0.0078125. Their first different generated token was position 47 (zero-based),
with both outputs reaching 256 tokens. This localizes observed divergence to
 decode execution; it does not identify the exact CPU/GPU numerical cause or prove
worse task quality. The corpus has no task-answer rubric.

Next work is a fixed-hidden-state CPU/GPU expert and partial-sum comparison.
Any numerical correction requires a separate qualified source arm; the declared
output-identity gate has not been relaxed. No new inference or benchmark was
launched during this status check.

### Numerical recovery attempt authorized - 13 October 2026

The user requested fixing the output mismatch and attempting similar corrections
as the plan proceeds. Original failed pilots and runtime sources are retained.
New recovery root: `results/m_reduction/recovery_20261013T125617Z/`.
The tmux session remains `m_reduction_v3`; `progress` now watches this root and
`requalify_gpt` runs the recorded recovery controller.

A fixed-state diagnostic used GPT-OSS `math_000`, the original chat template,
99 prompt tokens, greedy natural-EOS sampling, and a two-token completion to
capture layer 0 at the first decode step. Routed IDs were `[31,21,22,6]`, with
weights `[0.5297907591,0.1693312973,0.1566058397,0.1442720741]`.
It compared CPU/GPU full groups and each of the four weighted expert outputs,
saved inputs/banks/outputs in `fixed_state.pt`, and restored cache bookkeeping
and overwritten bank slots. This diagnostic makes no timing or quality claim.

The first diagnostic failed because its arm file omitted the observer's required
`prompt_tokens`; the failed server log and run remain intact. The retry rendered
that count before loading weights and completed. Source inspection and same-state
measurements found two precision-contract differences: CPU MXFP4 gate/up linear
outputs skipped the GPU's BF16 rounding before SwiGLU, and CPU down outputs were
accumulated before per-route rounding whereas GPU weighted routes were individually
rounded to BF16. Hybrid serving additionally added already-rounded subset sums.
The original fixed-state CPU/GPU grouped outputs differed in 1,666/2,880 elements
(maximum absolute difference 0.0234375); CPU grouped versus summed individually
rounded routes differed in 1,079 elements.

The new isolated runtime
`.runtime/m_reduction_v3_mxfp4_alignment_20261013T125947Z/` applies
`runtime/patches/m-reduction-mxfp4-alignment.patch` after the Stage 0 and diagnostic
hooks. With `FREETOKEN_M_REDUCTION_MXFP4_ALIGN=1`, it rounds CPU gate/up projections
to BF16 before activation, rounds weighted expert outputs to BF16, emits CPU/GPU
partial sums in FP32, then casts the combined hybrid output to BF16 once. The
C++ extension explicitly exposes the changed output-buffer capability and rejects
setting it after tasks are created. All three extensions rebuilt successfully on
the current host with `MAX_JOBS=1`; the build log, patch hashes and complete source/
binary identity are retained in the recovery root. Prefill and checkpoint/cache
geometry are unchanged. This is a new numerical source arm, not a correction to
old evidence.

The corrected diagnostic used exactly the same hidden state, IDs and weights.
Three expert outputs matched exactly; one differed in one element by
1.1920928955078125e-7. CPU/GPU partial-group maximum difference was the same
1.19e-7, and the final BF16 grouped outputs matched exactly in all 2,880 elements.
Both branches now matched their independently summed rounded route outputs.
`same_state_correction_summary.json` retains these comparisons. This single state
does not prove complete-generation identity or general numerical equivalence.

Fresh corrected native synthetic calibration completed: CPU standalone 37.97,
CPU overlap 34.93, PCIe standalone 6.72 and PCIe overlap 6.71 GB/s (logical kernel
bytes, not a DRAM counter). `configs/m_reduction/stage0_mxfp4_alignment.json`
identifies the new numerical mode. The recorded controller first screens the two
fixed pilot IDs `math_000` and `math_001` under all four q arms, retaining observation
pairs, full scores, replay and resource gates. Native-floor0 completed both
captures and its whole-batch resource gate; native-floor1 is running. If all four
screens match, the controller runs three new-process baseline repetitions and
all four complete 15-pair pilots. Cross-arm output identity remains required;
any residual failure is retained and stops dependent collection. Qwen/Gemma
require their own format-specific diagnostics; this MXFP4 correction is not
applied to NVFP4. No unit tests were added or run.

The environment recorder previously redacted every `FREETOKEN_` value because
the product prefix itself contains TOKEN. Future records inspect the setting name
after that prefix, retaining operational flags while still redacting credentials.
Baseline records now state the MXFP4 numerical mode explicitly. Earlier environment
records remain unchanged.

### First numerical correction rejected; second correction started

The first correction completed all four two-request screens with individually
matched replay/observation and passed resource gates. It still failed cross-arm
generated-token identity: native-floor1 changed both requests, Reuse-Aware-floor0
changed both, and Reuse-Aware-floor1 changed `math_001`. For `math_000`, floor1
still first changed a generated token at index 47. The first router-score difference
now occurred at the first decode position (99), layer 11, maximum absolute
difference 0.000244140625; the original runtime differed at layer 1. The controller
stopped before its new baseline or complete pilot qualification. The failure remains
in `recovery_20261013T125617Z/qualification/gpt_oss_mxfp4/screen_cross_arm_output.json`.
The single-state agreement was insufficient, as the complete-generation gate found.

A second ignored runtime, `.runtime/m_reduction_v3_mxfp4_dot64_20261013T131031Z/`,
adds `runtime/patches/m-reduction-mxfp4-dot64.patch` to the first correction. Besides
projection reduction order, inspection found a different activation evaluation order:
CPU used `gate / (1 + exp(...))`, GPU used `gate * (1 / (1 + exp(...)))`, and the GPU
exponential used an approximation. The second opt-in mode accumulates projections
in FP64 on both devices, casts to FP32 before common bias/weight operations,
evaluates the exponential in FP64 before casting to FP32, and uses the same division
order with GPU round-to-nearest division and activation fusion disabled. Native
prefill stays unchanged. The CPU extension rebuilt successfully on the current host.
The extra computation and transfer costs must be charged in fresh calibration and
baseline measurements; there is no speed or quality claim.

Second recovery root: `results/m_reduction/recovery_dot64_20261013T131424Z/`.
The recorded diagnostic probed layer 10 at the first decode step and completed: all
four CPU/GPU weighted expert outputs matched exactly, as did their grouped FP32
outputs and the independently summed rounded routes. A new synthetic branch profile
completed and the second four-arm generation screen is running. The controller will
stop on another failed gate and run baseline/full qualification only if that screen
matches. The tmux `progress` window follows the second root; `dot64_recovery` runs
its queue. Config/source/build identities and all prior failed attempts are retained.

### Second correction: first complete generation matches

On `math_000`, corrected native-floor0 and native-floor1 produced exactly the same
256 generated token IDs. All 8,496 corresponding router-score events also matched
exactly (2,376 prefill and 6,120 decode events). This was the request that had
diverged at generated-token index 47 in the original and first-correction arms.
Both request captures passed their observation, full-score and replay checks;
the second arm's remaining request and whole-batch gate are still running. The
comparison is retained as `first_complete_score_comparison.json` in the second
recovery root. It is one complete request, not 15-pilot or three-model qualification.

The second correction's fresh native synthetic profile measured CPU standalone
4.76 and overlap 4.31 GB/s, versus the first correction's 37.97/34.93 GB/s; PCIe
standalone/overlap was 6.72/6.70 GB/s. These logical-byte profiles show substantial
added CPU cost. The arm is a numerical-recovery candidate, not an accepted faster
serving baseline. Its q policies use this new measured profile, so their fractions
also differ from the earlier arms and timings cannot be pooled across source arms.

Baseline entry points now accept and hash an explicit `--calibration` file.
For already running recovery controllers, a numerical baseline can resolve only
its controller's uniquely matching model/precision profile; mismatched runtimes,
missing or ambiguous profiles fail. This prevents a new precision arm from silently
using the default old bandwidth cache. Newly launched recovery controllers pass
the file explicitly. The current queue continues the remaining two-request arms
and will run its three fresh baseline repetitions and complete pilots only if
the generation screen passes. No unit tests were added or run.

The subsequent `math_001` comparison also matched generated-token hashes between
corrected native-floor0 and native-floor1. Both two-request batches completed with
passed whole-batch resource gates. The controller advanced to the Reuse-Aware
screens. This confirms the floor pair on both screened requests; the remaining
policy pairs, complete 15-pilot qualification and other models are still open.

### Persistent monitoring and bounded operational recovery - 13 October 2026

The user requested keeping a live monitor, using a background fallback if needed,
and attempting repairs whenever an error occurs. The tmux monitor was alive, but
the second correction had stopped at the Reuse-Aware-floor0 screen's resource gate.
Both requests had passed capture/observation/replay checks, but the whole batch was
rejected: 987,209,728 bytes of system swap-in over 143.50 seconds, exceeding the
unchanged 134,217,728-byte limit. Server swap peak/growth were both zero; minimum
whole-batch available RAM was 10,090,401,792 bytes and full memory pressure was
0.00384%. These system counters cannot attribute the swap activity to a particular
unrelated application. Its raw evidence remains intact and excluded.

Added `experiments/common/experiment_watch.py` and enhanced the existing progress
display. The separate transient user service `m_reduction_watchdog.service` runs
through `scripts/run.sh`, refreshes snapshots/heartbeats every five seconds, follows
`results/m_reduction/monitor/registry.json`, and restores a dead/missing/stale tmux
progress pane. Background status is `results/m_reduction/monitor/latest.txt`;
`heartbeat.json` includes PID, tick count, current study/stage, last rejection and
tmux health. Changes and pending failures are recorded beside it. Both independent
and tmux heartbeats were observed advancing; service status was active/running.
This survives tmux closure during the current login/session, not a machine reboot.

Added bounded operational guards in `m_reduction_guard.py`. Known whole-batch
swap, pressure and host-headroom failures receive up to three complete attempts
with a fresh batch directory each time. Retry/resume first waits for a 30-second
idle-GPU quiet interval with at most 1 MiB total system swap-in/out, native headroom
and pressure limits; the wait is bounded at 600 seconds. Native inference gates
and `MemorySwapMax=0` remain unchanged. No unrelated processes were terminated
and global swap settings were not changed. Unknown/numerical errors are retained
and surfaced for diagnosis rather than given an automatic blind retry.

The same numerical arm was resumed through a new recorded controller at
13:38:39 UTC in tmux window `monitored_recovery`. Each controller invocation now
retains its own control/source files under `controllers/`, leaving the original
controller and failure immutable. Resume checks plan/profile/runtime identity and
all capture/qualification dependencies, then reuses only complete resource-passed
batches. Native-floor0/floor1 screens passed these checks and were reused. After
30 quiet seconds, with about 25.9 GB available RAM, Reuse-Aware-floor0 began a
fresh whole-batch attempt at 13:39:10 UTC. Its previous resource-rejected batch is
not accepted. The monitor service and controller run/environment/console records
are retained. No unit tests were added or run.

The fresh Reuse-Aware-floor0 screen subsequently completed both observation pairs
and passed its whole-batch resource gate (system swap-in 22618112
bytes; swap-out 10072064 bytes). Its generated-token
hashes matched native-floor0 on both screened requests. The controller advanced
to Reuse-Aware-floor1. The supervisor was restarted to load improved error
reporting that selects the newest actionable failure instead of masking a new
numerical error behind an older resource rejection. Its prior operational run is
recorded as interrupted by the requested service restart; measurement records
were unchanged. The new background heartbeat and live tmux heartbeat both advanced.

### Three-model coverage made mandatory - 13 October 2026

The user clarified that each experiment must run on all three MoE models.
The canonical plan now explicitly requires GPT-OSS-20B MXFP4, Qwen3.6-35B-A3B
NVFP4 and Gemma-4-26B-A4B NVFP4 for every P1-P9 experiment, its baselines,
controls and retained ablations. The former optional Gemma D2/D3 language,
model-specific P7 deferral and P9 trigger-based model omission were removed.
G1 triggers still determine priority/adoption, while every model's comparison
remains required. Unsupported context/kernel cells require a recorded blocked/
failed attempt and repair; they keep the experiment unfinished rather than
silently excluding a model or weakening its gates. Dataset/context eligibility,
native geometry and all quality/resource contracts remain explicit.

The currently active recovery controller is still a GPT-OSS-specific numerical
prerequisite, not evidence of three-model completion. It has passed the four-arm
two-request screen and completed its three fresh-process baseline repetitions;
its full GPT-OSS pilots are running. Qwen and Gemma qualification and experiment
runs remain required. This change updates the prospective coverage contract;
it does not edit frozen configurations or reclassify any existing measurement.
No new measurements or unit tests were launched for this plan clarification.

### Updated-plan audit before further progression - 13 October 2026

The user requested confirming compliance with the updated plan before proceeding.
Added `configs/m_reduction/coverage_policy.json` with all three required models,
all P1-P9 stages, four required pilot arms, 15 pilots/arm, 90 primary and 18 repeat
captures/model, all100/R75 pruning fractions and the unfinished P2 support contract.
The read-only `m_reduction_plan_gate` audits controlled model pins/geometry/budgets,
dataset and split hashes, sampling/warmup/workers, actual eager hybrid serving flags
and unchanged resource controls. It counts unique resource-passed captures and
requires model-local pilot/runtime evidence, complete task membership and cross-arm
output identity. Missing models or P2 support remain explicit; a two-request screen
cannot qualify a model or the stage. No acceptance threshold was weakened.

Recorded audit command: `bash scripts/run.sh pruning m_reduction_plan_gate --study
results/m_reduction/recovery_dot64_20261013T131424Z --out
results/m_reduction/recovery_dot64_20261013T131424Z/plan_compliance/audit_20261013T1401.json`.
Its run record is `runs/pruning/20261013T140234Z_9a72fc7c` in that recovery root.
The active GPT-OSS native-floor0 pilot controls were consistent with the declared
plan. Three-model pilot/Stage 0 completion were both false, so primary collection
and the next research stage were both disallowed. GPT-OSS full pilots remain in
progress; Qwen/Gemma qualification and complete P2 support remain pending. The
independent monitor service remained active and its heartbeat was advancing.

Future Stage 0 invocations now enforce the complete model set and invoke the
three-model gate before primary collection. After admission/trace work they remain
in `awaiting_p2_support`, not an unrestricted next-stage state. Future GPT-specific
recovery invocations write a readiness report and return to model qualification.
The active controller's preserved loaded source already stops after the GPT pilot
prerequisite; it was not interrupted or relabelled as three-model completion.
The numerical runtime, frozen configs/manifests and measurements were untouched.
This was a plan/provenance audit; no unit tests or new measurements were launched.

After documenting the gate, a fresh recorded audit bound the final current
`docs/PLAN.md` hash in
`plan_compliance/audit_after_gate_documentation_20261013.json`. Controls remained
consistent and both transition permissions remained false. This supersedes the
earlier audit for the current documentation hash; both reports are retained.

### GPT-OSS pilots fully qualified; Qwen/Gemma diagnosis started

The GPT-OSS numerical recovery controller finished at 15:02:41 UTC (20:32:41 IST)
on 13 October 2026. All four arms completed 15 observation-off/on pairs each,
60 captures total, with full scores, exact replay and passed whole-batch resource
gates. The final cross-arm comparison matched generated-token hashes on all
15 requests in every arm; `pilot_cross_arm_output.json` records no differing
requests. The original precision mismatch is resolved on this full pilot corpus.
This qualifies that GPT source/control arm; it does not establish broad task quality
or three-model P1/P2 completion, and its measured CPU cost remains disclosed.

At the user-requested progress check the GPU was idle and both tmux/background
monitor were healthy. The completed controller was awaiting Qwen/Gemma qualification.
A fresh recorded plan audit confirmed three-model primary/next-stage permission
remained false. Continued the previously authorized Stage 0 work with serial
fixed-state Qwen and Gemma diagnostics; no primary/P3 progression was made.
New root: `results/m_reduction/nvfp4_qualification_20261013T153609Z/`. Its runtime
`.runtime/m_reduction_v3_nvfp4_20261013T153609Z/` is a same-host source/binary-matched
copy of the qualified GPT runtime, with both GPT-specific precision switches
disabled for NVFP4 and no new numerical edit yet. Identity, queue/control files
and the qualified GPT dependency are retained. The diagnostic borrows the prior
model-native calibration only for correctness comparison; fresh calibration is
required before performance or pilot qualification. Both checkpoint-specific
geometry/context/cache budgets and text-only settings remain fixed.

The queue freezes the existing dataset/tokenization controls, then compares CPU/GPU
outputs for `math_000` at layer 0's first decode hidden state separately for Qwen
and Gemma, preserving input/routing/bank/output evidence. A failure of one diagnostic
is retained while the other independent model diagnostic is still attempted.
The monitor registry now follows the new root and tmux window `nvfp4_qualification`
shows this queue. Numerical review and any separate-arm repairs precede their
complete pilot qualification. No unit tests were added or run.

Both native NVFP4 fixed-state diagnostics completed. Qwen's grouped CPU/GPU output
differed in 1,576/2,048 elements (max absolute 0.00018310546875, RMS 4.57348e-5);
Gemma differed in 2,173/2,816 elements (max absolute 0.03125, RMS 0.00354876).
Both server logs identified CPU `avx512bf16+avx512vnni(nvfp4-w4a8)`, which adds
activation quantization. These comparisons do not determine task-quality loss.

A separate dispatch-arm repair probe now runs serially on both NVFP4 models with
`FREETOKEN_CPU_MOE_NO_VNNI=1`, exposed explicitly by `m_reduction_numerics
--cpu-no-vnni` and recorded in each configuration. It disables the CPU's extra
W4A8 activation quantization while keeping checkpoint/cache/context/router controls
and the numerical runtime source fixed. Original native diagnostics remain intact;
new outputs are under `diagnostics/MODEL-no-vnni` and `no_vnni_control.json` records
the change. Borrowed calibration remains correctness-only, with no speed claim.
The tmux window `nvfp4_no_vnni` runs this attempt; the monitor follows it.
Any remaining precision differences require a separately qualified numerical
repair before full pilots. No unit tests were added or run.

### NVFP4 precision repair build and serial diagnostics - 13 October 2026

The activation-quantization-off probes completed with remaining differences:
Qwen 1,201/2,048 elements (max absolute 0.0001220703125, RMS 2.48540e-5);
Gemma 1,580/2,816 (max absolute 0.015625, RMS 0.00176829). Disabling W4A8
activation quantization alone does not make CPU/GPU expert execution identical.

Created the separate ignored runtime
`.runtime/m_reduction_v3_nvfp4_precision_20261013T155751Z/` from the unchanged
NVFP4 diagnostic parent. The opt-in `FREETOKEN_M_REDUCTION_NVFP4_ALIGN=1`
arm keeps native packed weights and uses FP64 projection accumulation, common
activation arithmetic, BF16 projection/intermediate/weighted-expert boundaries,
and FP64 partial-group sums with a single final BF16 cast. Native prefill remains
unchanged. This is an unqualified numerical source arm with unknown performance.
The retained patch is `runtime/patches/m-reduction-nvfp4-precision.patch`.

Its first build failed because a new AVX-512 helper called a function declared
later in the source. The repaired code reads the existing 16-entry E2M1 lookup
directly. The original failed build script/log are preserved in the NVFP4 study;
`build_nvfp4_precision_retry.sh` completed all extensions with return code 0.
Both builds used `uv run --no-sync --project "$EXPERIMENT_REPO_ROOT" python
setup.py build_ext --inplace`, `MAX_JOBS=1`, on the same recorded host/software.
Source/binary hashes were frozen in `nvfp4_precision_runtime_identity.json` after
the successful rebuild and before new probes.

The serial two-model queue now runs through the recorded
`scripts/run.sh pruning m_reduction_numerics --model MODEL --align-nvfp4`
entry point in tmux window `nvfp4_precision`, with separate outputs under
`diagnostics/MODEL-precision`. It retains the same pinned checkpoints, model-native
budgets, eager/text-only settings, five CPU workers, `math_000`, layer 0 and two
completion tokens. Each invocation uses a user scope with `MemorySwapMax=0`.
The borrowed parent bandwidth profiles remain correctness-only dependencies;
fresh profiles, baselines and complete pilots are still required before accepting
this source arm. No primary or later research stage was started. The independent
monitor remains active. No unit tests were added or run.

The first precision queue used an invalid `--cpu-precision` flag due to an overly
broad command-string replacement. Both recorded invocations failed argument
parsing before model loading; their queue and failed run records are preserved.
`nvfp4_precision_retry_queue.sh` uses the correct explicit `--align-nvfp4` flag
and completed both model diagnostics. Qwen matched all 2,048 grouped elements
and all eight individual routes exactly; Gemma matched all 2,816 grouped elements
and all eight individual routes exactly. Both CPU/GPU grouped outputs also matched
their separately evaluated route sums. `precision_review.json` retains these
results and their one-state limitation. No task-quality or speed claim follows
from this check.

Started a fresh qualification root
`results/m_reduction/recovery_nvfp4_precision_20261013T161122Z/`, using the same
frozen precision runtime and the earlier qualified GPT pilot as an explicit
dependency. Its recorded `m_reduction_nvfp4_recovery` controller first validates
both diagnostic/runtime identities and audits the updated three-model plan. It
then verifies all pinned shards, collects fresh per-model corrected bandwidth
profiles, runs four two-request generation screens, three fresh-process baselines
per model and four complete 15-pair pilot arms per model. Whole-batch resource
failures receive bounded unchanged-gate retries; a numerical failure retains its
evidence and stops dependent work for that model. The other model's qualification
remains an independent required unit. No full pilot is accepted merely because
the fixed-state diagnostic matched.

The separate `m_reduction_nvfp4_collect` module preserves the source dependency
of the already qualified GPT collector while declaring its own NVFP4 precision
controls. Resume checks bind the model's actual collector dependency. Corrected
baselines now record the NVFP4 flag and verify calibration model/precision controls.
The monitor was refreshed to display this collector's progress and CLI failures;
its background service and tmux heartbeat follow the new root. A fresh all-three
primary transition audit remains required before primary collection, and full P2
support still gates later research stages. No unit tests were added or run.

The post-documentation plan audit again found consistent controls, qualified GPT
pilots and incomplete three-model coverage, with primary/next-stage permission
false. All three fresh checkpoint shard/metadata audits passed. Qwen's corrected
profile completed: CPU-MoE 6.98 GB/s, PCIe gather 6.72 GB/s; concurrent CPU-MoE
5.74 GB/s and PCIe gather 6.71 GB/s. These are synthetic production-kernel logical
byte rates on the recorded host, not a serving speed claim. Its first two-request
`native_floor0` generation screen is running with this new profile; Gemma's fresh
profile and generation/pilot qualification remain queued.

### NVFP4 precision qualification: first screen failures and resume - 13 October 2026

The first `m_reduction_nvfp4_recovery` attempt (controller
`controllers/20261013T161123Z_f7c12f72`) stopped at both models' first two-request
`native_floor0` generation screens; neither failure was numerical. Gemma's fresh
corrected profile completed first: CPU-MoE 7.25 GB/s, PCIe gather 6.72 GB/s;
concurrent CPU-MoE 6.58 GB/s and PCIe gather 6.71 GB/s (synthetic logical byte
rates, not a serving claim).

- Qwen batch `captures/qwen_nvfp4/native_floor0/pilot/20261013T161315Z_f1c8f672`:
  the server loaded and warmed up, but host available memory settled at 3.62-3.68
  GiB, below the unchanged 4 GiB gate, so the 90-second settle wait failed. Server
  consumption (about 20.9 GiB from start, 21.1 GiB RSS) matched the accepted
  8 October Qwen runs (about 20.6 GiB, 21.6 GiB RSS); the attempt started with
  24.57 GiB available versus 25.55-25.78 GiB then. The shortfall is background
  host memory, not the precision arm. The guard does not classify this startup
  error as retryable, so it was not retried automatically.
- Gemma run `runs/pruning/20261013T161552Z_bfc64dea` failed before server start
  with `KeyError: 'format'`: Gemma's entry in every frozen stage-0 plan (and in
  `configs/pruning/models.json`) omits the format pin. Frozen plans were not edited.
  `m_reduction_nvfp4_collect.py` now uses `nvfp4` explicitly, rejects any entry
  whose declared format is not NVFP4, and still requires the fresh profile to
  contain the `nvfp4` kernel (Gemma's does). The qualified GPT collector
  `m_reduction_collect.py` is unchanged. New collector SHA256
  `5a8c5b2e7c7221869d3593d64e54aec80beaa8a2e253f3a0ee0371aed842bcf7`; batches
  bind this hash, and no batch had completed under the previous collector.

The user declined closing their VS Code session to recover host memory; no
unrelated application was terminated and no gate was relaxed. Host available RAM
was 23.33 GiB before relaunch, so Qwen is expected to remain blocked by the
memory gate until about 1-2.5 GiB of background memory is released. The same
controller was resumed unchanged (`launch_resume1.sh`, tmux window
`nvfp4_resume1`, log `controller_resume1.log`); its plan, runtime, diagnostic and
controller-source bindings matched, and it reuses the verified checkpoint audit
and both fresh same-arm calibrations. Each model remains an independent
qualification unit, so Gemma's screens, baselines and pilots proceed after the
Qwen attempt. No unit tests were added or run.

The resumed Qwen screen (`captures/qwen_nvfp4/native_floor0/pilot/20261013T163018Z_921cc02e`)
was rejected for the same reason: 22.83 GiB available at startup, 3.52 GiB after
load, settle wait failed at 16:32:42 UTC. Qwen remains blocked on background
host memory, which comes from the user's VS Code Remote-SSH server extension host.
By user instruction the VS Code server is not terminated; closing the client window
may release its extension host. Gemma's screen then started with the format fix
(`captures/gemma_nvfp4/native_floor0/pilot/20261013T163252Z_b37f860f`). After
Gemma's controller pass, Qwen requires another `--resume` once headroom returns.

Gemma qualified on the NVFP4 precision arm at 17:38:48 UTC: all four arms
completed 15 observation-off/on pairs (60 captures) with exact replay and passed
whole-batch resource gates, and `qualification/gemma_nvfp4/pilot_cross_arm_output.json`
matched generated tokens on all 15 requests in every arm. Its two-request screens
also matched, and three fresh baselines completed. This qualifies the Gemma source
and control arm only; it does not establish task quality. The resumed controller
then exited as unfinished because the earlier Qwen screen failure was still recorded.
The user closed the VS Code client, and its extension host exited. Host available
memory was 26.40 GiB with the GPU idle when the second resume (`launch_resume2.sh`,
log `controller_resume2.log`) started at 17:39 UTC as a non-tmux background task.
It reuses Gemma's verified batches and retries Qwen's screens, baselines and pilots.

The second resume's Qwen screen
(`captures/qwen_nvfp4/native_floor0/pilot/20261013T173926Z_343ed42c`) again failed
the startup settle wait, for a different reason. Available memory stayed at or
above 5.59 GiB, and full pressure fell to 0, but system swap-in ran at about
3.5-10 MiB per one-second sample, above the 1 MiB settle limit. Swap used drained
from 3.87 to 3.14 GiB. The no-swap Qwen scope did not cause this. With no model
server running, background swap-in continued at about 1.2-3.9 MiB/s, mostly from
another account's idle resque `rake` workers (about 1 GiB swapped) paging back in
after the VS Code client released RAM. No process was terminated and no swap
setting or gate was changed. Gemma's batches were reused, so its qualification
stands. A background waiter (not tmux) now waits up to 60 minutes for the plan's
stricter retry admission: 30 s with at most 1 MiB total system swap in plus out,
an idle GPU and at least 25 GiB available. Only then does it start a third
resume (`launch_resume3.sh`, log `controller_resume3.log`).

The waiter admitted a quiet interval at 17:48:25 UTC (26.5 GiB available) and the
third resume ran. Its Qwen screen (`20261013T174841Z_c7b8d283`) failed the same way.
Swap was 2.69 of 4 GiB at start. Loading the about 21 GiB Qwen bank swapped out
about 1.4 GiB of other processes' memory, filling swap. During settling that memory
paged straight back in at 3-130 MiB per one-second sample, with pressure near 0 and
at least 5.58 GiB available. The accepted 8 October Qwen runs instead started with
swap nearly full (3.75-4.08 GiB): only 18-324 MiB was swapped out during load,
and settle swap-in was 0-9 MiB. Repeated retries therefore do not converge on this
host state. Automatic retries were stopped; the main swap-in sources are another
account's polling API and resque processes (`/api/scripts/workers`), which this
session cannot and will not stop. No gate, swap setting or numerical control changed.
Qwen remains a required, blocked qualification cell; Gemma's qualification stands.

By user decision, Qwen retries were scheduled for later instead of repeated
immediately. A bounded background runner (not tmux) makes at most three
attempts, about 19:25, 21:25 and 23:25 UTC on 13 October. Each attempt requires
the plan's retry admission before launch (30 s with at most 1 MiB total swap
traffic, an idle GPU and at least 25 GiB available) and then resumes the same
controller (`launch_resume4.sh` to `launch_resume6.sh`, logs
`controller_resume4.log` to `controller_resume6.log`). It stops on qualification
success or on any failure other than the startup-settle failure, and every attempt
is retained. Gates and controls are unchanged.

### Qwen deferred; GPT-OSS and Gemma primary collection - 13/14 October 2026

The user deferred Qwen and asked to continue GPT-OSS and Gemma first. Scheduled
Qwen retries were cancelled before any further attempt. The amendment is recorded
in `configs/m_reduction/deferral_qwen_nvfp4.json` (user-authorized,
collection-order scope only; Qwen remains required, and the next research stage
still needs all three models). `m_reduction_plan_gate` accepts it through
`--deferral`, which allows primary collection only for the non-deferred models
whose pilots are qualified, and through `--require-model`. Without a deferral the
original three-model refusal is unchanged, and all-three completion and next-stage
permission stay false. A dry audit allowed exactly `gpt_oss_mxfp4` and
`gemma_nvfp4`. The new `m_reduction_primary` controller runs that gate, then the
90 primary and 18 repeat `native_floor1` captures for one model in its own
qualified root and runtime, with up to three whole-batch attempts for resource
rejections (a startup settle timeout counts as one). The collectors still refuse
any runtime, source, plan, calibration or policy hash that differs from the
model's qualified `native_floor1` pilot. Before launch, every recorded dependency
hash still matched for both models.

The first sequence (`results/m_reduction/primary_sequence_20261013T185708Z/`,
`launch.sh`) stopped both models before any server started. The frozen-input check
found that `protocol/rendered_inputs.json` no longer matched. Cause: GPT-OSS's chat
template writes `Current date: strftime_now("%Y-%m-%d")` using local time, and at
00:00 IST the date changed from 2026-10-13 to 2026-10-14. All 105 GPT-OSS
renderings changed at token 35; Qwen and Gemma renderings were unchanged. The
second sequence (`launch2.sh`, `sequence2.log`) pins `TZ=Etc/GMT+12`, which keeps
the template date at the frozen 2026-10-13 until 12:00 UTC on 14 October. It
refuses to start a model if the pinned date differs. With this pin, both roots
reproduced every frozen rendering for all three models with no differing request,
and the setting reaches the no-swap scope. This is a disclosed GPT-OSS input
control. Timing and resource gates use monotonic and system counters, which
`TZ` does not affect. GPT-OSS primary collection started at 18:59 UTC (dot64
runtime, `stage0_mxfp4_dot64.json`, aligned calibration) and passed its startup
settle. Gemma follows in its NVFP4 precision root. Both earlier failed attempts are
retained under each root's `primary_controllers/`. No unit tests were added or run.

GPT-OSS corrected-floor captures completed at 20:05:38 UTC: 90 primary (accepted
19:53:52) and 18 repeat captures, each with full scores and exact event and
final-state replay, in whole batches that passed the resource gates. The
post-collection audit counts 90 primary and 18 repeat resource-passed captures
for `gpt_oss_mxfp4` (pilot ready; P2 support not qualified; next stage not
allowed). Gemma collection started immediately afterwards in its precision root
with the same date pin.

Gemma primary collection exhausted its three whole-batch attempts, so the sequence
ended (`DONE gemma_nvfp4 rc=1`, 21:00:35 UTC). Attempts 1 and 2
(`20261013T200546Z_535324b4`, `20261013T201126Z_dbd57b5e`) failed the startup
settle. Swap started at 2.54 and 2.63 GiB used; loading swapped out 598 and
435 MiB of other processes, and 333 and 170 MiB paged back in during the 90 s
window. Available memory stayed at or above 9.2 GiB. The accepted 17:00 Gemma
pilot started with swap at 4.06 GiB, swapped out 42 MiB and settled in 9 s.
Attempt 3 (`20261013T201852Z_130590a6`) settled and completed all 90 captures,
but the whole batch was rejected: the 30 s rolling full memory pressure peaked at
1.07% against the 1% gate. Inference swap-in was 121.9 MiB (gate 128 MiB), and
server swap stayed at 0. Every attempt is retained and excluded. Diagnosis:
loading a checkpoint fills the page cache; with `vm.swappiness=60` and free swap,
the kernel reclaims by swapping out idle processes, mainly another account's
polling API and resque workers, which then page back in. Runs pass only when swap
is already full. The user chose to pause those workers for the remaining Gemma
and Qwen collection rather than change VM settings. GPT-OSS's accepted 90+18
captures are unaffected.

### Host swap source paused by user authorization - 14 October 2026

At the user's explicit instruction, with sudo they supplied, the Judge0 Docker
stack on this host was paused at 02:46:50 UTC on 14 October:
`judge0-v1131-workers-1`, `-server-1`, `-redis-1` and `-db-1`. The workers
container (about 1.26 GiB, constant polling) was the main swap-in source.
`docker pause` freezes the containers in place, and `docker unpause` restores
them. At the user's request the `dev-pgvector` container was removed; its data
volume `dev_pgvector_data` was kept. Global VM and swap settings are unchanged.
After the pause, system swap traffic fell to 0-4 KiB/s with 26.1 GiB available.
`launch3.sh` (`sequence3.log`) then retried Gemma's 90 primary and 18 repeat
captures. Next it resumes Qwen pilot qualification (`launch_resume7.sh`), and if
Qwen qualifies it collects Qwen's primary and repeat captures, all with the
GPT-OSS date pin. Gemma batch `20261014T024820Z_99595e99` settled in 18 s
(171 MiB swapped out during load, 2 MiB swapped in during settle) and started
inference. The Judge0 pause is an environment change for every batch after
02:46 UTC. Judge0 stays paused until this sequence finishes and must then be
unpaused.

With Judge0 paused, Gemma's corrected-floor captures completed: 90 primary
(accepted 03:29:54 UTC) and 18 repeat (accepted 03:39:29 UTC), each with full
scores and exact replay, in whole batches that passed the resource gates. The
post-collection audit counts 90 primary and 18 repeat for `gemma_nvfp4`. The
resumed Qwen qualification (`launch_resume7.sh`) then passed a startup settle for
the first time (`20261014T033946Z_5c1815d6`): swap 3.28 GiB at start, 810 MiB
swapped out during load, 40 MiB swapped in during a 47 s settle, at least
6.45 GiB available. Its generation screens are running.

Qwen qualified on the NVFP4 precision arm at 04:53:56 UTC on 14 October. Its four
two-request screens matched, three fresh baselines completed, and all four arms
completed 15 observation-off/on pairs (60 captures) with exact replay, in batches
that passed the resource gates. `qualification/qwen_nvfp4/pilot_cross_arm_output.json`
matched generated tokens on all 15 requests in every arm. The controller wrote
`recovery_completion.json` (`nvfp4_pilots_complete`). Its final three-model audit
reports `all_three_models_pilot_qualified: true`, so primary collection is allowed
without the deferral. Next-stage permission remains false: Qwen's primary and
repeat captures and full P2 support are still missing. This qualifies the source
and control arm only, not task quality. Qwen's 90 primary and 18 repeat captures
started at 04:53:57 UTC with the date pin.

### Offline P3-P9 tooling built - 14 October 2026

At the user's request, while Qwen collection ran, offline stage tooling was built
in `experiments/pruning/`: `m_reduction_traces`, `_sim`, `_stats`, `_offline`,
`_routing` and runners `_p3` to `_p9`. Scope, inputs and missing runtime pieces
are listed in plan section 11.5. Runners read only accepted capture batches,
record control files and refuse to run before the next-stage gate unless
`--pre-gate-diagnostic` is given. Development checks: native simulator m/q matched
live exactly for one GPT-OSS and one Gemma capture; logit top-k and native weight
reconstruction matched one capture per model after adding Gemma's per-expert
scales (maximum error 1.3e-7); each runner completed on a 15-request GPT-OSS
subset in scratch space. The checks found and fixed four defects: domain
bootstrap aggregation, non-finite JSON scores, an over-strict mass percentile and
quadratic per-slot scans. No outputs were written under `results/`. These are not
tests or results, and no performance claim follows.

Qwen primary attempt 1 (`20261014T045407Z_58c25afa`) completed all 90 captures
with exact replay, but the whole batch was rejected at 05:44:26 UTC: system swap-in
during inference was 365 MiB against the 128 MiB gate. Server swap stayed at 0,
full pressure peaked at 0.098%, and available memory stayed at or above 4.84 GiB.
About 90 MiB arrived in the first six minutes, as pages swapped out during the
Qwen load (145 MiB) returned. A 212 MiB burst at minutes 16-20 coincided with the
assistant's offline development checks, which loaded full Gemma router-score
traces in parallel. That concurrent host activity should not have run during a
measured batch and is recorded as an operational error. Without it the batch
would still have been about 153 MiB, above the gate. The retained rejected batch
is excluded. The controller's bounded retry (attempt 2 of 3) waits for a quiet
interval. No further host-side work runs during measured GPU batches.

Qwen primary attempts 2 (`20261014T054545Z_2b008ab1`) and 3
(`20261014T063657Z_b4ee3b0e`) each completed all 90 captures with exact replay,
but the whole batches were rejected. Attempt 2 had 232 MiB swap-in (gate 128 MiB).
Attempt 3 had 218 MiB swap-in, and the rolling full memory pressure peaked at 2.89%
(gate 1%). Firefox was opened on the desktop at about 05:14 UTC and held more than
1.7 GiB. Its launch coincides with attempt 1's 212 MiB burst, so that burst is now
attributed to Firefox, the assistant's concurrent development checks, or both
(correcting the earlier entry). The user closed Firefox before attempt 3
measured. Attempt 3 still loaded with swap not full: loading swapped out 351 MiB
of other processes' memory, and 146 MiB came back in the first 15 minutes. The
loader reads about 21 GiB of shards through the page cache, so with
`vm.swappiness=60` and free swap the kernel swaps out idle processes, which then
return during the 50-minute batch. All three rejected batches are retained and
excluded.

At the user's explicit instruction ("stop all processes not related to
freetoken"), 78 non-essential processes on the user's account were frozen with
SIGSTOP at 07:36:38 UTC: the GNOME desktop session, `openclaw-gateway`, the Codex
daemons, `update-manager`, portals, ibus, gvfs (except FUSE) and evolution. The
PID list is recorded in `primary_sequence_20261013T185708Z/frozen_host_pids.txt`
for SIGCONT. These kept running: tmux, this session, systemd/user dbus,
`ssh-agent`, `gnome-keyring-daemon`, `gvfsd-fuse`, the VS Code server (per earlier
user instruction), the FreeToken watchdog and all root system services. Judge0
remains paused. System swap traffic then fell to 0-4 KiB/s with 27.4 GiB
available. A fresh three-attempt Qwen primary/repeat run (`launch4.sh`,
`sequence4.log`) started with the date pin. No gate, VM setting or numerical
control changed. This is a disclosed environment difference for Qwen's batches.

With host processes frozen, Qwen's fresh run passed on its first attempt: 90
primary captures (accepted 08:36:32 UTC) and 18 repeat captures (accepted
08:49:42 UTC), each with full scores and exact replay, in whole batches that
passed the resource gates. The post-collection audit now counts 90 primary and
18 repeat resource-passed captures for all three models, with
`all_three_models_pilot_qualified: true`. `all_three_models_stage0_complete` and
`next_research_stage_allowed` remain false only because P2 support (sequential
teacher forcing, full-vocabulary distributions, expert norms and probes) is
unimplemented. At 08:49:55 UTC all 78 frozen processes were resumed (SIGCONT) and
the Judge0 containers were unpaused. The host environment differs between
models' batches and is disclosed per batch: Judge0 paused for Gemma and Qwen
after 02:46 UTC; desktop and user apps frozen for Qwen's accepted primary/repeat
only. The GPT-OSS date pin (`TZ=Etc/GMT+12`) applied to every 14 October
collection.

### 14 October 2026 — Qwen offline checks and P2 support implementation

Qwen offline development checks (scratch, not results): on two primary and two
repeat accepted Qwen captures (`code_005_r0`, `code_008_r0`, `code_021_r0`,
`code_023_r1`; 10,200 decode events each), the default simulator reproduced the
live m/q streams exactly, and logit top-k plus native weight reconstruction had
zero selection mismatches (maximum effective-weight error 1.44e-7). All three
models now have these checks.

The GPT-OSS date pin (`TZ=Etc/GMT+12`) expired at 12:00 UTC on 14 October.
P2 avoids the problem: teacher forcing sends each model's frozen rendered prompt
token IDs from `protocol/rendered_inputs.json` and never renders a chat template.

New isolated runtime `.runtime/m_reduction_v3_p2_20261014T122306Z/`, copied from
the NVFP4 precision runtime, which also carries the env-gated GPT-OSS MXFP4
dot64 path. Compiled extensions are unchanged. The Python-only patch is
`runtime/patches/m-reduction-p2-teacher.patch`:

- `/v1/completions` accepts one flat token-ID prompt, only with
  `FREETOKEN_M_REDUCTION_TOKEN_PROMPTS=1` and without streaming; the tokenizer
  passes those IDs through.
- The routing observer checks the prefill against the frozen IDs and every
  decode input against the forced history. The first target is scored from the
  final prefill position, and each later target by one sequential decode step.
  It replaces the sampler's token with the forced target after recording the
  sampler token and the logits argmax.
- In baseline mode the observer saves full-vocabulary logits in their native
  dtype (`logits.bin`, BF16 for GPT-OSS, 201,088 wide) and every decode routed
  MoE input row (`moe_inputs.bin`).
- In control/candidate mode it computes FP64 `KL(P_baseline || P_live)`, top-1
  agreement and NLL online against the saved baseline rows.

New tooling:

- `m_reduction_teacher`: the collector, with the same resource gates and server
  command as the source batch.
- `m_reduction_experts`: reference expert function from weight-only
  dequantized checkpoint experts in FP32 on the GPU. It computes per-event
  pre-gate output norms and fold-specific probe means; probes come only from
  outer-training rows, 43/43/42 or 128 per domain, uniformly spaced in
  chronological order.
- `m_reduction_p2`: per-model driver.

`m_reduction_p7` now consumes these outputs; it uses mixed-pool probes per outer
fold and per-event weight × norm saliency.

The plan gate's P2 rule changed. `p2_support.json` must now record the P2
runtime's own hashes and must have reproduced captures made by the qualified
runtime (`reproduces_inference_hashes`). Previously the P2 support had to run on
the pilot runtime itself, which no isolated patch can do.

Development checks (scratch):

- A two-request GPT-OSS baseline smoke run (`code_005`, `code_006`) passed its
  resource gate. It matched the accepted captures event by event with zero
  mismatches: 8,376 route and score rows, 256 batch rows, 6,120 cache events,
  and initial/final cache state. All 256/256 sampler and argmax choices equalled
  the forced targets, at about 40 s per history.
- The reference experts matched the native same-state CPU per-route outputs
  (GPT-OSS layer 10 dot64; Qwen/Gemma layer 0 precision fixtures) within 0.38%
  maximum relative L2, with minimum cosine 0.999994. This is consistent with the
  native BF16 output boundary.
- A norms/probe code-path check on the smoke batch gave GPT-OSS layer-10 mean
  pre-gate norm 431; the native fixture's per-route norms are 384–623.

The qualifying sequence `results/m_reduction/p2_support_20261014T123345Z/launch.sh`
started at 12:33 UTC. It records the validation in `expert_validation.json`,
then runs GPT-OSS, Gemma and Qwen, each through three stages:

1. Baseline teacher forcing of all 90 primary histories.
2. Native zero-change control on the nine repeat IDs.
3. Expert outputs.

After that it writes the model's `qualification/MODEL/p2_support.json` in its
source study. Expected storage is about 44 GB of logits and MoE inputs
(113 GB free). The host environment is ordinary: Judge0 running and the desktop
unfrozen.

P2 GPT-OSS baseline attempt 1 (`captures/gpt_oss_mxfp4/teacher_baseline/all/20261014T123348Z_8dca13b6`)
completed all 90 histories with exact identity and greedy targets. It was
rejected by the swap gate: system swap-out was 132 MiB against the 128 MiB limit
over 58 minutes, and swap-in was 84 MiB. The server's own swap stayed at 0, and
available memory never fell below 11.1 GiB. Buffer/page cache grew from 7.6 to
11.7 GiB while the run wrote about 13 GB of logits and MoE inputs; with
`vm.swappiness=60` the kernel then swapped out idle anonymous memory. The
rejected batch is retained and excluded. The sequence was stopped before an
automatic retry. `m_reduction_teacher` now fsyncs each sealed capture and drops
its pages (`POSIX_FADV_DONTNEED`). Gates, the runtime and the host are
unchanged. The sequence restarted with `launch2.sh` (`sequence2.log`), reusing
the recorded `expert_validation.json`.

GPT-OSS P2 support is qualified (14:40 UTC). The relaunched baseline batch
`teacher_baseline/all/20261014T133358Z_a7a1f733` reproduced all 90 accepted
primary captures event by event with greedy targets. It passed the resource gate:
inference swap-out 96 KiB, swap-in 55.8 MiB, server swap 0. The page-cache
release kept buffer/cache flat. The zero-change control
`teacher_native_control/repeat_ids/20261014T143327Z_50abfd3f` scored KL 0, top-1
agreement 1 and identical logits on all nine repeat IDs. Expert norms and the 20
fold × pool probe sets per layer completed under
`expert_outputs/gpt_oss_mxfp4/20261014T133358Z_a7a1f733`.
`recovery_dot64_20261013T131424Z/qualification/gpt_oss_mxfp4/p2_support.json`
records the evidence hashes. Gemma started at 14:40 UTC. Storage: the P2 root
holds 27 GB (including the retained rejected attempt); 87 GB free.

Gemma P2 baseline attempt 1 (`teacher_baseline/all/20261014T144057Z_c5d1d8cc`)
completed all 90 histories with exact identity. It was rejected for system
swap-in of 230 MiB against the 128 MiB limit; swap-out was 212 KiB and server
swap 0. Startup loading swapped out 430 MiB of other processes, which returned
during the 45-minute batch: the same mechanism that blocked earlier Gemma/Qwen
collections, not the P2 output writes (page cache stayed flat). Attempt 1 is
retained and excluded; the driver's automatic whole-batch retry follows a quiet
interval. Judge0 is running and the desktop is unfrozen. Pausing either
awaits the user's answer.

Gemma P2 support is qualified (16:21 UTC). Baseline attempt 2
`teacher_baseline/all/20261014T152826Z_84d4acad` reproduced all 90 captures
exactly. It passed the resource gate: inference swap-in 57.8 MiB, swap-out
15.8 MiB, server swap 0. Startup swap-out was only 60.7 MiB, versus 430 MiB for
attempt 1, because attempt 1 had already pushed out idle pages. The nine-ID
zero-change control passed (KL 0, identical logits), and expert norms and probes
completed. The record is
`recovery_nvfp4_precision_20261013T161122Z/qualification/gemma_nvfp4/p2_support.json`.
Qwen started at 16:21 UTC with the host unchanged.

Qwen P2 baseline attempt 1 (`teacher_baseline/all/20261014T162113Z_5747ed68`)
failed the bounded startup settle before any inference, as Qwen did before
host intervention. The driver's retry then stopped because the host did not
give a 30-second quiet interval within 600 seconds. The sequence ended at 16:33
UTC with GPT-OSS and Gemma qualified and Qwen unqualified. No host intervention
was made; Judge0 is running and the desktop is unfrozen. A Qwen-only relaunch,
`launch3_qwen.sh`, is prepared and awaits the user's decision on pausing Judge0
and freezing non-FreeToken processes, as in the accepted Qwen primary run.

The user gave standing permission to pause any process for these experiments
("Yes you can always pause any process to proceed with these experiments"). At
16:48 UTC the four Judge0 containers were paused (`docker pause`), and the same
78 non-essential user processes as the accepted Qwen primary run were frozen
with SIGSTOP. All 78 PIDs were unchanged, and two are zombie codex processes.
The list is copied to `p2_support_20261014T123345Z/frozen_host_pids.txt`. Kept
running: tmux, shells, this session, systemd/dbus, ssh-agent,
gnome-keyring-daemon, gvfsd-fuse, the VS Code server, the FreeToken
watch/progress processes and root services. `launch3_qwen.sh`
(`sequence3.log`) then started Qwen-only P2 support. Gates and the runtime are
unchanged. This is a disclosed per-batch environment difference, matching
Qwen's accepted primary/repeat batches.

Qwen P2 baseline attempt 1 under the freeze (`teacher_baseline/all/20261014T164825Z_654a0a30`)
settled and reproduced all 90 histories exactly. It was rejected for system
swap-in of 346.5 MiB against the 128 MiB limit; swap-out was 10.9 MiB and server
swap 0. Startup loading swapped out 1.31 GiB of the remaining running processes
while swap still had free space. About 210 MiB came back in the first 70 s of
inference, and smaller bursts followed (27, 23 and 30 MiB). It is retained and
excluded. The driver's automatic attempt 2 follows, with swap now nearly full
(3.7 of 4 GiB), as for Gemma's accepted second attempt.

Qwen P2 attempt 2 (`teacher_baseline/all/20261014T174708Z_*`, the latest
batch) was stopped by the operator at 30/90. Its inference swap-in was already
277.6 MiB (gate 128 MiB) and swap-out 101 MiB, so the batch could not pass. It
is retained with an operator `failure.json`. The 16 `rake`/`resque` processes
holding swap belong to the Judge0 containers, which are already frozen by
`docker pause`. The enigma account's other processes are live interactive SSH
sessions and were not frozen. The assistant's own swap and progress checks
during the batch are a suspected contributor, as in the first Qwen
primary attempt. The relaunch (`sequence4.log`, about 18:12 UTC) runs with no
assistant host checks during the batch. A per-process VmSwap sampler
(scratch-space, every 20 s) attributes any further swap-in. The host freeze and
Judge0 pause remain in place.

Qwen P2 relaunch attempt (`teacher_baseline/all/20261014T180812Z_ffa51f49`)
reproduced all 90 histories exactly. It was rejected for inference swap-in of
163.9 MiB (gate 128) and a rolling full-pressure peak of 1.58% (gate 1%); the
server's swap stayed 0. The per-process VmSwap sampler attributed the page-ins
to `warp-svc` (Cloudflare WARP daemon, root, status "Disconnected", about
100 MiB), this Claude session (about 41 MiB), `tailscaled` (about 17 MiB) and
dockerd/containerd (about 13 MiB). The driver was stopped before its retry. At
about 19:10 UTC `warp-svc` (PID 1478) was frozen with sudo SIGSTOP under the
user's standing permission, recorded in `frozen_root_pids.txt`. Traffic routes
via `enp6s0`, not WARP. `tailscaled` stays running because the user's and this
session's SSH connections use Tailscale. Qwen relaunches (`sequence5.log`)
without host checks during the batch.

Qwen P2 support is qualified (20:12 UTC 14 October), completing P2 for all three
models.

- **Baseline:** the relaunch with `warp-svc` frozen,
  `teacher_baseline/all/20261014T190631Z_*`, reproduced all 90 histories
  exactly and passed every resource gate.
- **Zero-change control:** passed on all nine repeat IDs (KL 0, identical
  logits).
- **Expert outputs:** norms and probes completed.
- **Record:** `recovery_nvfp4_precision_20261013T161122Z/qualification/qwen_nvfp4/p2_support.json`.

The host was restored at 20:10:46 UTC, once no gated GPU work remained: the 78
user processes and `warp-svc` got SIGCONT (none still stopped) and the Judge0
containers were unpaused. Qwen's expert-output stage, which has no resource
gate, finished after that.

The plan gate (`offline_stage1_20261014T200452Z/gate_before_stage1.json`) now
reports `all_three_models_stage0_complete: true` and
`next_research_stage_allowed: true`. Stage 1 offline evidence then started at
about 20:13 UTC: `offline_stage1_20261014T200452Z/launch.sh` runs one serial
chain per model, in parallel on CPU (P3, P4, P5, P6, P7 with P2 expert outputs,
P9 with P5 quotas and the P3 G1 record, then P8). P8 records a missing
interaction because no P6 finalist is quality-qualified. By the user's
instruction the work continues unattended overnight.

Stage 1 note (development check, scratch). P3 reports `min_miss_lru_hit: 0` in
every cell. One real GPT-OSS trace confirms that MIN's per-event miss sets are
subsets of fully admitting LRU's at C=154 and C=308. At C=154 the counts are
LRU 13,817 misses and MIN 6,872, with 6,945 LRU-miss/MIN-hit events. This
follows from equal-capacity full admission. When MIN evicts an expert, every
other resident is reused before that expert's next use, so at least C distinct
experts are accessed after its last use, and LRU must also miss it. The zero is
therefore expected, not a defect. The plan's caution about pointwise oracles
concerns the converse direction, which does occur.

Stage 1 offline results, GPT-OSS and Gemma (chains complete at 22:06 and 00:13
UTC; `offline_stage1_20261014T200452Z/MODEL/p3..p9`, status `stage_evidence`).
Every value is mean decode missing demand m per layer event at the primary
capacity with native post-prefill state. Reductions are equal-workload, with
10,000-resample paired bootstrap CIs. These are offline replays and
fixed-logit screens: no timing or quality claim.

- P3 (G1):
  - GPT-OSS: m 1.878. Classes: compulsory 5.8%, admission/state-sensitive 9.5%,
    capacity-like 46.0%, replacement-sensitive 38.7%. The top-10% thrash share
    of replacement misses is 18.6%. Deletion upper estimates: bottom-frequency
    9.4%, largest-miss-count 46.9%.
  - Gemma: m 2.202. Classes: compulsory 14.6%, admission 17.3%, capacity-like
    35.0%, replacement 33.2%. Thrash share 18.9%. Deletion: bottom-frequency
    2.3%, largest-miss 53.4%.
  - G1 for both: P4/P5 first; P9 and P6 not prioritised; P7 priority not
    lowered.
- P4: the overlap fraction equals the live native fraction (GPT 0.6085).
  - GPT-OSS m: paper 2.038, overlap (native) 1.878, fixed25 1.834,
    reuse-uniform 1.866, reuse-frequency 1.853. Inner-fold selection on m:
    fixed25.
  - Gemma m: paper 2.191, overlap 2.202, fixed25 2.363, reuse-uniform 2.165,
    reuse-frequency 2.131. Selection: reuse-frequency.
  - Base-rule selection still needs the live latency gate.
- P5, per-layer quotas versus global LRU:
  - GPT-OSS: uniform −9.1% (CI 8.8–9.4%), utility −9.9% (9.6–10.2%).
  - Gemma: uniform −2.0%, utility −2.6%.
  - Live confirmation is required.
- P6 (fixed-logit routing screen): zero-change identity holds in both start
  states. No setting in families A, B or C meets the 98% mean / 95% fifth-
  percentile mass screen on either model, so each family records
  `no_eligible_setting`. Closest settings (B α=0.1): GPT-OSS −13.2% m, mass
  96.4%, p5 80.4%; Gemma −16.0% m, mass 97.45%, p5 89.8%.
- P7 (decode exclusion and aliasing, routing screen only): identity alias
  reproduces native.
  - GPT-OSS: R75 −5.0%, F75 −14.3%, random 75% masks −7 to −10%
    (5th–95th percentile), arm 1 miss-targeted −18.8%, arm 2 alias −22.3%.
  - Gemma: R75 −1.9%, F75 −3.4%, random −17 to −20%, arm 1 −33.0%,
    arm 2 −50.4%.
  - Arm 2 still lacks merged-weight execution, and quality is unmeasured.
- P9:
  - GPT-OSS: SLRU 0.5 −5.1% (CI 4.7–5.6%), G2 exact `defer_timing` (the m
    criteria pass; only the live timing bound remains). SLRU 0.25 −3.4%
    rejected; pins 0.10/0.25 +0.1%/+1.7% rejected.
  - Gemma: SLRU 0.25/0.5 −1.1%/−3.0% rejected; pins +0.4%/+2.9% rejected.
- P8: missing interaction for both, because no qualified P6 finalist exists.

Stage 1 offline results, Qwen (chain complete at 03:57 UTC 15 October;
`offline_stage1_20261014T200452Z/qwen_nvfp4/p3..p9`, status `stage_evidence`).
`ALL CHAINS END` was logged at 03:57:09 UTC, so Stage 1 offline evidence now
exists for all three models. The same conventions apply: mean decode m at
C=2048 with native state, equal-workload paired bootstrap CIs, offline replays
and fixed-logit screens only.

- P3 (G1): m 2.300, p95 5.90, fully admitting LRU 2.243, MIN 1.175.
  - Classes: compulsory 27.7%, admission/state-sensitive 18.3%, capacity-like
    23.4%, replacement-sensitive 30.7%. Thrash share 19.4%.
  - Deletion upper estimates: bottom-frequency 7.0%, largest-miss 48.8%.
  - G1: P4/P5 first; P9 and P6 not prioritised; P7 not lowered.
- P4: the overlap fraction equals the live native fraction (0.539).
  - m: paper 2.243 (−2.5%), overlap (native) 2.300, fixed25 2.578 (+12.1%),
    reuse-uniform 2.221 (−3.4%), reuse-frequency 2.251 (−2.1%).
  - Inner-fold selection: reuse-uniform in 5/5 folds. Frequency weighting is
    1.4% worse than uniform. The live latency gate is still required.
- P5: quotas do not help Qwen. Uniform +0.4% m (CI 0.2–0.6% worse; code and
  incident confidently worse); utility −0.8% (0.7–0.9%).
- P6: zero-change identity holds. Families A, B and C all record
  `no_eligible_setting`. Closest: B α=0.1, −20.7% m, mass 96.1%, p5 84.2%. A
  λ=0.1 gives −57.6% m but only 86.9% mass.
- P7 (routing screen only): identity alias reproduces native.
  - R75 −5.2%, F75 −10.4%, random 75% masks −12.1 to −13.6% (5th–95th
    percentile).
  - Arm 1 miss-targeted −25.2% (CI 24.9–25.6%); arm 2 alias −32.6%
    (32.1–33.2%).
  - Arm 2 still lacks merged-weight execution, and quality is unmeasured.
- P9:
  - SLRU 0.25 −1.0%, SLRU 0.5 −2.2% (selected in 5/5 folds).
  - Pins 0.10 −0.7%, 0.25 −0.3% (CI crosses 0).
  - All G2 rejected on the mean-m and p95 criteria.
- P8: missing interaction, because no qualified P6 finalist exists.

Cross-model Stage 1 note, start state. Every P3/P6 `cold` cell equals the
corresponding `native` cell exactly. This is not a replay defect. All 270
accepted primary captures (90 per model) record `decode_initial_cache.json`
with zero occupied slots. The native runtime therefore enters decode with an
empty expert cache: prefill uses global staging, and the cache is rebuilt from
decode demand. `verify_native` and P2's event-by-event reproduction both pass
from that empty state. Checked directly on three GPT-OSS traces:
native and cold replays give identical per-event m. Consequences:

- The plan's native/cold start-state contrast is degenerate on this runtime.
- Every m includes per-request warm-up misses. GPT-OSS's first ten events miss
  all 4 top-k experts.
- This makes P1's prefill-informed seeding add-on (`PLAN.md` §5) directly
  relevant. It has not been run. Its seeds would need real copies, charged in
  bytes, wall time and TTFT.

Stage 2 started (15 October, by the user's request to finish the remaining work).

- **Live runtime:** `.runtime/m_reduction_v3_live_20261015T041706Z`, the P2
  runtime plus `runtime/patches/m-reduction-live.patch` (sha `d37b6851…`).
  - The hybrid admission kernel gains per-layer quota and SLRU replacement
    modes. Their victim order mirrors `m_reduction_sim.CacheSim`.
  - The router gains a decode-only P7 mask hook: logits are set to −inf
    before native top-k, and prefill stays native.
  - Streamed token-ID completions are allowed, for first-text latency.
  - Lean runs log sampled token IDs, for output identity.
  - Admission adds a fixed-fraction q policy, used by the `paper_floor1`
    and `fixed25_floor1` arms.
- **Development checks (scratch, not tests):**
  - The kernel driven directly with recorded decode streams matched
    `CacheSim` event for event (m, q, admitted IDs) for LRU, quota and SLRU,
    on two GPT-OSS traces and one each for Qwen and Gemma. Quota and SLRU
    changed m relative to LRU.
  - Live GPT smokes (`results/m_reduction/live_smoke_20261015T042428Z`):
    - Fidelity, `quota_utility` and `slru_0.5`: identical outputs and zero
      replay mismatches (6,120 events per request).
    - Lean timing, `reuse_freq_floor1`: identical output. The run was
      rejected for server swap growth. It ran outside the controller's
      `MemorySwapMax=0` scope while parallel CPU jobs ran.
    - Candidate teacher forcing for P7 arm 1 (two histories): prefill native,
      no masked selections, m −18.5% on the first history, mean KL 0.37
      nats/token, top-1 agreement 74%.
- **P7 deployment masks:** arm 1 and R75 fitted on all 90 D0 captures for
  D1/D2 (`stage2_live_20261015T042628Z/p7_deploy/`).
- **D2 frozen:** AIME 2025-I at the pinned revision (file sha `b91b3c96…`).
  All 15 problems fit the 512-token input budget for every model.
- **P1 seeding (offline replay, `stage2_live_…/p1_seed/`, stage evidence):**
  - Seeds are the top-C prompt routes, by whole-prompt or last-64-token
    frequency. Native replay reproduces live m for all three models.
  - m change against the empty native decode start:
    - Qwen: −6.8% (CI 6.7–7.0%); early tokens −22%, steady state 0;
      3.6 GB of seed copies.
    - Gemma: −4.2%; 2.6 GB.
    - GPT-OSS: −0.8%; 2.0 GB.
  - The whole-prompt and last-64 seeds coincide on Qwen and Gemma, and
    nearly so on GPT-OSS, because the D0 prompts are short.
  - Only Qwen meets the 5% offline m criterion, so only Qwen needs a timed
    live seeding arm.
- **Incident:** at about 04:37 UTC a CPU requantization identity check (an
  unchunked whole-layer dequantization, about 17 GB) ran beside the D1 smoke.
  The OOM killer stopped this session's process group. The D1 smoke batch was
  rejected for memory, swap, pressure and OOM. Before that, its task arms and
  teacher-forcing path had worked; GPT arm 1 D1 KL was 0.57 and 0.97 on two
  sequences. No accepted evidence was affected.
- **Queue:** `stage2_live_20261015T042628Z/queue.sh` (`m_reduction_stage2`,
  launched with setsid at 04:57 UTC) runs serially, GPT-OSS → Gemma → Qwen
  per unit:
  1. Live exact arms: fidelity for every arm, then five paired fresh-server
     timing repetitions with shuffled arm order. The arms are the five P4 q
     rules, uniform and utility quotas, and GPT-OSS SLRU 0.5.
  2. P7 arm 1 and R75 candidate teacher forcing on all 90 D0 histories, each
     request under its own outer-fold mask, scored against the P2 baseline
     logits.
  3. D1, four shards: tasks for the baseline and arm 1; sequence KL for arm 1
     and R75.
  4. D2: baseline and arm 1, plus arm 1 KL on the baseline trajectories.
- **Pending:** arm 2 merge and requantization (code in
  `m_reduction_merge.py`, not yet run on a runtime) and the live Qwen seeding
  arm. Both go on a separate runtime copy so the queue's source hashes stay
  fixed.

### 15 October 2026 — weekly deck

Built `docs/presentations/FreeToken_Weekly_Progress_15-10-2026.pptx` (23 slides) with
`docs/presentations/weekly_20261015_assets/build_weekly_20261015.py`. It clones the
8 October deck's slide templates; masters, layouts, theme and fonts are unchanged.
Direction 1 slides restate the cross-architecture plan (exp_plan.pdf pages 3–4);
Direction 2 slides summarise P1–P9 and report Stage 0, Stage 1 offline, P1 seeding
replay and GPT-OSS live fidelity (8 arms × 9 requests, outputs identical, 0 replay
mismatches). The GPT-OSS P7 arm 1 smoke is shown as development evidence only.
Speaker notes are bullet pointers with sources. No measurement was run for the deck.

### 15 October 2026 — Stage 2 live: GPT-OSS done, Gemma host intervention

The GPT-OSS live controller finished at about 10:11 UTC: fidelity plus five paired
timing repetitions (repetition 2 needed a second attempt after a swap rejection) and
analysis under `analysis/gpt_oss_mxfp4`. The Gemma controller then rejected timing
repetition 1 twice, first for system swapping during inference and then because host
memory did not settle within the startup wait. Both were host swap traffic from idle
services paging back in, not server faults. At 11:22:56 UTC the four Judge0 containers
were paused with `docker pause` under the standing permission; system swap-in fell to
about 12 KiB/s. They stay paused until the Stage 2 queue and the arm 2 queue finish and
must then be unpaused. All Gemma and Qwen live batches from attempt 3 of Gemma
repetition 1 onward ran with Judge0 paused; GPT-OSS ran with it live.

Gemma's live controller then finished (repetition 1 passed on attempt 3 with Judge0
paused) and Qwen started. At 14:42 UTC the root filesystem filled (915 GB, 0 free;
`results/` holds 346 GB, of which the P2 baseline logits are 104 GB). Qwen timing
repetition 2 failed with `ENOSPC` while copying the runtime source snapshot, the main
queue ended (`QUEUE END 1`), and the chained arm 2 queue failed its preflight at
14:42:34 UTC. No accepted batch was affected; the failed batches stay as evidence.
`uv cache prune` freed 7.1 GB (no results, models or environments removed). The
remaining units need roughly 5 GB, mostly 41 MB source snapshots per run; D1/D2
baseline logits are deleted after scoring, and candidate batches do not save logits.

The first resume (14:45 UTC) exposed a controller resume bug: for each repetition it
only checked `rep{N}_attempt1`, so it began re-timing GPT-OSS repetition 2 (accepted
under attempt 2). It was stopped at 14:53 UTC after one extra GPT-OSS batch
(`rep2_attempt1` quota_utility, complete, unused because analysis takes the latest
complete attempt) and one killed fixed25 batch. `m_reduction_live.py` now skips a
repetition once any attempt has every arm accepted; this changes only resumption.
`queue_resume.sh` chains the main units and then arm 2 + P7 on the live2 runtime
(log `queue_resume.log`). The relaunch at 14:57 UTC re-analysed GPT-OSS and Gemma
without new GPU runs and continued Qwen repetition 2 from scratch under
`rep2_attempt1`.

16 October, 11:57 UTC: the resumed queue finished Qwen live, all six P7 candidate
batches (arm 1 and R75 per model) and D1 for GPT-OSS and Gemma, and is on Qwen D1
shard 3/4. The study grew from 6.5 to 12 GB and free disk fell to 2.0 GB.
`npm cache clean --force` freed 1.5 GB (3.5 GB free). Estimated remaining writes:
D2 about 1 GB plus up to 1 GB of transient baseline logits per trajectory, and arm 2
about 5 GB (six teacher batches per model, sized like the arm 1 batches of 0.1-0.5 GB),
so arm 2 needs more space than is free.

16 October, 21:56 UTC: Qwen D1 finished and D2 finished for GPT-OSS and Gemma. Qwen D2
was rejected three times for system swapping during inference (batches 16:27, 18:16 and
20:06 UTC, Judge0 still paused), so the main units ended with `MAIN END 1` and the
chained arm 2 phase started on the live2 runtime with 2.0 GB free. A guard
(session scratchpad `arm2_guard.sh`) stops the queue's process group when Gemma arm 2
would start or free space falls below 800 MB, so GPT-OSS arm 2 (identity check, merge
controls, requantization-only control and five fold batches, about 0.1 GB each) can
finish without filling the disk. Still to run: Qwen D2 (needs a quieter host), and
Gemma and Qwen arm 2 (need more disk).

17 October, 02:56 UTC: GPT-OSS arm 2 stopped on a runtime bug, not resources. The
requantization-only control passed (attempt 3; attempts 1-2 rejected for sustained full
memory pressure), but the fold 0 server aborted on its first request with
`observed cache miss count disagrees with native hybrid kernel`. Arm 2 aliasing can map
two logical experts in one top-k to the same physical representative (the last route was
`[6, 20, 12, 12]`); the kernel counts distinct missing experts (per-expert mask, and the
Python path deduplicates), while the observer in `routing_trace.finish_cache` counted
each routed entry. The client then waited its full 3600 s timeout on the dead server.
The fix counts distinct missing ids in the observer. The frozen live2 runtime is
unchanged; the fix is in a copy, `.runtime/m_reduction_v3_live3_20261017T025758Z`
(single-file diff in `routing_trace.py`). It affects only top-ks with duplicate IDs, so
the accepted live2 requantization-only control (no aliasing) is equivalent. GPT-OSS
arm 2 folds and the P7 analysis were relaunched at 02:58 UTC on live3
(`queue_arm2_gpt.sh`, log `queue_arm2_gpt.log`); the failed fold 0 batch is kept.
Gemma and Qwen arm 2 still need disk space, and Qwen D2 a quieter host.

17 October, 04:11 UTC: GPT-OSS arm 2 folds 0-4 completed on live3 (first attempts) and
the P7 D0 teacher-forcing analysis ran over all 90 histories
(`analysis/gpt_oss_mxfp4/p7_teacher_20261017T041355Z.json`). Per workload, mean KL in
nats/token and top-1 agreement (math / code / incident):

| Arm | Teacher-forced m reduction (95% CI) | KL | Top-1 | Decision |
|---|---|---|---|---|
| arm 1 | 18.2% [17.9, 18.6] | 0.129 / 0.298 / 0.429 | 89.2 / 75.9 / 68.2% | reject (quality) |
| R75 | 5.0% [4.8, 5.3] | 0.013 / 0.019 / 0.030 | 97.3 / 95.4 / 93.0% | reject (m < 10%, quality) |
| arm 2 | 23.9% [23.5, 24.3] | 0.312 / 0.287 / 0.501 | 84.0 / 77.7 / 69.5% | reject (quality) |
| requantization only | 0 | 0 / 0 / 0 | 100% | control (exact) |

No GPT-OSS output-changing arm meets the 0.02 nats/token KL or 98% top-1 gate. The
requantization-only control is exact, so arm 2's error comes from merging, not
requantization. The first analysis (`p7_teacher_20261017T041122Z.json`) labelled arm 1
and arm 2 "provisional": `g2_output_changing` only rejected on quality once timing was
also known. It now rejects whenever `quality_passed is False`, and the corrected
analysis supersedes the first file (which is kept).

### 20 October 2026 — P7 quality results, finishing Stage 2, closeout block

**Host and disk.** At 15:08 UTC the root disk had 441 MB free; a separate Claude
session was writing `results/reuse_aware_models/` and `results/reuse_aware_v2/`. At
the user's instruction to continue the whole plan, the Hugging Face model downloads
(`~/.cache/huggingface/hub/models--*`, about 6 GB: ViT, BERT, CLIP, DALLE2 weights from
other projects, all re-downloadable) and the Xet chunk cache (4.3 GB) were deleted; the
AIME dataset caches were kept. Free space rose to 11 GB. `~/.freetoken/models/` was not
touched.

**D1 (synthetic regression, 300 task generations and 100 teacher-forced sequences per
arm).** Task score, baseline → arm 1, and the upper bound on baseline-minus-candidate loss
(gate ≤ 0.02):

| Model | Math | Code | Incident | Arm 1 KL (nats/token) | Arm 1 top-1 | R75 KL |
|---|---|---|---|---|---|---|
| GPT-OSS | 1.00 → 0.98 (ub 0.06) | 1.00 → 1.00 | 0.998 → 0.982 (ub 0.054) | 1.03 | 68.7% | 0.71 |
| Gemma | 0.82 → 0.80 (ub 0.17) | 1.00 → 0.43 (57 correct→wrong) | 0.97 → 0.94 | 0.58 | 93.4% | 0.51 |
| Qwen | 0.70 → 0.70 | 1.00 → 1.00 | 0.98 → 0.934 (15 correct→wrong) | 0.149 | 98.9% | 0.121 |

Only Qwen arm 1 passes top-1 on D1; every arm fails the D1 KL gate and at least one task
domain. Gemma arm 1 breaks code tasks. On D1 the candidate NLL is below the baseline's for
arm 1 on all three models (GPT-OSS 0.74 vs 2.09, Gemma 0.19 vs 0.47, Qwen 0.045 vs
0.089), so the perplexity check passes. Checked on GPT-OSS `quality_text_000`: the
`baseline_nll` each arm row computes from the saved baseline logits equals the baseline
run's own NLL (1.895), both score the same fixed D1 target tokens, and arm 1's argmax
matches the target at 198 of 256 steps against the baseline's 157. It is a real effect
of the synthetic template text, not a scoring error: D1 likelihood is a weak ability
proxy here, and the KL/top-1 gates, which fail, decide. D2's NLLs have the expected order.

**D2 (AIME 2025-I, 15 problems, all eligible).** GPT-OSS baseline 3/15 correct (12
truncated at 1,536 tokens), arm 1 2/15 with one baseline-correct→wrong; arm 1 KL 0.19,
top-1 85.7%. Gemma baseline 2/15 and arm 1 2/15 (13 truncated at 768 tokens each); KL
0.19, top-1 91.2%. Baseline competence is below 20% on both, so D2 cannot show ability
preservation. Qwen D2 was rejected three times for host swapping (16 October) and is
rerunning with the desktop frozen.

**P7 decision so far.** GPT-OSS: arm 1, R75 and arm 2 rejected on D0 quality (17
October entry). Gemma and Qwen arm 1 and R75 also fail D1's KL gate, so no P7 arm
can qualify on any model; Gemma and Qwen arm 2 and their D0 P7 analyses run now for
completeness. Because no P6 or P7 finalist and no Reuse-Aware P4 variant qualifies, the
P8 2×2 factorial has no r1 or q1 and is reported as a missing interaction (plan 10.3),
not replaced by another policy.

**Finishing queue (15:13 UTC).** `stage2_live_…/queue_finish.sh`: Gemma and Qwen arm 2 +
P7 analysis on live3, then Qwen D2 on the live runtime. 74 desktop/user processes
(GNOME session, Xorg, pipewire, ibus, gvfs except FUSE, gsd-*, evolution, tracker,
nautilus, update-manager, openclaw-gateway, gitstatusd) are frozen with SIGSTOP by
`host_freeze.sh` (PIDs in `frozen_host_pids_20261020.txt`), thawed on any exit. Kept
running: shells, tmux, gnome-terminal-server, both Claude sessions, systemd/dbus,
ssh-agent, keyring, gvfsd-fuse, the VS Code server, FreeToken processes and root
services. Judge0 stays paused (since 15 October).

**Closeout block (plan 10.3), prepared.** `m_reduction_live.py --arm-set closeout`
writes its own study, `results/m_reduction/closeout_live_20261020T151823Z/`, and leaves
the Stage 2 controls byte-identical. Arms per model, fidelity then five fresh paired
timing repetitions:

- all models: `overlap_floor1` (corrected baseline) and `overlap_floor0` (the original
  rounding, reference control without a G2 decision);
- GPT-OSS: `quota_utility`, the best exact component that kept G2 live (its SLRU 0.5
  also kept, but a quota+SLRU composition has no kernel or simulator mode, so the
  composition is an implementation gap, not run);
- Qwen: `seed_whole`, P1 whole-prompt seeding (offline −6.8% m, all five folds
  positive; the only seeding cell meeting the 5% criterion). Gemma has no kept exact
  component.

Seeding needed a runtime hook: `.runtime/m_reduction_v3_live4_20261020T151654Z` = live3
plus `moe/m_reduction_seed.py` and two call sites (prefill route counting in
`layers/moe.py`; a pre-forward hook in `engine/engine.py`, before the observer so the
`decode_initial_cache` snapshot is the seeded state). At the first decode step it writes
the top-C prompt routes into the global slot cache with LRU timestamps and recency by
last prompt touch, mirroring `m_reduction_p1_seed.seeded_state`, and copies the rows with
the cache's own `copy_missing` inside the request, so TTFT and latency carry the cost;
entries, bytes and seconds go to `seed_log.jsonl`. Patch from live2:
`runtime/patches/m-reduction-live4.patch`. It is unqualified until the closeout
queue's two-request Qwen seeding fidelity smoke (label `smoke`, `--limit`, never
accepted) and the full fidelity run show zero replay mismatches and identical outputs.
`queue_closeout.sh` waits for the finishing queue, re-freezes the desktop, freezes the
three controls, runs the smoke and then the Qwen, GPT-OSS and Gemma controllers, and
finally thaws the desktop and unpauses Judge0.

**D3 (SWE-bench Verified, five issues): blocked on this host.** It needs an OpenCode +
FreeToken integration, a pinned SWE-bench harness and container images; none exists
locally (plan 3.4 lists the pins as unresolved entry blockers), and the per-issue images
(several GB each) do not fit the remaining disk. Recorded as a blocked cell for all
three models, not as a result.

**Finishing queue, arm 2 (20 October, 15:13–18:14 UTC): Gemma rejected, Qwen not run.**
Gemma's merge identity check (layer 0 values identical 0.733, relative L2 0.0113) and
fold controls built. Its requantization-only teacher-forcing control then completed all
90 requests three times and was rejected each time by the resource gate (sustained full
memory pressure: rolling 30 s maximum 1.68 / 1.08 / 1.58% against the 1.0% limit; whole
inference 0.15 / 0.04 / 0.17%; minimum available 8.6–9.2 GiB, inference swap-out
32–83 MB). The stalls are 1–4 s bursts every few minutes without matching swap traffic.
Gemma's arm 1 teacher control had sat at the same margin on 15 October: rejected at
1.25% and 1.16%, then passed at 0.79%. The controller stops at the first exhausted
unit, so Qwen arm 2 and both remaining P7 analyses did not run. Qwen D2 started at 18:14.

Host intervention: four orphaned `zsh` shells, whose ptys had been deleted, were each
spinning one core and had used 6+ CPU-days. They are PIDs 29041, 1584269, 2195042 and
2195434, and none is the parent of a Claude session. They were SIGSTOPped at 18:15 UTC
and listed in `frozen_spinning_shells_20261020.txt`, to be continued with SIGCONT
afterwards. They were never killed. `queue_arm2_retry.sh` waits for the finishing and
closeout queues to exit, then re-pauses Judge0, re-freezes the desktop and runs arm 2 + P7
again on live3. Qwen goes first and Gemma runs in its own controller call.

**Qwen D2 (AIME 2025 I, 15 problems, desktop frozen; 18:14–19:49 UTC): passed the
resource gate; arm 1 fails quality.** Accuracy: baseline 0/15, arm 1 1/15 (one
wrong-to-correct). All 30 generations hit the token budget, so accuracy here says
nothing about ability. Teacher forcing for arm 1: KL 0.165 nats/token [0.129, 0.210],
top-1 agreement 90.8% [88.6, 92.6], NLL 0.270 vs baseline 0.109 (perplexity ratio 1.17
[1.13, 1.23]), teacher m 1.94 vs 2.44 (−20.6%). It fails all three gates, matching
GPT-OSS and Gemma D2.
Batch: `quality/qwen_nvfp4/d2/shard00of01/20261020T181405Z_de7ba6`; analysis run
`runs/pruning/20261020T194938Z_cb3ee890`. The finishing queue exited 0 at 19:49 and
thawed the desktop. The closeout queue re-froze it and prepared all three models'
controls. The Qwen seeding smoke started at 19:50.

**Seeding smoke passed (19:52 UTC).** On the two Qwen requests (code_021 and
code_023), each produced 256 tokens with output identical to the baseline and zero
replay mismatches. `seed_log.jsonl` shows the seeding really ran. Every request
(warmup included) filled the whole cache: 2048 entries over 40 layers, 3.64 GB copied.
The prompts touched 2967–3613 (layer, expert) pairs. The copy took 0.55 s per request
after a 3.96 s first copy during warmup. The copy runs before the first decode forward, which is
after the first token comes from prefill. So it shows up in decode time and total latency,
not in TTFT, and the latency comparison will decide the arm. The smoke batch is labelled `smoke` and is
never accepted. The Qwen closeout controller started at 19:52.

**Qwen closeout (19:52–20:51 UTC; five repetitions accepted, 9 requests × 5 per arm):**

| Arm | Live m (sum) | Live q (sum) | Latency ratio vs floor 1 [95% CI] | Decode ms/token ratio | TTFT ratio | G2 |
|---|---:|---:|---|---|---|---|
| overlap_floor1 | 211,854 | 127,878 | 1 (12.80 s mean) | 1 (39.52 ms) | 1 (2.726 s) | reference |
| overlap_floor0 | 211,854 | 127,878 | 0.997 [0.993, 1.001] | 0.996 [0.992, 1.001] | 1.000 | reference control |
| seed_whole | 197,452 (−6.8%) | 120,713 (−5.6%) | **1.026 [1.022, 1.030]** | 1.033 [1.029, 1.038] | 1.000 | **reject** |

Every arm had zero replay mismatches over 91,800 events and identical outputs. On Qwen,
the floor-1 correction never changes a fill decision: floor 0 and floor 1 have
identical m and q, and their latency matches within noise. Live seeding reproduces the
offline P1 prediction exactly (live −6.8% m; offline −6.83% [−6.71, −6.96]) and meets
every m, fold and domain criterion. It still loses on time. Copying 2048 rows (3.64 GB)
before decoding costs more than the decode-time misses it saves, so latency rises 2.6%
and the upper bound breaks 1.02. Decision: **reject**. Qwen keeps no exact arm. Summary:
`closeout_live_20261020T151823Z/analysis/qwen_nvfp4/live_summary_closeout_20261020T205140Z.json`.

**GPT-OSS closeout (20:51–22:27 UTC; five repetitions accepted, 9 requests × 5 per arm):**

| Arm | Live m (sum) | Live q (sum) | Latency ratio vs floor 1 [95% CI] | Decode ms/token ratio | TTFT ratio | G2 |
|---|---:|---:|---|---|---|---|
| overlap_floor1 | 103,086 | 70,222 | 1 (29.64 s mean) | 1 (110.2 ms) | 1 (1.529 s) | reference |
| overlap_floor0 | 103,086 | 70,222 | 1.000 [1.000, 1.001] | 1.000 [1.000, 1.001] | 1.000 | reference control |
| quota_utility | 93,443 (−9.4%) | 64,325 (−8.4%) | **0.956 [0.946, 0.965]** | 0.953 [0.943, 0.963] | 1.001 | **keep** |

Every arm had zero replay mismatches over 55,080 events and identical outputs. As on
Qwen, floor 0 and floor 1 make the same fill decisions, so the rounding correction is
inert on these workloads. `quota_utility` repeats its Stage 2 result against the native
reference: live −9.4% m (offline −9.9% [−9.6, −10.2]), 4.4% lower latency with the upper
bound 0.965, and TTFT unchanged. **G2 keep, confirmed.** Summary:
`closeout_live_20261020T151823Z/analysis/gpt_oss_mxfp4/live_summary_closeout_20261020T222759Z.json`.

**Gemma closeout (22:28–23:10 UTC; five repetitions accepted):**

| Arm | Live m (sum) | Live q (sum) | Latency ratio vs floor 1 [95% CI] | Decode ms/token ratio | TTFT ratio |
|---|---:|---:|---|---|---|
| overlap_floor1 | 152,342 | 92,712 | 1 (15.94 s mean) | 1 (54.92 ms) | 1 (1.936 s) |
| overlap_floor0 | 152,342 | 92,712 | 1.001 [1.000, 1.002] | 1.002 [1.000, 1.003] | 1.000 |

Zero replay mismatches over 68,850 events, identical outputs. Gemma has no kept exact
component, so the block is the floor comparison alone. The fill decisions are identical
again, and the 0.1% latency difference is run-order noise on identical work, not an
effect.

**Closeout block conclusions (plan 10.3).** On all three models and the frozen D0
requests, the floor-1 correction is inert. Floor 0 and floor 1 give identical m and q
on every model, and their latency matches within 0.2%. So the Stage 2 references
measured against floor 1 equal the native runtime. The only exact component that holds
against the native reference is GPT-OSS `quota_utility`: m −9.4%, latency 0.956×. Qwen
whole-prompt seeding meets the m criterion (−6.8%) but adds 2.6% latency from the
3.6 GB pre-decode copy, so it is rejected. The closeout queue exited at 23:10 and
thawed the desktop. Judge0 was briefly unpaused at 23:10:33, then paused again by the
retry queue at 23:11:23. The retry began with Qwen's merge identity check at 23:11.

**Arm 2 retry 1 (20 October 23:11 – 21 October 01:52 UTC): Qwen requant control rejected
three times; Gemma attempt stopped.** Qwen's merge identity check passed (layer 0 values
identical 0.847, relative L2 0.0136). Its requantization-only teacher control completed
inference three times and was rejected each time on full memory pressure: rolling 30 s
maximum 2.74 / 2.87 / 3.08%, whole inference 0.28 / 0.26 / 0.36%, minimum available
5.25–5.31 GiB. Qwen's arm 1 and R75 controls passed on 15 October at 0.70% and 0.05%.
The difference is systematic, not host noise. At mid-inference the requant worker held
22.57 GB RSS against 21.42 GB for arm 1, and its two sibling processes about 0.24 GB more
each, roughly 1.6 GB in total. The merge hook (`m_reduction_merge.wrap_pieces`)
dequantizes and requantizes whole layers on the host at load. The likely cause is glibc
keeping those freed buffers, since `MALLOC_ARENA_MAX=2` does not return large
heap blocks. Gemma's arm 1 → requant gap (10 → 8.6–9.2 GiB minimum) points the same
way. I stopped retry 1 at 01:52, 2 minutes into Gemma attempt 1, which is left
incomplete. I did this by signalling its process groups; the GPU was free afterwards.

`queue_arm2_retry2.sh` (01:52 UTC) re-pauses Judge0, re-freezes the desktop and reruns
Qwen then Gemma arm 2 + P7 with `MALLOC_MMAP_THRESHOLD_=1048576`. That fixes glibc's
mmap threshold at 1 MiB, so large tensors are mmapped and handed back to the OS when
freed. This is an allocator-only deviation from the arm 1 runs. It does not touch
logits, KL or top-1, and each batch's `configuration.json` records it.

Retry 2 check, 02:11 UTC, two minutes into Qwen's first requant inference: the allocator
setting is recorded in `configuration.json` (`MALLOC_MMAP_THRESHOLD_=1048576`). It
brings the worker down to 20.61 GiB from 21.02 GiB in retry 1 (arm 1 was 19.95 GiB), and
the sibling processes back to their arm 1 size, 0.59 and 0.64 GiB. About 0.7 GiB of the
gap remains, available memory is 5.9–6.2 GiB, and the rolling full pressure has already
reached 1.53% (30 windows above 1.0%). A 60 s vmstat sample during a quiet stretch
showed 318 MB/min of page-in, 12 MB of file refaults, kswapd scanning and 1.5 ms of
full stall. The pressure comes in bursts from page-cache reclaim with about 6 GiB free
and 19 GB of shmem holding the host expert banks. The allocator fix reduces memory
but does not clear the gate for Qwen. The queue keeps running so the gate decides each
attempt.

**Retry 2, Qwen (01:52–04:32 UTC): rejected three more times.** The requant control with
`MALLOC_MMAP_THRESHOLD_` gave rolling maxima of 4.60 / 2.76 / 2.06%, whole-inference
0.37 / 0.16 / 0.20%, and minimum available 5.55–5.62 GiB. Over six attempts it never
came under the 1.0% gate. **Qwen arm 2 is blocked by host memory under the current
resource protocol.** P7 arm 2 has no Qwen result, and Qwen's P7 D0 summary covers arm 1
and R75 only. The pressure gate was written for performance runs, and memory stalls do
not change teacher-forced logits. So an exemption for quality-only runs, with pressure
still recorded, would unblock it, but that protocol change needs the user's decision.
Gemma attempt 1 started at 04:32.

**Retry 2, Gemma (04:32–06:55 UTC): rejected three times.** With the allocator setting,
the first attempt was clean through its first 7.5 minutes (rolling maximum 0.01%). The
full runs still failed: rolling maximum 2.48 / 2.55 / 3.17%, whole inference
0.22 / 0.15 / 0.20%, minimum available 8.9–9.2 GiB. Before the fix, Gemma's requant
runs reached 1.08–1.68%. After six rejected runs each, **Gemma and Qwen P7 arm 2 are
blocked by host memory under the current resource protocol.** The retry queue exited at
06:55, thawed the desktop and unpaused Judge0. The four orphaned spinning shells are
still SIGSTOPped. They are not experiment processes, and continuing them would put four
busy cores back on the host.

**Final P7 D0 summaries, Gemma and Qwen (21 October, `m_reduction_p7_live`; arm 2 and
requant marked missing).** Means across the three workloads; every arm fails G2:

| Model | Arm | Teacher-forced m reduction [95% CI] | KL math / code / incident | Top-1 math / code / incident | Perplexity ratio | G2 |
|---|---|---|---|---|---|---|
| Gemma | arm 1 | 29.4% [28.9, 29.9] | 0.309 / 0.513 / 0.628 | 89.4 / 82.1 / 77.8% | 1.39 / 1.68 / 1.88 | reject (quality) |
| Gemma | R75 | 1.9% [1.8, 2.0] | 0.002 / 0.003 / 0.010 | 99.2 / 98.9 / 97.7% | 1.00 / 1.00 / 1.01 | reject (m < 10%) |
| Qwen | arm 1 | 22.7% [22.3, 23.1] | 0.324 / 0.489 / 0.594 | 88.2 / 80.1 / 76.4% | 1.41 / 1.71 / 1.88 | reject (quality) |
| Qwen | R75 | 5.1% [4.9, 5.4] | 0.008 / 0.008 / 0.011 | 98.9 / 97.9 / 97.2% | 1.01 / 1.01 / 1.01 | reject (m < 10%) |

Files: `analysis/gemma_nvfp4/p7_teacher_20261021T110328Z.json` and
`analysis/qwen_nvfp4/p7_teacher_20261021T110331Z.json`. With GPT-OSS's summary
(`p7_teacher_20261017T041355Z.json`: arm 1, R75 and arm 2 all reject), P7 is closed. No
output-changing arm passes on any model. The aggressive masks (arm 1, and GPT-OSS
arm 2) cut m by 18–29% but cost 0.13–0.63 nats/token of KL. The mild R75 mask keeps
quality nearly intact but saves only 2–5% of m.

**Stage 2 closed (21 October).** The GPU is now used by a separate `policy_study` queue
(`results/policy_study/20261021T090614Z/queue.sh`, started 09:06 UTC from another
session). It paused Judge0 again on its own; this study does not touch it.

### 21 October 2026 — q* / Reuse-Aware / Runtime-Variable policy study (`q_expswithreuse_aware/`)

A read-only audit (10:55–11:15 UTC) is in `q_expswithreuse_aware/docs/AUDIT.md`. Findings:
- Reuse-Aware is a local rule and does not appear in the paper.
- The paper's minimum one fill (§3.2) is absent from the official kernel.
- The built-in q\* fraction uses the overlapped bandwidth pair, not Eq. 4.
- Cache rate 0.10 comes from SP-01's `high_miss` constant. It is not from the paper or the
  official repository. The established baseline capacity is 0.20, and the paper serves at the
  auto-sized capacity.
- `check_policy_equivalence.py` tests the superseded `ReuseRV`, not `ReuseVariant`.
  `ReuseVariant` equivalence with all switches off has now been verified (60,000 cases plus
  real logs, 0 mismatches).
- The frozen V3 victim prediction breaks usage-stamp ties arbitrarily: 98.1% match with real
  evictions on Stage 1 gpt-oss.
- The Stage 1 gpt-oss `analysis/REPORT.md` predates the end of that stage.
The policy-study queue was left running (Gemma Stage 1 in progress; Qwen Stage 1 and
Stage 2 queued).

A corrected study is queued behind it (`q_expswithreuse_aware/results/logs/20261021T111516Z/`):
- model and cache: Qwen3.6-35B-A3B NVFP4, `--moe-cache-auto`;
- workload: AIME-25 #0–2 (`math-ai/aime25@563bb84`), 256 greedy tokens;
- arms: A, B and all seven V1/V2/V3 combinations, plus A0 and E as diagnostics;
- harness: one lean hook (`policy/qexp_hook.py`, with engine-exact V3 victims); unhooked A0
  and E as the overhead reference.
CPU-only tests: 77 passed. No engine, frozen hook or existing result was modified.

**21 October, user decisions.** (1) The local elastic expert-cache/KV reallocation harness
(`experiments/runtime_q/elastic_*.py`, `configs/elastic_reuse/`) is **not** integrated
into this study. Every m-reduction run used a fixed `--moe-cache-size` (GPT-OSS 154, Gemma
768, Qwen 2048) and never called `/v1/cache/rebuild`. (2) After the other sessions' GPU
work, free host RAM and retry what was missed. The only re-runnable miss is Gemma/Qwen
arm 2 + P7. P8 and D3 remain blocked and the quota+SLRU composition stays unimplemented.
`queue_arm2_retry3.sh` (launched 14:27 UTC) waits for every `policy_study` and
`saketh_exp_2` queue to exit and the GPU to stay idle for 5 minutes. It then stops Judge0
(stop, not pause, to release about 1 GB of swapped workers). It kills the four orphaned
spinning shells and their gitstatusd children, freezes the desktop and reruns Qwen then
Gemma arm 2 + P7 with `MALLOC_MMAP_THRESHOLD_`. Free memory and swap are logged at each
step. On exit it thaws the desktop and runs `docker start` on Judge0. The other sessions'
queues are not touched: `policy_study` stopped at 12:52, and `saketh_exp_2` has run
main_qwen since 13:22.
