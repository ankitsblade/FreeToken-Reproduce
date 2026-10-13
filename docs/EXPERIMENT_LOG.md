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
