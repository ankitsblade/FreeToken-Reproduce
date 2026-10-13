# Executed pruning protocol

This consolidates the completed static 75% protocol and its separately authorised actual-inference supplement. It describes recorded methods; current outcomes are in [report.md](report.md), and remaining work is in [the single plan](../../PLAN.md).

After the 13 October migration, sealed studies use their original collectors through
`scripts/frozen.sh`. The relocated packages define a new source identity. On another
GPU, qualify a new calibration and resource/pilot configuration before collection;
the old host's frozen bandwidth profile is historical evidence. See
[the reproduction guide](../../REPRODUCING.md).

## 1. Scope and model locks

All collected inference runs use native FreeToken on the RTX 4070 host. Kimi-K2.5 is excluded by user instruction. Shared branches remain active and are excluded from routed-expert rankings.

| Model | Checkpoint revision | L / E / k | Retained/layer | Cache slots / bytes | Memory ratio / sequence cap |
|---|---|---|---:|---|---|
| Qwen3.6 NVFP4 | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | 40 / 256 / 8 | 192 | 2,048 / 3,636,461,568 | 0.9 / 2,048 |
| GPT-OSS MXFP4 | `6cee5e81ee83917806bbde320786a8fb61efebee` | 24 / 32 / 4 | 24 | 154 / 2,038,417,920 | 0.9 / 2,048 |
| Gemma 4 NVFP4 | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` | 30 / 128 / 8 | 96 | 768 / 2,575,761,408 | 0.8 / 1,280 |

Exact repository IDs, original weight geometry and checkpoint metadata are in the [checkpoint lock](../../../results/moe_pruning/study_20261006_v1/closeout75/checkpoint_lock.csv). Quantization labels describe expert representations, not every model tensor.

Common controls: hybrid execution; concurrency one; naive KV cache; eager/no CUDA graphs; serial expert loading; prefill limit 1,024 for the pruning captures; recorded affinity and worker configuration. Qwen/Gemma use text-only arms. The frozen calibration is `results/baseline/calibration_repeat5/profile-run-04.json`. Gemma reuses Qwen's NVFP4 hybrid fraction and has no independently qualified optimal split.

Same prompt text across checkpoints does not create matching tokenizations, hidden states, training distributions or quantization layouts. Model-family differences are checkpoint-level observations, with architecture/precision confounding.

## 2. Research questions and endpoint hierarchy

| RQ | Evidence |
|---|---|
| Concentration | Same-request oracle coverage, 100% and equal-size random masks |
| Held-out generalisation | Macro out-of-fold access coverage, oracle gap, gate weight, weak requests |
| Sequence/layer sensitivity | Per-layer coverage, complete 32/64-token windows and position/cumulative curves |
| Workload/phase transfer | Math/code/incident/mixed calibration against all three held-out workloads in each phase |
| Identity stability | Disjoint calibration halves, random references, overlapping folds and cutoff ties |
| Checkpoint differences | Paired request comparisons with actual token counts and runtime metadata |
| Systems effect | U/F/R replay, fixed versus relative cache capacity, hybrid versus unconditional fill, cold/post-prefill state |
| Deferred ratios | Paired 60%-75%, 50%-75% and 50%-60% extensions; not yet run |

The original primary endpoint is same-workload, decode-derived, held-out macro access coverage at 75%, separately per checkpoint. Prefill, transfers, stability and cache effects are required secondary analyses.

## 3. Static data, folds and ranking

See [dataset.md](dataset.md) for exact corpus roles. The static study has 90 distinct primary requests/model, five globally assigned grouped folds, separate 15-request pilots, and 18 additional repeat captures/model.

- Freeze IDs, source metadata, prompt hashes, group-to-fold assignments and seed before collection.
- In each evaluation fold, exclude its groups from all calibration pools.
- Rank per-layer native access counts descending, then expert ID for deterministic ties. Retain `ceil(.75*E)` experts.
- Preserve full counts/order, zero-count expert universe, tie groups and exact mask identities.
- Keep prefill/decode and thinking/content token accounting explicit. Warmup and pilot rows are separate.
- Estimate macro coverage by request; also retain micro access coverage and effective routing-weight retention. Gemma's scaled weights are not probabilities.
- Oracle masks fitted to the same evaluation request are upper-bound controls, not deployable held-out masks.
- Random-mask controls and the 100% identity check must use the same native universe and retained count.

## 4. Observation integrity and native semantics

Capture native expert IDs and effective selected weights at the architecture's routing point. Record request/token/layer identity, prompt and generated IDs, phase boundaries, sampling/template settings, shared branches, native k, expert-bank bytes and cache counters.

Observation-off/on controls must agree within the declared identity/numerical tolerances. Reject incomplete token/layer geometry, invalid IDs, missing phases, nonfinite weights, source/control hash changes and resource failures. Never reconstruct an omitted phase as an observed trace.

Gemma's original static closeout omitted native R because full router scores/scales were unavailable. The separate repaired arm restores projection, learned scales and original dtypes; nine full-eligibility identities and 12 cold cache controls matched. The historical omission remains recorded in the original closeout.

## 5. Cache counterfactuals

| Mode | Meaning | Work preserved? |
|---|---|---|
| U | Replay original observed routes | Original routes preserved |
| F | Remove excluded original accesses | Active k may fall; removed work must be reported |
| R | Recompute native masked top-k at recorded original hidden states | Native k preserved; later changed activations not propagated |
| Actual M75 | New masked forwards and autoregressive histories | Native k preserved; changed activations propagate |

R is conditional replay, not autoregressive masked inference. F cannot establish a rerouting benefit. Neither cache replay mode measures serving latency.

Always compare original U at its original cache budget with the explicitly labelled alternative. Relative capacity is `ceil(S0 * retained_count / E)`: Qwen 1,536, GPT-OSS 116, Gemma 576 slots at 75%, versus original 2,048/154/768. Keep cache-byte and cache-fraction controls distinct, and separate calibrated hybrid from unconditional fill. Declare cold versus observed/projected post-prefill state; Gemma's observed pre-decode cache is empty.

Replay accounting retains missing experts/event, fills and bytes, CPU executions, active k, removed work and state transitions. Logical bytes/time and simulated cache demand are not hardware-counter bandwidth or measured serving speedup.

## 6. Original static activation gate

The mixed decode-derived candidate used the following prospective checks:

- Access-coverage 95% interval lower bound at least 98%, and effective-weight retention lower bound at least 99%, in each workload and relevant phase.
- Request fifth-percentile coverage at least 95%; every workload/layer mean at least 95%; every complete 64-token window at least 90%.
- Mean disjoint-half retained-ID Jaccard excess over random reference at least 0.10.
- Native R fixed-byte calibrated-hybrid missing-demand reduction at least 5% in every workload, with paired delta interval upper bound below zero.

No checkpoint passed all checks. The static closeout recorded `skipped_by_gate` for actual masking/quality/serving. It remains immutable.

## 7. Separately authorised actual 75% supplement

The user requested actual 25% routed-expert exclusion despite the failed static screens. Four deployment masks/model are fitted from all original calibration-design requests: mixed and the three workloads. Keep one mask in prefill and decode.

Exclusion happens before native selection; preserve original IDs, shared branches and native k. GPT selected-top-k softmax, Qwen normalization/shared gating and Gemma expert scaling retain native semantics. A 100%-eligible mask must reproduce baseline execution.

Full weight allocations and original cache budgets remain in place. This intervention establishes no compact checkpoint, file-size saving or host-RAM saving.

The 300 new synthetic tasks and 100 reference sequences were frozen before evaluation. Generation is greedy/natural EOS capped at 256, with Qwen/Gemma thinking off and GPT low reasoning effort. The original broader independent-task requirement is not met by this supplement.

### Qualification and paired execution

- Per model: 15 baseline/all100 identity pairs and 30 native 75% route audits, plus pilot fixed-input diagnostics. Startup rows are excluded from reported evaluation counts.
- Check emitted IDs, token usage, sampled routes and native pinned-bank labels.
- all100 mean NLL error at most `1e-5` and full-vocabulary mean KL at most `1e-7`.
- Twelve fresh-process shards; task/sequence ordinal modulo 12 freezes membership. Keep all configurations of each item together and reverse treatment order by the declared global ordinal.
- Reset expert cache per request; no prefix reuse. Each model has 900 task generations and 600 likelihood evaluations.
- Fixed-input likelihood scores 256 continuation tokens after a 64-token prefix. Vocabulary-head chunks and causal targets are identical within a checkpoint; Gemma softcapping is retained.
- Save NLL, KL, token agreement, layer L2/cosine, zero-norm handling and replayable lossless FP32 baseline references. References use aligned NPY v2/O_DIRECT storage; unsupported direct I/O rejects the arm rather than silently changing storage.

### Synthetic quality gate

Two percentage points task-score loss and 2% relative perplexity increase, using upper 95% paired template-cluster bootstrap bounds; unsafe-choice increases block serving eligibility. Wide intervals are inconclusive. Frozen code cases and incident substring rubrics have the limitations in [dataset.md](dataset.md).

### Serving diagnostics

Only eligible masks enter five paired fresh-process repetitions. Nine preselected tasks, three/domain, have nine 64-token warmup requests followed by two measured rounds of nine tasks/process. Actual task-specific claims use their named domain.

Run untraced with frozen geometry/cache/prompt settings. Save raw streams, token counts, output hashes, finish reasons, first raw API-delta latency, first visible text latency, end-to-end latency and throughput proxies. First raw model-token TTFT is unavailable without additional instrumentation. One-second sampled VRAM/RSS peaks do not guarantee subsecond peaks.

Incomplete or resource-rejected paired repetitions are preserved and repeated in full. Complete paired blocks may be reused only after configuration, membership, raw-response hash and resource validation. Output lengths may differ; throughput is not a fixed-work kernel speedup.

## 8. Resource acceptance and recovery

Final accepted arms use unchanged operational checks: at least 4 GiB available host RAM over startup/inference; at most 128 MiB system swap-in and swap-out during measured inference; at most 16 MiB server swap growth; at most 1% overall/worst observed 30-second full memory pressure; at least 30 samples; no monitor error or OOM; pinned native banks. Absolute server swap stock is reported without the initial absolute-stock rejection rule.

Require ten quiet seconds within a bounded 90-second settling wait after declared warmup. Final allocator settings include `MALLOC_ARENA_MAX=2`, mmap threshold 131,072 and trim threshold 131,072. Earlier failed policies remain rejected; they were not retrospectively requalified.

Recovery used text-only arms, serial loading, declared startup warmup, the separately qualified Qwen allocator arm, approved unused-application closure, and a transient experiment `MemorySwapMax=0` scope for serving. No global swap/sysctl changes were made. Older GPT paired blocks lacked that final scope, and the environment difference is disclosed.

For changed sources, use a qualified new arm or the preserved isolated runtime. The v6 resume runtime restores the recorded engine without overwriting the user's shared `offload_cache.py` change. All future commands use `uv`; raw invocation and amendment details are in [the experiment log](../../EXPERIMENT_LOG.md) and immutable run directories.

## 9. Reporting, freezing and extension gates

Completed reports must include every accepted/rejected/unavailable disposition, counts, commands/configuration identities, checkpoint/corpus/source hashes, controls, figures and limitations. Native replay and quality evidence have separate meanings; do not count missing metric cells as passed.

The actual finisher validated 11,155 lossless reference-array layouts, exported the supplement, and reverified all 44,155 artifact hashes. The original 22,554 offline artifacts were verified before supplemental closeout. Parent manifests and source snapshots remain untouched during cleanup.

Lower-ratio work must follow [PLAN.md](../../PLAN.md): complete 60% for all three checkpoints before opening 50%, reusing frozen full rankings and maintaining explicit qualification for any new actual-inference stage.
