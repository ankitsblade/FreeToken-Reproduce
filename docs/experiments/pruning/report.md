# Static and actual 75% expert-retention studies

The actual supplement completed **8 October 2026 at 12:45:37 IST**. The earlier static closeout and Gemma repair are summarised below. This document summarizes the frozen actual-inference supplement; the original offline static closeout remains preserved.


## Original static 75% study

Three qualified checkpoints, 90 primary requests/model (30/domain), five grouped folds, and 18 additional repeat captures/model. The original closeout freezes 22,554 artifacts with SHA256 `967c23a59581d36eedc64cd2f1f7eb82f053e41190dd66d92ad640b82b6a1aba`.

Same-workload, decode-derived held-out access coverage (%); 95% request-bootstrap intervals conditional on fitted rankings:

| Model | Workload | Coverage | 95% interval | Effective-weight retention |
|---|---|---:|---|---:|
| qwen_nvfp4 | math | 96.41 | 95.68-96.98 | 97.07% |
| qwen_nvfp4 | code | 98.47 | 98.29-98.63 | 98.85% |
| qwen_nvfp4 | incident | 98.91 | 98.72-99.07 | 99.19% |
| gpt_oss_mxfp4 | math | 96.52 | 95.90-97.01 | 97.26% |
| gpt_oss_mxfp4 | code | 96.15 | 95.81-96.47 | 96.70% |
| gpt_oss_mxfp4 | incident | 97.74 | 97.43-98.02 | 98.00% |
| gemma_nvfp4 | math | 99.32 | 99.07-99.52 | 99.41% |
| gemma_nvfp4 | code | 99.57 | 99.50-99.63 | 99.64% |
| gemma_nvfp4 | incident | 99.53 | 99.41-99.64 | 99.61% |

None passed every prospective static activation check. The original quality/masking/serving decision was `skipped_by_gate`; routing coverage is not answer quality. Mixed/task transfer, prefill transfer, weak layers/windows and calibrated-hybrid cache benefits are part of that negative result.

## Gemma native rerouting repair

The separate supplement repaired missing full scores, projection/scaling and dtype restoration without changing the original frozen closeout. Fifteen complete paired pilot controls and 18 full-score repeat captures passed. Nine preselected requests reproduced original/fresh histories and expert IDs; full eligibility reproduced effective weights exactly; all 12 native cold U/F/R cache controls matched.

At 768 slots, calibrated-hybrid R75 reduced mean missing demand by 0.46% over nine requests, with paired delta -0.01386 experts/event and 95% interval [-0.02013,-0.00728]. At 576 slots demand increased 9.51% relative to the original 768-slot baseline. Cold and projected post-prefill coincide because the observed pre-decode cache is empty. R evaluates original recorded hidden states, not new autoregressive histories or latency.

## Actual 75% supplement

## Deliverables

- Three checkpoints: `nvidia/Qwen3.6-35B-A3B-NVFP4`, `openai/gpt-oss-20b`, and `nvidia/Gemma-4-26B-A4B-NVFP4`. All inference uses FreeToken.
- Twelve accepted quality batches per checkpoint: 2,700 task generations and 1,800 fixed-input likelihood evaluations overall.
- GPT-OSS: 15 accepted serving arms; Qwen: ten. Both complete five paired fresh-process repetitions for their eligible masks. Gemma has no eligible serving mask.
- Lossless baseline reference layouts validated; 44,155 supplement artifacts verified by the finisher.
- Manifest SHA256: `7af5bd1fa883cede21cd4a41ec6998b0ff291335206e51dc48f2bd5e209df0b6`.

[Final report](../../../results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/closeout75_actual/report.md) and [completed controller record](../../../results/moe_pruning/study_20261006_v1/orchestration/actual75_scoring_and_closeout_v6_noswap_recovery_20261008/progress.json).

## Quality conclusions

GPT-OSS math/code masks and Qwen code passed the combined synthetic quality screen. Qwen's mixed mask lost 30 percentage points on math (95% paired interval 11-52). Gemma's combined mask decisions are inconclusive; its serving diagnostics were skipped by the quality screen. Individual task-score margin decisions must be read together with fixed-input likelihood results.

The corpus has ten template families/domain and ten authored-reference template families. These results do not fulfill the original requirement for 100 independent tasks/workload or establish broad real-world quality preservation. The full checkpoints remain allocated; expert-route exclusion does not demonstrate physical checkpoint compaction or RAM savings.

## Serving conclusions

Ratios are masked/baseline geometric means over five paired repetitions. Latency ratios below one are faster. These are end-to-end observations with potentially different generated token counts.

| Model / mask | End-to-end latency ratio | 95% paired interval | Interpretation |
|---|---:|---|---|
| GPT-OSS / math | 0.9990 | 0.9931-1.0038 | Near parity; interval includes one |
| GPT-OSS / code | 1.3451 | 1.3379-1.3529 | About 34.5% longer end-to-end latency |
| Qwen / code | 1.0255 | 1.0051-1.0462 | About 2.5% longer end-to-end latency |

These observations do not support a general serving-latency improvement. Throughput proxies must be read with output lengths, first-visible-text latency, and raw API-delta latency; they do not establish a fixed-work engine-kernel speedup.

[GPT-OSS paired measurements](../../../results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/performance75/gpt_oss_mxfp4_paired_summary.csv), [Qwen paired measurements](../../../results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/performance75/qwen_nvfp4_paired_summary.csv).

## Resource recovery and comparability

Firefox was confirmed unused and closed before the successful Qwen quality recovery. The frozen v6 runtime remained isolated from user changes in the shared engine workspace.

For serving recovery, a transient user systemd scope with `MemorySwapMax=0` protected the experiment subtree. Global swap/sysctl settings and frozen numerical workers, masks, task membership, serving order, and acceptance gates remained unchanged. This did not eliminate whole-machine swap-in or pressure rejections; those attempts were excluded and their full paired repetitions repeated.

GPT-OSS repetitions 0-2 were reused from the earlier environment; repetitions 3-4 use process-specific swap protection. All Qwen serving repetitions use that protection. Both treatments within each new paired repetition share the same protection. The different external environments between older and newer GPT repetitions are part of the result's limitations. Amendment and per-run environment records are preserved in the frozen supplement manifest.

## Deferred work

Kimi remains excluded. The 60% and 50% retention ablations have not started. The two reserved PowerPoint slides in the archived 8 October deck have not yet been populated with these completed actual-inference results.


## Methods, data and operational evidence

[Executed protocol](protocol.md) · [Dataset and scorer limitations](dataset.md) · [Resource diagnosis/recovery](resource_recovery.md) · [Master log](../../EXPERIMENT_LOG.md) · [Remaining plan](../../PLAN.md).

Use the immutable raw result reports for exact score tables, figures, source snapshots and per-attempt evidence. Those historical documents remain in result directories; this is the maintained cross-stage interpretation.
