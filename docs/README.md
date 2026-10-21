# Documentation index

This experiment-only workspace combines local reproduction studies with experiments from Kushal, Yeshwanth, and Saketh. FreeToken is an external dependency. Research status below reflects the records reviewed on **12 October 2026**. Cleanup did not launch experiments or rerun historical tests.

## Start here

- [Portable setup and reproduction](REPRODUCING.md): uv-only Bash entry points, baseline contracts, output layout and blocker handling.

- [Experiment log](EXPERIMENT_LOG.md): the single maintained Markdown history for every study, qualification, failure, retry, and closeout.
- [Detailed m-reduction experiment plan](PLAN.md): the only maintained plan for unfinished work, with P1–P9 parameters, pinned datasets, controls and execution gates. Completed protocols describe what was measured; they are not pending launch queues.
- [Dataset and evaluation contract](experiments/pruning/dataset.md): calibration versus evaluation, template dependence, scoring rules, and limits of the synthetic inference study.

## Study reports

| Study | Attribution | Recorded status | Read |
|---|---|---|---|
| Qwen/GPT-OSS bandwidth, traffic, geometry and q oracle | Local reproduction work | Finite supported study closed; 36 controlled captures and 8,000 measured q trials | [Report](experiments/bandwidth/report.md), [reproduction](experiments/bandwidth/reproduction.md) |
| Prefill overlap | Kushal | 32 recorded requests; four repeats per setting/length | [Report and commands](experiments/prefill_overlap/report.md) |
| Routing locality and LRU | Yeshwanth | One captured routing trace; five frequency-preserving shuffle seeds | [Report and commands](experiments/routing_locality/report.md) |
| q* versus fixed CPU/GPU splits | Saketh | Six policies, three cache regimes; separate branch-only oracle | [Report](experiments/split_policy/report.md), [tables](experiments/split_policy/tables.md) |
| Runtime-aware q selection | Saketh | 889 logged episode records; 19-policy main matrix and ablations | [Report](experiments/runtime_q/report.md), [tables](experiments/runtime_q/tables.md), [features](experiments/runtime_q/features.md), [reproduction](experiments/runtime_q/reproduction.md) |
| Static 75% retention, Gemma rerouting repair, actual 75% inference | Local pruning work | Frozen offline and actual-inference supplements complete; broad quality preservation unestablished | [Report](experiments/pruning/report.md), [executed protocol](experiments/pruning/protocol.md), [resource recovery](experiments/pruning/resource_recovery.md) |

The baseline split-policy study and runtime-aware q study have different graph/eager execution and logging overhead. Their absolute timings must be compared within each study. The original bandwidth study's captured-state oracle is also distinct from an end-to-end cache-admission policy experiment.

## Current conclusions and open work

- The 75% actual-inference supplement completed **8 October 2026 at 12:45:37 IST**: 2,700 task generations, 1,800 fixed-input evaluations, 15 accepted GPT-OSS serving arms, ten accepted Qwen serving arms, and 44,155 verified artifacts.
- GPT-OSS math/code and Qwen code passed the combined **synthetic-corpus** quality screen. Qwen mixed math lost 30 percentage points; Gemma mask decisions were inconclusive. End-to-end latency showed no general improvement.
- The original requirement for 100 independent quality tasks per workload remains unmet. Ten families with ten variants are not 100 independent tasks.
- Kimi remains excluded at the user's request.
- Runtime-aware q has not been validated on a second model/device or implemented as the measured device-side CUDA-graph policy.
- Hardware-counter DRAM bandwidth and actual SM occupancy remain unmeasured in the local bandwidth study.

## Runtime dependency

The application build and release source has been archived. The pinned external
runtime, experiment hook patch and observed dependencies are described in
[reproduction instructions](REPRODUCING.md) and [source metadata](../runtime/source.json).
Repository policies remain at [CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENTS.md](../AGENTS.md).

## Presentations and references

- [8 October deck with static retention slides](presentations/FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx)
- [Speaker notes for slides 1-10](presentations/speaker_notes_20261008_slides_1-10.md)
- [1 October progress deck](presentations/FREE_TOKEN_PROGRESS_1-10-2026.pptx)
- [FreeToken paper, archived v1](references/2608.16157v1.pdf)
- [Saketh's archived runtime-q PDF](../results/historical/runtime_q/reports/FreeToken_q_selection_report.pdf)

Presentation slides and PDFs preserve their historical scope. The two reserved slides in the 8 October deck have not yet been populated with the completed actual-inference results.

## Repository organization and provenance

| Location | Contents |
|---|---|
| `experiments/` | All contributors' collectors, analyses and baseline helpers, grouped by study |
| `configs/`, `scripts/`, `runtime/` | Declarative profiles/plans, uv-based Bash entry points, runtime pin and hook patch |
| `results/historical/` | Imported contributors' original measurements, figures, models and binary report; ignored |
| `results/` | Local historical studies, checkpoints and fresh run records; ignored |
| `.runtime/`, `.pruning-runtimes/` | External engine dependency and preserved sealed execution runtimes; ignored |
| `docs/` | Maintained reports, one plan, one experiment log, presentations and references |
| `archives/` | Verified layout backups, original colleague ZIPs and installer archive; ignored |

Raw server logs, JSONL/CSV measurements, source snapshots, and frozen result reports remain beside their results. They are evidence, not competing maintained Markdown logs.

Superseded working Markdown and plans were consolidated after saving exact source text and hashes in the [recovery archive](_archive/documentation_before_cleanup_20261012.zip). The [cleanup manifest](_archive/cleanup_manifest.json) maps each old document to its maintained replacement and records protected source/artifact hashes. Original archived paths are historical identifiers.

The [13 October layout manifest](_archive/layout_manifest_20261013.json) records the experiment-only migration and byte-preservation checks.
