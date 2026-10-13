# Remaining experiment plan

This is the single maintained plan for unfinished research. Status audited **13 October 2026**. The experiment-only repository cleanup takes priority; no experiment is launched by this document or by the cleanup.

Completed methods and results are in the [documentation index](README.md) and [experiment log](EXPERIMENT_LOG.md). The former bandwidth plan and overlapping pruning plans are superseded; their exact originals remain in the recovery archive.

## 1. Closed stages and claim boundaries

| Stage | Disposition |
|---|---|
| Same-host Qwen/GPT bandwidth and geometry study | Supported finite set complete. Captured-state m=5 results do not justify a geometry-wide runtime override. |
| Static 75% retention | Three qualified checkpoints measured; none passed every original static activation check. |
| Gemma native R75 repair | Complete; nine routing identities and 12 cold cache controls matched. |
| Actual 75% inference supplement | Complete on its frozen synthetic corpus; general real-world quality preservation unestablished. |
| 60% / 50% retention | Not started. Sequential offline ablations remain planned. |
| Kimi-K2.5 | Excluded by user; historical loader/resource blockers retained. |

An accepted synthetic screen is not qualification of a compact checkpoint or a general quality-preserving pruning method. The original 100-independent-tasks/workload requirement remains open.

## 2. Sequential 60% then 50% offline ablations

### Research questions

1. How much held-out access coverage and effective routing weight is lost at 60% and 50% relative to 75%?
2. Do losses concentrate in particular workloads, layers, phases, or complete rolling windows?
3. How do workload transfer and retained-ID stability change?
4. Does conditional native rerouting change miss demand at fixed versus reduced cache budgets?

### Entry gate

- Check both original offline and actual75 completion records and their manifest identities before opening an extension.
- Original offline manifest SHA256: `967c23a59581d36eedc64cd2f1f7eb82f053e41190dd66d92ad640b82b6a1aba`.
- Actual supplement manifest SHA256: `7af5bd1fa883cede21cd4a41ec6998b0ff291335206e51dc48f2bd5e209df0b6`.
- Reuse the frozen full rankings, fold/group assignments, native expert universes, raw routing requests, and statistical seeds. Record the parent identity in each extension.
- Keep parent result files and source snapshots immutable. Gemma primary inputs are in the `additional/gemma_v1` child study; its authoritative three-model offline closeout is at the parent.
- The existing replay/report code is specialized for 75%. Parameterization and complete extension accounting are prerequisites; running only a coverage command does not complete an ablation.

### Execution order and counts

| Model | 75% retained/layer | 60% retained/layer | 50% retained/layer | Native k |
|---|---:|---:|---:|---:|
| Qwen, E=256 | 192 | 154 | 128 | 8 |
| GPT-OSS, E=32 | 24 | 20 | 16 | 4 |
| Gemma, E=128 | 96 | 77 | 64 | 8 |

Counts use `ceil(retention * E)`. Actual fractions therefore need not equal the nominal percentage exactly.

1. Open a new 60% extension and take the first `ceil(.60*E)` experts in each frozen ranking. Preserve tie order, fold, pool and phase.
2. Repeat oracle/random/100% controls, held-out coverage, gate mass, tails, layers, complete 32/64-token windows, all workload-transfer cells, stability, and applicable U/F/R replay controls.
3. Use paired request-level intervals for 60%-75% deltas. Document missing or unavailable cells; do not fabricate native full-score evidence.
4. Validate exact retained counts, 100% identity and nested-mask monotonicity for original-trace coverage. Close the complete three-checkpoint 60% extension before 50% starts.
5. Open a separate 50% extension, repeat the same controls, and report paired 50%-75% and 50%-60% deltas.
6. Publish curves at 50/60/75/100%, with actual counts/fractions and separate oracle and held-out estimates.

### Blockers and fallbacks

- Missing full scores: use the qualified native full-score arm if applicable, or report R unavailable. Post-top-k deletion is F, not native rerouting.
- Missing child completion pointer: validate the parent closeout directly and record that validation; do not invent a child completion receipt.
- Resource failure: preserve the attempt, change the environment only through a recorded amendment, and restart the entire paired unit. Never pool partial rejected measurements.
- Changed numerical source: use an isolated recorded runtime or define and qualify a new arm. Do not overwrite the frozen engine snapshots.
- A change that invalidates 75% results blocks extension until the affected stage is repaired and closed.
- Original-trace coverage is monotone for nested masks with nonnegative weights. Cache misses and actual quality need not be monotone.

Actual 60%/50% masked inference is a separate stage requiring its own quality qualification. A 75% pass does not qualify a lower ratio. If previous results guide mask/data design, use a new unseen confirmation set.

## 3. Independent quality confirmation

The original comprehensive quality claim requires at least 100 independent tasks per workload, external or otherwise independently sourced, with references, licences, source revisions, and meaningful task breadth.

- Freeze calibration, design/selection data and confirmation data before judging the final mask.
- Group related variants across splits and audit source identity, exact text, and substantive task-family overlap.
- Evaluate meaningful math difficulty, code correctness beyond a small fixed case set, and incident evidence with an explicit rubric that supports valid paraphrases.
- Preserve task scores, baseline ability, failures, truncations and output lengths. A near-zero paired loss on two failing outputs is insufficient.
- Retain the original predeclared task-loss and likelihood margins unless a new study explicitly declares different ones before observation.
- Use the new evidence to qualify claims; do not retroactively relabel the synthetic supplement as comprehensive validation.

No new benchmark corpus or external agent harness has been selected or run during cleanup.

## 4. Runtime-aware q follow-up

Saketh's study supports an analytic residency-aware rule on one GPT-OSS/RTX 4070 eager configuration. The following remain unmeasured:

- Device-side integration and performance under CUDA graphs.
- A second model and a second hardware/bandwidth ratio.
- Bandit adaptation across changing conditions or multiple episodes.
- Broad independent prompt coverage and repeated individual-signal/history/ANN ablations.
- Byte-aware selection for heterogeneous expert sizes.

The source study also proposed an expert GPU/RAM/NVMe hierarchy and elastic host-RAM
residency, comparing expected reuse savings with exposed promotion cost. Neither was
implemented or measured here; they require a separate resource and correctness contract.

Any performance implementation needs matched baseline/changed runs, charged policy overhead, controlled cache state, exact output/quality checks, and clear treatment of divergent generated sequences. No performance improvement is implied by implementing the rule.

## 5. Bandwidth-study follow-up

Open directions from the completed study:

1. Joint expert/KV/scratch memory budgeting, particularly the Qwen prefill admission boundary.
2. CPU scheduling, affinity and contention, with kernel counters distinguished from logical-byte proxies.
3. Future-aware cache admission that charges fills and evictions and checks end-to-end latency.
4. Batching/concurrency after memory envelopes are characterized.

Hardware-counter DRAM bandwidth and actual SM occupancy remain unavailable in the recorded evidence. No profiling proxy should be relabelled as either measurement. The captured-state oracle and the end-to-end runtime-q results answer different questions.

## 6. Operational rules for subsequent work

- Use `uv` for Python execution and installation. Keep the existing environments and recorded CUDA 13 toolkit; the system-default compiler differed in historical runs.
- Check GPU ownership before starting a measurement and run one GPU experiment at a time.
- Preserve user engine changes and imported experiment artifacts.
- Log owner, question, model revision, environment, dataset/split, exact command, controls, counts, raw evidence, outcome, blockers and next gate in `docs/EXPERIMENT_LOG.md`.
- Give every run a fresh output directory and bind its source/control hashes.
- Keep accepted and rejected units separate. Runtime/allocator/cgroup changes must be disclosed with their qualification and paired-comparison scope.
- Do not create additional progress/handoff Markdown files. Append to the single log and update the relevant report/index.

## Portable reproduction prerequisite

Use [REPRODUCING.md](REPRODUCING.md) and the uv-based Bash entry points. Establish a
new unpruned baseline and qualify checkpoint/kernel/resource settings on each new
NVIDIA GPU before a comparison. The 13 October reorganization changes source paths;
sealed studies use their original frozen runtime, while new studies must record and
qualify the relocated source identity. No 60%/50% ablation or new baseline was launched
as part of repository cleanup.
