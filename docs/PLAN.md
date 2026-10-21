# Family-adaptive reduction of cache-missing experts (m)

**Priority experiment plan v3, expanded execution specification**

**Updated:** 21 October 2026

**Owner:** Ankit, with human review of implementations and conclusions

**Status:** Stage 0 and Stage 1 are complete for all three models; Stage 2 is closed
(21 October 2026). Results are in `EXPERIMENT_LOG.md`.

- **Done:**
  - Live exact arms, all three models (`stage2_live_20261015T042628Z/analysis/`):
    identical outputs and zero replay mismatches. G2 keep: GPT-OSS `quota_utility`,
    `quota_uniform` and `slru_0.5` (latency 0.96–0.98×). Every P4 q rule is rejected
    on every model (slower or no m gain). Gemma and Qwen keep no exact arm.
  - P7 arm 1 and R75 teacher forcing on D0, D1 for all three models, D2 for all three
    models, GPT-OSS arm 2; final P7 D0 summaries for all three models (21 October).
    No output-changing arm passes G2 on any model: arm 1 fails quality (KL 0.31–0.63),
    and R75 keeps quality but saves only 2–5% of m.
  - P6: no eligible setting on any model.
  - Closeout block (10.3, 20 October), five accepted repetitions per model: floor 0
    and floor 1 make identical fill decisions on all three models, so the correction
    is inert. GPT-OSS `quota_utility` is confirmed against the native reference
    (m −9.4%, latency 0.956× [0.946, 0.965]). Qwen P1 whole-prompt seeding is
    rejected (m −6.8%, latency 1.026× [1.022, 1.030]).
  - Qwen D2: arm 1 KL 0.165, top-1 90.8%, perplexity ×1.17; fails.
- **Queued:** `stage2_live_20261015T042628Z/queue_arm2_retry3.sh`. After the other
  sessions' GPU queues finish, it frees RAM and swap (Judge0 stopped, orphaned shells
  killed, desktop frozen), then retries Gemma/Qwen arm 2 + P7.
- **Blocked or not applicable:**
  - P8 2×2 factorial: no qualified Reuse-Aware P4 variant (q1) or P6 finalist (r1);
    reported as a missing interaction.
  - D3: no OpenCode/SWE-bench harness and no disk for its images; blocked cell for all
    three models.
  - GPT-OSS quota + SLRU composition: no kernel or simulator mode; implementation gap.
  - Gemma and Qwen P7 arm 2: the requant control failed the 1.0% full-memory-pressure
    gate six times each (rolling 1.1–3.2% Gemma, 2.1–4.6% Qwen; merge-at-load workers
    hold extra host memory beside 15–19 GB of host banks).
    Unblocking it needs a protocol decision: exempt quality-only teacher forcing from
    the pressure gate and record pressure instead.
  - D1 likelihood: candidate NLL is below baseline on all models (a real effect of the
    synthetic text, checked 20 October), so D1 perplexity is not evidence of preserved
    ability; KL/top-1 decide.
- **Start state:** every captured native decode starts from an empty expert cache,
  so the native and cold start states coincide.
- **Host state:** this study's freezes ended 21 October 06:55 UTC (desktop thawed,
  Judge0 unpaused; another session's `policy_study` queue has since paused Judge0
  again). Four orphaned spinning `zsh` shells remain SIGSTOPped since 20 October
  (`frozen_spinning_shells_20261020.txt`).

Execution was authorized on 13 October 2026; no experiment is launched merely by
reading this document.

**Live progress:** `tmux attach -t m_reduction_v3`, window `progress`.
The independent user service `m_reduction_watchdog` refreshes a background snapshot
and heartbeat every five seconds and restores a missing/stalled tmux display.
Fallback: `cat results/m_reduction/monitor/latest.txt`. Its active study comes from
`results/m_reduction/monitor/registry.json`; service logs are available through
`journalctl --user -u m_reduction_watchdog -f`.
Current qualification: `results/m_reduction/recovery_nvfp4_precision_20261013T161122Z/`.
NVFP4 diagnostic evidence: `results/m_reduction/nvfp4_qualification_20261013T153609Z/`.
Qualified GPT-OSS pilots: `results/m_reduction/recovery_dot64_20261013T131424Z/`.
First correction attempt: `results/m_reduction/recovery_20261013T125617Z/`.
Original failed study: `results/m_reduction/study_20261013T104934Z_6309f0f7/`.
The `gemma_baseline` and `p1_p2` windows retain the completed baseline retry and
stopped admission/trace controller. `requalify_gpt` retains the first correction's
failed screen; `monitored_recovery` retains the completed GPT-OSS controller.
`nvfp4_qualification` and `nvfp4_no_vnni` retain completed Qwen/Gemma diagnostics.
`build_nvfp4_precision` retains a compiler failure; `build_nvfp4_precision_retry`
retains its successful repair build. `nvfp4_precision` retains a queue flag error;
`nvfp4_precision_retry` retains both matched probes. `nvfp4_recovery` runs the
fresh two-model qualification queue with the accepted GPT pilot dependency;
`nvfp4_resume1` resumes it after the first screen failures.
`aligned_probe` retains the completed first same-state diagnostic. See [the log](EXPERIMENT_LOG.md) for accepted and failed
attempts. P3–P9 and P2 teacher-forcing/quality support remain unfinished.

**Next gate:** diagnose Qwen/Gemma CPU/GPU numerical paths at fixed hidden states,
qualify any format-specific corrections, then collect their fresh profiles and
four complete 15-pair pilot arms. Require cross-arm output identity for all three
models before primary collection. Full P2 support still gates later research stages.

This is the repository's single maintained plan for unfinished research. It expands
`m_reduction_experiment_plan_v3_priority.md` and supersedes the previous
execution queue. Completed methods and
results remain in the [experiment log](EXPERIMENT_LOG.md) and
[study reports](README.md). Read [CONTRIBUTING.md](../CONTRIBUTING.md) and
[REPRODUCING.md](REPRODUCING.md) before implementing or collecting measurements.

Values labelled **new specification** below are prospective choices made explicit
by this plan. They are not claims that the corresponding code or data collection
already exists. Existing frozen configurations are historical contracts; do not
edit them to make an old run appear to follow this plan.

## 1. Question, evidence and claim boundaries

Measure whether fewer routed experts are absent from the GPU cache during decode,
and whether that reduction improves serving on the same hardware and memory budget.
Separate cache admission, cache replacement, routing changes and physical expert
aliasing. A reduction in missing demand does not by itself imply fewer PCIe transfers,
lower latency, preserved task quality or a smaller checkpoint.

Historical inputs, with their original scope:

- Ankit's cold, original-hidden-state R75 replay found fixed-cache miss reductions
  of 0.87–2.24% for Qwen, 0% for GPT-OSS and 0.46% for Gemma. Reducing cache size with
  the pool increased misses. These are offline comparisons, not new live baselines.
- Saketh's original GPT-OSS split-policy study found that native q* admitted almost
  no experts. His eager runtime-q extension reported 1.52x geometric mean speedup
  for Reuse-Aware on one host/model. The earlier split study used CUDA graphs;
  absolute timings from those two studies are not interchangeable.
- Yeshwanth's routing-locality study compared ordered and shuffled routing on a
  limited captured trace. It motivates locality controls; it does not establish
  a 30% gain on the new three-model corpus.
- The actual 75% inference supplement **completed on 8 October 2026**. Its synthetic
  quality screen and serving results are in the [pruning report](experiments/pruning/report.md).
  It is not a running dependency, broad quality qualification, or a compact checkpoint.
- The archived [FreeToken paper](references/2608.16157v1.pdf), Sections 3.2 and 5.1,
  motivates a nonzero fill floor and W1/W2-style validation. Our quantized checkpoints,
  device, context limits and task sets differ from its main evaluation.

The output of this study is a per-family verdict on **tested configurations**.
Neither a small deletion estimate nor failed candidates rules out every pruning
or routing method. Kimi remains excluded. Shared/dense branches keep their native
computation and are excluded from routed-expert cache accounting and modifications.

### 1.1 Mandatory coverage: every experiment on all three models

**User requirement:** run every P1–P9 experiment, its baselines, controls and retained
ablations on all three pinned MoE checkpoints: GPT-OSS-20B MXFP4, Qwen3.6-35B-A3B
NVFP4 and Gemma-4-26B-A4B NVFP4. This requirement overrides earlier model-specific
optional/skip language. Keep the declared per-model geometry, context/cache budgets,
native router semantics and format-specific kernels; use the same experimental
question, dataset IDs, controls and evaluation contract across the three models.

| Experiment | GPT-OSS | Qwen | Gemma |
|---|---|---|---|
| P1 floor, warm-start and seeding comparisons | Required | Required | Required |
| P2 traces, replay and numerical controls | Required | Required | Required |
| P3 miss/capacity/replacement diagnostics | Required | Required | Required |
| P4 admission and fill selection | Required | Required | Required |
| P5 fixed-budget layer allocation | Required | Required | Required |
| P6 resident-biased routing | Required | Required | Required |
| P7 both 75% exclusion/merging arms | Required | Required | Required |
| P8 interaction, composition and component ablations | Required | Required | Required |
| P9 replacement/pinning comparisons | Required | Required | Required |

Attempt dataset/context and kernel qualification for every model, including D2/D3.
G1 signals change scheduling priority and adoption decisions, not model coverage.
Small expected benefit or a negative result does not justify omitting a model.
If a cell cannot run under its declared controls, retain the attempted qualification,
record it as **blocked** or **failed** with its cause, and attempt a separately
identified repair. Do not silently drop it, substitute another checkpoint or relax
quality/resource gates. An experiment remains unfinished while a required model
cell is blocked or missing; completion requires all three models' required runs.

The current MXFP4 recovery controller is a GPT-OSS-specific repair prerequisite.
Its successful pilot alone cannot complete P1/P2. Qwen and Gemma must each receive
their own numerical/backend qualification and the full corresponding experiment
runs next. Kernel fixes may differ by format; this never makes their scientific
experiment optional. Later-stage runners must retain an explicit experiment x
model x arm coverage matrix and enforce this completion rule.

The machine-readable coverage contract is
[coverage_policy.json](../configs/m_reduction/coverage_policy.json).
[m_reduction_plan_gate.py](../experiments/pruning/m_reduction_plan_gate.py)
audits the controlled Stage 0 dataset, model pins/budgets, serving/sampling settings
and resource gates against this contract. It counts unique resource-passed captures,
requires all four 15-request pilot arms and matched outputs for every model, and
keeps missing P2 support explicit. A two-request screen is insufficient.

Before primary collection run this gate with `--require-ready primary`; before
advancing past Stage 0 run it with `--require-ready next-stage`. Use a fresh
`--out` and explicit `--model-study MODEL_ID=STUDY` bindings when family-qualified
evidence lives in separate roots. It refuses the transition if required evidence
is missing. The original controller now invokes the primary gate; GPT-only recovery
returns to model qualification and does not advance the research stage. The active
controller was started before these code changes and already stops at that boundary;
its numerical runtime and measurements have not been changed.

## 2. Fixed machine, checkpoints and memory contract

### 2.1 Checkpoint pins

Use the repositories/revisions in [models.json](../configs/pruning/models.json).
Local paths come from the caller after `source scripts/env.sh`; paths are not pins.

| Model ID | Hugging Face repository | Revision | MoE layers L | Experts/layer E | Native k | Format |
|---|---|---|---:|---:|---:|---|
| `qwen_nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | 40 | 256 | 8 | NVFP4 |
| `gpt_oss_mxfp4` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 24 | 32 | 4 | MXFP4 |
| `gemma_nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` | 30 | 128 | 8 | NVFP4 |

Confirm routed layer IDs and native k from each loaded checkpoint. Preserve download
revision metadata and hash model metadata, tokenizer/template files and weight shards
in the new study manifest. A loader, dtype, kernel or quantization change is a new arm.

Start the isolated runtime from base commit
`4d0f84983700eba7dfb046fb4df3cbbcc488ecc2` plus the preserved hook patch
SHA256 `5db728e38ce6faec74894673eb722bc6e34138d116c0dd5dae35787dac74d35e`
in [runtime/source.json](../runtime/source.json). Apply new floor/routing/cache hooks
as a separate recorded patch and hash the resulting source and compiled kernels.
Use Python 3.13, the recorded CUDA 13 toolkit and the existing pinned runtime
dependencies, including `torch==2.11.0+cu130`; a changed environment is separately
qualified. The frozen v6 pruning runtime remains a different source arm.

### 2.2 Primary serving settings

The intended comparison host is RTX 4070, 12 GB VRAM, PCIe Gen4 x4, Ryzen 5 8500G,
approximately 30 GB host RAM. Record the observed GPU UUID, negotiated PCIe link,
RAM, driver, clocks/power settings, CPU affinity and package versions on every run;
the description above is not a current machine diagnostic. Another device requires
its own qualified baseline and separately reported results.

| Parameter | Qwen | GPT-OSS | Gemma |
|---|---:|---:|---:|
| Global expert cache slots C | 2,048 | 154 | 768 |
| Actual cache fraction C/(L E) | 20% | 20.0521% | 20% |
| Average slots/layer, descriptive only | 51.2 | 6.4167 | 25.6 |
| GPU memory ratio | 0.9 | 0.9 | 0.8 |
| Total sequence cap, including template/prompt/generated tokens | 2,048 | 2,048 | 1,280 |
| Primary prompt cap, after rendering | 1,024 | 1,024 | 1,024 |
| Primary generated-token cap | 256 | 256 | 256 |
| Two full-layer prefill buffers, within C | 512 slots | 64 slots | 256 slots |
| Routed expert bytes in recorded geometry | 1,775,616 | 13,236,480 | 3,353,856 |
| C x expert bytes | 3,636,461,568 | 2,038,417,920 | 2,575,761,408 |

Assert these slots and allocated cache bytes at startup and at every request boundary.
The slot budget includes the engine's prefill staging allocation; never count the
same memory again as extra decode capacity. All model-specific bank tensors, scales,
padding and cache metadata must be accounted for. Gemma's byte value comes from
the [executed protocol](experiments/pruning/protocol.md), since the current model
JSON omits it. Verify all three expected bank sizes against the newly loaded
runtime before G0; a different layout is a separately identified byte-budget arm.

Common settings, **new study contract** unless already supplied by the baseline profile:

| Setting | Exact value / behavior |
|---|---|
| Backend / concurrency | FreeToken `hybrid`, batch 1, one request active, one GPU experiment at a time; Qwen/Gemma text-model-only |
| Execution / KV | Eager; `cuda_graph_max_bs=0` / `--no-graph`; naive KV cache |
| CPU expert workers / loading | 5 workers; serial expert-bank loading; pinned host banks required |
| Allocator / library threads | `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| Scheduling | Disable overlap between separate requests; retain native CPU/PCIe/GPU overlap within a layer |
| Generation | Temperature 0, top_p 1, sampling top_k -1; greedy; natural EOS; no forced continuation |
| Seeds | Corpus/folds `20261006`; arm ordering/selection `20261013`; bootstrap `20261014` |
| Original routing corpus template | Thinking enabled, matching its original protocol; save exact rendered token IDs |
| Synthetic quality template | Qwen/Gemma thinking disabled; GPT-OSS low reasoning effort, matching its frozen scorer contract |
| W1/W2 template | Thinking enabled on Qwen; GPT-OSS medium reasoning effort; Gemma native template, same recorded template for every arm |
| Timing warmup | One unmeasured 64-token request using `math_000`; clear cache and policy histories afterwards |
| Primary request lifecycle | Reset cache and policy history before each request's prefill; capture actual post-prefill state |
| W2 lifecycle | Reset at session start only; preserve expert cache/history through all requests and user turns |

Template/effort differences are separate dataset contracts, not effects attributed
to a policy. All reasoning, tool-call and visible tokens count toward generation
and context limits. EOS-terminated requests can have fewer than 64 decode events;
report their lengths rather than manufacturing a steady-state window.

Use the existing [resource gates](../configs/pruning/resource_policy.json): available
host RAM at least 4 GiB, inference swap-in and swap-out each at most 128 MiB, server swap growth at most
16 MiB, full memory-pressure percentage at most 1% over 30 seconds, at least 30
inference monitor samples, 90-second settling timeout and 10 quiet seconds. Use a
user systemd scope with `MemorySwapMax=0`. Sample resources once/second. If a short
run lacks 30 samples, qualify the complete paired batch and retain its boundaries;
do not relabel missing observations as a pass. Reject incomplete/pressure-failed
paired timing blocks and repeat the whole block in a fresh directory.

### 2.3 Cache-capacity sensitivity, separate from equal-budget comparisons

| Nominal regime | Qwen slots | GPT-OSS slots | Gemma slots |
|---|---:|---:|---:|
| High miss, 10% of native pool | 1,024 | 77 | 384 |
| Medium miss, primary ~20% | 2,048 | 154 | 768 |
| Low miss, 40% of native pool | 4,096 | 308 | 1,536 |

Counts use `ceil(fraction x L x E)` and each policy pair uses the same capacity.
Run all three capacities offline. The 40% Qwen/Gemma allocations may not fit the
12 GB device: live sensitivity is conditional on fitting the unchanged KV/scratch
contract. An unavailable live cell stays unavailable. Do not reduce context or
memory allocations to present it as the primary arm. Workload domains are math,
code and incident; miss regimes above refer to capacity, not new datasets.

## 3. Exact datasets, splits and trajectory controls

### 3.1 D0: original routing/design corpus

Use [pruning_requests.jsonl](../experiments/prompts/pruning_requests.jsonl) and
[pruning_splits.json](../experiments/prompts/pruning_splits.json), without regeneration.

| Dataset field | Frozen value |
|---|---|
| Source ID / revision | `local-curated-with-assistant` / `20261006-v1` |
| Primary IDs | `math_005`–`math_034`, `code_005`–`code_034`, `incident_005`–`incident_034` |
| Primary size | 90/checkpoint: 30/domain; same text across three checkpoint tokenizers |
| Pilot IDs | `_000`–`_004` in each domain: 15/checkpoint; excluded from primary statistics |
| Outer folds | Existing 5 grouped folds; 6/domain held out, 24/domain for calibration per fold |
| Mixed calibration | 72 train / 18 held out per outer fold |
| Domain calibration | 24 train/domain; evaluate on the 6 held-out requests in each target domain |
| Repeated requests | `code_021`, `code_023`, `code_029`, `incident_010`, `incident_011`, `incident_025`, `math_011`, `math_015`, `math_033` |
| Repeat captures | Two additional captures of each of those 9 IDs; 18/checkpoint, not extra tasks |
| Task references | None: D0 supports routing/traffic analysis, not task-accuracy claims |
| Request-file SHA256 | `37062bc1d6b23d6d1b21ebcb643c8a33a70821834c77a71e1852df7b53bd8a6b` |
| Split-file SHA256 | `e8aaa6199a9c03912ba8fa2390a1860ed0398267411fed06a335d8cc7e63bd48` |

The old rows say `repository MIT`; the current repository retains Apache-2.0.
Preserve the original provenance field and disclose this mismatch; do not silently
rewrite the frozen dataset's licence attribution. Related/conceptually similar
requests are not guaranteed independent merely because IDs differ.

For parameter selection, use **nested selection**: within an outer training set,
use each of its four remaining original folds as inner validation, fitting on the
other three (54 mixed requests, 18/domain). Fit final fold quotas/rankings on all
72 outer training requests only after inner selection. Report out-of-fold results
on the 18 untouched outer requests. Fixed, predeclared candidate tables can also
be reported, but choosing the best outer result and reporting it as unseen
confirmation is prohibited. Select final deployment parameters by the most frequent
inner-selected configuration across outer folds; break ties by smaller routing
perturbation, then arm ID. Fit deployment artefacts on D0 only.

Every learned quota, miss ranking, saliency ranking, cluster and pin set has four
calibration variants: mixed, math, code and incident. Report their 4 x 3
calibration-pool/target-workload cells with the same excluded outer fold. No
secondary or quality task may train these artefacts or choose hyperparameters.

### 3.2 D1: existing synthetic quality regression set

Reuse the completed supplement's immutable corpus, described in the
[dataset contract](experiments/pruning/dataset.md):
`results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6/quality_corpus/`.

- `tasks.jsonl`: 300 tasks, 100/domain, ten families/domain x ten variants.
  SHA256 `5b3d5aac670b14daabfc37d27a1f4f1182d80035989cf15001fcfdfffd69242e`.
- `sequences.jsonl`: 100 operational narrative sequences, ten reference clusters.
  SHA256 `008900e85cc43b0bdcb418f5a3363f0bb84c87357ea1d8b84c3fa93ee24f85fd`.
- Task generation cap 256; fixed-input sequences have a 64-token prefix followed
  by 256 scored tokens, within a 320-token checkpoint-specific input.
- Preserve the exact rational-answer math scorer, code case sets and incident
  0.6 action + 0.2 + 0.2 evidence-substring rubric. Save invalid answers, failures,
  truncations, unsafe-choice flags and baseline ability.
- Preserve ordinal-modulo-12 execution shards: 25 tasks and 8–9 reference sequences
  per shard. Shards are resource partitions, not folds. Bootstrap task families
  and reference clusters, not 100 nominally independent variants.

D1 is a **known regression screen** whose historical outcomes have been seen.
It is not fresh independent confirmation. Recompute unmodified baseline and
candidate outputs on the new runtime; never use old baseline timings as a pair.
A 100-independent-tasks/domain corpus remains unselected and blocks any broad
quality-preservation claim. Do not equate D0's incident domain with paper W4.

For the new decode-policy likelihood/KL comparison, prefill the 64-token prefix,
score the first continuation token from the final prefix position, then process
the other 255 causal continuation positions through sequential teacher-forced
decode, maintaining the candidate's cache and hidden states. Save the phase of
each of the 256 scored targets. The old full-prefill likelihood path can remain
a separately labelled compatibility diagnostic; it cannot qualify a decode-only
cache-aware router because that router would never be activated.

### 3.3 D2 / W1-style: AIME 2025-I, 15 fixed problems

**New selection:** [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025),
revision `a6ad95f611d72cf628a80b58bd0432ef6638f958`, file `aime2025-I.jsonl`,
configuration `AIME2025-I`, split `test`. Use all 15 rows in source order; local
IDs `aime2025_I_01` through `aime2025_I_15`. The publisher lists MIT; retain its
card and source-attribution/licence information in the acquisition manifest.
The II file is not used for calibration or replaced into I after seeing results.

| Parameter | Qwen / GPT-OSS | Gemma, required |
|---|---:|---:|
| Rendered input budget, including all template tokens | 512 | 512 |
| Generated tokens, including reasoning | 1,536 | 768 |
| Total sequence cap | 2,048 | 1,280 |
| Tools / samples | None; one greedy sample/problem | Same |

Prompt suffix: `Return your final answer as \\boxed{n}, where n is an integer from 0 to 999.`
Score the last valid boxed integer against the dataset answer; missing/invalid or
unfinished answers score zero. Save all output channels, token counts and finish
reasons. Do not silently cut a problem to fit. Exclude an over-budget input before
policy evaluation and record its ID/token count; retain truncation at the generation
cap as an outcome. Report the full 15-task denominator and the eligible count.

Freeze one unmodified baseline trajectory/model/problem for teacher-forced
comparisons. Fixed-logit screens reuse its logits; candidate teacher forcing runs
the candidate's own activations on the same token history. Free-running evaluation
uses independent candidate outputs. This is context-limited W1-style evidence,
not the paper's unconstrained long reasoning reproduction. Model-training
contamination is unknown; held out here means excluded from policy calibration.

### 3.4 D3 / W2-style: five fixed SWE-bench Verified issues

**New selection:** [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, split `test`,
file `data/test-00000-of-00001.parquet`. Use the first five `psf/requests` IDs
in lexicographic order, selected without model outcomes:

| Instance ID | Repository base commit | Environment-setup commit |
|---|---|---|
| `psf__requests-1142` | `22623bd8c265b78b161542663ee980738441c307` | `ba25184ed5f0bf9b876dea3cf4312fa35b539a7c` |
| `psf__requests-1724` | `1ba83c47ce7b177efe90d5f51f7760680f72eda0` | `4bceb312f1b99d36a25f2985b5606e98b6f0d8cd` |
| `psf__requests-1766` | `847735553aeda6e6633f2b32e14ba14ba86887a4` | `4bceb312f1b99d36a25f2985b5606e98b6f0d8cd` |
| `psf__requests-1921` | `3c88e520da24ae6f736929a750876e7654accc3d` | `3eb69be879063de4803f7f0152b83738a1c95ca4` |
| `psf__requests-2317` | `091991be0da19de9108dbe5e3752917fea3d7fdc` | `091991be0da19de9108dbe5e3752917fea3d7fdc` |

This is a narrow, one-repository five-issue sample, not a representative Verified
benchmark. Preserve dataset and per-repository licences; no licence inference from
the harness applies to repository code. The public source metadata/IDs have been
inspected for planning; local frozen acquisition and harness qualification remain open.

Use OpenCode connected to FreeToken, no subagents, one active model request,
three scripted user turns per issue:

1. `Fix the following issue in this checkout. Inspect the relevant code and implement a minimal patch.\n\n{problem_statement}`
2. `Review your patch against the issue. Check relevant edge cases and correct any remaining problem.`
3. `Validate the final patch with the available checks and summarize the change. Do not modify tests to hide a failure.`

No gold patch, hidden test patch or hints are put in the model context. Pin OpenCode
commit/version, provider configuration, tool schemas, SWE-bench harness revision,
container image digest, timeouts and repository state in a manifest **before**
recording baseline sessions. These harness pins are unresolved entry blockers;
no existing local OpenCode/SWE-bench integration is claimed.

**New session limits:** 600 seconds/issue, at most 12 model calls total, at most
4 model calls/user turn, at most 6 tool invocations/turn, 30 seconds/tool invocation,
384 generated tokens/model call. For Qwen/GPT-OSS, every rendered model input must
be at most 1,664 tokens, leaving 384 within the 2,048 cap. Gemma is required:
qualify its complete harness against 896 input + 384 generated tokens within 1,280.
If it cannot fit, record a blocked Gemma cell and attempt a separately specified
repair; keep its experiment unfinished rather than making Gemma optional.
Tool responses expose at most 256 model-tokenized tokens; save their complete raw
output separately and append a recorded truncation marker to shortened responses.
Keep the same limit for every policy in a checkpoint.

Disable compaction, automatic context summarization, prefix/KV reuse and retries.
Carry the GPU expert cache across calls/turns while each call performs its own
prefill. If required issue/history/tool messages exceed the context budget, record
an ineligible session; do not cut user instructions/history, replace the issue,
shrink the expert cache or raise the cap. W2 may be infeasible on this device.
Report that outcome and the eligible denominator rather than substituting a claim.

For fixed-trajectory analysis, record one baseline session's prompts, calls, token
IDs and tool outputs and replay those identical inputs for every candidate, with
candidate-specific cache state carried forward. For live task evaluation, tools
execute on each candidate's own fresh checkout; tool outputs and call counts may
diverge. Recorded baseline tool outputs are not reused as if they came from a
candidate's live patch.

Task success uses the pinned [official SWE-bench evaluation contract](https://www.swebench.com/SWE-bench/guides/quickstart/):
required FAIL_TO_PASS and PASS_TO_PASS outcomes, with patch/application and harness
failures recorded. Also report all-three-turn completion and exact gold-patch match
as a paper-alignment diagnostic. A valid alternative patch can succeed without
being text-identical to gold. No issue-solving or test execution is authorized
merely by documenting this future evaluation.

### 3.5 Dataset freeze and exclusions

Before new collection, write a manifest containing dataset revision/file SHA256,
row ID, original text hash, grouping, fold, model-specific rendered token IDs and
hash, reference/scorer version, template, caps, exclusions and exclusion reason.
Materialize D2/D3 from the pinned revisions and verify the chosen IDs/commits; a
moving viewer page is insufficient provenance. Save baseline trajectories and
W2 tool outputs by content hash. Audit exact and substantive overlap with D0/D1.

D2/D3 are never used to choose candidates. A quality failure on them cannot be
repaired and then retested as fresh confirmation on the same tasks: a redesigned
arm needs a new predeclared confirmation set. Report primary, synthetic regression
and secondary estimates separately. W3/W4, concurrency, context editing and
dynamic cache resizing are deferred.

## 4. Measurement, replay and statistical contract

### 4.1 Units and metrics

For decode event (request r, token t, MoE layer l), let U be unique routed physical
experts **before** cache admission and C_t the resident set. `m = |U minus C_t|`;
`q` is the count admitted/fetched; CPU-executed misses are `m - q`.
Measure misses before fills and protect active experts from eviction during the
same event. In P7 aliasing, also save original logical top-k and physical unique k.

Every arm records:

- E[m] and P95(m) per request/layer, overall, tokens 0–63 and tokens 64 onward.
  P95 uses the `higher` empirical quantile. Empty steady-state windows are unavailable.
- Missing experts/token summed across layers; unique-demand miss rate; missing bytes,
  actual fills, CPU-executed misses and evictions/token; occupancy and residence ages.
- Full 32/64-token windows only, layer curves, domain summaries and complete request counts.
- For changed routing: substitution/drop/alias rates, original selection-probability
  mass retained, effective native weights and replacement-hit/replacement-miss counts.
- Live: decode ms/token, tokens/s, request latency, TTFT, first-visible-text latency,
  token interarrival P95, generated lengths, finish reasons and task outcomes.

`m x expert_bytes` is **missing demand**, not PCIe traffic. `q x expert_bytes` is a
logical expert-fill proxy. Record actual copy sizes for weights/scales, prefill,
seeding and CPU-result transfers separately; label hardware-counter PCIe totals
unavailable if counters are unavailable. Branch timings and exposed waits explain
latency; they cannot be added as if CPU/PCIe/GPU work were serial.

### 4.2 Replay fidelity and modes

| Mode | What is held fixed | Valid use / limit |
|---|---|---|
| Access replay | Baseline chronological routed IDs; policy's own evolving cache | P3/P4/P5/P9 m, if admission, fetch selection and eviction match live |
| Fixed-logit/cache simulation | Baseline per-layer scores; simulated routing and cache influence each other | P6/P7 screening only; upstream hidden states remain baseline |
| Candidate teacher forcing | Baseline prompt/output token history; candidate recomputes its own hidden states and scores | Paired m and full-vocabulary KL with propagated routing changes |
| Live free-running | Same task/start settings; each arm generates and calls tools independently | Latency and task outcome; tokenwise routes need not be paired |

The fixed-logit simulator is closed loop **with respect to cache/routing only**.
It is not an autoregressive model simulation. A fixed-logit win cannot qualify
quality or predict downstream hidden-state routing exactly.

Primary starts from captured **post-prefill** state, even when that state is empty.
Cold start is an explicitly labelled control. Each alternative capacity needs its
own captured state for a live claim; deterministic offline projection of the
primary state is a separate sensitivity arm. Include slot owners, usage timestamps,
expert recency, step counter, free slots, fetch settings and policy histories.

### 4.3 Statistics and timing repetitions

Primary analysis gives each request equal weight and each workload one-third of
the overall mean. Pool token events only in a separately labelled token-weighted
appendix. Define reduction as `1 - arm_mean_m / matched_baseline_mean_m`; report
absolute paired deltas as well. A zero baseline denominator makes relative reduction
undefined. Calculate per-request P95 before request-level summaries; also report
the pooled-event P95 with its distinct weighting.

Use 10,000 paired bootstrap resamples, seed `20261014`, resampling request groups
within each outer-fold/workload cell and recomputing equal-workload summaries.
Report 95% percentile intervals and all five fold point estimates. These intervals
are conditional on fitted artefacts and this known collection; five folds are
not five independent datasets. D1 resamples its ten families/domain or ten reference
clusters. D2 resamples problems; D3 resamples issues and discloses one-repository
correlation. Small-n secondary sets remain directional and are never pooled into D0.

**Live timing cohort:** the nine preselected repeat IDs in Section 3.1, three/domain.
Use five complete paired repetitions/cohort request/model. Every arm has a fresh
server, the same timing warmup/reset and the same request-specific prefill lifecycle.
Shuffle arm order within each block with seed `20261013 + repetition`; save the
realized order. A repeated execution is not another independent task. Pair on
request and repetition, report geometric latency ratios with paired intervals,
raw repetitions, policy overhead and accepted/rejected counts. End-to-end results
with different generated lengths are not fixed-work kernel speedups.

For branch diagnosis, retain instrumented traces. For performance, disable verbose
per-layer logging while retaining all policy feature computation and its overhead;
use equally instrumented controls and save the chosen logging/timing configuration.
Verify observation-off/on output equivalence on pilots before using lean timing.

## 5. P1 — fill floor and warm-start baseline

**Question:** does native q* fill, and what baseline is valid for later comparisons?

Inspect the isolated new runtime's native CPU reference and GPU kernel rounding.
The current preserved `offload_kernels.py` computes integer balance without an
explicit minimum-one clamp; confirm this with new trace evidence. The paper states
at least one fill. Do not reinterpret either historical study as already corrected.

Define `R(f,m)` exactly as existing `round_balance` in
[policy_q.py](../experiments/split_policy/policy_q.py): clamp/round f to Q16, compare
`lo = (m x f_Q16) >> 16` and `lo+1` by
`max(q x (65536-f_Q16), (m-q) x f_Q16)`, choose lo on a tie, clamp to `[0,m]`.
For corrected arms, `F(q,m)=0 if m=0 else min(m,max(1,q))`. Apply the floor after
rounding/selection in CPU reference, GPU execution and replay.

P1 arms on all 15 pilot IDs/model:

| Arm | q rule | Purpose |
|---|---|---|
| `native_floor0` | Shipped calibrated fraction, native rounding | Preserve paper/code discrepancy control |
| `native_floor1` | F(R(f_native,m),m) | Corrected main baseline |
| `reuse_freq_floor0` | Original Reuse-Aware rule | Historical-rule diagnostic |
| `reuse_freq_floor1` | F(q_reuse,m) | Same-floor policy comparison |

Capture all 90 primary IDs/model under `native_floor1`, plus the two additional
captures of each of nine repeat IDs. These are fresh captures; the historical
engine/manifest stays immutable. For every miss event verify integer m/q,
`0 <= q <= m`, nonzero admission for `floor1` when m>0, physical slot ownership,
pinned banks and exact baseline output versus the corresponding observation-off arm.
A legitimately empty post-prefill cache is valid; decode must subsequently fill.

**Prefill-informed seeding add-on:** compare native residual state against seeds
ranked by prompt routing frequency over the last 64 prompt tokens and over the
whole prompt. Select at most C unique `(layer,expert)` entries globally, descending
frequency, then layer ID/expert ID; selected-set ties are deterministic. Mark most
recent prompt touches in the recency record. Missing seed entries require real
copies; charge bytes, seeding wall time and TTFT. Apply the same final seed state
to compared decode policies and report early/steady-state m separately. This is
not assumed free. Leave native seeding as the primary baseline unless a timed,
equal-budget comparison meets the exact-component adoption gate.

**G0:** all three checkpoint kernels/resources qualified; cache bytes bound; main
floor arms admit as specified; observation does not change output. An unqualified
model is blocked rather than silently omitted from an all-model conclusion.

### 5.1 Numerical recovery before changing q

Changing q moves routed experts between CPU and GPU. Qualify those numerical
paths at the same hidden state before interpreting an output difference as a
routing or cache-state defect. Preserve the original failed captures. A corrected
source must use a fresh profile and baseline; earlier timings are dependencies
for diagnosis, not paired serving evidence.

The GPT-OSS correction is a separate prospective arm,
`configs/m_reduction/stage0_mxfp4_alignment.json`, using
`FREETOKEN_M_REDUCTION_MXFP4_ALIGN=1` on the isolated runtime:

- Round CPU gate/up linear outputs, including bias, to BF16 before clamped SwiGLU,
  matching the GPU projection boundary. Keep the activated intermediate in BF16.
- Round each weighted down-projection expert output to BF16 before accumulation.
- Accumulate each CPU/GPU subset in FP32, transfer the CPU partial in FP32 and
  cast the combined result to BF16 once. Include the larger result transfer and
  all changed computation in new timing/calibration measurements.
- Keep native prefill, routing weights, expert IDs, quantized weights, physical
  cache bytes, worker count and q policies fixed. The compiled extension exposes
  the FP32-output capability explicitly and rejects enabling it after tasks exist.

The first correction failed complete-generation identity on the two-request screen.
The second prospective arm uses
`configs/m_reduction/stage0_mxfp4_dot64.json` and additionally sets
`FREETOKEN_M_REDUCTION_MXFP4_DOT64=1`. It accumulates CPU/GPU split-K projections
in FP64, converts to FP32 at the same point before bias/weight operations, and uses
the same `gate / (1 + exp(-alpha*gate))` evaluation order. Both branches evaluate
the exponential in FP64 and round it to FP32; GPU division uses round-to-nearest
and disables activation multiply-add fusion. The activated/weighted routed outputs
still use the declared BF16 boundaries. This mode changes decode computation and
must be profiled as its own arm; no performance benefit or general equivalence is
assumed. Its initial diagnostic targets layer 10, immediately before the first
router difference seen in the first correction. Prefill remains on the native path.
The recorded layer-10 expert/group outputs matched exactly. The first complete
`math_000` floor0/floor1 comparison then matched all 256 generated tokens and all
8,496 router-score events; `math_001` also matched generated tokens. Both floor
batches passed resource gates. Reuse-Aware screens and complete pilots remain necessary. The
fresh CPU overlap profile fell from 34.93 to 4.31 GB/s in this arm; its substantial
cost is a result to report, not grounds to reuse the faster arm's calibration.

Run `m_reduction_numerics` for identical inputs/weights/IDs and retained expert
outputs, then `m_reduction_recovery` for fresh calibration, the two-request screen,
three fresh-process baseline repetitions and all four 15-pair pilots. The screen
is not full model qualification. Exact observation and cross-arm output gates
remain enforced; a residual failure stays recorded and blocks dependent primary
collection. Remaining FP32 reduction/activation differences must be investigated
at fixed inputs rather than hidden by a weaker tolerance.
Bind the new baseline to that arm's fresh `--calibration` file and numerical-mode
flags, retain its SHA256, and reject an ambiguous or missing profile.

Resource failures retry the whole rejected batch at most three times, preserving
all attempts. Before a retry/resume require 30 seconds with an idle GPU, native
host-memory/pressure headroom, and at most 1 MiB total system swap-in and swap-out
over that interval; wait at most 600 seconds. This is stricter retry admission and
does not change the 128 MiB inference swap gate, 4 GiB memory headroom, 1% pressure
gate or no-swap server scope. Do not terminate unrelated applications or change
global swap settings. Unrecognized/numerical failures remain rejected and recorded
for diagnosis. The monitor shows the actual resource/error cause and retains a
pending-error record rather than treating a stopped controller as running.

Use `m_reduction_recovery --resume` only with the same plan, calibration, numerical
runtime and capture/validation dependencies. It verifies completed batches before
reusing them, excludes a resource-rejected batch even when its individual requests
completed, and retains each controller attempt and source separately. A new runtime
or changed dataset/configuration starts a separate arm; do not resume it over old
evidence.

Qwen/Gemma need their own fixed-state numerical probes and format-specific
corrections if they exhibit the same symptom. MXFP4 alignment is rejected for
these NVFP4 models; it does not establish their numerical equivalence. In
particular, record CPU activation quantization/VNNI and GPU dequantization precision
when diagnosing NVFP4. Rebuild only a new runtime, retain the patch and source/binary
hashes, rerun affected qualification, and charge any correction's cost. Do not
substitute all-CPU or all-GPU execution for the declared hybrid comparison.

## 6. P2 — traces and faithful simulation

Existing [collectors](../experiments/pruning/README.md) provide token/layer access,
cache events and initial snapshots. Full scores were selected for repeat IDs rather
than every original primary request; expert-output vectors/norms and general
cache-aware routers are not provided as a complete P1–P9 runner.

Implement on an isolated new runtime, leaving sealed runtime sources untouched:

- Full native router logits for every prompt/decode token/layer on all 90 D0
  requests, native dtype/scaling metadata, logical IDs and effective weights.
- Complete chronological event IDs, generated/input token IDs, phase tags,
  cache snapshots before/after prefill and all admission/victim records.
- Expert pre-gate output L2 norms for calibration-selected experts; common-input
  expert output probes for P7 merging, never taken from an outer evaluation fold.
- Candidate teacher-forcing execution with sequential decode semantics. Evaluating
  a continuation through ordinary prefill is insufficient to exercise P6's cache bias.
- Full-vocabulary baseline/candidate next-token distributions for KL/agreement;
  router-score KL is not a substitute for language-model KL.

On the nine repeat IDs/model, compare live and replay **event by event**: m, q,
pre-admission slots, admitted IDs, victims, evicted IDs, usage/recency and final
state. Require zero mismatches. Native ties/dtypes must reproduce native decisions;
no arbitrary tie rule replaces the model's native zero-change control.

Controls: observation off/on; cold/native post-prefill; zero-bias P6; disabled
substitution; all100 mask; identity alias map; corrected integer rounding for all
`m=0..k`. These are future experiment qualifications, not a test suite executed by
this documentation task. Current replay also needs active-expert eviction
protection aligned with the native GPU kernel; do not assume the existing class
already covers all changed policies.

**G0 replay gate:** exact-policy streams match live and all zero-change numerical
controls pass. Missing full scores cannot be reconstructed from selected top-k
weights. Fixed-logit evidence stays unavailable for incomplete captures.

## 7. P3 — diagnose misses before choosing levers

Run D0 at all three capacities, native post-prefill and labelled cold/projected
states, with the corrected native q rule. P3 produces:

1. Operational demand under the actual partial-admission policy.
2. Fully admitting (`q=m`) LRU and Belady/MIN diagnostic streams on the same
   ordered accesses/initial state. Treat each token-layer top-k as one request
   group and protect all active experts during victim selection.
3. Per-layer fully admitting curves at every integer quota `k..E`, plus actual
   policy curves where supported. Use training requests only to derive P5 utilities.
4. Expert access/fill/eviction/refetch counts, inter-touch token gaps and distinct-
   expert reuse distance. Define a thrash refetch as an evicted expert returning
   after a distinct-expert reuse distance in `(C, ceil(1.10 x C)]` in a global stream;
   use layer quota instead of C for partitioned streams.
5. Router k/k+1 gaps, per-layer score ranges, and cached unselected candidates
   within the P6 alpha margins.

A three-way compulsory/capacity/policy partition is not exact for hybrid bypass
admission. Classify **actual** misses, in this order: first touch absent from the
initial state; otherwise admission/state-sensitive if fully admitting LRU hits;
otherwise capacity-like if both fully admitting LRU and MIN miss; otherwise
replacement-sensitive if LRU misses and MIN hits. Report these operational labels,
plus MIN-miss/LRU-hit disagreements. MIN's smaller total miss count does not mean
it hits on every LRU-hit event; do not use a pointwise subtraction as an oracle.
The LRU–MIN total gap is a diagnostic bound for the fully admitting problem, not
an achievable latency gain at fixed q.

**Deletion estimates:** per layer mark bottom-frequency 25% and largest-miss-count
25%, then report their share of baseline misses. These are upper estimates for
removing those original demands while holding state/other routes fixed. They are
**not upper bounds on native rerouting or merging**, which change future cache and
hidden states. At 75% retention remove 64/8/32 experts per layer for Qwen/GPT/Gemma.

**G1, exact decision rules (new specification):**

- Replacement-sensitive misses at least 15% of all baseline misses, with the top
  10% of experts by thrash count accounting for at least 50% of those misses:
  prioritize P9 for that model. Run its prescribed comparison on all three models
  and report whether each trigger passed. Diagnose admission-sensitive misses separately.
- Capacity-like misses at least 50% of baseline misses: prioritize P6; otherwise
  run P4/P5 first. This scheduling rule is not an exclusive mechanism claim.
- Largest-miss-count deletion share below 5%: lower the scheduling priority of P7
  miss-targeted exclusion for that model; still run both P7 arms and report any
  small/negative result. Do not call pruning generally ruled out.

## 8. P4/P5 — exact cache policies

### 8.1 P4: q and fill-selection policy

All main P4 arms use F's minimum-one floor, global native timestamp LRU, identical
initial state and no routing changes. The historical unfloored Reuse-Aware result
is a separate control. Calibrate bandwidth per format/model on the qualified host
and save standalone and concurrent CPU/PCIe values; do not reuse GPT's bandwidth
numbers for NVFP4 without qualification.

| Arm | Exact specification |
|---|---|
| `paper_floor1` | f = min(1, B_P_standalone/B_H_standalone), q=F(R(f,m),m) |
| `overlap_floor1` | f = B_P_overlap/(B_P_overlap+B_H_overlap), q=F(R(f,m),m); corrected main native baseline |
| `fixed25_floor1` | f=0.25, q=F(R(f,m),m) |
| `reuse_uniform_floor1` | Reuse-Aware below, p=k/E |
| `reuse_freq_floor1` | Reuse-Aware below, smoothed recent-frequency p |

For native/fixed q, fetch most recently routed missing experts, then lower expert
ID, using pre-action recency. For Reuse-Aware, fetch the specific missing experts
ranked by expected reuse, then recency and expert ID. Selecting q with frequency
but fetching unrelated IDs does not implement this arm. Record both q and fetch IDs.

Port Saketh's [ReuseAware](../experiments/runtime_q/instrumentation/runtime_q_hook.py)
with **kappa=1, t_free=32 layer-token visits, prior_n=8, history window=32 previous
layer-token visits, use_cache=true, rho_online=false**. Let
`rho=B_P_overlap/B_H_overlap` and
`delta(m,j)=max((j+1)/rho,m-j-1)-max(j/rho,m-j)`.
For each missing candidate, `R_e=p_e x T_res`, with T_res=32 while a free slot
remains, otherwise the legal LRU victim age converted to token visits (not raw
cross-layer timestamps). Frequency uses `(count_e + 8 x k/E)/(n+8)`; count/n
exclude the current event, n<=32, and count decode visits only. Rank candidates by p, apply marginal comparisons
`R_e >= delta(m,j)` from j=0 until the first failure, then apply the common floor.
Native uniform p is exactly 0.03125 / 0.125 / 0.0625 for Qwen/GPT/Gemma.

These are explicit new-runtime ports, not unchanged historical measurements.
Policy history resets per D0 request, carries per W2 session and uses only causally
observed routing. No learned ANN/bandit or kappa sweep is added to this stage.

Evaluate all five arms on D0 in faithful access replay at three capacities; live
confirm all five on the nine-request cohort with five paired repetitions. Select
a base q rule using inner-fold m, subject to the latency gate; break statistical
or numerical ties toward `overlap_floor1`, then lower fill traffic. Report the
frequency-vs-uniform comparison even when one fails to qualify.

### 8.2 P5: static per-layer allocation at fixed total bytes

Hold the P4-selected q/fetch rule fixed. Compare global LRU, uniform quotas and
utility quotas. Quotas are integers with `k <= c_l <= E` and `sum(c_l)=C`.
Uniform allocation at the primary capacity is exactly:

| Model | Uniform quota vector |
|---|---|
| Qwen | 8 layers x 52 slots + 32 layers x 51 |
| GPT-OSS | 10 layers x 7 slots + 14 layers x 6 |
| Gemma | 18 layers x 26 slots + 12 layers x 25 |

Assign remainder slots to lowest layer IDs. Derive utility curves on outer training
requests under the selected q policy, giving requests/domains equal weight. Use
non-increasing marginal-gain fits to suppress noisy quota reversals. Start each
layer at k; allocate the remaining slots one at a time to the greatest predicted
next-slot miss reduction; tie by lower layer ID. Save raw/fitted curves and quotas.
No held-out request determines a quota. Report fit error on held-out requests.

During prefill use the unchanged global staging mechanism; activate quotas at the
prefill→decode boundary. A native warm state may violate quotas. For the primary
controlled comparison construct a **common feasible seed**: per layer retain the
most recent initial residents up to `min(uniform_quota_l, utility_quota_l)`, with
native slot/ID tie rules; all three P5 arms begin from those exact retained
identities/recencies. Record dropped residents and the projection cost. Also show
global LRU from native state as a separate diagnostic, so the projection penalty
cannot disappear from the overall serving result. Quotas govern decode eviction;
free slots outside a layer's quota cannot be borrowed. Charge any reconfiguration
and restore global staging before the next prefill.

Replay mixed/domain-calibrated quota vectors in the 4 x 3 transfer cells. Live
confirm global/uniform/utility on the nine-request cohort with five paired
repetitions, including whole-request timing and prefill staging. Adopt quotas only
if their net serving result meets G2, not merely an offline quota curve.

## 9. P6/P7 — output-changing mechanisms

### 9.1 Native gating contract for both experiments

Selection may change; gate weights come from the **original unbiased logits** on
the selected set, using the checkpoint's native selected-weight normalization,
dtype, scaling and shared branch. In particular, preserve Gemma's per-expert
output scales; generic softmax alone is not its full effective-weight contract.
Use full softmax probabilities only as a nonnegative **comparison score** in P6
substitution and mass-retention statistics. Never multiply signed raw logits by
`1 +/- alpha` and call the result a probability threshold.

P6 and P7 miss-targeted exclusion apply at **decode only**, keeping prompt prefill
native. P7 merging changes weights and therefore affects both prefill and decode;
report that scope difference. All zero-change arms must reproduce native selection
and effective weights before nonzero candidates are evaluated.

### 9.2 P6: resident-biased routing

Shared parameters: native k unchanged; protected original top-J with J=2 for
Qwen/Gemma and J=1 for GPT-OSS; no shared-branch changes; cache membership sampled
before admission. Compute a per-layer original-logit range `d=max(z)-min(z)`.
Initialize its mean from calibration decode events; update causally after each
layer event using cumulative mean, with no held-out future data. Save sum/count.

| Candidate family | Sweep | Exact selection |
|---|---|---|
| A, Cache-Prior-inspired | lambda in {0.1, 0.2, 0.4} | Add `lambda x mean_range x 1[e resident OR e in original top-J]`; hard-include original top-J, choose remaining k-J from biased scores |
| B, low-score substitution | alpha in {0.1, 0.3} | Starting from native top-k, consider missing unprotected selections weakest first; swap each at most once with highest-scoring unused resident candidate that satisfies the thresholds below |
| C, A then B | All 3 x 2 pairs | Apply A, then B using original probabilities and the original top-J protection |
| Zero-change controls | A lambda=0; B disabled; C disabled | Exact native route/weight/cache identity |

For B define S_e=softmax(z)_e over all E and S_(k+1) from the original native
ordering. A missing selected expert is eligible if `S_e < (1+alpha) x S_(k+1)`.
A resident unselected replacement must satisfy `S_c >= (1-alpha) x S_(k+1)`.
Use descending S_c/native tie order; replacements are unique; stop when no
candidate qualifies. Alpha=0 can still swap score ties, so **disabled** is the
identity control. The hard top-J constraint is an explicit study choice; adding
a bonus alone does not mathematically guarantee protection.

Screen all 11 nonzero settings/model at primary capacity and both cache starts.
Tune within inner folds. Select exactly one A, one B and one C finalist/model,
ranked by mean-m reduction, then smaller perturbation and arm ID. A screen candidate
must retain at least 98% mean original selected probability mass and at least
95% at its fifth percentile on inner validation; these are new screening cutoffs,
not a quality qualification. If a candidate family has no eligible setting, record
its rejected screen for that model; do not omit the model's P6 experiment. An
unqualified finalist blocks dependent deployment/interaction cells.

For every finalist, run candidate teacher forcing on D0 outer-held-out histories,
then D1 quality, D2 and eligible D3 frozen histories/live tasks with parameters
locked. Live timing uses the nine-request cohort/five paired repetitions.
Report expert utilization and route concentration to detect collapse. A KL or
m benefit from fixed baseline logits cannot pass the numerical/task gate.

### 9.3 P7: two 75% physical/routing arms

Retain exactly 192/24/96 physical experts per layer for Qwen/GPT/Gemma: remove
64/8/32. Keep the original C and cache-byte budget. Run both arms on all three
models after their G0 qualification. Missing tooling/kernel support is a recorded
blocked arm to repair; a blocked model prevents complete P7 coverage.

**Arm 1, miss-targeted decode exclusion:** on calibration only, compute per-expert
mean `native effective gate weight x pre-gate expert output L2 norm` over routed
decode events. Normalize saliency within a layer to [0,1] by min/max; all equal
saliencies map to 0, unseen experts to 0. Rank removal score
`miss_count x (1-normalized_saliency)` descending; ties by larger miss_count then
lower expert ID. Remove the required count per layer. Decode masks logits to
minus infinity **before native top-k**, retaining k and native scaling; prefill
stays unmasked. Track new replacement hits/misses and changed downstream routes.
Run this arm on all three models after qualification; record the G1 deletion
estimate as a diagnostic and scheduling signal, including small/negative results.

**Arm 2, common-input clustering and router aliasing:**

1. Per layer, take 128 routed-input hidden vectors from outer training decode,
   allocated 43 math / 43 code / 42 incident for mixed calibration, or all 128
   from the specified domain pool. Select uniformly spaced chronological indices
   within each domain and save exact probe IDs; do not duplicate missing probes.
2. Evaluate every original expert on the same probes, before gating. Average its
   output vector, normalize to unit L2; zero-vector experts stay singleton unless
   merging zero vectors is necessary to reach the declared count.
3. Deterministic average-linkage cosine-distance clustering until exactly 75% of
   experts remain; tie by minimum original expert IDs. Preserve cluster assignments.
4. Merge corresponding expert matrices in BF16 with normalized calibration usage
   weights, equal weights if a whole cluster has zero usage. Requantize to the
   original expert format with a pinned/qualified packing implementation; keep
   layouts and routing/scaling metadata explicit. Averaging nonlinear expert
   parameters is an approximation, not an exact average of their functions.
5. Keep the original router IDs and native selected weights, map IDs to a physical
   cluster expert, sum the effective weights when aliases co-occur, and evaluate
   that physical expert once. Cache physical IDs; log both logical k and physical k.

Required controls: unmodified all100; identity alias map; dequantize/requantize-only
all100; calibration-frequency R75 reference; 200 random 75% masks/layer with seeds
`20261013 + mask_index` for offline sensitivity. Post-top-k deletion F75 is a
labelled dropped-demand diagnostic, never represented as native R75. Random/F75
controls are not automatically 200 live task runs.

Use native per-expert output scales with original logical gates before aggregation.
The same physical format/entry size keeps C fixed; clustering or projected warm
states must not silently remove baseline cache bytes. Record merge artefact hashes,
new weights and quantizer provenance separately from immutable original checkpoints.
If format packing or identity requantization cannot be qualified, merging is blocked.
No checkpoint/RAM saving is claimed without separate physical compaction evidence.

Run all applicable mixed/domain transfer cells, native/projected/cold-start
sensitivities, propagated teacher-forcing and the Section 10 quality gate. Report
incremental merge error against the requantization-only control as well as total
error against the original baseline.

**Verdict:** a tested arm works for m if it meets G2's >=10% reduction and all
quality/latency gates. Otherwise report failure or inconclusive qualification for
that arm/model. A skipped arm, small D2/D3 sample or low original deletion share
cannot support a universal negative pruning claim.

## 10. Quality, adoption and P8/P9

### 10.1 Quality gate, locked before new results

Every P6/P7 or composed output-changing candidate needs all of:

| Check | Decision contract |
|---|---|
| Numerical identities | Zero-change route IDs/effective weights and baseline generated tokens match exactly; otherwise stop qualification |
| Full-vocabulary divergence | Teacher-forced `KL(P_baseline || P_candidate)` mean upper paired 95% bound <=0.02 nats/token, separately per available workload; top-1 agreement lower bound >=98% |
| Synthetic task loss | Per-domain upper 95% paired bound on baseline-minus-candidate score <=0.02 absolute (two percentage points), using ten-family resampling |
| Synthetic likelihood | Preserve <=2% relative perplexity-increase upper bound for Qwen/Gemma; report GPT-OSS NLL/perplexity as diagnostic, not its gate |
| Baseline competence | Report success/score and correct-to-wrong and wrong-to-correct counts; if baseline score <20% in a domain, a non-loss result cannot qualify useful ability preservation there |
| Unsafe choices / failures | No new incident unsafe-choice outcome; crashes, invalid outputs and truncations retained and counted according to the frozen scorer |
| D2 / D3 confirmation | Show individual paired outcomes, score-loss intervals, truncations and eligible denominators; no new baseline-correct→candidate-wrong problem/issue for a provisional finalist |

The KL, agreement and competence cutoffs are **new specification**; the D1 task
and Qwen/Gemma likelihood margins preserve the previous regression contract.
KL is language-model output KL, measured through the candidate's own hidden states
on fixed text; D0 additionally uses its recorded baseline histories. Use workload-
matched teacher-forced task continuations in D1, not only its incident narratives.
No post-hoc switch between KL and JSD or between absolute and relative accuracy.

With 15 AIME problems and five SWE issues, even zero observed flips does not
establish a 1–2% population noninferiority margin. Report independent-confirmation
status as **directional/provisional** or **inconclusive** when intervals cannot
resolve the margin. Do not weaken a margin after observing uncertainty or call
D1's ten template families independent task confirmation. Broad quality adoption
requires the separate >=100 independent tasks/domain contract in Section 12.

### 10.2 G2: keep, reject or defer a component

- **Exact cache components:** at least 5% mean-m reduction with a positive lower
  95% bound, or at least 10% relative per-request P95 reduction with a positive
  lower bound; no changed task outputs. Timing ratio upper 95% bound <=1.02 is
  the new practical non-regression margin; disclose up to 2% tolerance rather
  than describing it as proof of zero regression.
- **Output-changing components:** at least 10% mean-m reduction with a positive
  lower bound, all applicable quality gates, and the same <=1.02 timing bound.
  If independent evidence is insufficient, retain only as a provisional research
  candidate; do not adopt a broad quality-preserving serving claim.
- Require positive point m reductions in at least four of five outer folds and
  no domain with a confidently negative mean-m reduction. Report unavailable
  domains/models. A candidate can be a scoped domain result without an all-domain claim.
- Parameter selection uses D0 inner folds. Secondary results confirm or reject
  the frozen candidate; they never select a replacement winner.

### 10.3 P8: interaction and complete serving result

Hold the P1 seed and P5 allocation fixed across the 2 x 2 cells. Define q0 as
`overlap_floor1`, q1 as the selected Reuse-Aware P4 variant, r0 as native routing,
r1 as the frozen qualified P6 finalist. If no Reuse-Aware/P6 finalist qualifies,
report the missing interaction instead of replacing it with a new policy.

| Arm | Routing | q policy |
|---|---|---|
| 00 | r0 | q0 |
| 01 | r0 | q1 |
| 10 | r1 | q0 |
| 11 | r1 | q1 |

On fixed trajectories compute miss interaction
`I_m = M_10 + M_01 - M_00 - M_11`. Positive I_m means the combined mean miss
reduction exceeds the sum of separate reductions; report a paired 95% interval.
For timing, compute `I_logT=log(T_10)+log(T_01)-log(T_00)-log(T_11)` on matched
requests/repetitions; positive values indicate improvement beyond multiplicative
independent speedups. Beating both individual arms alone does not prove compounding.

Run all four on D0 fold-specific artefacts and the nine-request timing cohort;
D2/D3 use frozen deployment artefacts and free-running evaluation, five paired
fresh-process repetitions for timing where feasible. Score each unique greedy
problem/issue once for task success; repeated identical generations are repeats,
not independent problems. Baseline tool trajectories remain a separate paired
m/KL analysis. Report per-turn prefill bytes/time and cache occupancy on D3.

Finally compare original unmodified native-floor0, corrected native-floor1, best
exact composition and best provisionally qualified output-changing composition.
Remove each included component one at a time, preserving all other settings.
Include P7/P9 only if qualified; never auto-stack every positive screen. Charge
seeding, feature calculation, reconfiguration and requantized-model effects.
Output a family table explaining the result with measured slots, expert bytes,
miss classes, k/E, margins, traffic, latency and quality scope.

### 10.4 P9: replacement protection and conditional adoption

Run the prescribed comparisons on all three models, recording whether each
P3 G1 trigger passed. The trigger determines priority and eligibility for adoption,
while the experiment's model coverage remains mandatory. Hold q, cache bytes,
dataset and calibration folds fixed.

| Arm | Parameters |
|---|---|
| LRU | Native timestamp LRU reference |
| Segmented LRU | Protected capacity fractions {0.25, 0.50}; new admissions enter probation; a second hit promotes; protected overflow demotes its LRU entry; evict probation LRU first |
| Thrash-targeted pinning | Pin fractions {0.10, 0.25}; IDs ranked by training thrash count, then training misses, then lower ID; fixed until session/request ends |

For partitioned caches, protected/pinned counts are `floor(fraction x c_l)`;
for a global cache use `floor(fraction x C)` with pin selections apportioned by the
P5 quota vector. Ensure total slots exactly C and leave at least k unpinned slots
per layer in a partitioned arm; record an infeasible choice as a blocked cell to
resolve without silently removing that model's comparison.
Decode pins do not reserve extra memory or obstruct the two full-layer prefill
buffers: release protection for prefill and charge any subsequent repopulation.
No adaptation to held-out future reuse. Select SLRU fraction/pin fraction in inner
folds, evaluate all 4 x 3 transfer cells, confirm live and include only G2 winners.

## 11. Implementation order, artefacts and reproducible commands

### 11.1 Stage gates and work to implement

The repository provides baseline/route/replay infrastructure; it does not yet
provide an end-to-end m-reduction runner for this specification. Keep study code
under `experiments/`, new declarative controls under `configs/`, runtime edits in
an isolated ignored `.runtime/` checkout. The implementation units below are **planned
interfaces**, not commands claimed to work today.

Stage 0 implementations are now in
[stage0.json](../configs/m_reduction/stage0.json),
[m_reduction_stage0.py](../experiments/pruning/m_reduction_stage0.py),
[the collector](../experiments/pruning/m_reduction_collect.py),
[admission policy](../experiments/pruning/m_reduction_policy.py),
[replay qualifier](../experiments/pruning/m_reduction_replay.py),
[checkpoint audit](../experiments/pruning/m_reduction_audit.py), and
[calibration wrapper](../experiments/pruning/m_reduction_calibrate.py).
The isolated runtime hooks are preserved as
[a separate patch](../runtime/patches/m-reduction-stage0.patch).
Implementation is not qualification: the controller stops on any failed child
command and retains its evidence. It advances through the 12 complete pilot-arm
batches and native-floor1 primary/repeat captures only after their own gates.
Sequential teacher forcing, output norms/probes and later policy stages are still
to implement; Stage 0 completion cannot be labelled complete P1–P9 execution.

| Stage | Implementation / experiment work | Dataset and exit artefact |
|---|---|---|
| 0 | P1 isolated floor port and resource/byte qualification; P2 all-score collector, event replay, teacher-forcing support | 15 pilots/model; 90 primary + 18 repeats/model; qualified source/config/trace manifest; G0 |
| 1 | P3 diagnostics and P4 five-arm admission evaluation | D0, three capacities; diagnosis tables, q/fetch traces; G1 |
| 2 | P5 quotas; P6 11-setting screen; both P7 arms; P9 comparisons on all three models | Nested D0 fits and held-out cells; D1 regression; locked finalist artefacts and quality reports |
| 3 | Acquire/pin D2/D3; qualify context/harness; record baseline trajectories; fixed-trajectory and live confirmation | D2 15 problems, D3 five issues or explicit blocked/excluded cells; no retuning |
| 4 | P8 factorial plus one-component-at-a-time ablation and serving closeout | Paired raw timings, interaction intervals, per-family verdict and reproduction record |

Execute GPU work serially. P1/P2 are logically related prerequisites; the old
instruction to run them or P3/P4 in parallel does not authorize extra agents or
concurrent GPU experiments. A failed gate blocks only dependent work; retain
completed independent offline analysis with its limitations.

Planned implementation units: m-reduction collector; faithful policy/cache replay;
quota and pin calibrator; native decode router hooks; merge/probe/packing worker;
sequential teacher-forcing quality worker; OpenCode trajectory adapter; paired
serving controller; analysis/closeout exporter. Each must expose a machine-readable
control file with model/runtime IDs, all parameters, dataset/split hashes, source
hashes, arm order and output root. Record missing features as work, not capabilities.

### 11.2 Declared execution counts

Counts below are per checkpoint and per declared budget/setting unless stated.
Additional calibration/inner-fold fits are analysis work, not independent requests.

| Unit | Declared count |
|---|---:|
| P1 pilots | 15 requests x 4 arms = 60 request executions, plus observation controls |
| P1/P2 primary baseline captures | 90 primary + 18 repeat captures = 108; full-score collection on every capture |
| P4 primary-capacity timing | 9 requests x 5 repetitions x 5 arms = 225 fresh-server arm executions |
| P5 timing | 9 x 5 x 3 arms = 135, plus the separately labelled native-state diagnostic |
| P6 screen | 3 A + 2 B + 6 C = 11 nonzero configurations; three finalist families maximum |
| P6 finalist timing | At most 9 x 5 x (baseline + 3 finalists) = 180 |
| D1 task quality | 300 generations/arm; each nonidentity deployed finalist and a fresh baseline |
| D1 fixed-input references | 100 likelihood evaluations/arm; baseline and numerical identity controls retained |
| D2 | 15 unique AIME tasks/arm; timing repeats do not increase task count |
| D3 | Five unique issues/arm, three user turns/issue, subject to declared eligibility/call limits |
| P8 primary timing factorial | 9 x 5 x 4 = 180 fresh-server arm executions |
| P9 candidates | Two SLRU fractions + two pin fractions + matched LRU, on all three models; report trigger status |

For study-wide totals account for all three required models in every experiment.
Finalists, P7/P9 arms, seeding controls and blocked secondary cells must have their
own explicit three-model run matrix after qualification; do not count a blocked
or absent cell as completed. A baseline can be reused only within a declared
matched block with identical source, settings, artefacts and resource acceptance;
do not subtract separately collected baseline executions from the paired contract.

### 11.3 Commands already available, for future instructed execution

Run Bash explicitly so `scripts/env.sh` resolves its Bash paths correctly. These
examples are not launches performed while editing the plan.

```bash
source scripts/env.sh
export MODEL=/path/to/checkpoints/gpt-oss-20b
bash scripts/doctor.sh --require-runtime --out results/machine_before_m_study.json
bash scripts/baseline.sh --profile gpt_oss_mxfp4 --model "$MODEL" --gpu 0 --repeats 3
bash scripts/run.sh --list
bash scripts/run.sh pruning collect_moe_routes --help
bash scripts/run.sh runtime_q driver --help
```

The existing baseline uses its own fixed local request, 128 generated tokens,
three repeats and warmup lifecycle. It is a machine-qualification prerequisite; it does **not** execute
D0–D3 or implement corrected-floor/cache-aware policies. Existing collectors and
runtime-q plans contain their historical reset/context conventions; review their
arguments and adapt new workers rather than invoking them on sealed studies.
Future m-study workers must launch through `scripts/run.sh STUDY MODULE ...` and
write fresh `results/runs/STUDY/UTC_UUID/` records. Do not invent a currently
available CLI for the planned interfaces above.

### 11.4 Required closeout evidence

Each accepted/rejected invocation retains `run.json`, `environment.json`, exact
command/control file, console/server logs, resource samples, source/kernel hashes,
checkpoint/quantizer IDs and status/return code. Per-arm artefacts include:

- Dataset/acquisition/rendering/exclusion manifests and chronological trajectories.
- Fold/inner-selection records, quotas, rankings, probes, merge maps/weights and pin IDs.
- Initial/final cache states; route/gate/cache events; copy-byte accounting.
- Request-level metrics, paired deltas/intervals, all transfer cells and unavailable cells.
- Quality outputs, reference/scorer identity, flips, failures, truncations and baseline ability.
- Timing repetitions, realized order, instrumentation overhead and rejected whole blocks.
- G0/G1/G2 decisions and scoped family verdicts; exact reproduction commands.

Maintain status/results only in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and the
applicable maintained report. Preserve source/control manifests before any run;
never overwrite a parent result or sealed source. Original offline manifest:
`967c23a59581d36eedc64cd2f1f7eb82f053e41190dd66d92ad640b82b6a1aba`;
actual75 manifest:
`7af5bd1fa883cede21cd4a41ec6998b0ff291335206e51dc48f2bd5e209df0b6`.
Read reproduction rules before opening either sealed study; new baselines use a
new qualified source identity and do not resume them.

### 11.5 Offline stage tooling built 14 October 2026

These offline units read only accepted `native_floor1` captures, meaning whole batches
that completed, passed resource gates and match the model's qualified pilot hashes.
Each writes a `control.json` with stage, models, study bindings, parameters,
dataset/split hashes, source hashes and gate status. Each refuses to run while the
next-stage gate is closed, unless `--pre-gate-diagnostic` labels every output as
diagnostic rather than stage evidence. They are built, not qualified evidence.

| Module | Implements | Inputs still missing |
|---|---|---|
| `m_reduction_traces` | Accepted-batch discovery, decode access streams, full router logits, routed weights, Gemma per-expert scales, bound calibration | None |
| `m_reduction_sim` | Global LRU / quota / SLRU / pin-protected cache with native, fixed, full and Reuse-Aware q rules and F floor; projected and cold starts; MIN; stack-distance curves; reuse distance | None |
| `m_reduction_stats` | Request metrics (section 4.1), equal-workload means, 10,000-resample fold x workload paired bootstrap, fold estimates, G2 exact/output-changing gates, nested selection | Live timing bound |
| `m_reduction_p3` | Miss classes, LRU/MIN, per-layer curves k..E, thrash refetches, router gaps and alpha margins, held-out deletion estimates, G1 | None |
| `m_reduction_p4` | Five floor-1 arms plus unfloored control, three capacities, native/projected/cold, frequency vs uniform, inner-fold selection | Live latency gate |
| `m_reduction_p5` | Per-layer utility curves under the selected q rule, non-increasing fits, greedy quotas, uniform vector, common feasible seed, 4 x 3 transfer cells | Live confirmation |
| `m_reduction_p6` | Families A/B/C and zero-change controls, causal range tracker, mass-retention screen, finalist selection | Teacher forcing, D1-D3, timing |
| `m_reduction_p7` | Decode masking, F75, R75, random masks, identity alias; arm 1 saliency and arm 2 fold-specific aliasing from P2 expert outputs | Qualified P2 expert outputs; merge/requantization |
| `m_reduction_p8` | 2 x 2 miss interaction with bootstrap; timing interaction from paired timing JSONL | Qualified P6 finalist; live timing |
| `m_reduction_p9` | SLRU 0.25/0.50, thrash-ranked pins 0.10/0.25 apportioned by quotas, transfer cells, inner selection, G1 trigger record | Live confirmation |

Development checks performed: the default simulator reproduced live m/q exactly for one
GPT-OSS and one Gemma capture; logit top-k and native weight reconstruction matched one
capture per model (GPT-OSS softmax over selected logits; Gemma also multiplies its
per-expert scales, maximum error 1.3e-7); and each runner completed on a 15-request
GPT-OSS subset in scratch space. The same simulator and routing checks passed on four Qwen
captures (maximum weight error 1.44e-7). These are not tests or results.

P2 support units built 14 October 2026 (qualification in progress, see the log):
`m_reduction_teacher` (baseline teacher forcing that must reproduce each accepted capture
event by event, and native zero-change KL control), `m_reduction_experts` (validated
reference experts; per-event pre-gate norms; 128-row fold/pool probes) and the
`m_reduction_p2` driver. P7 consumes their outputs through `--expert-outputs MODEL=DIR`.

Runtime-side units still to implement on an isolated runtime copy, each a new arm to
qualify: live decode router hooks for P6/P7, live quota and
SLRU/pin victim selection for P5/P9, P7 merge/requantize worker, paired fresh-server
timing controller, and D1 quality, D2 AIME and D3 OpenCode/SWE-bench workers. P2 support
is first because it gates every later stage.

## 12. Deferred work and revival conditions

| Work | Disposition / revival condition |
|---|---|
| Independent quality qualification | Still open: >=100 independent tasks/domain, references/licences/revisions and grouped source-family audit. Freeze design/calibration/confirmation before judging masks. Original task-loss/likelihood margins remain; D1 and small D2/D3 do not satisfy this. |
| CUDA-graph/device-side Reuse-Aware | After the eager exact policy qualifies; measure full overhead and establish a fresh matched graph baseline. |
| Second host/device and changing bandwidth | After same-host results; independent qualified baseline, causal adaptation episodes and per-host reporting. |
| Heterogeneous expert sizes / byte-aware policies | Requires explicit bank-size and allocation contract; slot-equivalent quotas are insufficient. |
| GPU/RAM/NVMe hierarchy and elastic RAM residency | Separate promotion/reuse/correctness/resource study; not implemented or measured here. |
| Full ARC/LFU/LRFU replacement survey | Revive if P9 yields a substantial qualified gain. |
| Cross-layer prefetch | Revive for high compulsory demand; charge copies/evictions and timing, since hiding latency differs from reducing m. |
| Larger pinning/refresh and pin x routing/q interactions | Only after P9 qualifies; per-turn refresh requires eligible multi-turn traces. |
| Margin substitution / top-k trimming | P6 B covers the immediate substitution question; trim only if m headroom is large and the present routers fail quality. |
| Low-precision resident copies | Only after format feasibility and separate numerical/resource qualification. |
| REAP-vs-frequency compact checkpoint comparison | Revive when physical checkpoint/RAM saving is an explicit goal. |
| Pruning cache-size sweep / pin-prune-LRU composition | After positive P7; pin/prune composition also needs P9. |
| Prefill/TTFT pruning, pinned skip, semantic anchors | Separate prefill study after the decode question; prompt streaming costs are retained here. |
| Learned family-adaptive selector | Three families do not qualify a learned general rule; requires a fourth runnable family and held-out family validation. |
| Dynamic partitioning/cache resizing | Static quota win and a separately defined variable-memory budget; outside equal-budget primary comparisons. |
| W3/W4, concurrent subagents, long-context editing | Larger memory device and new context/concurrency contract; user authorization required for delegation. |
| Joint expert/KV/scratch budget, CPU affinity/contention | Separate memory/scheduling arms after primary baseline; retain logical-byte versus hardware-counter distinctions. |
| Actual SM occupancy / DRAM hardware bandwidth | Counters remain unavailable in recorded evidence; profiling proxies cannot be relabelled measurements. |

The v3 input and previous canonical plan were backed up byte-for-byte before
consolidation in `archives/m_reduction_plan_before_detail_20261013T102535Z.zip`
(SHA256 `dad301f9eeafbc6760168393ce68a73e1600ad7dde76a4e4b131b956fe3da4fe`).
The old v2 reference is superseded by the explicit revival conditions here; do not
require a missing working v2 file to execute the priority plan.
