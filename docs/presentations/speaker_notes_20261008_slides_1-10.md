# Speaker notes: FreeToken weekly progress, 8 October 2026

**Deck:** [FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx](FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx)

These notes follow the updated deck's first ten slides. Slides 1-2 introduce the project and weekly objectives; slides 3-10 present the completed static expert-retention study. Slides 11-12 are reserved for the separate actual-inference study.

Suggested pace: approximately 10-12 minutes. Read the talk tracks naturally; use the figure cues and technical details when explaining a result or answering questions.

## Slide 1 - FreeToken reproduction and experimental context

**Talk track**

"We started by reading the FreeToken paper, reproducing the supported experiments, and studying how the system behaves on our hardware.

"The earlier work gave us three useful observations. First, overlapping prefill work improved time to first token as prompts became longer. Second, expert-access locality mattered: keeping the original routing order helped the LRU cache. Third, Qwen and GPT-OSS had different expert sizes and routing patterns, so expert count alone did not explain execution or transfer cost.

"We also explored static expert popularity. A ranking that looked useful on its calibration workload could lose access coverage on held-out workloads. That observation motivated a more structured retention study alongside the work on CPU/GPU split selection."

**Emphasise**

- Define TTFT as time to first token and LRU as least recently used if the audience needs them.
- The earlier retention observation concerns routed accesses. It does not establish model quality after pruning.

**Transition:** "These observations led to two objectives for this week."

## Slide 2 - Two objectives for this week

**Talk track**

"The first objective is runtime-aware q selection. Here, m is the number of selected experts missing from the GPU cache at a routing event. We choose q of those experts to fetch and execute on the GPU, leaving m minus q for CPU execution.

"The second objective asks whether expert retention can reduce the missing work itself. We study whether a smaller eligible expert set preserves useful routing across tasks and model families, and whether it improves cache demand.

"These objectives connect: retention can change which experts are selected, while the execution policy decides how to handle the resulting misses. In this presentation, we cover the static-retention evidence first, then return to the runtime-aware policy experiments."

**Emphasise**

- q controls the CPU/GPU split for the current missing set.
- A 75% eligible set means retaining 75% of the expert IDs in each layer. Native top-k can remain unchanged when the router selects replacement experts from that set.
- A smaller eligible set does not automatically imply fewer cache misses.

**Transition:** "We began by asking what the recorded routing could tell us about a fixed expert set."

## Slide 3 - Can a fixed expert set reduce missing work?

**Talk track**

"Our static experiment retains the most frequently accessed 75% of routed experts in each layer. We fit separate rankings for math, code, incident tasks, and a mixed pool.

"For each checkpoint, we collected 90 distinct requests: 30 in each workload. We used five grouped folds so held-out requests were excluded from the data used to fit their rankings. We also collected 18 additional captures of nine preselected requests to examine repeat sensitivity.

"We examined access coverage, retained routing weight, transfer between workloads and phases, and ranking stability. We then replayed cache behaviour to ask whether the retained sets could reduce missing work.

"At this stage, the checkpoints remained complete. The evidence came from recorded native routing and conditional replay. Task quality after actual masking and serving latency required a separate inference experiment."

**Technical detail if asked**

- Access coverage is the fraction of recorded expert selections whose IDs are included in the retained set.
- Retained routing weight measures how much observed effective expert weight remains inside the set. Gemma includes learned expert scaling, so its effective weights should not be described as probabilities.
- Shared expert branches remain active and are excluded from the popularity ranking.
- The additional captures assess repeat sensitivity; they do not add independent tasks.

**Transition:** "We ran this study with three checkpoint configurations that were qualified on our device."

## Slide 4 - Three supported MoE configurations on one device

**Talk track**

"All three configurations ran through FreeToken on the same RTX 4070 with 12 GB of VRAM and a Ryzen 5 8500G.

"Qwen3.6 uses NVFP4, with 40 MoE layers and 256 routed experts per layer. We retain 192 and preserve a native top-k of eight. GPT-OSS uses MXFP4, with 24 layers and 32 experts per layer; we retain 24, with top-k four. Gemma 4 uses NVFP4, with 30 layers and 128 experts per layer; we retain 96, with top-k eight.

"Their GPU expert caches also differ: 2,048 slots for Qwen, 154 for GPT-OSS, and 768 for Gemma. We froze the checkpoint revisions and configurations, used hybrid execution with one request at a time, and disabled CUDA graphs. Gemma needed its separately qualified text-only configuration and a lower memory ratio."

**Table reference**

| Checkpoint | Layers | Routed experts per layer | Retained per layer | Native top-k | GPU cache slots |
|---|---:|---:|---:|---:|---:|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 40 | 256 | 192 | 8 | 2,048 |
| `openai/gpt-oss-20b` | 24 | 32 | 24 | 4 | 154 |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 30 | 128 | 96 | 8 | 768 |

**Emphasise**

- Retained counts are per layer. Cache slots describe the model's GPU expert cache, not a per-layer retained count.
- Slot counts cannot be compared directly as memory budgets because expert sizes differ.
- Gemma used memory ratio 0.8, a 1,280-token sequence limit, and a 1,024-token prefill limit. Kimi was excluded.
- Differences between these checkpoints include precision, geometry, tokenization, and generated routes. The comparison does not isolate architecture as a causal factor.

**Transition:** "First, we looked at coverage when calibration and evaluation used the same workload."

## Slide 5 - Task-specific masks preserve most decode accesses

**Talk track**

"Each bar shows held-out decode access coverage for a mask fitted on the same workload. The three colours represent math, code, and incident tasks. We average coverage across 30 held-out requests in each workload, and the error bars show 95% request-bootstrap intervals.

"Gemma retains about 99.32% to 99.57% of the recorded accesses. Qwen ranges from about 96.41% on math to 98.91% on incident. GPT-OSS ranges from about 96.15% to 97.74%.

"This tells us that expert access is concentrated enough for popularity-based retention to preserve many selections within a workload. However, the bars measure access coverage. An omitted expert could still matter to the output, and a task-specific ranking may behave differently on another workload."

**Figure cues**

- Point to the model groups, then identify the workload colours and intervals.
- The dashed line is the 98% access screen. The activation decision uses the interval's lower bound and additional checks, not just the bar height.
- The vertical axis starts at 90%, which makes differences easier to see.

**Values to keep handy: mean coverage (%)**

| Model | Math | Code | Incident |
|---|---:|---:|---:|
| Qwen3.6 | 96.41 | 98.47 | 98.91 |
| GPT-OSS | 96.52 | 96.15 | 97.74 |
| Gemma 4 | 99.32 | 99.57 | 99.53 |

**Technical detail:** The intervals describe request variation conditional on the fitted rankings. They do not treat the overlapping training folds as independent datasets.

**Transition:** "The next question was how well those rankings transferred to other tasks."

## Slide 6 - A task-specific mask can lose coverage on another task

**Talk track**

"In each heatmap, the rows identify the workload used to calibrate the mask. The columns identify the held-out workload used to evaluate it. The numbers are percentages of recorded decode accesses covered by that mask.

"The diagonal is usually strong, but some off-diagonal cells fall substantially. For example, the GPT-OSS incident mask covers about 97.7% of incident accesses but only 63.2% of math accesses. Qwen's math mask covers about 96.4% on math but about 75.1% on incident.

"The mixed row shows the compromise when one ranking must serve several tasks. Mixing calibration data improves some transfers, while giving up some of the coverage available from a task-specific ranking. The trade-off differs across the three checkpoints."

**Figure cues**

- Read one row horizontally to demonstrate transfer from a fixed mask to different evaluation workloads.
- Compare that row with the mixed row.
- All panels use the same 60-100% colour scale. The printed values are the measured means; the scale's lower endpoint is not an activation threshold.

**Emphasise:** These cells evaluate membership in a retained set on original recorded routes. They do not measure answer quality under autoregressive masked inference.

**Transition:** "Even a mixed ranking must also work across different phases of inference."

## Slide 7 - Decode rankings transfer poorly to prefill

**Talk track**

"For this comparison, every mask is fitted from mixed-workload decode routing. We then evaluate that same mask during decode and during prefill. Blue shows decode coverage and orange shows prefill coverage.

"GPT-OSS shows the largest phase difference: its decode-derived mixed mask covers roughly 93% to 95% of decode accesses, but only 74% to 76% of prefill accesses. Gemma also drops, from roughly 99% in decode to about 94% in prefill. Qwen's change is smaller, although its mixed-mask coverage still misses several screening requirements.

"The implication is that decode popularity alone does not reliably describe prefill routing. A static mask intended for the full inference pipeline has to be evaluated in both phases."

**Figure cues**

- Compare the blue and orange bars within one workload and model.
- The error bars are 95% request-bootstrap intervals; the dashed line is the 98% access screen.
- The vertical axis starts at 70%.

**If asked about the phases:** Prefill processes the input prompt; decode generates subsequent tokens. These measurements show a routing difference between the phases but do not establish its causal mechanism.

**Transition:** "We also checked whether the ranking itself changed substantially with calibration data."

## Slide 8 - Stable rankings can still omit important routes

**Talk track**

"We split calibration data into disjoint halves, fitted retained expert sets separately, and compared them using Jaccard overlap: the intersection of the sets divided by their union.

"The observed mean overlap is about 0.891 for Qwen, 0.943 for GPT-OSS, and 0.937 for Gemma. Because each set already includes 75% of the experts, random sets also overlap substantially. Their empirical reference overlap is about 0.60.

"Subtracting that reference gives excess overlap of about 0.291, 0.342, and 0.336. All three clear the 0.10 stability screen. This suggests that the calibration data identifies repeatable popularity patterns, even though those patterns can still omit routes needed by another task or inference phase."

**Table cues**

- Explain the observed Jaccard column before the random reference and excess-overlap columns.
- Jaccard 0.943 does not mean 94.3% access coverage. It measures similarity between two retained expert-ID sets.
- The layer and subset comparisons are dependent; they are not thousands of independent task samples.
- The 18 additional captures per checkpoint address repeat sensitivity separately.

**Transition:** "Stable, high-coverage sets are useful candidates, but we still need to ask whether they reduce cache misses."

## Slide 9 - Rerouting gives little benefit at the fixed cache budget

**Talk track**

"This table reports native masked-router replay. We recompute top-k selection from the retained eligible experts at the original recorded hidden states, then replay the resulting cache accesses. Native top-k remains eight for Qwen and Gemma and four for GPT-OSS.

"At the original fixed cache budget, Qwen reduces missing experts per routing event by about 2.24% on math, 0.96% on code, and 0.87% on incident. GPT-OSS shows no reduction under the calibrated hybrid policy. Gemma's repaired rerouting capture shows a 0.46% reduction averaged over nine requests.

"The next column reduces the cache to roughly 75% of its original slots, while keeping the original fixed-cache baseline for comparison. Missing demand increases for Qwen and Gemma: the negative reductions mean more misses. Gemma's increase is about 9.51%.

"These observations did not meet the 5% fixed-cache benefit screen. They are conditional demand estimates: changed routes are replayed at original hidden states, so actual latency and changes in subsequent autoregressive hidden states still require direct inference."

**Table cues**

- Positive means fewer missing experts per event; negative means more.
- Qwen and GPT-OSS rows use three preselected requests per workload. Gemma's row aggregates nine requests across workloads, so it is not a per-workload estimate.
- The reduced caches use 1,536 Qwen slots, 116 GPT-OSS slots, and 576 Gemma slots, compared with original budgets of 2,048, 154, and 768.

**Technical detail if asked**

- **U replay:** preserve original expert accesses.
- **F replay:** delete accesses to excluded experts; active top-k can decrease.
- **R replay:** rerun native masked top-k selection at original recorded hidden states, preserving native top-k. This table uses R.
- GPT-OSS's calibrated hybrid replay stays at m = 4 with q = 0. Unconditional-fill diagnostics are a different policy and should not be substituted for these results.
- Gemma's repaired capture restored native router projection, learned expert scales, and original dtypes. Full eligibility reproduced recorded effective weights exactly, and all 12 native cold-cache controls matched.
- Gemma's hybrid split fractions reuse the frozen NVFP4 calibration measured on Qwen geometry. This does not establish a Gemma-specific optimal split.

**Transition:** "Combining the coverage, stability, and cache results gives the static-study disposition."

## Slide 10 - No checkpoint passed every static activation check

**Talk track**

"We defined several requirements before activating a static mask. Access coverage needed a 95% confidence interval whose lower bound was at least 98%. Retained effective routing weight needed a lower bound of at least 99%. We also checked weak requests, layers, rolling windows, ranking stability, and systems benefit.

"All three checkpoints passed the ranking-stability screen, but none passed every activation requirement. Coverage transfer remained a problem, especially between decode and prefill. Native rerouting also failed to establish at least 5% fixed-cache miss reduction in every workload. Gemma's repaired aggregate result was only 0.46%.

"The static study therefore did not qualify these masks for deployment. We proceeded with a separately authorised exploratory study that runs actual autoregressive inference with 25% of routed experts excluded. That study evaluates task quality and matched serving performance directly. Its results belong in the two reserved slides after closeout. The 60% and 50% retention ablations follow completion of the 75% study."

**Additional thresholds if asked**

- Request-level fifth-percentile coverage at least 95%.
- Every workload/layer mean coverage at least 95%.
- Every complete 64-token rolling-window coverage at least 90%.
- Mean disjoint-half excess Jaccard at least 0.10.
- Native R fixed-cache hybrid miss reduction at least 5% in each workload, with the paired delta's 95% interval entirely below zero.

**Emphasise**

- Failure of these prospective screens does not prove that every task-specific mask fails every task. It means the evidence did not satisfy the complete static activation criteria.
- The original static closeout recorded Gemma native R as unavailable. A later supplement repaired that gap; the historical closeout remains preserved.
- The separate inference study excludes expert routes while keeping the complete weight allocation. Physical checkpoint compaction and measured memory savings are not established by that intervention.

**Transition:** "Once the actual-inference report is final, slides 11 and 12 will present those results. We then return to runtime-aware q selection."

## Source reference

The paths below are relative to `results/moe_pruning/study_20261006_v1/` unless stated otherwise.

| Slides | Evidence |
|---|---|
| 1 | Previous progress deck: `docs/presentations/FREE_TOKEN_PROGRESS_1-10-2026.pptx`; `docs/experiments/bandwidth/report.md` |
| 2 | Weekly objectives in the supplied 8 October deck |
| 3 | `closeout75/report.md`; `protocol/protocol.json`; grouped-fold analysis and repeatability records |
| 4 | `closeout75/checkpoint_lock.csv`; `closeout75/qualified_resources.csv` |
| 5 | `closeout75/primary_summary.csv` |
| 6-7 | `retention75/summary.csv`; `additional/gemma_v1/retention75/summary.csv` |
| 8 | `retention75/stability.csv`; `additional/gemma_v1/retention75/stability.csv` |
| 9 | `replay75/summary.csv`; `additional/gemma_rerouting_v2/native_controls75/R75_summary.csv`; `additional/gemma_rerouting_v2/report.md` |
| 10 | `closeout75/masking_decisions.json`; `closeout75/report.md`; `docs/experiments/pruning/protocol.md` |

Exact plotted rows are also saved in `docs/presentations/pruning_20261008_assets/chart_source_rows.json`. These notes cover the static study and its repaired rerouting supplement. Actual-inference performance results are in the completed pruning report and are outside these ten slides.
