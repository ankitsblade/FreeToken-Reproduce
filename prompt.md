# FreeToken Research: Paper-Faithful q* vs. Reuse-Aware vs. Runtime-Variable-Aware Policies

We are starting a fresh research-planning and implementation conversation because the previous chat drifted away from the actual research objective.

Act as a careful research engineer. Prioritize experimental correctness, reproducibility, source-code verification, and fidelity to the original FreeToken paper. Do not invent configurations, assume experimental settings, fabricate results, or make unrelated modifications.

# 1. Research Objective

I want to investigate alternative expert-execution decision policies in FreeToken.

FreeToken performs MoE inference by deciding whether missing experts should be fetched from host memory to GPU memory or executed on the CPU. My research focuses exclusively on improving this decision.

Compare these three policy families:

1. **Original paper q\* policy:** Reproduce the policy defined in the original FreeToken paper and official implementation.
2. **Original Reuse-Aware policy:** Preserve the existing policy using the decision rule \(R_i > \Delta(m,j)\).
3. **Runtime-Variable-Aware Reuse policy:** Extend the original Reuse-Aware policy using runtime-measured execution costs, calibrated reuse predictions, and eviction-aware reuse valuation.

The primary research question is:

**Can runtime information improve the original Reuse-Aware expert-execution policy while preserving FreeToken's original experimental setup and improving end-to-end serving performance?**

The only intended research variable is the expert-execution decision policy.

Do not turn this into a study of elastic memory management, KV-cache resizing, alternative workloads, model selection, scheduler redesign, or general runtime optimization.

# 2. Source of Truth: Original Paper and Official Repository

The original paper PDF is available at:

`@/docs/references/2608.16157v1.pdf`

**Read and refer to this PDF throughout the audit, implementation, and analysis.** Do not rely on memory, previous chat summaries, or unverified interpretations of the equations.

Also inspect the official FreeToken repository and the exact implementation corresponding to our current working environment.

Before implementing any policy:

- Read the relevant paper sections and equations.
- Inspect the original q\* implementation.
- Locate the existing Reuse-Aware implementation and its \(R_i > \Delta(m,j)\) decision rule.
- Trace how the engine selects routed experts, determines cache misses, executes CPU experts, fetches GPU experts, and evicts cache entries.
- Verify the definitions, units, rounding rules, boundary conditions, and minimum-fill behavior used by the paper and the implementation.
- Distinguish what the paper specifies from what the repository implements and what our research proposes.

When citing the paper internally in your report, provide the exact section, equation, figure, or page where possible.

Do not silently change the meaning of the paper's q\* equation or substitute a convenient approximation without documenting and justifying it.

# 3. STRICTLY Preserve the Original Experimental Setup

**The original FreeToken paper and official repository are the experimental source of truth.**

Do not invent, arbitrarily choose, or silently modify:

- Cache rates or GPU expert-cache capacities.
- Models or model variants.
- Workloads, prompts, input lengths, or generation lengths.
- Batch sizes or decoding settings.
- CPU/GPU allocation or hardware settings.
- Memory budgets or KV-cache configurations.
- Warm-up procedures.
- Number of repetitions.
- Timing boundaries or performance metrics.
- Evaluation methodology or baseline commands.

Derive these settings from the paper, official repository, and existing established baseline experiments.

For every setting, establish its provenance. Record the exact source, configuration file, command, or baseline result supporting it.

If the paper does not specify a setting, explicitly say so. Then inspect the official repository and our established experiments to determine the supported choice. Do not claim that a repository default is a paper-mandated setting unless the evidence supports that conclusion.

If an existing experiment uses a different configuration from the paper, document the deviation rather than silently treating it as paper-faithful.

## Special requirement: cache rate 0.10

The previous conversation reported that cache rate `0.10` was being used.

Do not assume this is correct.

Determine:

1. Where `0.10` originated.
2. Whether it is specified by the paper, the official repository, an established baseline command, or a later experimental choice.
3. Which cache rates and cache capacities the paper actually evaluates.
4. Which cache rates were used in our existing baseline experiments.
5. Whether the current results at `0.10` are directly comparable to the original paper's results.

Cite the exact source or configuration for every conclusion. If provenance cannot be established, mark it as unresolved.

Do not automatically rerun experiments using a different cache rate. First finish the read-only audit and explain the implications.

# 4. Mandatory Initial Step: Read-Only Audit

Before changing anything, conduct a comprehensive read-only audit.

**During this initial audit, do not edit files, launch experiments, terminate jobs, restart servers, modify configurations, or overwrite results.**

Inspect the following.

## A. Paper and repository

- Read `@/docs/references/2608.16157v1.pdf`.
- Identify the paper's exact q\* derivation and its assumptions.
- Identify the original Reuse-Aware decision rule and its implementation.
- Inspect the official source and current runtime source.
- Establish the repository commit and any local patches.
- Identify all differences between the upstream implementation and the current working tree.

## B. Experimental configurations

Recover the exact:

- Model names and model revisions.
- Workloads and prompts.
- Batch sizes and decoding settings.
- Cache capacities and cache rates.
- CPU/GPU execution configuration.
- Warm-up and repetition procedures.
- Baseline commands.
- Timing boundaries and reported metrics.

For every configuration, state whether it comes from the paper, official repository, or established experiments.

## C. Working tree and existing experiments

Inspect:

- Git status and uncommitted changes.
- Existing experiment scripts and configuration files.
- Existing result files and analysis scripts.
- Existing experiment plans and logs.
- Output directories and timestamps.
- Current running processes and GPU utilization, using read-only inspection.
- The exact state of the current policy-study run.

The previous conversation reported the following state. Treat this only as a hypothesis to verify:

- Stage 1 for `gpt-oss-20b` completed.
- Stage 1 for Gemma was running.
- Stage 1 for Qwen and Stage 2 were queued.
- Phase A runtime cache-resize validation was completed separately.

Verify whether these statements are still true. Do not assume queued jobs have not started or that running jobs are still alive.

## D. Report the audit before proceeding

Produce a concise, evidence-based audit report with:

1. What the paper actually specifies.
2. What the official implementation actually does.
3. What the current experimental implementation changes.
4. The provenance of every cache rate and workload currently in use.
5. The current status of every experiment and running job.
6. Which existing results are valid, incomplete, or potentially confounded.
7. The smallest defensible plan for comparing the three policies.

Do not automatically continue queued experiments. Complete the audit first.

# 5. Implementation Location and Isolation

All new research code, configurations, logs, analysis scripts, and results must be organized in a separate directory:

`q_expswithreuse_aware/`

Use this directory for the new policy-comparison experiment and its artifacts.

Requirements:

- Do not scatter new research files throughout the original repository.
- Do not overwrite existing experiments or results.
- Do not modify the official q\* or Reuse-Aware implementation in place.
- Prefer isolated experiment hooks, wrappers, or configuration-driven policy selection.
- If a runtime modification is unavoidable, use the smallest possible patch and document it separately.
- Preserve a clean way to select the original q\* policy, original Reuse-Aware policy, and proposed runtime-aware policy.
- Keep a record of the base commit, patch files, configuration hashes, commands, and output locations.

The directory must clearly distinguish:

- Experiment configuration.
- Policy implementation or hooks.
- Launch scripts.
- Raw results.
- Process/runtime logs.
- Analysis scripts.
- Aggregated results and plots.
- Documentation and reproducibility notes.

Do not start implementation until the read-only audit has identified the correct baseline and resolved the experimental configuration as far as the available evidence permits.

# 6. Policy A: Original Paper q\*

Implement or reuse the original paper's q\* policy exactly as defined in the PDF and official implementation.

Do not assume that a simplified equation, a minimum-one-fetch rule, or a no-minimum-fill variant is equivalent to the original.

Specifically verify:

- The exact q\* equation.
- The definitions and units of every variable.
- The bandwidth assumptions.
- How q\* is converted to an integer number of experts.
- The behavior when q\* rounds to zero.
- Whether the policy imposes a minimum fill.
- The behavior at zero misses, full cache, and other boundary conditions.
- Whether the paper and the current repository implementation differ.

If the paper and repository disagree, document the disagreement and reproduce the appropriate original reference faithfully. Do not silently select whichever variant produces better results.

Any additional q\* variant should be included only if justified by the original paper or necessary to diagnose an implementation difference. Label it clearly as a diagnostic rather than replacing the main baseline.

# 7. Policy B: Original Reuse-Aware

Use the existing original Reuse-Aware policy, preserving its actual decision rule:

`R_i > Δ(m,j)`

Read the code to establish the exact definitions of `R_i`, `Δ(m,j)`, `m`, `j`, and any other variables.

Do not replace the original policy with a reconstructed version from memory.

Verify that the policy behaves identically when runtime-aware features are disabled. Use deterministic or randomized unit tests as appropriate, but do not use synthetic tests as a substitute for real inference benchmarks.

The original Reuse-Aware policy is a required baseline, not merely an ablation of the new method.

# 8. Policy C: Runtime-Variable-Aware Reuse

Develop Policy C as an extension of Policy B. Preserve the original Reuse-Aware decision as the reference behavior, and introduce each additional signal independently so its effect can be measured.

Policy C will initially investigate three components: V1, V2, and V3.

Do not implement them until the audit has verified the actual runtime behavior and the meanings of the original variables.

## V1 — Measured Execution Costs

### Research question

Are the paper's fixed bandwidth-derived cost units an adequate model of actual execution cost on this machine?

The proposed cost model is:

`C(m, j) = max(g0 + j * t_p, [m > j] * (c0 + (m - j) * t_c))`

The proposed fetch rule is:

`fetch while p_j * T_j * v_hit >= C(m, q+1) - C(m, q)`

All quantities used in the comparison must have consistent units. Verify the exact meaning of every variable and the inequality before implementing it.

### Proposed measurements

- `g0` and `t_p`: estimate the GPU branch time as a function of the number of experts fetched, using GPU timing for the relevant transfer and computation.
- `c0` and `t_c`: estimate CPU execution time as a function of the number of CPU-executed experts.
- Fit the cost models using least squares over appropriate samples from the current episode/layer, if that is consistent with the actual runtime timing boundaries.
- Use a fallback only when the fit is insufficiently supported. The proposed fallback is zero intercept and bandwidth-derived slopes based on `S/B_P` and `S/B_H`.
- `v_hit`: estimate the value of one future cache hit using the observed distribution of miss counts and the measured cost model.

### Mandatory validation

- Confirm whether CUDA event timings and CPU task timings measure comparable portions of execution.
- Account for asynchronous work, synchronization, and CPU/GPU overlap.
- Verify the dimensions of the decision rule.
- Define the sample sufficiency condition explicitly; the proposal is at least eight samples at two or more distinct x-values with positive fitted slope.
- Track fit failures, fallback frequency, prediction error, and policy overhead.
- Never use information unavailable at decision time.

Treat the equation above as a research proposal to validate, not as a proven or paper-defined formula.

## V2 — Calibrated Reuse

### Research question

Does the original reuse prediction systematically under- or overestimate realized reuse?

The proposed calibration is:

`R_j = α * p_j * T_j`

`α = (realised hits + 8) / (predicted hits + 8)`

At fetch time, record the predicted reuse quantity `p * T`. While the expert remains resident, count its realized hits. When it is evicted, aggregate the corresponding predicted and realized values to update calibration.

### Mandatory validation

- Verify how `p_j` and `T_j` are defined in the original implementation.
- Confirm that the calibration is causally valid and does not leak future information into online decisions.
- Verify that realized hits and predicted hits are counted over compatible horizons and populations.
- Determine whether calibration is global, per-layer, or per-expert based on the actual design; do not silently change the scope.
- Evaluate whether the `+8` smoothing constant has a documented rationale or is a new research hyperparameter.
- Record calibration sample counts and the frequency of insufficient evidence.
- Keep the original Reuse-Aware behavior intact when calibration is disabled.

## V3 — Eviction-Aware Reuse

### Research question

Should the policy account for the future reuse opportunities lost when a fetch evicts an expert?

The proposed adjusted value is:

`R_j = (p_j - p_victim(j)) * T_j`

The predicted victim should match the actual cache engine's eviction behavior, including:

- Free-slot handling.
- LRU usage stamps.
- Exclusion of experts required by the current layer, if applicable.
- Tie-breaking rules.
- Layer-local reuse probabilities and prediction horizons.

### Mandatory validation

- Verify that the victim predicted by the policy matches the actual victim selected by the runtime.
- Count evictions and subsequent hits lost by the evicted expert where observable.
- Verify that the two reuse probabilities and the horizon in the formula are comparable.
- Establish whether subtracting the victim's expected reuse is dimensionally and conceptually correct for the decision being made.
- If the formula is inconsistent with the actual runtime semantics, report the problem and propose a justified correction rather than silently changing the equation.

Treat the formula as a hypothesis to validate, not as an established result.

# 9. Required Ablation Study

Ablation is a required part of this research. Do not base the final claim only on the full-feature policy.

After the initial audit and validation, evaluate the following policy configurations under the same original FreeToken experimental conditions:

1. Original paper q\*.
2. Original Reuse-Aware B.
3. B + V1.
4. B + V2.
5. B + V3.
6. B + V1 + V2.
7. B + V1 + V3.
8. B + V2 + V3.
9. B + V1 + V2 + V3.

Keep the feature definitions fixed across runs and document every difference.

If a component cannot be implemented faithfully, mark it as blocked and explain why. Do not substitute a different feature without documenting and separately evaluating that change.

# 10. Fair Evaluation and Instrumentation

All policy comparisons must use the same:

- Model and model revision.
- Workload and prompts.
- Batch size and decoding parameters.
- Cache capacity and cache rate.
- Hardware allocation and runtime configuration.
- Warm-up and repetition procedure.
- Timing boundaries.
- Baseline serving implementation.

The only intended change between policy variants is the policy logic and its unavoidable, explicitly measured overhead.

Use the original paper's metrics and procedures wherever specified. If additional diagnostics are necessary, report them separately.

At minimum, record the metrics supported by the original setup and relevant to the policy decision, including:

- End-to-end latency.
- Time to first token (TTFT), if included in the established evaluation.
- Inter-token latency and throughput, where applicable.
- GPU expert-cache hit/miss rate.
- Number of experts fetched and CPU-executed.
- CPU execution time.
- GPU execution time.
- Transfer time.
- CPU/GPU overlap and branch completion time, if measurable.
- Policy decision overhead.
- Runtime-aware prediction error.
- Eviction behavior and observed reuse.
- Variability across repeated trials and confidence intervals where appropriate.

Do not invent a new metric definition halfway through the experiment.

Instrumentation must be consistent across policies. Measure its overhead, and do not compare heavily instrumented results against an uninstrumented baseline without accounting for the difference. Preserve raw results and run metadata.

Use paired, repeated trials when feasible within the original evaluation methodology. Report effect sizes and uncertainty; do not select only favorable runs.

# 11. Existing Results and Running Jobs

The prior conversation reported:

- Stage 1 for `gpt-oss-20b` had completed.
- Stage 1 for Gemma was running.
- Stage 1 for Qwen and Stage 2 were queued.
- Phase A cache-resize validation had been completed separately.

Verify this against actual processes, logs, and result files.

Preserve all existing code, data, and logs. Do not terminate jobs or restart servers during the initial audit. Do not overwrite or delete existing results.

After the audit, identify which results remain scientifically usable, which are preliminary, and which may be confounded by configuration differences or instrumentation overhead.

Do not automatically continue queued jobs until the audit has established whether the configurations are valid and comparable.

# 12. Explicit Scope Exclusions

The primary study is about the expert-execution decision policy.

Do not introduce any of the following into the primary policy comparison:

- Elastic expert-memory management.
- KV-cache resizing or KV-pressure workloads.
- New models or workloads not justified by the paper or established baseline.
- Arbitrary cache rates.
- Unrelated scheduling or memory-management optimizations.
- New learned policies, neural networks, or contextual bandits.
- Future-information leakage into online decisions.

The previously completed Phase A cache-resize validation is separate supporting work and must remain separate.

# 13. Required Deliverables

Store new artifacts under `q_expswithreuse_aware/`.

After completing the read-only audit, provide:

1. **Paper and repository audit:** Exact source references and confirmed baseline behavior.
2. **Configuration provenance table:** Every cache rate, model, workload, and relevant setting with its source.
3. **Current experiment status:** Running, completed, queued, failed, or unresolved, supported by logs/process evidence.
4. **Validity assessment:** Which existing results are comparable and which are potentially confounded.
5. **Minimal corrected experiment plan:** The smallest fair experiment needed to compare q\*, B, and C.
6. **Policy specification:** The verified equations, variable definitions, units, boundary conditions, and fallback behavior.
7. **Implementation and tests:** Isolated policy hooks, unit tests, integration checks, and proof that disabling V1/V2/V3 restores the original Reuse-Aware behavior.
8. **Experiment artifacts:** Commands, configurations, logs, raw results, and analysis scripts.
9. **Ablation report:** Individual feature contributions and interactions.
10. **Final research report:** Results, limitations, and claims supported by the evidence.

# 14. Working Rules

- Be precise, evidence-based, and direct.
- Refer to `@/docs/references/2608.16157v1.pdf` throughout implementation and evaluation.
- Do not hallucinate results or source-code behavior.
- Do not assume a setting is paper-faithful without checking its provenance.
- Do not silently change the baseline.
- Do not discard negative results.
- Do not edit files or launch experiments during the initial read-only audit.
- Do not automatically continue queued experiments.
- Keep all new research artifacts in `q_expswithreuse_aware/`.
- If something cannot be verified, state exactly what is missing.
- If a formula is inconsistent with the paper or runtime, stop and explain the inconsistency before implementing it.
- After completing the audit, present the evidence-based report and the minimal corrected plan before making implementation changes.

**Begin with the read-only audit only. Do not implement or launch anything yet.**
