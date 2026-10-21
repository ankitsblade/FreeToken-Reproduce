# Read-only audit: paper q\* vs original Reuse-Aware vs runtime-variable-aware policy

Date: 2026-10-21 (audit about 12:55–13:15 UTC). Specification: repository-root `prompt.md`; the user
asked for its artifacts in `saketh_exp_2/` instead of `q_expswithreuse_aware/`.
Host: RTX 4070 12 GB (PCIe 4.0 x16, driver 595.84), Ryzen 5 8500G 6c/12t, 30 GiB RAM.

Nothing was edited, launched or overwritten during the audit except this directory. Before the
audit, at the user's request ("stop everything"), four jobs were stopped. Their state is in §5.

Labels used below: **[P]** paper (`docs/references/2608.16157v1.pdf`), **[O]** official
implementation, **[L]** local experiment code, **[R]** this proposal.

## 1. What the paper specifies

| Item | Paper text | Location |
|---|---|---|
| Miss split | M = F ∪̇ C, q = \|F\|; F is filled into the cache and run on the GPU; C runs on the CPU, leaving residency unchanged | §3.2, Eq. 1, p.7 |
| Residual bandwidth | B_R = max(B_H − B_P, 0), because DMA and the CPU share host memory | §3.2, Eq. 2, p.7 |
| Branch times | T_fill(q) ≈ qS/B_P, T_cpu(m−q) ≈ (m−q)S/(B_H − B_P) | Eq. 3, p.7 |
| q\* | balancing gives q/(m−q) ≈ B_P/(B_H − B_P), so **q\* ≈ m·B_P/B_H** | Eq. 4, p.7 |
| Integer and minimum fill | "rounds q\* to an integer … It always retains at least one fill" (rounding mode not given) | §3.2, p.7 |
| Fill selection | "delegates the specific selection of F to the cache replacement policy" (shared LRU) | §3.2, p.7 |
| B_P, B_H | "empirically profiled on the target hardware at deployment". Table 1: B_P = measured host→device expert transfer; B_H = "measured effective bandwidth of the CPU-side MoE expert kernel", on deployed tensor shapes | §3.2 p.7; Table 1 and §5.1, p.9–10 |
| Device side | one kernel per layer derives q, picks K LRU victims in one pass, rewrites ids; graph-captured | §4.1, p.8 |
| Warm-up | "Warming is unnecessary by construction"; the first request runs cold | §3.3, p.8 |
| Reuse argument | CPU-only execution "forfeits the future hits that a cache fill would buy" (motivation only; no formula) | §2.2, p.4 |
| Models | Qwen3.6-35B-A3B in BF16 (the 8 GB laptop serves the **NVFP4** release), DeepSeek-V4-Flash, GLM-5.2 | §5.1, p.9 |
| Workloads | W1 AIME (single turn, decode-dominated), W2–W4 agent harnesses | §5.1, p.10 |
| Metrics | decode throughput (per-request mean tok/s) and TTFT (per-request mean) | §5.1, p.10 |
| Cache capacity | no cache-rate parameter; serving capacity is what fits (RTX 5090: 37% of Qwen3.6's pool). Fig. 4b sweeps cache size only in an offline trace replay | §5.3, p.11; Fig. 4b |
| CPU threads | rented servers capped at 6 threads; Qwen3.6 main runs at 6 threads | §5.1 p.9; §5.2 p.10 |
| Not specified | repetitions, decode length, sampling, warm-up request, cache rate for decode timing, rounding mode | — |

**Reuse-Aware does not appear in the paper.** Policy B is local research [L], and so is C [R].

## 2. What the official implementation does [O]

The runtime source is base commit `4d0f849` plus `runtime/patches/experiment-hooks.patch`
(sha256 `5db728e3…`). I checked this by applying the patch to `git archive 4d0f849` and
comparing the trees: there are no other differences. The `python/` and `benchmarks/`
directories of `4d0f849` are byte-identical to upstream FlashML-org `cab110e` (#469). The patch
touches `engine.py`, `layers/moe.py`, `models/gpt_oss/moe.py`, `moe/offload_cache.py` and
`cpu_moe_ext.cpp`, and adds `decode_trace.py`, `layer_replay.py` and `routing_trace.py`.
Everything it adds is gated behind environment variables. When none is set, it changes no
decision.

**q in the kernel** (`moe/offload_kernels.py:_ensure_experts_hybrid_kernel`; CPU mirror at L129–165):
- `lo = floor(m·f)`, with f in Q16 fixed point.
- `q = argmin over x ∈ {lo, lo+1} of max(x(1−f), (m−x)f)`; a tie goes to `lo`. This is a balance rounding consistent with Eq. 3.
- **There is no minimum fill.** q = 0 whenever m·f is small. This disagrees with §3.2.

**f from the profile** (`bench_profile.load_hybrid_fetch_fraction`, `engine._resolve_hybrid_fetch`):
- It prefers the overlapped pair, `f = pcie_ov/(pcie_ov + cpu_ov)`, falling back to `pcie/cpu` (exactly Eq. 4). If no profile exists, it uses a fixed cap of 1.
- Under the paper's model, cpu_ov = B_H − B_P, so the overlapped form equals B_P/B_H. It is a measured instantiation of Eq. 2–4, not a different equation.
- On this host, however, pcie_ov + cpu_ov (42.4 GB/s) exceeds standalone B_H (38.1 GB/s). The two forms therefore give different numbers, and the shared-bandwidth assumption of Eq. 2 does not hold exactly here.
- The dtype entry is used before the per-model workload entry.

**Other engine behaviour:**
- **Fill selection:** the most recently active misses (`BY_RECENCY`), with ties going to the lower id.
- **Victims:** `argmin(usage)` over every cache slot (a global LRU across layers). Slots owned by this layer's active experts are excluded. Free slots have usage 0. Equal usage goes to the lowest slot index (Triton `argmin` with `tie_break_left`). Several victims are taken sequentially.
- **Reset:** `OffloadCache.reset()` clears slots, usage, step **and** `expert_recency` (`offload_cache.py:904–910`), so per-episode resets are truly cold.

**Official benchmark** (`benchmarks/bench_decode_moe.py` @ `4d0f849`):
- AIME-25 (`math-ai/aime25`) problem 0 sent as chat with thinking enabled, D = 256, `ignore_eos`.
- Checkpoint-recommended sampling; `--greedy` is an option offered "for the stricter comparison".
- `--moe-cache-auto` unless a rate or size is given; memory ratio 0.9; max-seq 8192 + D; CUDA graph on; `--moe-hybrid-max-fetch -1` (auto f).
- A fresh server, then one identical warm request, then the measured request.
- Metric: `(completion_tokens − 1)/(t_last − t_first)` over SSE events. TTFT is the warm first-token latency.

## 3. What the local experiment code changes [L]

| Component | File | Difference from [P]/[O] |
|---|---|---|
| Driver | `experiments/runtime_q/driver.py` | Many episodes per server with a cache reset per episode; 3 burn-in episodes; **eager** (`--no-graph`); greedy; `--cache-type naive`; warm-up and measured prompts carry different tags; fixed `--cache-rate` |
| Hook loader | `offload_cache.py` patch (`FREETOKEN_MOE_RUNTIME_Q_MODULE`) | A host-side policy overrides `hybrid_max_fetch` for one layer call; requires eager decode |
| Policy B | `runtime_q_hook.py:ReuseAware` | See §6 |
| V1/V2/V3 | `reuse_aware_v2_hook.py:ReuseVariant` (+ `FractionMin1`) | Defaults reproduce B exactly; lean harness |
| Policy-study hook | `reuse_aware_rv_hook.py` | Subclass that adds an unrelated `reuse_aware_rv` (exec/xfer/cache/hist) kind; the policy study ran `reuse_aware_v`, not `reuse_aware_rv` |
| Hook timing | `start_layer`/`finish_layer` | `torch.cuda.synchronize` at the start and end of **every** MoE layer, plus host reads of the slot state |

**Measured harness overhead** (policy study Stage 1, same policy, hooked vs unhooked, problem 0, ms/token):

| Model | Hooked − unhooked | Ratio |
|---|---|---|
| gpt-oss | +16.8 … +22.3 | 1.48–1.64× |
| Gemma | +13.0 … +16.2 | 1.50–1.59× |
| Qwen | +10.0 … +20.3 | 1.26–1.57× |

The harness costs as much as the policy differences it measures. Every hooked result is
conditional on this harness, and none is a serving-performance claim (§7).

## 4. Configuration provenance

| Setting | Value(s) in use | Source | Paper-faithful? |
|---|---|---|---|
| Model (paper-evaluated) | Qwen3.6-35B-A3B **NVFP4** `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6` (local metadata matches the pin) | [P] §5.1 (NVFP4 is the paper's own 8 GB laptop configuration; BF16 does not fit 30 GiB host RAM) | Yes (laptop variant) |
| gpt-oss-20b MXFP4 `@6cee5e8` | used in SP-01, runtime-q and the policy study | [L] chosen because it fits this host | **Not a paper model** |
| Gemma-4-26B-A4B NVFP4 `@a19cfe0` | policy study | [L] | **Not a paper model** |
| Workload | `experiments/prompts/reuse_aware_tasks.jsonl` #0–2 (code, summarization, math) | [L] reuse_aware_models study | **No**; the paper and benchmark use AIME |
| Decode length | 128 | [L] | No; the official benchmark uses 256; the paper is silent |
| Sampling | greedy | [O] option `--greedy` | Default sampling differs; greedy is an upstream option |
| Graph | eager (`--no-graph`) | [L], required by host hooks | **No**; the paper/[O] q is graph-resident (§4.1) |
| KV cache | naive | [L] | [O] default differs |
| CPU threads | 5 (`--cpu-threads 5`) | [L] baseline profiles | Paper uses 6; this host has 6 cores |
| Memory ratio / seq | 0.9 / 2048 (Gemma 0.8 / 1280) | [L] `configs/baselines.json` | [O] 0.9 / 8192 + D |
| Warm-up | per episode: cold reset → 1 warm generate → measure; 3 burn-in episodes per server | [L] | [O]: fresh server + 1 warm request; [P]: no warm-up |
| Repetitions | 3 timed + 1 logged round | [L] | Not specified by [P]/[O] |
| B_P/B_H for the "paper q\*" arm | B_P / **cpu_overlap** (gpt-oss 6.70/31.76 = 0.211; Qwen 6.71/31.38 = 0.214; Gemma 6.71/36.83 = 0.182) | [L] policy study | **Mislabelled**: Table 1 B_H is the standalone CPU kernel bandwidth (gpt-oss 35.39 → 0.190; Qwen 32.90 → 0.204; Gemma: no Gemma-shape entry, unresolved) |
| Engine f (built-in q\*) | 0.178 (MXFP4 dtype entry), 0.154 (NVFP4 dtype entry) | [O] overlapped pair | Official implementation |

Integer q from each f, using the [O] rounding, for m = 1..8:
- gpt-oss: Eq. 4 with standalone B_H gives q = 0 for all m ≤ 4. Since top_k = 4, m ≤ 4, so **q\* never fetches without the minimum fill**. The policy study's "A0" (f = 0.211) fetches 1 at m = 4.
- Qwen: Eq. 4 gives q = 1 for m ≥ 4. The engine gives q = 1 only for m ≥ 6.

### Cache rate 0.10 (special requirement)

1. **Origin.** `experiments/split_policy/run_experiment.py:52–54`, SP-01 (Saketh): `WORKLOADS = {"low_miss": 0.40, "med_miss": 0.20, "high_miss": 0.10}`, with the comment that 0.10 is "the smallest cache the overlap double-buffer allows (cache_size ≥ 2·E)" for gpt-oss. It was copied into `experiments/runtime_q/driver.py:37`. The experiment log (SP-01, line 108) records an older README value of 0.08 that conflicts with the saved 0.10 / 77 slots.
2. **Classification.** A later experimental choice that creates a high-miss regime. It comes from neither [P] nor [O].
3. **What the paper evaluates.** Decode timing uses the serving capacity, i.e. automatic sizing (RTX 5090: 37% of Qwen3.6's pool, 11% of DSV4-Flash's). Fig. 4b varies cache size from 0 to 100% only in offline trace replay. [O] defaults to `--moe-cache-auto`.
4. **Rates in our experiments:**
   - SP-01 and runtime-q (gpt-oss): 0.10 / 0.20 / 0.40.
   - Baseline profiles (`configs/baselines.json`): 154 / 2048 / 768 slots, which is 0.20 for every model.
   - reuse_aware_models and the policy study: gpt-oss 0.10 / 0.20 / 0.40; Gemma 0.10 / 0.15 / 0.20; Qwen 0.05 / 0.10 / 0.20. These were "chosen per model to fit 12 GB with the KV cache" (`docs/reuse_aware_rtx4070.pdf`, Table 6).
   - Auto on this host: Qwen NVFP4 = **4354 slots = 42.5%** of 40 × 256 (`q_expswithreuse_aware/results/raw/20261021T111516Z/.../smoke_qwen_auto_episodes_servers.jsonl`). Auto for gpt-oss and Gemma is **unresolved**: nothing on this host records it.
5. **Comparability.** Results at 0.10 are not directly comparable to the paper. The cache rate is not the paper's serving capacity, and the model, workload, graph mode and hardware also differ. They are a valid high-miss stress regime, but they belong in a labelled sensitivity analysis.

## 5. Status of every experiment and job (verified 12:55–13:15 UTC)

| Item | Evidence | Status |
|---|---|---|
| Policy-study queue `results/policy_study/20261021T090614Z/queue.sh` | `queue.log` | **Stopped** (user request, 12:52 UTC); judge0 restored by its trap |
| Smoke (3 models) | `queue.log` 09:06–09:10, rc = 0 | Completed |
| Stage 1 gpt-oss / Gemma / Qwen | rc = 0 at 10:13 / 11:13 / 12:27; 162 episodes each, every (rate, policy) cell full (12 hooked + 3 unhooked) | **Completed, all three.** The earlier hypothesis ("Gemma running, Qwen queued") is out of date |
| Stage 1 `analysis/REPORT.md` | written 15:40 IST, before Gemma and Qwen ended | **Stale**: covers gpt-oss only |
| Stage 2 gpt-oss | attempt 1 rc = 143 (12:50), attempt 2 stopped; 70 / 144 episodes, rate 0.10 only | **Incomplete** |
| Stage 2 Gemma, Qwen | never started | Not run |
| `q_expswithreuse_aware` Qwen auto-cache AIME ablation (stamp 20261021T111516Z) | smoke 3 episodes rc = 0 (12:54); ablation run `results/runs/runtime_q/20261021T125446Z_7f7c1e84` started 12:54:46 and was stopped about 1 min later (user's choice) | **Aborted**, no ablation data. Its interrupted restore left 3 judge0 containers paused; they were unpaused during this audit |
| m_reduction watchdog + tmux progress | `systemctl --user` | Stopped (user request); other tmux windows untouched |
| Phase A cache-resize validation | `results/elastic_reuse/20261021T074158Z/queue_phaseA_r3.log`: "PHASE A R3 QUEUE DONE 07:59:13Z" | **Completed separately**; `queue_exp1.sh` has no log, so it never ran. Out of scope |
| GPU / processes now | `nvidia-smi`: no compute apps; no experiment processes | Idle |

## 6. Policy definitions as implemented (verified by reading code)

**A, paper q\*** [P + O rounding]:
- `q = max(1, round_balance(f, m))` for m > 0, otherwise 0.
- `f = B_P/B_H` from Table-1-style standalone profile values on the model's shapes.
- `FractionMin1` in `reuse_aware_v2_hook.py` implements this; the existing runs used the overlap B_H.

**A0** (diagnostic): the same without the minimum fill.

**E, built-in q\*** [O]: the engine kernel with the overlapped-pair f and no minimum fill.

**B, original Reuse-Aware** [L] (`runtime_q_hook.py:130–174`):
- ρ = B_P/B_H, from env `FREETOKEN_MOE_RUNTIME_Q_BP/BH` (default 6.71/35.67, i.e. the Sept gpt-oss overlap pair; the policy study used B_P/cpu_ov per model).
- δ(m,q) = max((q+1)/ρ, m−q−1) − max(q/ρ, m−q). This is the change in exposed layer time, in units of one CPU-expert time S/B_H, if q+1 instead of q misses are filled.
  - **Note:** the CPU branch here costs S/B_H per expert, while Eq. 3 uses S/(B_H − B_P). It is consistent with Eq. 3 only if B_H is read as the overlapped CPU bandwidth.
- p_e = (f32_e + (k/E)·8)/(n + 8):
  - f32_e = the number of the last n ≤ 32 tokens in which expert e was routed in this layer;
  - k/E is the prior, with 8 pseudo-counts.
  - Candidates are sorted by p, descending.
- T_res:
  - 32 tokens if free slots > q;
  - otherwise the age in tokens of the globally least-recently-used occupied slot, (step − usage)/L. This is not the engine's exact victim, because the engine excludes this layer's active slots.
- Rule: increase q while **p_q·T_res ≥ κ·δ(m,q)**, with κ = 1. The code uses **≥**, not the strict ">" in `prompt.md`; this matters only at exact ties.
- B chooses only the **count** q (`hybrid_max_fetch = q`, f = 0). The engine then fills the q **most recently active** misses (`BY_RECENCY`). These need not be the q highest-p candidates that B scored.

**B ≡ C with every switch off** (`ReuseVariant`, `cost="paper", reuse="window", evict=False`). The
code reduces to the same loop: v = 1, scale = 1, r = p, cost(j) = max(j/ρ, m−j). The repository's
`check_policy_equivalence.py` tests `ReuseRV`, **not** `ReuseVariant`. In this directory the
equivalence still has to be proven by test (deliverable 7).

**V1, measured costs** [R] (`ReuseVariant cost="runtime"`):
- `C(j) = max(g0 + j·tp, [j<m](c0 + (m−j)·tc))` in ms; `v_hit` = observed-m-weighted mean of `c(m) − c(m−1)` with no fetch. Units check out: count of hits × ms per hit, compared with ms.
- Problems to resolve before implementing:
  - (a) `g0 + j·tp` is fitted on pcie + gemm against n_fetch only, but the grouped GEMM runs over G = H ∪ F, so it also depends on hits h.
  - (b) The timer boundaries do not match. The CPU timer runs on the host clock from `submit` to the last worker finishing (`cpu_moe_ext.cpp` patch); it excludes the D2H input copy and the H2D result. The GPU timers are CUDA events on `copy_missing` and the GEMM.
  - (c) The per-layer synchronize in the hook changes the very overlap being modelled.
  - (d) `v_hit` assumes a future miss would run on the CPU.
- Sufficiency rule as coded: ≥ 8 samples at ≥ 2 distinct x with a positive slope; otherwise the prior is (0, S/B_P) or (0, S/B_H).

**V2, calibrated reuse** [R] (`reuse="calibrated"`):
- α = (realised hits + 8)/(predicted p·T + 8). It is global (all layers) and per episode, and it is updated only when a fill is evicted.
- It is causal (past evictions only). But it is survivorship-biased: residents that are still in the cache never enter the calibration.
- The "+8" reuses B's pseudo-count. It has no separate rationale, so it is a new hyperparameter.
- The earlier pilot measured V2 at 0.905× B at 0.10 (`docs/reuse_aware_v1_v2_v3.md`, line 163).

**V3, eviction-aware** [R] (`evict=True`):
- r = p_q − p_victim(q). The victim's p is its own layer's window estimate, and it is multiplied by the **candidate's** T_res.
- Dimensionally both terms are per-token probability × tokens. Conceptually, though, the victim's loss should run over its own remaining residency, which is unknown. This needs a justified correction or an explicit approximation note.
- Free slot → p_victim = 0, so V3 equals B while free slots exist.
- **Bug:** the frozen v2 hook predicts victims with `argpartition` followed by `lexsort`. When several slots tie on usage at the k-th position, this picks arbitrary ones, whereas the engine takes the lowest slot index. The prior session recorded a 98.1% match against real evictions. A correct prediction must reproduce the engine's sequential lowest-index argmin.

## 7. Validity of existing results

| Result | Verdict | Reason |
|---|---|---|
| Policy study Stage 1, all 3 models | **Valid as a hooked-harness comparison only** | Paired, complete and same harness. But cache rates, workload, 128 tokens and eager mode are all non-paper. The 1.3–1.6× hook overhead dominates. "A0 paper q\*" is mislabelled (overlap B_H). Output text differs across arms within a cell (CPU vs GPU numerics), so the decoded workloads are not identical |
| My recomputation (paired vs A, geo-mean): | | |
| ↳ gpt-oss | B 1.42 / 1.54 / 1.61× at 0.1 / 0.2 / 0.4. A0 1.42 / 1.49 / 1.10×. E 1.15 / 1.03 / 0.73× | The paper's min-1 fill is the slowest arm on gpt-oss. B equals A0 except at 0.40 |
| ↳ Gemma | B 1.11–1.19×, ≈ A0 | B adds nothing over A0 |
| ↳ Qwen | B 1.02–1.05×, A0 1.02–1.08×, E 1.05–1.09× | B is no better than q\* without the minimum on the only paper model |
| Stage 1 `REPORT.md` | Stale | gpt-oss only |
| Stage 2 gpt-oss (70 rows) | Preliminary, incomplete | Rate 0.40 missing; stopped mid-run |
| V1/V2/V3 pilot (gpt-oss, 0.10/0.40) | Preliminary | Non-paper model, workload and rates; synced hook |
| reuse_aware_models study (B vs q\*: gpt-oss 1.15×, Qwen 1.00×, Gemma 0.99×) | Hooked-harness only | Same caveats |
| `q_expswithreuse_aware` | Smoke only | Ablation aborted |
| Phase A | Separate supporting work | Out of scope |

## 8. Smallest defensible plan

**Fixed conditions.** Hold these constant across every arm:
- **Model:** Qwen3.6-35B-A3B NVFP4 `@1355db6`, the only paper model that fits this host.
- **Workload:** the official W1 benchmark contract, AIME-25 `math-ai/aime25`, D = 256. Use problems #0–2 so the result rests on more than one prompt; the problem count is a choice, recorded as such.
- **Cache:** `--moe-cache-auto` (paper and [O]); Qwen auto = 4354 slots.
- **Engine settings:** memory ratio 0.9; text-only; 5 CPU threads (paper: 6, host: 6 cores; recorded deviation); greedy (an upstream option).
- **Server and timing:** one server per paired round with a cold reset, one warm generate and one measured generate, using the [O] decode metric. Rotate arm order each round, run 5 rounds, and pair every comparison within its round.

**Arms.** All arms run on **one** lean hook, so harness cost is identical:
- A (paper: Eq. 4, standalone B_H on Qwen shapes = 6.71/32.90, min-1);
- E (official f, through the hook's native passthrough);
- B (original, ρ from the same profile);
- B+V1, B+V2, B+V3, B+V1V2, B+V1V3, B+V2V3, B+V1V2V3.

Diagnostics:
- A0 (no minimum fill).
- **Unhooked** E in eager mode **and** in CUDA-graph mode, to report the harness cost and the gap to real serving speed.

**Before timing.** The hook in `saketh_exp_2/policy/` (new file; frozen hooks untouched) needs:
- (i) an exact V3 victim replica (sequential lowest-index argmin), checked against logged real evictions;
- (ii) unit tests proving that B+no-V reproduces `runtime_q_hook.ReuseAware` exactly;
- (iii) the V1 timer-boundary audit;
- (iv) logging of m, q, fills, CPU/GPU/PCIe times, decision µs, α, fits, fallback counts and victim matches.

**Claims.** Effects are reported only as "under the eager hook". A serving-performance claim
needs B/C implemented device-side, inside the graph (§4.1). That is a separate engine patch,
justified only if C beats B in this study (already in `docs/PLAN.md:1215`).

**Sensitivity analysis (optional, labelled non-paper).** One fixed low capacity (0.10) for
the same arms. It is not required for the primary question.

## 9. Decisions that need the user

1. Whether a hooked-eager-only study (claims restricted to the harness) is acceptable, or whether a device-side port of B/C is wanted for a real serving claim.
2. Whether Qwen alone (the only paper model here) is the primary scope; gpt-oss and Gemma would be labelled non-paper.
3. AIME problems #0–2 × 5 rounds (about 12 arms × 15 episodes × ~25 s ≈ 1.5–2 h on Qwen), or a different repetition budget.
