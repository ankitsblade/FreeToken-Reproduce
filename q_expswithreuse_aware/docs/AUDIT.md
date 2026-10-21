# Read-only audit: paper q\*, original Reuse-Aware, Runtime-Variable-Aware Reuse

Audit date 2026-10-21 (10:55–11:15 UTC). Host: RTX 4070 12 GB (PCIe 4.0 x4), Ryzen 5 8500G
(6 cores / 12 threads), 30 GB RAM, driver 595.84. Nothing was edited, stopped or launched
before this audit was complete. All analyses for it ran on existing logs. Its outputs are in
`results/aggregated/*existing*.json`.

Paper: `docs/references/2608.16157v1.pdf` (arXiv 2608.16157v1, 17 Aug 2026, sha256
`65433248…`). Citations are written as §section / Eq. / page.

---

## 1. What the paper specifies

### 1.1 Decode decision (§3.2, p. 7; Fig. 2, p. 6)

- At each MoE layer the router and cache lookup run on the GPU. Resident experts `H` run on
  the GPU. The `m` unique missing experts `M` are split into a fill set `F` and a CPU set `C`:
  `M = F ∪ C`, `q = |F|` (Eq. 1). Fills are copied into cache slots, run on the GPU, and stay
  resident. CPU experts run from host memory and leave residency unchanged.
- Residual-bandwidth model. `S` is the bytes of one expert, `B_P` the pinned expert-transfer
  bandwidth and `B_H` the host-side expert-processing bandwidth.
  `B_R = max(B_H − B_P, 0)` (Eq. 2), `T_fill(q) ≈ qS/B_P`, `T_cpu(m−q) ≈ (m−q)S/(B_H − B_P)` (Eq. 3).
  Balancing the two gives `q/(m−q) ≈ B_P/(B_H−B_P)` and `q* ≈ m·B_P/B_H` (Eq. 4).
- "In practice, FreeToken rounds q\* to an integer and delegates the specific selection of F to
  the cache replacement policy. **It always retains at least one fill**, so the cache continues
  warming even when the CPU handles most misses." (p. 7). The rounding mode is not given.
  Fig. 2 example: `m = 4`, `B_P:B_H ≈ 1:4`, `q* = 1`.
- `B_P` and `B_H` are "empirically profiled on the target hardware at deployment" (p. 7), and
  Table 1 (p. 10) gives them "measured on the deployed tensor shapes". Eq. 2 models contention
  explicitly, so `B_H` is the uncontended CPU-kernel bandwidth.
- Execution order: the CPU branch is launched first, then the GPU miss path (cache update, batch
  copy of F, grouped GEMM of `G = H ∪ F`). The layer's exposed latency is the slower branch (p. 8).
- Cache: one shared LRU over all layers. A hit refreshes recency, a fill admits, and eviction
  removes the least recently demanded expert (§3.2 p. 7; §4.1 p. 8: a single pass finds the
  K LRU candidates and the miss path takes the first q).

### 1.2 What the paper does not specify

- **Reuse-Aware (Policy B) does not appear in the paper.** It is a local research rule (§2.2 below).
- Rounding mode of q\*. Behaviour when `B_H ≤ B_P` is given only through Eq. 2 (`B_R = 0`).
- Cache capacity of the serving runs as a rate. Serving uses the VRAM left after weights and
  runtime state (§3.3 p. 8). §5.3 (p. 11) quotes "RTX 5090 serving capacity (37% of Qwen3.6's
  expert pool, 11% of DSV4-Flash's)". Fig. 4b sweeps 0–100% of the pool as an **offline trace
  replay**, not live serving.
- Decode length, repetitions, warm-up count (§3.3 says warming is unnecessary by construction),
  batch size (single-stream is implied), and per-run cache reset.
- Metrics (§5.1 p. 10): decode throughput (per-request mean tok/s) and TTFT (per-request mean).

### 1.3 Paper setup vs. this host

| Item | Paper | Here |
|---|---|---|
| Models | DeepSeek-V4-Flash, Qwen3.6-35B-A3B (BF16; **NVFP4 on the 8 GB laptop**), GLM-5.2 (§5.1) | Qwen3.6-35B-A3B NVFP4 is the only paper model that fits 30 GB RAM |
| Workloads | W1 AIME (single-turn, decode-dominated), W2–W4 agent harnesses (§5.1) | Only W1 is reproducible here; W2–W4 need OpenCode/Claude Code/OpenClaw (D3 blocked, EXPERIMENT_LOG 20 Oct) |
| CPU threads | rented servers capped at 6 threads; desktop/laptop at natural threads (§5.1) | established runs use 5 CPU workers |
| Graphs | CUDA-graph-resident control and CPU branch (§4.1) | host-side policies need eager decode |
| `B_P`, `B_H` | measured per machine (Table 1) | benchbw profile `results/reuse_aware_models/study_20261020T071921Z/calibration/benchbw.json` (sha256 `2293cff3…`): Qwen NVFP4 standalone 6.71 / 32.90 GB/s, overlap 6.71 / 31.38 GB/s |

Integer q\* for Qwen (`m` = 0…8) is identical for the standalone fraction `6.71/32.9` and the
overlap fraction `6.71/31.38` used by the established plans, under both balance and nearest
rounding (checked, and asserted in `tests/test_policies.py`). The choice is inert on Qwen. On
gpt-oss it changes A0 at `m = 4` (q = 1 with overlap, 0 with standalone). It does not change A,
because A has the minimum fill.

## 2. What the official implementation does

The runtime is `.runtime/freetoken`. Its `python/` tree is byte-identical to upstream
`FlashML-org/FreeToken` `cab110e` (the fork merge `4d0f849` changes nothing under `python/`)
plus `runtime/patches/experiment-hooks.patch` (sha256 `5db728e3…`). Verified by
`git archive 4d0f849 python | patch` followed by `diff -r` (no differences outside
`__pycache__` and binaries).

### 2.1 Built-in q\* (Policy E)

- `--moe-hybrid-max-fetch -1` (auto) → `engine.py::_resolve_hybrid_fetch` →
  `bench_profile.load_hybrid_fetch_fraction`. The fraction is **`pcie_ov / (pcie_ov + cpu_ov)`**
  from the overlapped benchbw pair, using the dtype-level entry first (`bench_profile.py:182-187`).
  It falls back to `pcie/cpu` (= Eq. 4) only when the profile has no overlap pair. If there is no
  profile at all, it uses a fixed cap of 1 fill. On this host: 0.178 (gpt-oss MXFP4) and 0.154
  (NVFP4, which Qwen and Gemma both use), against Eq. 4's 0.20–0.21.
- Integer q: in-kernel **branch-balance rounding**. `lo = ⌊m·f⌋`, then pick `lo` or `lo+1`
  minimising `max(F(1−f), (m−F)f)`. That is Eq. 3's cost with `t_p/t_c = (B_H−B_P)/B_P`
  (`offload_kernels.py:331-341` GPU, `:158-162` CPU mirror). **No minimum fill.** `m = 0` gives `q = 0`.
- Fill selection: the `q` most recently active missing experts (`FREETOKEN_HYBRID_FETCH=recency`,
  the default). Victims: repeated `argmin(usage)` over slots not owned by this layer's routed
  experts. Free slots have usage 0, ties go to the lower slot, and every fill in one call gets
  the same usage stamp (`offload_kernels.py:367-394`).
- Ordering matches the paper: CPU submit first, then copy + GEMM, then sync and merge
  (`layers/moe.py::_decode_hybrid`).

**Paper vs. repository:** (i) the minimum fill is in the paper, not the code; (ii) the built-in
fraction uses the overlapped pair, which differs from Eq. 4; (iii) the rounding is the code's
choice, not the paper's. Policy A follows the paper (Eq. 4 + balance rounding + minimum one
fill). A0 (Eq. 4, no minimum) and E (built-in) are kept as **labelled diagnostics** that isolate
(i) and (ii).

### 2.2 Original Reuse-Aware (Policy B)

`experiments/runtime_q/instrumentation/runtime_q_hook.py:130-174` (sha256 `68b093ad…`, Saketh,
6–7 Oct; unchanged since import). It runs host-side through the experiment hook
(`FREETOKEN_MOE_RUNTIME_Q_MODULE`). That needs eager decode, because the decision reads routed
ids and cache state on the host. Definitions are in POLICY_SPEC.md §2. Audit points:

- The implemented inequality is **`R_j ≥ κΔ(m, j)`**, with κ = 1 (`break if p·T < κ·Δ`), not a strict `>`.
- `R_j = p_j · T_j`. Here `p_j` is the j-th largest smoothed 32-token routing frequency among
  the misses, and `T_j` is 32 tokens if `free_slots > j`, else the age of the single LRU slot in
  tokens. That age is the same for every `j`.
- `Δ(m, j) = max((j+1)/ρ, m−j−1) − max(j/ρ, m−j)` with `ρ = B_P/B_H`. Units are "one CPU
  expert". It uses `t_p/t_c = B_H/B_P`, i.e. the CPU at the full `B_H`. **That is not Eq. 3**,
  which uses `(B_H−B_P)/B_P`. So with `R = 0`, B balances at `m·B_P/(B_P+B_H)` (0.176m on Qwen),
  not Eq. 4's `m·B_P/B_H` (0.214m). The docstring's "nests q\*" is approximate. This is B's
  definition and is preserved.
- B chooses only the **count** q. The engine fetches the q most *recent* misses, while B ranks
  candidates by *frequency*. `R_j` is therefore the value of the j-th most frequent miss, which is
  not necessarily the expert that gets fetched. V2 and V3 inherit this.

## 3. What the experimental implementation changes

| Layer | Change | Effect on decisions |
|---|---|---|
| `experiment-hooks.patch` | env-gated hook loader, CUDA events around copy/GEMM, `FREETOKEN_MOE_FORCE_FETCH_FRACTION`, routing/replay traces | none unless enabled. The FORCE path reuses the engine's fraction path |
| `runtime_q_hook.RuntimeQ` | host decision per layer; `torch.cuda.synchronize` at layer start and end | sets `hybrid_max_fetch = q`, `fraction = 0` per call |
| `reuse_aware_v2_hook` | lean harness (one D2H copy of slot ids/usage/step per layer), `FractionMin1` (A), `ReuseVariant` (V1/V2/V3), `BranchFit` | B unchanged (subclass) |
| `reuse_aware_rv_hook` | superseded feature-group rule plus extra bookkeeping. **Harness of the queued Stage 2** | same A/B/V decisions, but more per-layer work |
| `driver.py` (uncommitted, sha256 `3d2ec98a…`) | per-group hook/env/reset, `log_rep`, server records | none |

Harness cost: on gpt-oss the hooked server is 1.47–1.62× slower than the unhooked server
(+16–22 ms/token), from Stage 1's own unhooked arms (`analysis/REPORT.md`, instrumentation table).

## 4. Configuration provenance

P = paper, R = official repository, E = established local experiments, **D** = deviation from P/R.

| Setting | Value in use (policy study / this study) | Source | Status |
|---|---|---|---|
| Model | Qwen3.6-35B-A3B NVFP4 `1355db6a…` | P §5.1 (NVFP4 laptop build); E `configs/pruning/models.json` | P-justified |
| Model | gpt-oss-20b `6cee5e81…`, Gemma-4-26B-A4B NVFP4 `a19cfe00…` | E only (`configs/baselines.json`) | **D**: not paper models |
| Cache, paper/official | VRAM left after weights + runtime state | P §3.3; R default `--moe-cache-auto` (`server/args.py:835-846`), KV floor 8192 tokens (`engine/config.py:43`) | not run before this study |
| Cache, established baseline | 154 / 2048 / 768 slots = **0.20** of the pool (gpt-oss / Qwen / Gemma) | E `configs/baselines.json`, `scripts/baseline.sh` | E |
| **Cache rate 0.10** | gpt-oss and Gemma 0.10 / Qwen 0.10 | E: SP-01 `high_miss` regime, `driver.py::WORKLOADS = {0.40, 0.20, 0.10}`; EXPERIMENT_LOG SP-01 records that an old README said 0.08 while the saved data are 0.10 / 77 slots | **D**: later experimental choice, not P or R |
| Rates 0.05 (Qwen), 0.15 (Gemma), 0.40 (gpt-oss) | per-model grids | E `docs/reuse_aware_rtx4070.pdf` Table 6: "chosen per model to fit 12 GB with the KV cache" | **D** |
| Workload (policy study) | `reuse_aware_tasks.jsonl` 0–2: code, summarization, math (`routing_workloads#0,#1`, `pruning_requests#math_011`) | E | **D**: not AIME |
| Workload (this study) | AIME-25 problems 0–2, `math-ai/aime25@563bb84` (file sha256 `b4e273c0…`), chat + thinking + boxed instruction | P W1 (AIME); R `benchmarks/bench_decode_moe.py` default dataset and problem 0 | P/R |
| Decode tokens | 128 (policy study) / **256** (this study) | E `baseline.sh` 128; R benchmark default 256; P unspecified | as cited |
| Sampling | greedy, `ignore_eos` | R benchmark `--greedy` option, `ignore_eos`; E | R option |
| Warm-up | one warm-up generate per episode from a reset cache; 3 burn-in episodes per server (2 unhooked) | R benchmark sends one warm-up request; E burn-in from RQ studies (first episodes slower) | E |
| Repeats | 3 timed rounds + 1 logged round, rotated order, paired on one server | E (runtime_q / reuse_aware studies) | E; P unspecified |
| Batch / requests | bs = 1, `--max-running-requests 1` | R benchmark single-stream | R |
| Memory ratio / seq len | 0.9 / 2048 (Qwen, gpt-oss); 0.8 / 1280 (Gemma) | R benchmark default 0.9; E profiles (Gemma lowered for startup memory) | E |
| CPU threads | 5 workers | E | **D**: P uses 6 on capped servers |
| Graphs | eager (`--cuda-graph-max-bs 0`) | E, required by host hooks | **D** from P §4.1 / R default |
| Scheduler | `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`, naive KV, 1024-token prefill cap, serial load | E | **D** (harness) |
| `B_P`, `B_H` | 6.71 / 31.38 (Qwen), 6.70 / 31.76 (gpt-oss), 6.71 / 36.83 (Gemma), overlap mode | E benchbw profile (above) | **D** in source (P implies standalone); inert for integer q\* on Qwen |
| Metrics | ms/token = (t_last − t_first)/(tokens − 1), TTFT, end-to-end | R benchmark formula; P §5.1 | R/P |

**Cache rate 0.10, conclusions.**

1. It originates in SP-01's `high_miss` regime (`driver.py` constant). It was inherited by
   runtime_q, reuse_aware_models and the policy study.
2. It is not specified by the paper or the official repository, and it is not the established
   baseline capacity (0.20). It is a later experimental choice.
3. The paper evaluates live serving only at auto-sized capacity (37% / 11% of the pool on the
   5090), and 0–100% only in an offline replay (Fig. 4b).
4. The established baseline uses 0.20. The policy-study grids are gpt-oss {0.10, 0.20, 0.40},
   Qwen {0.05, 0.10, 0.20} and Gemma {0.10, 0.15, 0.20}, so 0.20 is the only rate shared with
   the established baseline.
5. Results at 0.10 are **not directly comparable** with the paper's serving results. They are a
   sweep point under a deliberately small cache. The paper's comparable live configuration is
   `--moe-cache-auto`, which had never been run on this host.

## 5. Status of experiments and jobs (11:04–11:15 UTC)

| Item | Status | Evidence |
|---|---|---|
| Policy-study queue `results/policy_study/20261021T090614Z/queue.sh` | **running** (PID 3152711, since 09:06 UTC) | `ps`; `queue.log` |
| Smoke (3 models) | completed rc=0 09:06–09:10 | queue.log `SMOKE OK 09:10:50Z` |
| Stage 1 gpt-oss | **completed** rc=0 10:13:32Z; 162 episodes, all 127 decode steps | `stage1_gpt_oss_episodes.jsonl` |
| Stage 1 gpt-oss analysis | **stale**: `analysis/REPORT.md` was written 10:10Z, before the stage ended (0.40/math shows 1 or 0 pairs although all 4 episodes exist) | file mtimes; episode counts |
| Stage 1 Gemma | **running** (rate 0.20, problem 1 at 11:04Z; 131 episodes written) | `stage1_gemma_episodes.jsonl`; live server PID 3166954 |
| Stage 1 Qwen; Stage 2 all models | **queued, not started**. The queue script will start them on its own after Gemma | queue.sh loop |
| Phase A cache-resize validation | **completed separately** (R3 queue done 07:59Z; e.g. Gemma 11 pass / 0 fail). A later probe queue failed (rc=1, 08:11Z) | `results/elastic_reuse/20261021T074158Z/queue_phaseA_r3.log`, `queue_probe.log` |
| Other GPU users | none besides the queue. Two m_reduction monitors (CPU only) are running | `nvidia-smi`, `ps` |

The prior summary's hypotheses (gpt-oss done, Gemma running, Qwen and Stage 2 queued, Phase A
done separately) are **confirmed**. The queue was not stopped. Its later stages use the same
established configuration as Stage 1 gpt-oss and Gemma, and §6 finds them internally valid,
so letting it finish does not contaminate anything. This study is queued behind it
(one GPU job at a time).

## 6. Validity of existing results

| Result | Valid for | Confounds and limits |
|---|---|---|
| Stage 1 gpt-oss (A, A0, E, B; unhooked A0, E) | paired policy comparison inside the hooked eager harness, established grid | rates are not paper rates; prompts are not AIME; eager plus sync harness (×1.5); 5 threads; the stale report must be regenerated |
| Stage 1 Gemma (running), Qwen, Stage 2 (queued) | same as above once complete; Stage 2 at 0.20 (Qwen, Gemma) equals the established baseline capacity | V3 arms use the frozen victim prediction (F7); Stage 2 uses the heavier `reuse_aware_rv` harness |
| V1/V2/V3 pilot (gpt-oss, 0.10 / 0.40, 12 pairs per variant) | preliminary | one model, two non-paper rates, three rounds; V3 victim ties (F7) |
| reuse_aware_models study (B vs paper q\* without minimum) | B vs A0 | its "paper q\*" has no minimum fill (A0, not A); the study harness is not the lean one |
| SP-01 / runtime_q (gpt-oss) | historical, single model | original harness; rates from SP-01 |

## 7. Findings

- **F1** Reuse-Aware is not in the paper. "Original" means the local `runtime_q_hook.ReuseAware`.
- **F2** The paper's minimum one fill is absent from the official kernel.
- **F3** The built-in q\* fraction (overlap pair) differs from Eq. 4.
- **F4** B's Δ uses the CPU at full `B_H` (not Eq. 3), so B does not reduce exactly to q\* when `R = 0`.
- **F5** B, V2 and V3 rank candidates by frequency, while the engine fills by recency.
- **F6** B implements `≥`, not `>`. `T_j` uses the single LRU age for every j.
- **F7** **Defect** in the frozen V3 victim prediction (`reuse_aware_v2_hook.py:284`):
  `np.argpartition` breaks ties at the k-th usage stamp arbitrarily, while the engine takes the
  lower slot. Ties are common because fills share a stamp. Measured match rate on real logs:
  98.08% on Stage 1 gpt-oss (97.57% under A's heavy filling, 99.90% under B) and 99.88% in the
  pilot. A test reproduces a concrete mismatch (`test_frozen_victim_prediction_breaks_ties_arbitrarily`).
  The frozen file is not modified. The study hook selects victims exactly as the engine does
  (0 mismatches against the engine's CPU reference over 40 randomised histories) and logs how
  often the two predictions differ (`victim_fix`).
- **F8** `experiments/runtime_q/check_policy_equivalence.py` tests the superseded `ReuseRV`, not
  `ReuseVariant`. The documented "0 mismatches with all switches off" was therefore unverified
  for the arms Stage 2 runs. It is now verified:
  - 60,000 randomised cases (3 model shapes), `ReuseVariant` with all switches off equals
    `ReuseAware` and an independent reference written from the specification;
  - on real inference logs, every pilot arm's q equals its own rule's shadow decision
    (4 arms × 12,192 layers, 0 mismatches), and the Stage 1 B arm also has 0 mismatches.
- **F9** Cache rate provenance is as in §4.
- **F10** Harness overhead is ×1.47–1.62 on gpt-oss.
- **F11** The Stage 1 gpt-oss report is stale.
- **F12** V1 timers are comparable:
  - on CPU-bound layers, GPU-observed copy-start→layer-end time minus the CPU task timer has
    median −0.019 ms (10–90%: −0.035 to 0.000 ms; 58,976 layers);
  - V1's C(m, q) predicts that time with MAE 0.03–0.05 ms (median APE 1.2–4.8%) for arms that fetch;
  - the fitted per-expert costs (t_p 1.76–1.91 ms, t_c 0.38–0.41 ms) match `S/B_P` and `S/B_H`,
    so V1 differs from B through the intercepts and the measured value of a hit (0.30–0.38 ms,
    against a full CPU expert in B).
  Source: `results/aggregated/v1_timer_audit_existing_*.json`.
- **F13** V2 has no future leakage: on real logs α changed 24,618 times (Stage 1) and 6,237
  times (pilot), always right after a layer with an eviction. Its scope is global per episode;
  hits are gross and censored to finished residencies; `+8` is the existing `PRIOR_N` convention
  with no other documented rationale, so it is a research hyperparameter.
- **F14** V3 is dimensionally consistent. The conceptual caveat is in POLICY_SPEC.md §4.3.

## 8. Smallest defensible corrected experiment

Goal: compare A, B and C under the configuration closest to the paper and official repository
that this host can run, with V1/V2/V3 as a full 2³ factorial. This does not replace the
established-grid policy study; it complements it.

| Factor | Choice | Why |
|---|---|---|
| Model | Qwen3.6-35B-A3B NVFP4 only | the only paper model runnable here; gpt-oss and Gemma are covered by the established-grid study |
| Cache | `--moe-cache-auto` (rate 0 → auto in the runtime) | P §3.3 and R default; size recorded per server |
| Workload | AIME-25 #0, #1, #2 (R benchmark dataset; P W1) | three tasks for task-level replication |
| Decode | 256 tokens, greedy, one warm-up per episode | R benchmark default length; greedy is R's comparison option |
| Arms | A, B, B+V1, B+V2, B+V3, B+V1V2, B+V1V3, B+V2V3, B+V1V2V3; diagnostics A0, E | spec §9; F2/F3 diagnostics |
| Harness | one lean harness for all hooked arms (`policy/qexp_hook.py`); unhooked A0 and E on #0 | equal instrumentation; overhead measured |
| Repeats | 3 timed rounds + 1 logged round per task, rotated, 3 burn-in | established protocol; 9 pairs per arm |
| V3 definition | engine-exact victims (F7 corrected), labelled | matches the documented V3 definition |

Cost: 151 episodes on 5 servers (about 1.5–2 h). It is queued after the policy-study queue as
`results/logs/20261021T111516Z/`.
