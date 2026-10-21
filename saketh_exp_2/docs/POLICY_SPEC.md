# Policy specification (verified against code; see AUDIT.md for sources)

Notation, per MoE layer call during decode (batch 1):

| Symbol | Meaning | Units |
|---|---|---|
| k | routed experts per token (top_k) | — |
| m | unique routed experts not resident in the GPU cache | experts |
| q | misses filled into the cache over PCIe; the other m − q run on the CPU | experts |
| h | routed experts already resident (hits) | experts |
| S | bytes of one complete expert | bytes |
| f | fill fraction | — |
| ρ | B_P / B_H, B's bandwidth ratio | — |
| L | MoE layers | — |
| E | experts per layer | — |

The policy chooses **only q**. Which q misses are filled is decided by the engine: the most
recently active misses first, ties to the lower id. The victims are the q least-recently-used
slots across all layers, excluding slots of this layer's routed experts, ties to the lowest
slot.

## A — paper q\* (primary paper baseline)

- **Equation:** paper §3.2, Eq. 4, p.7: q\* ≈ m·B_P/B_H. B_P and B_H are the standalone profile bandwidths on the model's shapes (Table 1 definitions):
  - Qwen: 6.71/32.90 = 0.203951;
  - gpt-oss: 6.71/35.39 = 0.189602;
  - Gemma: 6.71/38.84 = 0.172760, from the generic NVFP4 entry because no Gemma-shape entry exists.
- **Integer:** FreeToken balance rounding (`offload_kernels.py`). With lo = ⌊m·f⌋ (Q16), q = lo if max(lo(1−f), (m−lo)f) ≤ max((lo+1)(1−f), (m−lo−1)f), else lo + 1. The paper does not specify the rounding mode.
- **Minimum fill:** q = max(1, q) whenever m > 0 (paper: "always retains at least one fill").
- **Boundaries:** m = 0 → q = 0. A full cache does not change q; the victims are LRU slots.

## A0 — paper q\* without the minimum fill (diagnostic)

As A, but q may be 0. For gpt-oss (m ≤ 4), A0 never fills.

## E — built-in q\* (official implementation)

The engine kernel with f = pcie_ov/(pcie_ov + cpu_ov) from the dtype entry of the same profile
(MXFP4 0.1778, NVFP4 0.1541), the same rounding, and no minimum fill.

## B — original Reuse-Aware (`runtime_q_hook.ReuseAware`, unchanged)

- **Cost term:**
  - δ(m, q) = max((q+1)/ρ, m−q−1) − max(q/ρ, m−q), in units of one CPU-expert time S/B_H.
  - ρ = B_P/B_H uses the overlapped CPU bandwidth (Qwen 6.71/31.38, gpt-oss 6.70/31.76, Gemma 6.71/36.83). This is the convention of the original runtime-q study and the policy study.
- **Reuse estimate:**
  - p_e = (f32_e + 8·k/E)/(n + 8), where f32_e counts the last n ≤ 32 decode tokens in which e was routed in this layer.
  - Candidates are sorted by p, descending.
- **Residency horizon:** T_res = 32 tokens if free slots > q; otherwise (step − min usage over occupied slots)/L, the age in tokens of the LRU slot.
- **Rule:** q = 0; while q < m and p_q·T_res ≥ δ(m, q): q += 1. The comparison is "≥" as coded, κ = 1.
- **Boundaries:**
  - m = 0 → 0.
  - δ ≤ 0 means the fill is hidden behind the CPU branch, so it always fills (p > 0 because of the prior).
  - If the cache has no free slot and the LRU age is 0, nothing is filled.

## C — B + V1/V2/V3 (`reuse_aware_v2_hook.ReuseVariant`)

General rule: fill while scale·(p_q − [V3]·p_victim(q))·T_res·v ≥ cost(q+1) − cost(q).

| | Off (= B) | On |
|---|---|---|
| V1 cost | cost(j) = max(j/ρ, m−j); v = 1 (CPU-expert units) | cost(j) = max(g0 + j·tp, [j<m](c0 + (m−j)·tc)) in ms; v = v_hit in ms |
| V2 scale | 1 | α = (realised hits + 8)/(predicted p·T + 8) |
| V3 victim | 0 | p of the j-th predicted victim (0 for a free slot) |

With every switch off, the rule is identical to B. This is tested on 20,000 random cases, and
each switch is also tested to be inert on neutral input (`tests/test_policies.py`).

**V1 details.**
- Online least-squares fits per episode, across all layers:
  - g0 + tp·x, with y = PCIe + GEMM CUDA-event time and x = fills;
  - c0 + tc·x, with y = CPU task time (host clock, submit → last worker) and x = CPU experts.
- A fit is used only with ≥ 8 samples at ≥ 2 distinct x and a positive slope. Otherwise the prior applies: (0, S/B_P) or (0, S/B_H). Fallbacks are logged.
- v_hit = Σ_m n_m·(c(m) − c(m−1))/Σ n_m, where c(n) = max(g0, c0 + n·tc) and n_m counts observed miss counts (capped at k).
- **Known limitations** (AUDIT.md §6): the GPU fit ignores h; the CPU and GPU timers have different boundaries; the per-layer synchronisation changes the overlap being modelled.

**V2 details.**
- α is global and per episode. It is updated only when a hook-tracked fill is evicted: its predicted p·T at fill time against the hits it realised while resident, in later tokens only.
- It is causal but survivorship-biased: residents are never scored until they are evicted.
- The "+8" is B's pseudo-count reused; no separate rationale exists.

**V3 details.**
- Victims are predicted by the engine-exact replica (`sx2_hook.engine_victims`), tested against the real kernel on GPU (900 states, exact order).
- Every real fill is checked against that prediction.
- The victim's p comes from its own layer's window but is multiplied by the candidate's T_res. This is an approximation: the victim's lost reuse should be counted over its own remaining residency, which is unknown.

**Causality.** All inputs are read before the fill kernel runs (routing ids, slot state, past
timer samples, past evictions). No future information is used.
