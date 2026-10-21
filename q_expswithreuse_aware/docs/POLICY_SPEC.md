# Policy specification (verified against code)

Every policy decides one integer per (token, MoE layer): `q ∈ {0..m}`, the number of this layer's
`m` unique missing experts to copy to the GPU. The other `m − q` run on the CPU. The engine
chooses *which* misses to fetch (most recently active first) and *which* slots to evict
(global LRU, ties to the lower slot); no policy changes that. A host-side policy writes
`hybrid_max_fetch = q`, `hybrid_fetch_fraction = 0` before the engine's fill kernel runs
(`runtime_q_hook.py:443-450`). Code references are to frozen files, which are not modified.

Common symbols:

| Symbol | Definition | Unit |
|---|---|---|
| `k` | routed experts per token per layer (top-k): 8 on Qwen | experts |
| `m` | unique routed experts of this layer not resident at the decision | experts |
| `h` | resident routed experts (`h = unique − m`) | experts |
| `S` | bytes of one complete expert (all banks) | bytes |
| `B_P`, `B_H` | profiled PCIe gather and CPU-MoE bandwidths (benchbw) | GB/s |
| `ρ` | `B_P / B_H` | – |
| `p_e` | `(count_32(e) + 8·k/E) / (n + 8)`: smoothed routing frequency of expert `e` in its layer over the last `n ≤ 32` tokens | probability per token |
| `p_j` | j-th largest `p_e` among the misses (j = 0, 1, …) | probability per token |
| `T_j` | 32 if `free_slots > j`, else LRU age = `(step − min usage of occupied slots) / L` | tokens |

Boundary conventions shared by B and C:
- `m = 0` gives `q = 0`.
- The loop never exceeds `m`.
- When the cache is full and the LRU slot was used this token, `T = 0`, so only "free" fetches
  (cost ≤ 0) are taken.
- `n = 0` (first token) gives `p = k/E`.

## 1. Policy A: paper q\* (`fraction_min1`, `reuse_aware_v2_hook.FractionMin1`)

`q = 0` if `m = 0`, else `max(1, round_balance(f, m))` with `f = B_P / B_H` (Eq. 4).

`round_balance` reproduces the engine's in-kernel rounding (Q16 fixed point):
1. `lo = ⌊m·f⌋`.
2. Choose `lo` if `max(lo(1−f), (m−lo)f) ≤ max((lo+1)(1−f), (m−lo−1)f)`, else `lo + 1`.
3. Clamp to `[0, m]`.

Step 2 is the integer neighbour minimising the slower branch of Eq. 3. The minimum fill is the
paper's text (§3.2 p. 7). The rounding mode is the repository's, because the paper gives none.

On Qwen, `f = 0.21383` (6.71/31.38), and q by m = 0…8 is `[0,1,1,1,1,1,1,1,1]`. The standalone
`f = 6.71/32.9` gives the same. Boundary: `B_H ≤ B_P` gives `f ≥ 1`, which is clamped to 1, so `q = m`.

Diagnostics:
- **A0** `fraction`: the same without the minimum, q = `[0,0,0,0,1,1,1,1,1]`.
- **E** `native`: the engine's own fraction `pcie_ov/(pcie_ov+cpu_ov)`, 0.1541 on NVFP4,
  q = `[0,0,0,0,0,0,1,1,1]`.

## 2. Policy B: original Reuse-Aware (`reuse_aware`, κ = 1; `runtime_q_hook.ReuseAware`)

```
q = 0
while q < m and  R_q ≥ κ·Δ(m, q):  q += 1
R_j     = p_j · T_j                                   expected future hits of the j-th fill (hits)
Δ(m, j) = max((j+1)/ρ, m−j−1) − max(j/ρ, m−j)          marginal exposed cost of fill j+1, in units
                                                       of one CPU-expert time S/B_H
```

Both sides are in "CPU-expert times". One future hit is valued at one CPU expert, because a hit
removes one expert from the CPU branch. `Δ ≤ 0` means the fill hides behind the CPU branch.

The comparison is `≥`. The prompt's `R_i > Δ(m, j)` differs only at exact ties.

Cost model: `T_fill(j) = j·S/B_P`, `T_cpu = (m−j)·S/B_H`. This is not Eq. 3 (AUDIT F4).

## 3. Policy C: B with switches V1, V2, V3 (`reuse_aware_v`; `reuse_aware_v2_hook.ReuseVariant`)

```
fetch the j-th candidate while   a · (p_j − b·p_victim(j)) · T_j · v  ≥  ΔC(m, j)
a  = α if V2 else 1        b = 1 if V3 else 0
v  = v_hit (ms) if V1 else 1
ΔC = C(m, j+1) − C(m, j) (ms) if V1 else Δ(m, j)
```

With all switches off this is B, operation for operation. Verified as follows:
- `tests/test_policies.py`: 60,000 randomised states across 3 model shapes, 0 mismatches against
  `ReuseAware` and against an independent reference written from this document;
- real logs: 0 mismatches between every arm's q and its own rule's shadow decision (AUDIT F8).

### 3.1 V1: measured costs

```
C(m, j) = max(g0 + j·t_p,  [m > j]·(c0 + (m − j)·t_c))                        ms
v_hit   = Σ_m n_m [c(m) − c(m−1)] / Σ_m n_m,  c(n) = max(g0, [n>0](c0 + n·t_c))  ms per hit
```

| Term | Measurement |
|---|---|
| `g0`, `t_p` | least squares of `pcie_ms + gemm_ms` (CUDA events on the compute stream: copy_missing, then the grouped GEMM of `H ∪ F`) against fills `n_fetch`, over every layer sample of this episode |
| `c0`, `t_c` | least squares of the CPU worker's task timer against CPU experts `m − n_fetch` (samples with ≥ 1 CPU expert) |
| `n_m` | count of past layers with `m` misses in this episode |

- Sufficiency: a fit is used only with ≥ 8 samples, ≥ 2 distinct x values and slope > 0.
  Otherwise it falls back to the prior: intercept 0, slope `S/B_P` (GPU) or `S/B_H` (CPU).
  The intercept is clamped at ≥ 0. That clamp is an implementation detail not in the original
  proposal (`BranchFit.line`).
- Causality: fits are updated in `finish_layer`, after the layer, so decisions use only earlier layers.
- Dimensions: `p·T` is hits, `v_hit` is ms/hit, `ΔC` is ms. With zero intercepts and prior
  slopes, V1 reduces to B scaled by `S/B_H` on both sides; that is tested.
- Timer comparability (AUDIT F12): on CPU-bound layers the stream-observed copy-start→end time
  equals the CPU task time to a median of −0.02 ms. The GPU fit omits `pre` (router, decision,
  fill kernel) and the merge, which are common to every q.
- Logged per row:
  - fit sample counts and prior state at decision time (`fit_g_n`, `fit_c_n`, `fit_g_prior`, `fit_c_prior`);
  - `C(m, q_taken)` (`c_pred_ms`), for prediction error against `pcie + gemm + exposed`;
  - the decision time (`decide_us`).

### 3.2 V2: calibrated reuse

`α = (H + 8) / (P + 8)`, global per episode.
- When an expert is filled, `P_e = p_e·T_{rank(e)}` is recorded (B's prediction for that fill).
- Each later token on which it is routed while still resident adds one to `H_e`. The fill
  token itself does not count.
- When it is evicted, `H += H_e` and `P += P_e`.

Properties:
- Causal: α changes only after a layer that evicted something; 0 violations in real logs.
- The horizons match in kind (predicted hits over the expected residency, realised hits over the
  actual residency).
- Residencies still open at the end of the episode are never scored (censoring), and hits are
  gross, i.e. not net of the evicted expert's lost hits.
- `+8` is the existing `PRIOR_N` pseudo-count, so `α = 1` before any evidence. It has no other
  documented rationale and is treated as a research hyperparameter.
- Insufficient evidence means `calib_n = 0` scored residencies, and then `α = 1`. Logged as
  `calib_n`, `hits_done`, `pred_done`.

### 3.3 V3: eviction-aware

`R_j = (p_j − p_victim(j))·T_j`. `victim(j)` is the j-th slot the engine will evict:
- ascending usage stamp, over slots not owned by this layer's routed experts;
- free slots first (usage 0, `p_victim = 0`);
- ties to the lower slot.

`p_victim` is the victim's own-layer `p` over that layer's window.

**This study's V3 uses an engine-exact victim order** (`qexp_hook.engine_victim_slots`). The
frozen hook breaks ties at the k-th stamp arbitrarily (AUDIT F7). The difference is logged
(`victim_fix`), and the match against the engine's own CPU reference kernel is tested.

Dimensions: both `p` values are per-token routing probabilities, each estimated in its own layer
(every layer is visited once per token), and `T_j` is in tokens. So `(p_j − p_victim)·T_j` is in hits.

Conceptual caveat (not changed, reported). Charging the victim over the same horizon `T_j` is a
first-order LRU-stack argument: one fill shifts the stack by one slot for about `T_j` tokens, and
the slot at the boundary loses about `p_boundary·T_j` hits. The specific victim's own
counterfactual residency is shorter when the cache churns, roughly until the next fill. To
check this empirically, `n_remiss` / `remiss_tok` count misses of experts this episode had
cached and evicted earlier, separating capacity misses from cold misses.

## 4. Fallbacks and failure behaviour

| Condition | Behaviour |
|---|---|
| no bench profile | E: fixed cap 1 fill (engine). A, A0 and B use `B_P`, `B_H` from the plan env |
| `m = 0` | every policy returns 0 |
| V1 fit not supported | prior line (intercept 0, slope `S/B_P` or `S/B_H`) |
| V2 no finished residency | `α = 1` |
| V3 free slot | `p_victim = 0` |
| cache rebuilt mid-episode | capacity re-read; residency and eviction records cleared (`_on_resize`) |
| hook in a captured graph | raises; eager decode required |
