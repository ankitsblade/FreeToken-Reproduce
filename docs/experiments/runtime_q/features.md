# Runtime feature inventory

Source: Saketh's frozen feature analysis and instrumentation. All 89 candidates, screening decisions, 63 kept features, pre-action restrictions, post-action outcomes and offline labels are retained below. See [report.md](report.md) for measured policy scope.

Decision point: inside `ensure_experts_hybrid`, after routing and the cache lookup (so `m` is known) and before the fetch kernel. Everything under *pre-action* is computed from the current routing/cache state or from measurements of **earlier** layer calls. Outcomes and offline labels are never policy inputs.

## A. Pre-action features

| group | feature | class | status | mean | std | Spearman vs cost_h8 | perm. importance (ms) | note |
|---|---|---|---|---|---|---|---|---|
| BASE | `m` | measured | kept | 1.97 | 1.3 | +0.63 | 0.272 | cache-missing experts of this layer call |
| BASE | `bp_gbps` | measured | removed: constant (6.71) on this machine/model |  |  |  |  | calibrated PCIe expert-gather bandwidth |
| BASE | `bh_gbps` | measured | removed: constant (35.67) on this machine/model |  |  |  |  | calibrated CPU MoE bandwidth |
| G1_routing | `layer_id` | measured | kept | 11.5 | 6.92 | -0.15 | 0.080 |  |
| G1_routing | `tok` | measured | kept | 122 | 72.2 | -0.12 | 0.070 |  |
| G1_routing | `top_k` | measured | removed: constant (4.0) on this machine/model |  |  |  |  |  |
| G1_routing | `n_unique` | measured | removed: constant (4.0) on this machine/model |  |  |  |  |  |
| G1_routing | `n_hits` | measured | removed: redundant with m (|rho|=1.000) |  |  |  |  |  |
| G1_routing | `miss_ratio` | derived | removed: redundant with m (|rho|=1.000) |  |  |  |  |  |
| G2_cpu | `cpu_ms_per_exp_ewma` | derived | kept | 0.413 | 0.0456 | +0.24 | -0.002 | EWMA CPU task ms per CPU-executed expert (this layer) |
| G2_cpu | `cpu_ms_prev` | measured | kept | 0.635 | 0.477 | +0.44 | 0.003 |  |
| G2_cpu | `bh_recent_gbps` | derived | kept | 32.4 | 1.87 | -0.24 | 0.011 |  |
| G2_cpu | `cpu_threads` | measured | removed: constant (5.0) on this machine/model |  |  |  |  |  |
| G3_gpu | `gemm_ms_prev` | measured | kept | 0.223 | 0.0688 | -0.15 | 0.000 |  |
| G3_gpu | `gemm_ms_ewma` | derived | kept | 0.223 | 0.0635 | -0.27 | 0.001 |  |
| G4_transfer | `pcie_ms_per_exp_ewma` | derived | kept | 1.99 | 0.0572 | -0.07 | 0.003 | EWMA PCIe ms per fetched expert |
| G4_transfer | `bp_recent_gbps` | derived | removed: redundant with pcie_ms_per_exp_ewma (|rho|=0.999) |  |  |  |  |  |
| G4_transfer | `expert_mb` | measured | removed: constant (13.23648) on this machine/model |  |  |  |  |  |
| G4_transfer | `q_prev` | measured | kept | 0.435 | 0.76 | +0.39 | 0.000 |  |
| G4_transfer | `h2d_mb_prev` | derived | removed: redundant with q_prev (|rho|=1.000) |  |  |  |  |  |
| G5_overlap | `tcpu_prev` | measured | removed: redundant with cpu_ms_prev (|rho|=1.000) |  |  |  |  |  |
| G5_overlap | `tgpu_prev` | derived | kept | 1.11 | 1.46 | +0.50 | -0.000 |  |
| G5_overlap | `imbalance_prev` | derived | kept | 0.665 | 0.229 | +0.02 | -0.000 | previous visit: |T_CPU-T_GPU|/max |
| G5_overlap | `exposed_cpu_prev` | measured | kept | 0.291 | 0.385 | +0.03 | 0.001 | previous visit: time the GPU stream waited for the CPU branch |
| G5_overlap | `wall_prev` | measured | kept | 1.82 | 1.36 | +0.55 | 0.000 |  |
| G5_overlap | `tcpu_over_tgpu_prev` | derived | kept | 1.59 | 1.62 | +0.11 | 0.001 |  |
| G6_cache | `occ` | measured | kept | 169 | 94.5 | -0.66 | 0.002 |  |
| G6_cache | `cap` | measured | kept | 180 | 96 | -0.65 | 0.000 |  |
| G6_cache | `free_slots` | derived | kept | 10.4 | 38.8 | -0.02 | -0.000 | unoccupied expert-cache slots |
| G6_cache | `occ_frac` | derived | removed: redundant with free_slots (|rho|=0.999) |  |  |  |  |  |
| G6_cache | `layer_resident` | measured | kept | 6.83 | 4.47 | -0.68 | 0.073 |  |
| G6_cache | `ins_w32` | derived | kept | 315 | 276 | +0.62 | 0.086 |  |
| G6_cache | `evi_w32` | derived | kept | 282 | 287 | +0.58 | 0.123 |  |
| G6_cache | `churn_w32` | derived | kept | 2.68 | 3.54 | +0.65 | 0.121 | evictions in the last 32 tokens / capacity |
| G6_cache | `lru_age_tok` | derived | kept | 11.4 | 13.6 | -0.76 | 0.574 | tokens since the LRU victim was last used |
| G6_cache | `mean_age_tok` | derived | kept | 2.67 | 1.73 | -0.76 | 0.036 |  |
| G6_cache | `hit_ewma` | derived | kept | 0.493 | 0.259 | -0.69 | 0.116 | EWMA of this layer's hit fraction |
| G7_reuse | `miss_rec_min` | derived | kept | 5.8 | 12.3 | +0.15 | 0.002 |  |
| G7_reuse | `miss_rec_mean` | derived | kept | 10 | 13.9 | +0.27 | 0.003 |  |
| G7_reuse | `miss_freq32_mean` | derived | kept | 4.51 | 3.79 | +0.41 | 0.000 | mean count of the missing experts in this layer's last 32 tokens |
| G7_reuse | `miss_freq32_max` | derived | kept | 6.31 | 5.44 | +0.48 | 0.000 |  |
| G7_reuse | `miss_freq8_mean` | derived | kept | 1.22 | 1.18 | +0.34 | 0.000 |  |
| G7_reuse | `n_miss_seen8` | derived | kept | 1.37 | 1.19 | +0.53 | 0.000 |  |
| G7_reuse | `distinct8` | derived | kept | 13.5 | 3.06 | +0.15 | 0.008 |  |
| G7_reuse | `distinct32` | derived | kept | 20.3 | 4.29 | +0.09 | 0.001 |  |
| G7_reuse | `entropy32` | derived | kept | 2.68 | 0.31 | +0.13 | 0.004 | routing entropy of this layer over 32 tokens |
| G7_reuse | `top4_share32` | derived | kept | 0.452 | 0.109 | -0.14 | 0.003 |  |
| G8_util | `cpu_util` | measured | kept | 26.1 | 4.19 | -0.02 | 0.054 |  |
| G8_util | `gpu_util` | measured | kept | 52.1 | 18.7 | +0.43 | 0.049 |  |
| G8_util | `gpu_mem_util` | measured | kept | 16.4 | 6.7 | -0.67 | 0.015 |  |
| G8_util | `load1` | measured | kept | 2.7 | 0.508 | +0.10 | 0.003 |  |
| G9_memory | `gpu_free_gb` | measured | kept | 1.09 | 0.00787 | +0.41 | 0.000 |  |
| G9_memory | `host_avail_gb` | measured | kept | 13.9 | 0.0679 | +0.29 | -0.006 |  |
| G9_memory | `kv_pos` | derived | removed: redundant with tok (|rho|=1.000) |  |  |  |  |  |
| G10_history | `hist1_m` | derived | kept | 1.97 | 1.3 | +0.56 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist1_hits` | derived | removed: redundant with hist1_m (|rho|=0.982) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist1_q` | derived | removed: redundant with q_prev (|rho|=1.000) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist1_wall` | derived | removed: redundant with wall_prev (|rho|=1.000) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist1_cpu` | derived | removed: redundant with cpu_ms_prev (|rho|=1.000) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist1_gpu` | derived | removed: redundant with tgpu_prev (|rho|=1.000) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist1_pcie` | derived | kept | 0.883 | 1.5 | +0.49 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_m` | derived | kept | 1.98 | 1.1 | +0.65 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_hits` | derived | kept | 2 | 1.1 | -0.67 | 0.008 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_q` | derived | kept | 0.44 | 0.511 | +0.52 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_wall` | derived | kept | 1.83 | 0.936 | +0.69 | -0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_cpu` | derived | kept | 0.639 | 0.388 | +0.50 | 0.006 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_gpu` | derived | kept | 1.12 | 0.98 | +0.59 | -0.003 | rolling mean over the last N visits of this layer |
| G10_history | `hist4_pcie` | derived | removed: redundant with hist4_gpu (|rho|=0.993) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist8_m` | derived | kept | 2 | 1.05 | +0.67 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist8_hits` | derived | removed: redundant with hit_ewma (|rho|=0.987) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist8_q` | derived | kept | 0.445 | 0.462 | +0.58 | -0.001 | rolling mean over the last N visits of this layer |
| G10_history | `hist8_wall` | derived | kept | 1.84 | 0.854 | +0.75 | 0.012 | rolling mean over the last N visits of this layer |
| G10_history | `hist8_cpu` | derived | kept | 0.643 | 0.366 | +0.51 | 0.002 | rolling mean over the last N visits of this layer |
| G10_history | `hist8_gpu` | derived | removed: redundant with hist8_q (|rho|=0.989) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist8_pcie` | derived | removed: redundant with hist8_q (|rho|=0.989) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist16_m` | derived | kept | 2.02 | 1.01 | +0.67 | -0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist16_hits` | derived | kept | 1.96 | 1.01 | -0.68 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist16_q` | derived | kept | 0.452 | 0.438 | +0.62 | -0.001 | rolling mean over the last N visits of this layer |
| G10_history | `hist16_wall` | derived | kept | 1.86 | 0.81 | +0.78 | 0.057 | rolling mean over the last N visits of this layer |
| G10_history | `hist16_cpu` | derived | kept | 0.649 | 0.351 | +0.50 | -0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist16_gpu` | derived | removed: redundant with hist16_q (|rho|=0.997) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist16_pcie` | derived | removed: redundant with hist16_q (|rho|=0.997) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist32_m` | derived | kept | 2.05 | 0.977 | +0.65 | -0.002 | rolling mean over the last N visits of this layer |
| G10_history | `hist32_hits` | derived | kept | 1.93 | 0.976 | -0.67 | -0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist32_q` | derived | kept | 0.461 | 0.423 | +0.64 | 0.057 | rolling mean over the last N visits of this layer |
| G10_history | `hist32_wall` | derived | kept | 1.88 | 0.779 | +0.79 | 0.654 | rolling mean over the last N visits of this layer |
| G10_history | `hist32_cpu` | derived | kept | 0.658 | 0.34 | +0.48 | 0.000 | rolling mean over the last N visits of this layer |
| G10_history | `hist32_gpu` | derived | removed: redundant with hist32_q (|rho|=0.998) |  |  |  |  | rolling mean over the last N visits of this layer |
| G10_history | `hist32_pcie` | derived | removed: redundant with hist32_q (|rho|=0.999) |  |  |  |  | rolling mean over the last N visits of this layer |

## B. Action

`q` = experts fetched to the GPU for this layer call, q in {0..m} (logged as `n_fetch`, the count the kernel actually fetched).

## C. Post-action outcomes (measured after the action)

| column | meaning |
|---|---|
| `n_fetch` | q actually executed |
| `wall_ms` | layer wall time, synced at both ends |
| `wall_net_ms` | wall time minus the decision overhead |
| `cpu_task_ms` | native CPU-pool task time (T_CPU) |
| `pcie_ms` | CUDA-event time of the H2D expert copy (T_transfer) |
| `gemm_ms` | CUDA-event time of the GPU expert GEMM (hits + fetched together) |
| `pre_ms` | ensure-experts kernel + D2H of activations + CPU submit |
| `exposed_ms` | time the GPU stream waited for the CPU branch after its own work |
| `decide_us` | host time for features + policy |
| `n_evict` | experts evicted by this fetch |
| `occ_post` | cache occupancy after the action |
| `routed_ids` | routed expert ids |
| `missing_ids` | missing expert ids |
| `fetched_ids` | expert ids fetched |
| `evicted_gids` | global ids evicted |

## D. Offline-only labels (built from future rows; analysis and training targets only)

| label | meaning |
|---|---|
| `cost_h{0,1,4,8,16,32}` | this layer's net wall time now plus its next H visits |
| `fut_hits_h*`, `fut_m_h*`, `fut_fetch_h*` | hits / misses / fetches of this layer over the next H tokens |
| per-fetch `future_hits`, `residency_tok`, `first_reuse` | what each fetched expert bought before eviction (cache_analysis) |
| single-step `q_oracle` | argmin over q of the measured wall(m, q) table (uniform-random logging) |

## E. Unavailable signals (not measured; reason)

| signal | why |
|---|---|
| PCIe utilization / queue depth | no driver counter exposes link utilization; only negotiated gen/width (static) is readable |
| CPU / GPU memory-bandwidth utilization | needs uncore PMU / vendor counters not available unprivileged on this host |
| CPU worker queue depth / idle time | the native pool exposes only per-task start/end timing, not queue state |
| pinned-memory usage / budget | not exported by the runtime |
| KV cache occupancy | not visible to the MoE cache hook; decode position (kv_pos) is the proxy |
| GPU expert execution time alone | the GEMM kernel covers hits + fetched experts together; reported as gemm_ms, not per-source |
| thermal / throttling | GPU clock/temperature only via 30 ms nvidia-smi calls; too slow to sample per decision |
| true future reuse probability | only as an OFFLINE label (future_hits_*), never a decision feature |

## F. Counts

- candidate pre-action features: 89
- kept after constant / redundancy screening: 63
- removed: 26 (constants on one machine+model, or |Spearman| > 0.98 with an earlier feature)
- decisions logged: 402336 (310896 with a randomized action)
