"""Feature schema for the runtime-aware q study.

Every name in PRE_ACTION is computed from state available BEFORE q is chosen for the
current (token, layer) decision: the routed experts and cache state of this step, plus
measurements of *earlier* steps. POST_ACTION columns are outcomes of the action. OFFLINE
labels are built later from future rows and are never fed to a policy.
"""

from __future__ import annotations

HIST_WINDOWS = (1, 4, 8, 16, 32)
HIST_SIGNALS = ("m", "hits", "q", "wall", "cpu", "gpu", "pcie")

GROUPS: dict[str, list[str]] = {
    # Baseline: what q* uses. B_P / B_H are calibration constants on a single machine.
    "BASE": ["m", "bp_gbps", "bh_gbps"],
    "G1_routing": ["layer_id", "tok", "top_k", "n_unique", "n_hits", "miss_ratio"],
    "G2_cpu": ["cpu_ms_per_exp_ewma", "cpu_ms_prev", "bh_recent_gbps", "cpu_threads"],
    "G3_gpu": ["gemm_ms_prev", "gemm_ms_ewma"],
    "G4_transfer": ["pcie_ms_per_exp_ewma", "bp_recent_gbps", "expert_mb", "q_prev",
                    "h2d_mb_prev"],
    "G5_overlap": ["tcpu_prev", "tgpu_prev", "imbalance_prev", "exposed_cpu_prev",
                   "wall_prev", "tcpu_over_tgpu_prev"],
    "G6_cache": ["occ", "cap", "free_slots", "occ_frac", "layer_resident", "ins_w32",
                 "evi_w32", "churn_w32", "lru_age_tok", "mean_age_tok", "hit_ewma"],
    "G7_reuse": ["miss_rec_min", "miss_rec_mean", "miss_freq32_mean", "miss_freq32_max",
                 "miss_freq8_mean", "n_miss_seen8", "distinct8", "distinct32", "entropy32",
                 "top4_share32"],
    "G8_util": ["cpu_util", "gpu_util", "gpu_mem_util", "load1"],
    "G9_memory": ["gpu_free_gb", "host_avail_gb", "kv_pos"],
    "G10_history": [f"hist{n}_{s}" for n in HIST_WINDOWS for s in HIST_SIGNALS],
}

PRE_ACTION: list[str] = [f for g in GROUPS.values() for f in g]

ID_COLS = ["episode", "phase", "policy", "workload", "cache_rate", "problem", "seed",
           "t_abs", "token_wall_ms"]
ACTION = ["q", "q_native"]
POST_ACTION = ["n_fetch", "wall_ms", "wall_net_ms", "cpu_task_ms", "pcie_ms", "gemm_ms",
               "pre_ms", "exposed_ms", "decide_us", "n_evict", "occ_post",
               "routed_ids", "missing_ids", "fetched_ids", "evicted_gids"]

COLUMNS = ID_COLS + PRE_ACTION + ACTION + POST_ACTION

# Why a candidate signal is not in the schema (documented, not silently dropped).
UNAVAILABLE = {
    "PCIe utilization / queue depth": "no driver counter exposes link utilization; only "
                                      "negotiated gen/width (static) is readable",
    "CPU / GPU memory-bandwidth utilization": "needs uncore PMU / vendor counters not "
                                              "available unprivileged on this host",
    "CPU worker queue depth / idle time": "the native pool exposes only per-task start/end "
                                          "timing, not queue state",
    "pinned-memory usage / budget": "not exported by the runtime",
    "KV cache occupancy": "not visible to the MoE cache hook; decode position (kv_pos) is "
                          "the proxy",
    "GPU expert execution time alone": "the GEMM kernel covers hits + fetched experts "
                                       "together; reported as gemm_ms, not per-source",
    "thermal / throttling": "GPU clock/temperature only via 30 ms nvidia-smi calls; too "
                            "slow to sample per decision",
    "true future reuse probability": "only as an OFFLINE label (future_hits_*), never a "
                                     "decision feature",
}
