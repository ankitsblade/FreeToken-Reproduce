"""Write the documented inventory of every candidate signal and its fate."""

from __future__ import annotations

import json
import re

import pandas as pd

from experiments.runtime_q.qlib import RQ
from schema import GROUPS, POST_ACTION, UNAVAILABLE

MEASURED = {"m", "layer_id", "tok", "top_k", "n_unique", "n_hits", "cpu_ms_prev", "cpu_threads",
            "gemm_ms_prev", "q_prev", "wall_prev", "tcpu_prev", "exposed_cpu_prev", "occ", "cap",
            "cpu_util", "gpu_util", "gpu_mem_util", "load1", "gpu_free_gb", "host_avail_gb",
            "bp_gbps", "bh_gbps", "expert_mb", "layer_resident"}
DESC = {"m": "cache-missing experts of this layer call", "bp_gbps": "calibrated PCIe expert-gather bandwidth",
        "bh_gbps": "calibrated CPU MoE bandwidth", "lru_age_tok": "tokens since the LRU victim was last used",
        "free_slots": "unoccupied expert-cache slots", "hit_ewma": "EWMA of this layer's hit fraction",
        "churn_w32": "evictions in the last 32 tokens / capacity", "miss_freq32_mean": "mean count of the missing experts in this layer's last 32 tokens",
        "cpu_ms_per_exp_ewma": "EWMA CPU task ms per CPU-executed expert (this layer)",
        "pcie_ms_per_exp_ewma": "EWMA PCIe ms per fetched expert", "exposed_cpu_prev": "previous visit: time the GPU stream waited for the CPU branch",
        "imbalance_prev": "previous visit: |T_CPU-T_GPU|/max", "entropy32": "routing entropy of this layer over 32 tokens"}


def main() -> None:
    rep = json.loads((RQ / "data" / "clean" / "feature_report.json").read_text())
    st = pd.read_csv(RQ / "results" / "feature_analysis" / "feature_stats.csv", index_col=0)
    lines = ["# Runtime feature inventory\n",
             "Decision point: inside `ensure_experts_hybrid`, after routing and the cache lookup "
             "(so `m` is known) and before the fetch kernel. Everything under *pre-action* is "
             "computed from the current routing/cache state or from measurements of **earlier** "
             "layer calls. Outcomes and offline labels are never policy inputs.\n",
             "## A. Pre-action features\n",
             "| group | feature | class | status | mean | std | Spearman vs cost_h8 | perm. importance (ms) | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g, feats in GROUPS.items():
        for f in feats:
            cls = "measured" if f in MEASURED else "derived"
            if f in rep["removed"]:
                status, mean, std, sp, pi = "removed: " + rep["removed"][f], "", "", "", ""
            else:
                r = st.loc[f]
                status, mean, std = "kept", f"{r['mean']:.3g}", f"{r['std']:.3g}"
                sp, pi = f"{r['spearman_cost_h8']:+.2f}", f"{r['perm_importance_mae_ms']:.3f}"
            note = DESC.get(f, "rolling mean over the last N visits of this layer" if re.match(r"hist\d+_", f) else "")
            lines.append(f"| {g} | `{f}` | {cls} | {status} | {mean} | {std} | {sp} | {pi} | {note} |")
    lines += ["\n## B. Action\n", "`q` = experts fetched to the GPU for this layer call, q in {0..m} "
              "(logged as `n_fetch`, the count the kernel actually fetched).\n",
              "## C. Post-action outcomes (measured after the action)\n",
              "| column | meaning |", "|---|---|"]
    post = {"wall_ms": "layer wall time, synced at both ends", "wall_net_ms": "wall time minus the decision overhead",
            "cpu_task_ms": "native CPU-pool task time (T_CPU)", "pcie_ms": "CUDA-event time of the H2D expert copy (T_transfer)",
            "gemm_ms": "CUDA-event time of the GPU expert GEMM (hits + fetched together)",
            "pre_ms": "ensure-experts kernel + D2H of activations + CPU submit", "exposed_ms": "time the GPU stream waited for the CPU branch after its own work",
            "decide_us": "host time for features + policy", "n_evict": "experts evicted by this fetch", "occ_post": "cache occupancy after the action",
            "n_fetch": "q actually executed", "routed_ids": "routed expert ids", "missing_ids": "missing expert ids",
            "fetched_ids": "expert ids fetched", "evicted_gids": "global ids evicted"}
    lines += [f"| `{c}` | {post.get(c, '')} |" for c in POST_ACTION]
    lines += ["\n## D. Offline-only labels (built from future rows; analysis and training targets only)\n",
              "| label | meaning |", "|---|---|",
              "| `cost_h{0,1,4,8,16,32}` | this layer's net wall time now plus its next H visits |",
              "| `fut_hits_h*`, `fut_m_h*`, `fut_fetch_h*` | hits / misses / fetches of this layer over the next H tokens |",
              "| per-fetch `future_hits`, `residency_tok`, `first_reuse` | what each fetched expert bought before eviction (cache_analysis) |",
              "| single-step `q_oracle` | argmin over q of the measured wall(m, q) table (uniform-random logging) |",
              "\n## E. Unavailable signals (not measured; reason)\n", "| signal | why |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in UNAVAILABLE.items()]
    lines += ["\n## F. Counts\n", f"- candidate pre-action features: {sum(len(v) for v in GROUPS.values())}",
              f"- kept after constant / redundancy screening: {len(rep['kept'])}",
              f"- removed: {len(rep['removed'])} (constants on one machine+model, or |Spearman| > 0.98 with an earlier feature)",
              f"- decisions logged: {rep['rows']} ({rep['randomized_rows']} with a randomized action)"]
    target = RQ / "reports/features.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print("wrote", target)


if __name__ == "__main__":
    main()
