# q* vs Reuse-Aware vs Runtime-Variable-Aware Reuse (FreeToken expert-execution policy)

Research question: can runtime information improve the original Reuse-Aware expert-execution
policy under FreeToken's original setup? The only variable is the per-layer decision `q`:
how many missing experts to fetch to the GPU, with the rest run on the CPU.

| Arm | Policy | Code (frozen unless noted) |
|---|---|---|
| A | paper q\* (Eq. 4, minimum one fill) | `reuse_aware_v2_hook.FractionMin1` |
| B | original Reuse-Aware, `R_j ≥ Δ(m, j)` | `runtime_q_hook.ReuseAware` |
| C_V1 … C_V1V2V3 | B + measured costs (V1) / calibrated reuse (V2) / eviction-aware reuse (V3), all 7 combinations | `reuse_aware_v2_hook.ReuseVariant` |
| A0, E (diagnostic) | q\* without minimum / engine built-in q\* | `runtime_q_hook.Fraction` / engine |

Start with [the audit](docs/AUDIT.md). It covers what the paper, the official code and the
existing experiments actually do, the provenance of every setting (including cache rate 0.10),
job status, validity, defects found, and the minimal corrected plan. Then read
[the policy specification](docs/POLICY_SPEC.md), the [reproducibility record](docs/REPRODUCIBILITY.md)
and the [final report](docs/FINAL_REPORT.md).

| Directory | Content |
|---|---|
| `config/` | `make_plan.py` (provenance of every setting), frozen prompt file, generated plans |
| `policy/` | `qexp_hook.py`: the study hook (frozen decision classes + validation counters + engine-exact V3 victims) |
| `scripts/` | `launch.sh`: queued smoke → ablation → analysis |
| `tests/` | `test_policies.py`: CPU-only policy and engine-equivalence tests |
| `analysis/` | `analyze.py` (paired ratios, 2³ factorial, diagnostics, plots), `integration_checks.py`, `v1_timer_audit.py` |
| `results/` | `raw/` and `logs/` (git-ignored raw evidence), `aggregated/` (tables, plots, generated reports) |
| `docs/` | audit, specification, reproducibility record, final and ablation reports |

No engine source, frozen hook, existing configuration or existing result was modified.
