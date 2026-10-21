# saketh_exp_2 — paper q\* vs original Reuse-Aware vs runtime-variable-aware Reuse-Aware

This study implements the specification in the repository-root `prompt.md`, redone in a fresh
directory at the user's request (2026-10-21). The only research variable is the
expert-execution decision q, the number of a layer's cache misses that are filled over PCIe
rather than run on the CPU.

| Path | Content |
|---|---|
| `docs/AUDIT.md` | Read-only audit: paper vs official code vs local code, provenance (incl. cache rate 0.10), job status, validity, plan |
| `docs/POLICY_SPEC.md` | Verified definitions of A, A0, E, B and V1/V2/V3 |
| `docs/REPRODUCIBILITY.md` | Exact inputs, commands, deviations and output locations |
| `config/make_plans.py` | Generates `config/plans/{smoke,main}_<model>.json` |
| `config/prompts_aime25.jsonl` | `math-ai/aime25@563bb8404243c5f09de6ec262f2db674fe5bce9b` `test.jsonl`, verbatim |
| `policy/sx2_hook.py` | Isolated hook; subclasses the frozen lean harness, engine-exact victims, per-episode counters |
| `tests/test_policies.py` | B ≡ C(all off), switch neutrality, minimum fill, fit fallback, kernel-vs-replica on GPU |
| `scripts/launch.sh` | Queue: smoke → main → unhooked references per model, then analysis |
| `analysis/analyze.py` | Paired speedups, 2³ factorial ablation, diagnostics, harness cost |
| `results/raw/<stamp>/` | Driver episodes, server logs, decision logs, hook counters, reference rows (ignored by Git) |
| `results/logs/<stamp>/` | Queue log, input hashes, Git status, analysis log |
| `results/aggregated/<stamp>/` | CSV tables, `summary.json`, `RESULTS.md` |
| `docs/FINAL_REPORT.md` | Results, ablation, limitations, supported claims (written after the run) |

No engine source, frozen hook, existing configuration or existing result was modified.
