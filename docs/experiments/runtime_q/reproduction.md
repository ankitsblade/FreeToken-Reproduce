# runtime_q — runtime-aware selection of FreeToken's CPU/GPU expert split

Commands execute from the repository root through the shared uv environment. Use a fresh output root for reproduction; preserve completed artifacts.

Extension of `saketh_exps/` (the q\* vs fixed-q study, which is untouched). Question: can
the per-layer fetch count `q` be chosen better than the bandwidth rule
`q* ≈ m · B_P / B_H` by using runtime state?

Everything here ran on one configuration: **gpt-oss-20b (MXFP4), RTX 4070 12 GB (PCIe gen4
x4), Ryzen 5 8500G, 30 GB RAM, hybrid backend, batch 1, greedy decode**. Nothing in this
directory has been validated on another GPU, CPU, PCIe link or model.

## Result in one paragraph

q\* is the correct answer to a single-step question (it equals the measured single-step
optimum, q = 0 on this host) but it never fetches, so the GPU expert cache stays empty and
every expert is recomputed on the CPU forever. Adding one term — the expected number of
times a fetched expert will be reused while it stays resident — gives an analytic rule that
is **1.52× faster than q\* (geometric mean over three cache sizes, worst case 1.16×)**,
1.70× on an unseen prompt and unseen cache sizes, and costs under 1 µs to evaluate. Learned
policies (linear, ridge, tree, GBM, GELU MLP, LinUCB, Thompson sampling) do not beat it and
generalize worse. Read [report.md](report.md) for the argument and [tables.md](tables.md) for the measured numbers.

## The rule

```
rho      = B_P / B_H                    (= t_cpu / t_pcie per expert; 0.188 here)
delta(m,j) = max((j+1)/rho, m-j-1) - max(j/rho, m-j)     exposed cost of the (j+1)-th fetch,
                                                          in CPU-expert units
R        = p * T_res                    expected reuses of a fetched expert
           p     = top_k / E            (or the expert's recent routing frequency)
           T_res = T_free tokens if the cache has a free slot, else the LRU victim's age
q        = number of consecutive j = 0,1,... with  R >= kappa * delta(m,j)      (kappa = 1)
```

With `R = 0` the rule selects q=0 on the measured slow-PCIe configuration, matching
the recorded local q* decision. This does not establish algebraic equivalence on every
hardware balance. In the stated PCIe-bound branch it reads
"fetch one more while `(m - j) + R >= (j + 1) · B_H / B_P`". Implemented as
`ReuseAware` in `instrumentation/runtime_q_hook.py`.

## Layout

| path | contents |
|---|---|
| `instrumentation/runtime_q_hook.py` | host-side hook: feature extraction, all policies, per-decision logging |
| `instrumentation/schema.py` | feature groups G1–G10, pre/post-action split, unavailable signals |
| `instrumentation/policy_models.py` | numpy-only cost models shared by training and runtime |
| `scripts/driver.py` | runs episodes against real `ft serve` processes (burn-in, rotated repeats, resumable) |
| `scripts/build_dataset.py` | raw logs → cleaned dataset, offline labels, splits |
| `scripts/feature_analysis.py`, `feature_inventory.py` | feature statistics, correlations, MI, VIF, permutation importance |
| `scripts/cache_analysis.py` | what each fetched expert bought (reuse, residency, net time) |
| `scripts/train_all.py`, `qlib.py` | models and every ablation family |
| `scripts/analyze_eval.py`, `figures.py` | tables and the 28 figures |
| `config/*.json` | every run plan (policies, seeds, cache sizes) |
| `data/raw/`, `data/clean/` | per-decision logs; cleaned dataset + `feature_report.json` |
| `models/` | runtime artifacts for every trained configuration |
| `results/` | `eval/` (main, generalization, stress, long), `ablations/`, `overhead/`, `feature_analysis/`, `cache_analysis/`, `statistical/`, `ann/`, `pilot/` |
| `figures/` | `01`–`28` |
| `reports/` | `reproducibility.json`; Markdown tables/features are under `docs/experiments/runtime_q/` |
| `docs/EXPERIMENT_LOG.md` | what was run, in order, including problems hit |
| `docs/experiments/runtime_q/report.md` | findings, answers to the research questions, limitations |

## How it hooks into FreeToken

One opt-in loader was added to `python/freetoken/moe/offload_cache.py`: if
`FREETOKEN_MOE_RUNTIME_Q_MODULE` names a file, its `build(cache)` object replaces the
decode trace. Without that variable the engine is unchanged. The hook uses call sites that
already existed (`decode_trace.start_layer/begin/finish_layer`, the `_layer_replay_events`
CUDA events, the native CPU-task timer). It requires eager decode (`--no-graph`), because
the decision needs the routed ids and cache state on the host.

Decision point, per MoE layer per token: routing → cache lookup → missing set `M`, `m` →
**features computed** → policy picks `q ∈ {0..m}` → `q` experts fetched, `m − q` on the CPU,
concurrently → outcomes measured. Features use only the current routing/cache state and
measurements of earlier layer calls. Future quantities exist only as offline labels.

## Reproduce

Run from the repository root after full setup. Set MODEL to the original compatible
GPT-OSS checkpoint representation and use a fresh output root. These commands retain
the original study's plan order and filenames; each driver call has a separate run record.
The original raw/processed artifacts remain in `results/historical/runtime_q`.

```bash
source scripts/env.sh
export MODEL=/path/to/gpt-oss-20b-ftw
export EXPERIMENT_OUTPUT_ROOT="$EXPERIMENT_REPO_ROOT/results/reproductions/runtime_q_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EXPERIMENT_OUTPUT_ROOT/runtime_q"
export RQ="$EXPERIMENT_OUTPUT_ROOT/runtime_q"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/pilot_plan.json --out "$RQ/results/pilot/pilot_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/collect_plan.json --out "$RQ/data/raw/explore_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/round2_plan.json --out "$RQ/data/raw/round2_episodes.jsonl"
bash scripts/run.sh runtime_q build_dataset
bash scripts/run.sh runtime_q feature_analysis
bash scripts/run.sh runtime_q cache_analysis
for h in 8 16 32; do
  bash scripts/run.sh runtime_q train --family main --h "$h" --tag "_r2_h$h"
done
bash scripts/run.sh runtime_q train --family horizon
bash scripts/run.sh runtime_q train --family chain logo indiv cache history --kinds ridge --h 8 --tag _ridge
bash scripts/run.sh runtime_q train --family ann --h 16
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/main_plan.json --out "$RQ/results/eval/main_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/ablation_plan.json --out "$RQ/results/ablations/e2e_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/ablation_rep_plan.json --out "$RQ/results/ablations/e2e_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/overhead_plan.json --out "$RQ/results/overhead/overhead_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/generalization_plan.json --out "$RQ/results/eval/generalization_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/stress_plan.json --out "$RQ/results/eval/stress_episodes.jsonl"
bash scripts/run.sh runtime_q run --model "$MODEL" --plan configs/runtime_q/long_plan.json --decode 384 --out "$RQ/results/eval/long_episodes.jsonl"
bash scripts/run.sh runtime_q analyze
bash scripts/run.sh runtime_q feature_inventory
bash scripts/run.sh runtime_q plot
bash scripts/run.sh runtime_q make_report_pdf
```

Policy-model paths in plans are relative to the runtime-q study directory. Missing
artifacts are a blocker, not an invitation to substitute a different model. For new
devices, qualify the pilot before the full matrix; keep eager, logging and graph
measurements in distinct arms. See [the shared reproduction guide](../../REPRODUCING.md).

## Reading the numbers

All latencies are client-side ms/token through the real server, in eager mode with the hook
active for every policy including q\*. That costs about 19 ms/token against the CUDA-graph
numbers in `docs/experiments/split_policy/report.md` (q\* is 65 ms/token here, 41 there), so compare
policies within this harness, not across the two.
