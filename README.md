# FreeToken experiments

An experiment repository for bandwidth, expert locality, CPU/GPU split policies,
runtime-aware q selection and MoE expert pruning. It combines Ankit's experiments
with Kushal's, Saketh's and Yeshwanth's work while preserving attribution and raw evidence.
FreeToken is an external runtime dependency. This repository does not package or
release the FreeToken application.

## Start

Linux x86_64 with an NVIDIA GPU is the reproduction target. All Python work uses `uv`.

```bash
bash scripts/setup.sh --analysis-only
bash scripts/doctor.sh
# For inference: install the pinned external runtime and its CUDA dependencies.
bash scripts/setup.sh
bash scripts/baseline.sh --profile gpt_oss_mxfp4 --model /path/to/gpt-oss-20b --gpu 0
```

See [reproduction instructions](docs/REPRODUCING.md) for checkpoint pins, other GPUs,
resource limits, baseline settings, study commands and recovery.

## Layout

```text
configs/       baseline profiles, pruning protocols and runtime-q plans
experiments/   study packages, shared collectors, analyses and baseline helpers
scripts/       Bash setup, diagnostics, baseline, run and frozen-runtime launchers
runtime/       dependency source pin, experiment hook patch and observed package versions
docs/          reports, presentations, one plan and one experiment log
results/       original measurements and new run directories (ignored)
archives/      verified ZIP backups and original imports (ignored)
.runtime/      external FreeToken source dependency (ignored)
```

[Documentation](docs/README.md) | [Experiment log](docs/EXPERIMENT_LOG.md) |
[Remaining plan](docs/PLAN.md) | [Contributing](CONTRIBUTING.md)

The 75% offline and actual-inference supplements are complete within their recorded
scope. Independent quality confirmation and the sequential 60%/50% ablations remain
open. No new performance numbers are implied by the repository reorganization.
