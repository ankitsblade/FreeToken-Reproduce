# Instructions for agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first; it is binding.

This is an experiment repository. Study packages live in `experiments/`, plans and
model profiles in `configs/`, Bash entry points in `scripts/`, and maintained research
documents in `docs/`. FreeToken source belongs only in ignored `.runtime/` or an
explicit external checkout. Do not restore application build/release folders at the root.

- Use `uv` for all Python work; use `scripts/env.sh` to resolve paths.
- Keep original results, manifests and frozen runtime sources immutable.
- Write fresh runs with `scripts/run.sh` or `scripts/baseline.sh`; keep their environment and status records.
- Keep ZIP backups ignored. Never remove raw evidence without a verified backup.
- Preserve each contributor's attribution and distinguish hardware, graph/eager settings and dataset scope.
- Do not spawn agents unless the user explicitly requests delegation or parallel agent work.
- Do not add or run tests unless the user asks. Never report tests/benchmarks as run when they were not.
- Do not push, create PRs, comment on PRs or create issues for the user. Commit only when asked.
- A human must understand and own every contribution; do not contribute autonomously without human oversight.

The single maintained research history is `docs/EXPERIMENT_LOG.md`; the single
unfinished-work plan is `docs/PLAN.md`. Check [reproduction instructions](docs/REPRODUCING.md)
before collecting new measurements or using sealed studies.
