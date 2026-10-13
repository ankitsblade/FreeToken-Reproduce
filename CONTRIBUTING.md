# Contributing to the experiment repository

This repository reproduces FreeToken research. The engine is an external dependency;
its original application contribution instructions are preserved in the ignored
engine backup. Keep the upstream attribution and [Apache-2.0 licence](LICENSE).

## Research changes

- Use `uv` for Python environments, dependencies and commands. Use the Bash entry points in `scripts/`.
- Put study code in the matching `experiments/` package and declarative plans in `configs/`.
- Keep raw measurements, failed attempts, checkpoints and per-run logs under ignored `results/`.
- Preserve completed measurements and sealed manifests. Record changed source, settings, hardware or data as a new arm.
- Update the single [experiment log](docs/EXPERIMENT_LOG.md) and the applicable maintained report. Unfinished work belongs in [the plan](docs/PLAN.md).
- Report hardware, driver, checkpoint ID/revision, runtime identity and exact commands with any new results.
- Run tests or benchmarks only when requested, and only report checks actually performed. Performance claims require comparable paired measurements on real hardware.

## AI policy

AI-assisted work is welcome when a human understands and owns the changes. Do not
submit work the contributor cannot explain or defend. Fully autonomous agents must
not contribute without human oversight. Never claim tests or benchmarks were run
when they were not.

Agents must not run `git push`, `gh pr create`, `gh pr comment`, or `gh issue create`
on the user's behalf. Commit only when the user asks. Use one focused change per
contribution and search existing upstream issues before proposing engine changes.
Upstream roadmap features require discussion with maintainers before implementation.

## Style and commits

Use familiar names and ASCII comments explaining a non-obvious reason. Keep comments
short; configuration files have no comments. Use Conventional Commits: one imperative,
lowercase line without a trailing period, for example `chore(experiments): organize study runners`.
If attribution is requested, use `Assisted-by`, not `Co-authored-by`.
