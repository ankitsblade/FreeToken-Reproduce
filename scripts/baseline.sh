#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export EXPERIMENT_NO_SWAP="${EXPERIMENT_NO_SWAP:-1}"
preview=0
for arg in "$@"; do
  case "$arg" in -h|--help|--dry-run) preview=1 ;; esac
done
if [[ $EXPERIMENT_NO_SWAP == 1 && $preview == 0 ]]; then
  command -v systemd-run >/dev/null || { echo 'systemd-run is required for MemorySwapMax=0; set EXPERIMENT_NO_SWAP=0 to declare a different arm.' >&2; exit 1; }
  systemd-run --user --scope --quiet -p MemorySwapMax=0 \
    uv run --no-sync --project "$EXPERIMENT_REPO_ROOT" python -m experiments.common.launch baselines baseline "$@"
else
  experiment_python -m experiments.common.launch baselines baseline "$@"
fi
