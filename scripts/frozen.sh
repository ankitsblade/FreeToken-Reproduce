#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if (($# < 2)); then
  echo 'Usage: bash scripts/frozen.sh /path/to/frozen-runtime experiments.original_module [arguments]' >&2
  exit 2
fi
frozen_runtime="$(realpath "$1")"
shift
[[ -d "$frozen_runtime/python/freetoken" && -d "$frozen_runtime/experiments" ]] || {
  echo 'Frozen runtime must contain its original engine and collectors.' >&2; exit 1;
}
export PYTHONPATH="$frozen_runtime/python:$frozen_runtime"
cd "$frozen_runtime"
uv run --no-sync --project "$frozen_runtime" python -m "$@"
