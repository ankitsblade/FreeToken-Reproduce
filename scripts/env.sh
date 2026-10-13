#!/usr/bin/env bash
export EXPERIMENT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FREETOKEN_SOURCE_DIR="${FREETOKEN_SOURCE_DIR:-$EXPERIMENT_REPO_ROOT/.runtime/freetoken}"
export UV_PROJECT_ENVIRONMENT="$EXPERIMENT_REPO_ROOT/.venv"
export PYTHONPATH="$FREETOKEN_SOURCE_DIR/python:$EXPERIMENT_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
if [[ -z ${CUDA_HOME:-} && -d "$EXPERIMENT_REPO_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13" ]]; then
  export CUDA_HOME="$EXPERIMENT_REPO_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13"
fi
if [[ -n ${CUDA_HOME:-} ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

experiment_python() {
  uv run --no-sync --project "$EXPERIMENT_REPO_ROOT" python "$@"
}
