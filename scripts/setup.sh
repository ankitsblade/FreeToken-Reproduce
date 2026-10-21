#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
analysis_only=0
while (($#)); do
  case "$1" in
    --analysis-only) analysis_only=1; shift ;;
    --source) export FREETOKEN_SOURCE_DIR="$(realpath "$2")"; shift 2 ;;
    -h|--help)
      echo 'Usage: bash scripts/setup.sh [--analysis-only] [--source /path/to/FreeToken]'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.' >&2; exit 1; }
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'This reproduction runtime is qualified only for Linux x86_64.' >&2
  exit 1
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
setup_log_root="$EXPERIMENT_REPO_ROOT/results/setup"
mkdir -p "$setup_log_root"
setup_log_dir="$(mktemp -d "$setup_log_root/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee -a "$setup_log_dir/setup.log") 2>&1
trap 'setup_exit_code=$?; printf "%s\n" "$setup_exit_code" > "$setup_log_dir/exit_code"' EXIT
echo "Setup log: $setup_log_dir/setup.log"
echo "uv downloads: HTTP read timeout ${UV_HTTP_TIMEOUT}s, retries $UV_HTTP_RETRIES, concurrency $UV_CONCURRENT_DOWNLOADS"
uv sync --project "$EXPERIMENT_REPO_ROOT" --locked --inexact
if ((analysis_only)); then
  echo 'Analysis environment ready. CUDA inference dependencies were not installed.'
  exit 0
fi
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver is missing.' >&2; exit 1; }
if [[ ! -f "$FREETOKEN_SOURCE_DIR/pyproject.toml" ]]; then
  if [[ $FREETOKEN_SOURCE_DIR != "$EXPERIMENT_REPO_ROOT/.runtime/freetoken" ]]; then
    echo 'The supplied external source does not contain pyproject.toml.' >&2
    exit 1
  fi
  experiment_python -m experiments.common.runtime restore
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -r "$EXPERIMENT_REPO_ROOT/runtime/observed_environment.txt"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export CC="${CC:-gcc}"
command -v nvcc >/dev/null || { echo 'CUDA 13 nvcc is required; set CUDA_HOME.' >&2; exit 1; }
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" setuptools wheel
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation --no-deps -e "$FREETOKEN_SOURCE_DIR"
bash "$EXPERIMENT_REPO_ROOT/scripts/doctor.sh" --require-runtime
