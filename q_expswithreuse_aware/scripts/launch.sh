#!/usr/bin/env bash
# Queue the policy-comparison study behind any running GPU work: smoke -> ablation -> analysis.
# One GPU job at a time; Judge0 is paused for the GPU part and restored on exit (standing
# permission, recorded per run). Usage: bash q_expswithreuse_aware/scripts/launch.sh [WAIT_PID]
# WAIT_PID: a queue to wait for first (default: none). Writes everything under
# q_expswithreuse_aware/results/{raw,logs}/<stamp>/.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
Q=q_expswithreuse_aware
RAW=$PWD/$Q/results/raw/$STAMP
LOG=$PWD/$Q/results/logs/$STAMP
MODEL=results/checkpoints/qwen3.6-35b-nvfp4
mkdir -p "$RAW" "$LOG"
exec > >(tee -a "$LOG/queue.log") 2>&1
J="judge0-v1131-workers-1 judge0-v1131-server-1 judge0-v1131-db-1 judge0-v1131-redis-1"
PAUSED="$LOG/judge0_paused.txt"
ts() { date -u +%FT%TZ; }
gpu_busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; }
restore() {
  [ -f "$PAUSED" ] && for c in $(cat "$PAUSED"); do docker unpause "$c" >/dev/null && echo "unpaused $c"; done
  rm -f "$PAUSED"; echo "judge0 restored $(ts)"
}
run() {  # run <plan> <decode>
  for attempt in 1 2 3; do
    echo "== $1 attempt $attempt start $(ts)"
    EXPERIMENT_OUTPUT_ROOT="$RAW" systemd-run --user --scope --quiet -p MemorySwapMax=0 \
      bash scripts/run.sh runtime_q run --plan "$Q/config/plans/$1.json" --model "$MODEL" --gpu 0 --decode "$2"
    rc=$?; echo "== $1 attempt $attempt rc=$rc $(ts)"
    [ $rc -eq 0 ] && return 0; sleep 20
  done
  return 1
}
analysis() { bash -c "source scripts/env.sh && experiment_python $*"; }

echo "launch $STAMP $(ts) git $(git rev-parse HEAD) wait_pid=${1:-none}"
sha256sum $Q/config/plans/*.json $Q/config/prompts_aime25.jsonl $Q/policy/qexp_hook.py \
  experiments/runtime_q/driver.py experiments/runtime_q/instrumentation/*.py runtime/patches/experiment-hooks.patch \
  > "$LOG/input_sha256.txt"
if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do echo "waiting for pid $1 $(ts)"; sleep 120; done
fi
while gpu_busy; do echo "waiting for GPU $(ts)"; sleep 60; done
trap restore EXIT
: > "$PAUSED"
for c in $J; do [ "$(docker inspect -f '{{.State.Paused}}' "$c" 2>/dev/null)" = "false" ] && docker pause "$c" >/dev/null && echo "$c" >> "$PAUSED"; done
echo "judge0 paused: $(tr '\n' ' ' < "$PAUSED") $(ts)"
free -g; nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv

run smoke_qwen_auto 32 || { echo "SMOKE FAILED $(ts)"; exit 2; }
SLOG=$(ls -t "$RAW"/runtime_q/logs/serve_auto_p0_*.log | head -1)
grep -q "moe_cache_auto=True" "$SLOG" || { echo "SMOKE: server did not resolve --moe-cache-auto ($SLOG)"; exit 3; }
grep -o '"moe_cache_size": [0-9]*' "$RAW/runtime_q/results/smoke_qwen_auto_episodes_servers.jsonl"
echo "SMOKE OK $(ts)"

run ablation_qwen_auto 256 || echo "ABLATION FAILED $(ts)"
echo "GPU PART DONE $(ts)"
restore; trap - EXIT

analysis $Q/analysis/analyze.py --root "$RAW" --out "$PWD/$Q/results/aggregated/$STAMP"
analysis $Q/analysis/integration_checks.py "$RAW/runtime_q/results/ablation_lean_qwen.csv" \
  --out "$PWD/$Q/results/aggregated/$STAMP/integration_checks.json" > /dev/null
analysis $Q/analysis/v1_timer_audit.py "$RAW/runtime_q/results/ablation_lean_qwen.csv" \
  --out "$PWD/$Q/results/aggregated/$STAMP/v1_timer_audit.json" > /dev/null
echo "QUEUE DONE $(ts)"
