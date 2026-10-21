#!/usr/bin/env bash
# saketh_exp_2 queue: per model (Qwen, gpt-oss, Gemma) smoke -> main policy run -> unhooked
# official-benchmark references (graph and eager), then analysis. One GPU job at a time.
# Judge0 is paused for the GPU part (standing permission) and restored on any exit.
# Usage: setsid bash saketh_exp_2/scripts/launch.sh   (STAMP and MODELS may be overridden)
# Writes saketh_exp_2/results/{raw,logs,aggregated}/<stamp>/.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
MODELS=${MODELS:-"qwen gpt_oss gemma"}
X=saketh_exp_2
RAW=$PWD/$X/results/raw/$STAMP
LOG=$PWD/$X/results/logs/$STAMP
AGG=$PWD/$X/results/aggregated/$STAMP
mkdir -p "$RAW/reference" "$LOG"
QLOG=$LOG/queue.log
exec >> "$QLOG" 2>&1   # no tee: a killed pipe must not abort the restore below
declare -A CK=([gpt_oss]=results/checkpoints/gpt-oss-20b [qwen]=results/checkpoints/qwen3.6-35b-nvfp4 [gemma]=results/checkpoints/gemma4-26b-nvfp4)
J="judge0-v1131-workers-1 judge0-v1131-server-1 judge0-v1131-db-1 judge0-v1131-redis-1"
PAUSED=$LOG/judge0_paused.txt
ts() { date -u +%FT%TZ; }
gpu_busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; }
restore() {
  if [ -f "$PAUSED" ]; then
    for c in $(cat "$PAUSED"); do docker unpause "$c" >/dev/null 2>&1 && echo "unpaused $c" || true; done
    rm -f "$PAUSED"
  fi
  echo "judge0 restored $(ts)" || true
}
trap restore EXIT
trap 'echo "signal received $(ts)"; exit 143' INT TERM

plan_env() {  # print `export K=V` for the plan's first group env (references use the same env)
  python3 -c "import json,shlex,sys; g=json.load(open(sys.argv[1]))['groups'][0]
for k,v in g['env'].items(): print(f'export {k}={shlex.quote(str(v))}')" "$1"
}
plan_serve() { python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['groups'][0]['serve_args']))" "$1"; }

run_plan() {  # run_plan <plan stem> <model> <decode>; the driver resumes completed episodes
  for attempt in 1 2 3; do
    echo "== $1 attempt $attempt start $(ts)"
    EXPERIMENT_OUTPUT_ROOT="$RAW" systemd-run --user --scope --quiet -p MemorySwapMax=0 \
      bash scripts/run.sh runtime_q run --plan "$X/config/plans/$1.json" --model "${CK[$2]}" --gpu 0 --decode "$3"
    rc=$?; echo "== $1 attempt $attempt rc=$rc $(ts)"
    [ $rc -eq 0 ] && return 0
    sleep 20
  done
  return 1
}

reference() {  # reference <model> <graph|eager> <problem> <repeat>: fresh server, official contract
  local plan=$X/config/plans/main_$1.json flag=""
  [ "$2" = eager ] && flag=--no-graph
  echo "== ref $1 $2 p$3 r$4 start $(ts)"
  ( eval "$(plan_env "$plan")"
    EXPERIMENT_OUTPUT_ROOT="$RAW" systemd-run --user --scope --quiet -p MemorySwapMax=0 \
      bash scripts/run.sh baselines bench_decode_moe --model "${CK[$1]}" --backend hybrid \
      --hybrid-fetch -1 --problem "$3" --decode 256 --greedy --aime "$X/config/prompts_aime25.jsonl" \
      --cache-type naive --gpu 0 --json "$RAW/reference/$1_$2.jsonl" $(plan_serve "$plan") $flag )
  echo "== ref $1 $2 p$3 r$4 rc=$? $(ts)"
}

echo "launch $STAMP $(ts) git $(git rev-parse HEAD) models=[$MODELS]"
sha256sum $X/config/plans/*.json $X/config/prompts_aime25.jsonl $X/policy/sx2_hook.py \
  experiments/runtime_q/driver.py experiments/runtime_q/instrumentation/*.py \
  experiments/baselines/bench_decode_moe.py runtime/patches/experiment-hooks.patch \
  results/reuse_aware_models/study_20261020T071921Z/calibration/benchbw.json > "$LOG/input_sha256.txt"
git status --short -- $X experiments/runtime_q experiments/baselines > "$LOG/git_status.txt"
while gpu_busy; do echo "waiting for GPU $(ts)"; sleep 60; done
: > "$PAUSED"
for c in $J; do
  [ "$(docker inspect -f '{{.State.Paused}}' "$c" 2>/dev/null)" = "false" ] && docker pause "$c" >/dev/null && echo "$c" >> "$PAUSED"
done
echo "judge0 paused: $(tr '\n' ' ' < "$PAUSED") $(ts)"
free -g; nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv

for M in $MODELS; do
  if ! run_plan "smoke_$M" "$M" 32; then echo "SMOKE FAILED $M $(ts)"; continue; fi
  SLOG=$(ls -t "$RAW"/runtime_q/logs/serve_auto_p0_*.log | head -1)
  grep -q "moe_cache_auto=True" "$SLOG" || { echo "SMOKE $M: --moe-cache-auto not resolved ($SLOG)"; continue; }
  python3 - "$RAW/runtime_q/results/smoke_lean_$M.csv" <<'EOF' || { echo "SMOKE $M: hook log check failed"; continue; }
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
checked = [r for r in rows if r["victim_match"] != ""]
bad = [r for r in checked if r["victim_match"] != "1"]
print(f"smoke log rows {len(rows)}, victim checks {len(checked)}, mismatches {len(bad)}")
sys.exit(1 if not rows or bad else 0)
EOF
  grep -o '"moe_cache_size": [0-9]*' "$RAW/runtime_q/results/smoke_${M}_episodes_servers.jsonl"
  echo "SMOKE OK $M $(ts)"
  run_plan "main_$M" "$M" 256 || echo "MAIN FAILED $M $(ts)"
  for p in 0 1 2; do for r in 1 2; do for mode in graph eager; do reference "$M" "$mode" "$p" "$r"; done; done; done
  echo "MODEL DONE $M $(ts)"
done
echo "GPU PART DONE $(ts)"
restore; trap - EXIT

bash -c "source scripts/env.sh && experiment_python $X/analysis/analyze.py --root '$RAW' --out '$AGG'" > "$LOG/analysis.log" 2>&1
echo "analysis rc=$? -> $AGG/RESULTS.md"
echo "QUEUE DONE $(ts)"
